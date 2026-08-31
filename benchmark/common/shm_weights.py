import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shm_alloc_python import SHMRegion

SYNC_NAME = "sync_ready"
SYNC_WRITER_READY = 1
SYNC_READER_DONE = 2


def add_backend_args(parser):
    parser.add_argument(
        "--backend",
        choices=["shm", "dax"],
        default="shm",
        help="Memory backend: POSIX shared memory or DAX device",
    )
    parser.add_argument(
        "--path",
        default="/pytorch_model_weights_shm",
        help="POSIX shm name or DAX device path",
    )
    parser.add_argument(
        "--region-size-gb",
        type=float,
        default=2.0,
        help="Shared region size in GB (POSIX SHM only)",
    )
    return parser


def backend_from_args(args):
    return args.backend == "dax", args.path


def region_size_bytes(args):
    return int(args.region_size_gb * 1024 * 1024 * 1024)


def tensor_nbytes(tensor):
    return tensor.numel() * tensor.element_size()


def total_weight_bytes(state_dict):
    return sum(tensor_nbytes(tensor) for tensor in state_dict.values())


def allocate_sync(region):
    _, sync_offset = region.alloc(SYNC_NAME, 8)
    sync_tensor = region.get_tensor(sync_offset, [1], torch.int64, writable=True)
    sync_tensor[0] = 0
    return sync_tensor


def wait_for_writer(sync_tensor, poll_seconds=0.1):
    while sync_tensor[0].item() != SYNC_WRITER_READY:
        time.sleep(poll_seconds)


def signal_writer_ready(sync_tensor):
    sync_tensor[0] = SYNC_WRITER_READY


def signal_reader_done(sync_tensor):
    sync_tensor[0] = SYNC_READER_DONE


def wait_for_reader(sync_tensor, poll_seconds=0.5):
    while sync_tensor[0].item() != SYNC_READER_DONE:
        time.sleep(poll_seconds)


def alloc_and_write_tensor(region, name, tensor, recorder=None, perms=None):
    """Allocate a named SHM object sized for `tensor` and copy it in.

    Returns (obj_id, offset). If `recorder` is a TimingRecorder, records an
    "alloc" sample (alloc + get_tensor) and a "copy" sample (the actual
    memcpy-equivalent) tagged with `name`.
    """
    shape = list(tensor.shape)
    dtype = tensor.dtype
    nbytes = tensor_nbytes(tensor)

    if recorder is not None:
        with recorder.time("alloc", name):
            kwargs = {} if perms is None else {"perms": perms}
            obj_id, offset = region.alloc(name, nbytes, **kwargs)
            shm_tensor = region.get_tensor(offset, shape, dtype, writable=True)
        with recorder.time("copy", name):
            shm_tensor.copy_(tensor)
    else:
        kwargs = {} if perms is None else {"perms": perms}
        obj_id, offset = region.alloc(name, nbytes, **kwargs)
        shm_tensor = region.get_tensor(offset, shape, dtype, writable=True)
        shm_tensor.copy_(tensor)
    return obj_id, offset


def lookup_and_attach_tensor(region, name, shape, dtype, recorder=None, writable=False):
    """Look up a named SHM object and attach it as a (zero-copy) tensor.

    If `recorder` is a TimingRecorder, records a "lookup" sample (name
    resolution + pointer resolution) tagged with `name`.
    """
    if recorder is not None:
        with recorder.time("lookup", name):
            _, offset = region.lookup_by_name(name)
            tensor = region.get_tensor(offset, list(shape), dtype, writable=writable)
    else:
        _, offset = region.lookup_by_name(name)
        tensor = region.get_tensor(offset, list(shape), dtype, writable=writable)
    return tensor


def copy_state_dict_to_shm(region, state_dict, recorder=None):
    for name, tensor in state_dict.items():
        alloc_and_write_tensor(region, name, tensor, recorder=recorder)


def attach_state_dict_from_shm(region, state_dict_template, recorder=None):
    attached = {}
    for name, template in state_dict_template.items():
        attached[name] = lookup_and_attach_tensor(
            region, name, template.shape, template.dtype, recorder=recorder, writable=False
        )
    return attached


def open_writer_region(args, weight_bytes, extra_bytes=64 * 1024 * 1024):
    use_dax, path = backend_from_args(args)
    size_bytes = max(region_size_bytes(args), weight_bytes + extra_bytes)
    region = SHMRegion(
        name=path,
        size_bytes=size_bytes,
        create=True,
        use_dax=use_dax,
        dir_capacity=1024,
    )
    return region, use_dax, path


def open_reader_region(args):
    use_dax, path = backend_from_args(args)
    region = SHMRegion(name=path, create=False, use_dax=use_dax, dir_capacity=1024)
    return region, use_dax, path
