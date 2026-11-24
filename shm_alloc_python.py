import ctypes
import os
import zlib
import torch

# libshm_alloc.so is now built (via build_dmalloc_lib.sh) from the ROOT
# repo's v0.3 dmalloc engine (src/shm_slim.c, src/dmalloc.c, src/shm_ns.c),
# not from allocator/'s own v2 sources (src/shm_alloc.c). The load path is
# unchanged; only what produces the .so changed.
lib_path = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "./allocator/build/libshm_alloc.so",
    )
)
lib = ctypes.CDLL(lib_path)

# v0.3's dmalloc.h has no per-object permission model (see dmalloc.h's file
# header, point 6: v2's SHM_PERM_ADMIN-style bypass was deliberately
# dropped) and no separate open-time "dir_capacity" concept -- the name
# table sizes itself via DMALLOC_NAME_CAPACITY (an engine env var, defaults
# fine for these workloads). These constants and the `perms`/`dir_capacity`
# constructor arguments are kept only so callers that still pass them
# (benchmark/common/shm_weights.py, tests/writer.py) don't need to change.
SHM_PERM_DEFAULT = 0x0F
SHM_PERM_READ_ONLY = 0x01

D_CREATE = 0x01

# All processes attaching to the same region must present the exact same
# key -- d_attach() has no admin-bypass (unlike v2's SHM_PERM_ADMIN). A
# single fixed key is enough here since every workload in this repo is one
# writer + one-or-more readers all trusting each other, never multi-tenant
# isolation between mutually distrustful principals.
_DMALLOC_KEY = 1

lib.d_init.argtypes = [ctypes.c_int, ctypes.c_uint32, ctypes.c_uint32]
lib.d_init.restype = ctypes.c_void_p

lib.d_publish.argtypes = [
    ctypes.c_void_p,
    ctypes.c_char_p,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_uint32,
]
lib.d_publish.restype = ctypes.c_bool

lib.d_attach.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32]
lib.d_attach.restype = ctypes.c_void_p

lib.d_detach.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
lib.d_detach.restype = ctypes.c_bool

lib.d_deinit.argtypes = [ctypes.c_void_p]
lib.d_deinit.restype = None

# v0.3's engine caps names at include/shm_slim.h's SHM_SLIM_NAME_MAX (24
# bytes incl. NUL) and shm_slim_alloc() REJECTS (ENAMETOOLONG), rather than
# truncates, anything longer -- confirmed directly, not assumed. PyTorch
# state_dict keys routinely exceed this (e.g.
# "layers.0.input_layernorm.weight" is 32 chars). Rather than widening the
# engine's fixed on-disk name field (a layout change with no smaller-scoped
# alternative would have to justify touching every other v0.3 consumer),
# names at or above the limit are mapped to a short, deterministic alias
# here. CRC32 is used (not Python's str hash, which is randomized per
# process by PYTHONHASHSEED and would produce a different alias in the
# writer vs. the reader) -- a 32-bit collision across the handful of names
# any one region holds is not a realistic risk.
_SHM_SLIM_NAME_MAX = 24


def _engine_name(name):
    if len(name) < _SHM_SLIM_NAME_MAX:
        return name
    return f"h{zlib.crc32(name.encode('utf-8')):08x}"


_DTYPE_TO_NUMPY = None


def _numpy_dtype(dtype):
    global _DTYPE_TO_NUMPY
    if _DTYPE_TO_NUMPY is None:
        import numpy as np

        _DTYPE_TO_NUMPY = {
            torch.float32: np.float32,
            torch.float64: np.float64,
            torch.float16: np.float16,
            torch.bfloat16: np.float16,
            torch.int32: np.int32,
            torch.int64: np.int64,
            torch.int8: np.int8,
            torch.uint8: np.uint8,
        }
    if dtype not in _DTYPE_TO_NUMPY:
        raise TypeError(f"Unsupported dtype for shared memory tensor: {dtype}")
    return _DTYPE_TO_NUMPY[dtype]


class SHMRegion:
    def __init__(
        self,
        name="/pytorch_shm_pool",
        size_bytes=16 * 1024 * 1024,
        create=True,
        uid=1000,
        use_dax=False,
        dir_capacity=512,
    ):
        self.name = name
        self.size_bytes = size_bytes
        self.uid = uid
        self.create = create
        self.use_dax = use_dax
        self._name_to_id = {}

        if use_dax:
            self.fd = os.open(name, os.O_RDWR)
        else:
            # glibc's shm_open() is just open() against tmpfs mounted at
            # /dev/shm, with the leading '/' stripped from the name -- doing
            # it directly here avoids a ctypes binding for shm_open/shm_unlink.
            self._shm_path = "/dev/shm/" + name.lstrip("/")
            if not create:
                # Writer and reader are typically launched back-to-back (the
                # writer backgrounded, the reader started right after) -- wait
                # for the file to exist AND be truncated to its real size
                # before attaching, instead of racing shm_open()+ftruncate().
                import time as _time

                deadline = _time.monotonic() + 10.0
                while True:
                    try:
                        if os.stat(self._shm_path).st_size >= 4096:
                            break
                    except FileNotFoundError:
                        pass
                    if _time.monotonic() > deadline:
                        break
                    _time.sleep(0.05)
            flags = os.O_RDWR | (os.O_CREAT if create else 0)
            self.fd = os.open(self._shm_path, flags, 0o660)
            if create:
                os.ftruncate(self.fd, size_bytes)

        flag = D_CREATE if create else 0
        self.handle = lib.d_init(self.fd, flag, _DMALLOC_KEY)
        if not self.handle and not create:
            # Narrow remaining window: the file was truncated to size but the
            # creator's d_init(D_CREATE) hadn't finished writing the region
            # header (magic/version) yet. Retry briefly rather than fail.
            import time as _time

            deadline = _time.monotonic() + 5.0
            while not self.handle and _time.monotonic() < deadline:
                _time.sleep(0.05)
                self.handle = lib.d_init(self.fd, flag, _DMALLOC_KEY)
        if not self.handle:
            raise RuntimeError(f"d_init failed for region {name}")

    def alloc(self, name, size_bytes, perms=SHM_PERM_DEFAULT):
        # d_publish() copies an existing local buffer in -- there is no
        # "reserve N empty bytes under this name" primitive in v0.3. A
        # zero-filled local buffer of the right size is published instead,
        # then immediately attached; callers overwrite it via get_tensor()
        # right after, exactly as they already do for the id/offset API.
        engine_name = _engine_name(name)
        zero_buf = ctypes.create_string_buffer(size_bytes)
        ok = lib.d_publish(self.handle, engine_name.encode("utf-8"), zero_buf, size_bytes, _DMALLOC_KEY)
        if not ok:
            raise RuntimeError(f"d_publish failed for name {name} (engine name {engine_name})")
        ptr = lib.d_attach(self.handle, engine_name.encode("utf-8"), 0)
        if not ptr:
            raise RuntimeError(f"d_attach (post-publish) failed for name {name}")
        obj_id = len(self._name_to_id) + 1
        self._name_to_id[name] = ptr
        return obj_id, ptr

    def lookup_by_name(self, name):
        engine_name = _engine_name(name)
        ptr = lib.d_attach(self.handle, engine_name.encode("utf-8"), 0)
        if not ptr:
            raise RuntimeError(f"d_attach failed for name {name} (engine name {engine_name})")
        obj_id = self._name_to_id.setdefault(name, ptr)
        return obj_id, ptr

    def resize(self, obj_id, new_size_bytes, perms=SHM_PERM_DEFAULT):
        raise NotImplementedError(
            "v0.3's dmalloc engine has no resize primitive (d_publish/d_attach "
            "only) -- PT-KVCache's in-place growth cannot be ported as-is. "
            "See WIP.md."
        )

    def free(self, obj_id, perms=SHM_PERM_DEFAULT):
        raise NotImplementedError(
            "not used by any workload in this repo; d_delete() exists on the "
            "engine but there's nothing exercising a by-id free path here."
        )

    def get_tensor(self, offset, shape, dtype, writable=True):
        # `offset` is the absolute pointer returned by alloc()/lookup_by_name()
        # above (d_attach() resolves names straight to a local pointer -- v0.3
        # has no separate offset-to-pointer step the way v2's shm_ptr() did).
        ptr_val = offset

        element_size = torch._utils._element_size(dtype)
        num_elements = 1
        for dim in shape:
            num_elements *= dim
        nbytes = num_elements * element_size

        import numpy as np

        np_dtype = _numpy_dtype(dtype)
        buf = (ctypes.c_byte * nbytes).from_address(ptr_val)
        arr = np.frombuffer(buf, dtype=np_dtype)
        tensor = torch.from_numpy(arr).view(shape)
        if dtype == torch.bfloat16:
            tensor = tensor.view(torch.bfloat16)
        return tensor

    def close(self, unlink=False):
        if getattr(self, "handle", None):
            lib.d_deinit(self.handle)
            self.handle = None
        if getattr(self, "fd", None) is not None:
            os.close(self.fd)
            self.fd = None
        if unlink and not self.use_dax and hasattr(self, "_shm_path"):
            try:
                os.unlink(self._shm_path)
            except FileNotFoundError:
                pass
