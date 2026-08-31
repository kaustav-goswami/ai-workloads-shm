import ctypes
import os
import torch

lib_path = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "./allocator/build/libshm_alloc.so",
    )
)
lib = ctypes.CDLL(lib_path)

shm_off_t = ctypes.c_uint64
shm_perm_t = ctypes.c_uint32
shm_user_id_t = ctypes.c_uint32

SHM_PERM_NONE = 0x00
SHM_PERM_READ = 0x01
SHM_PERM_WRITE = 0x02
SHM_PERM_FREE = 0x04
SHM_PERM_RESIZE = 0x08
SHM_PERM_ADMIN = 0x80
SHM_PERM_DEFAULT = SHM_PERM_READ | SHM_PERM_WRITE | SHM_PERM_FREE | SHM_PERM_RESIZE
SHM_PERM_READ_ONLY = SHM_PERM_READ

SHM_BACKEND_POSIX = 0
SHM_BACKEND_DAX = 1

SHM_OPEN_CREATE = 0x01
SHM_OPEN_ENFORCE_NS = 0x02
SHM_OPEN_ENFORCE_CGROUP = 0x04


class ShmRegionOpenOpts(ctypes.Structure):
    # Must mirror shm_region_open_opts_t in shm_alloc.h field-for-field.
    # A missing trailing field here means shm_region_open() reads past the
    # end of this (correctly-sized-for-Python, too-small-for-C) struct when
    # it accesses opts->guard_pages - undefined behavior that usually reads
    # zero but occasionally reads adjacent heap garbage, producing bogus
    # "region too small for N guard pages" failures that appear only rarely.
    _fields_ = [
        ("backend", ctypes.c_int),
        ("flags", ctypes.c_uint32),
        ("dir_capacity", ctypes.c_uint32),
        ("mode", ctypes.c_uint32),
        ("guard_pages", ctypes.c_uint32),
    ]


class shm_region_t(ctypes.Structure):
    pass


shm_region_p = ctypes.POINTER(shm_region_t)

lib.shm_region_open.argtypes = [
    ctypes.c_char_p,
    ctypes.c_size_t,
    ctypes.POINTER(ShmRegionOpenOpts),
    ctypes.POINTER(shm_region_p),
]
lib.shm_region_open.restype = ctypes.c_int

lib.shm_region_close.argtypes = [shm_region_p, ctypes.c_bool]
lib.shm_region_close.restype = None

lib.shm_alloc.argtypes = [
    shm_region_p,
    ctypes.c_size_t,
    ctypes.c_char_p,
    shm_user_id_t,
    shm_perm_t,
    ctypes.c_uint32,
    ctypes.POINTER(ctypes.c_uint64),
    ctypes.POINTER(shm_off_t),
]
lib.shm_alloc.restype = ctypes.c_int

lib.shm_lookup_by_name.argtypes = [
    shm_region_p,
    ctypes.c_char_p,
    shm_user_id_t,
    shm_perm_t,
    ctypes.POINTER(ctypes.c_uint64),
    ctypes.POINTER(shm_off_t),
]
lib.shm_lookup_by_name.restype = ctypes.c_int

lib.shm_lookup.argtypes = [
    shm_region_p,
    ctypes.c_uint64,
    shm_user_id_t,
    shm_perm_t,
    ctypes.POINTER(shm_off_t),
    ctypes.POINTER(ctypes.c_size_t),
]
lib.shm_lookup.restype = ctypes.c_int

lib.shm_resize.argtypes = [
    shm_region_p,
    ctypes.c_uint64,
    ctypes.c_size_t,
    shm_user_id_t,
    shm_perm_t,
]
lib.shm_resize.restype = ctypes.c_int

lib.shm_free.argtypes = [
    shm_region_p,
    ctypes.c_uint64,
    shm_user_id_t,
    shm_perm_t,
]
lib.shm_free.restype = ctypes.c_int

lib.shm_ptr.argtypes = [
    shm_region_p,
    shm_off_t,
    shm_user_id_t,
    shm_perm_t,
    shm_perm_t,
]
lib.shm_ptr.restype = ctypes.c_void_p

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
        self.name = name.encode("utf-8")
        self.size_bytes = size_bytes
        self.uid = uid
        self.create = create
        self.use_dax = use_dax

        opts = ShmRegionOpenOpts()
        opts.backend = SHM_BACKEND_DAX if use_dax else SHM_BACKEND_POSIX
        opts.flags = SHM_OPEN_CREATE if create else 0
        opts.dir_capacity = dir_capacity
        opts.mode = 0o660

        self.region_ptr = shm_region_p()
        rc = lib.shm_region_open(
            self.name, self.size_bytes, ctypes.byref(opts), ctypes.byref(self.region_ptr)
        )
        if rc != 0:
            raise RuntimeError(f"Failed to open shm region, code {rc}")

    def alloc(self, name, size_bytes, perms=SHM_PERM_DEFAULT):
        out_id = ctypes.c_uint64(0)
        out_off = shm_off_t(0)
        rc = lib.shm_alloc(
            self.region_ptr,
            size_bytes,
            name.encode("utf-8"),
            self.uid,
            perms,
            0,
            ctypes.byref(out_id),
            ctypes.byref(out_off),
        )
        if rc != 0:
            raise RuntimeError(f"shm_alloc failed for name {name}, code {rc}")
        return out_id.value, out_off.value

    def lookup_by_name(self, name):
        out_id = ctypes.c_uint64(0)
        out_off = shm_off_t(0)
        rc = lib.shm_lookup_by_name(
            self.region_ptr,
            name.encode("utf-8"),
            self.uid,
            SHM_PERM_READ,
            ctypes.byref(out_id),
            ctypes.byref(out_off),
        )
        if rc != 0:
            raise RuntimeError(f"shm_lookup_by_name failed for name {name}, code {rc}")
        return out_id.value, out_off.value

    def lookup(self, obj_id):
        out_off = shm_off_t(0)
        out_used_size = ctypes.c_size_t(0)
        rc = lib.shm_lookup(
            self.region_ptr,
            obj_id,
            self.uid,
            SHM_PERM_READ,
            ctypes.byref(out_off),
            ctypes.byref(out_used_size),
        )
        if rc != 0:
            raise RuntimeError(f"shm_lookup failed for id {obj_id}, code {rc}")
        return out_off.value, out_used_size.value

    def resize(self, obj_id, new_size_bytes, perms=SHM_PERM_DEFAULT):
        rc = lib.shm_resize(self.region_ptr, obj_id, new_size_bytes, self.uid, perms)
        if rc != 0:
            raise RuntimeError(f"shm_resize failed for id {obj_id}, code {rc}")
        return self.lookup(obj_id)

    def free(self, obj_id, perms=SHM_PERM_DEFAULT):
        rc = lib.shm_free(self.region_ptr, obj_id, self.uid, perms)
        if rc != 0:
            raise RuntimeError(f"shm_free failed for id {obj_id}, code {rc}")

    def get_tensor(self, offset, shape, dtype, writable=True):
        request_perm = SHM_PERM_READ | SHM_PERM_WRITE if writable else SHM_PERM_READ
        ptr_val = lib.shm_ptr(
            self.region_ptr,
            offset,
            self.uid,
            request_perm,
            SHM_PERM_READ,
        )
        if not ptr_val:
            raise RuntimeError("Failed to resolve shm pointer")

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
        if hasattr(self, "region_ptr") and self.region_ptr:
            should_unlink = unlink if not self.use_dax else False
            lib.shm_region_close(self.region_ptr, should_unlink)
            self.region_ptr = None
