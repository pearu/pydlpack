"""Expose array and buffer objects to DLPack consumers.
"""

__all__ = ['asdlpack']

from .buffer import Buffer, PyBUF_RECORDS_RO
from .capsule import Capsule
import atexit
import ctypes
import warnings

DLPACK_CAPSULE_VALID_NAME = b"dltensor"


class DLStructure(ctypes.Structure):
    
    def __repr__(self):
        args = []
        for field, typ in self._fields_:
            value = getattr(self, field)
            if typ.__name__.startswith('LP_') and hasattr(self, "ndim"):
                if ctypes.cast(value, ctypes.c_void_p).value is None:
                    value = None
                else:
                    value = tuple(value[i] for i in range(self.ndim))
            args.append(f'{field}={value}')
        args = ", ".join(args)
        return f'{type(self).__name__}({args})'

    __str__ = __repr__

    def todict(self):
        d = {}
        for field, typ in self._fields_:
            value = getattr(self, field)
            if typ.__name__.startswith('LP_') and hasattr(self, "ndim"):
                if ctypes.cast(value, ctypes.c_void_p).value is None:
                    value = None
                else:
                    value = tuple(value[i] for i in range(self.ndim))
            elif isinstance(value, DLStructure):
                value = value.todict()
            elif isinstance(value, (DLDeviceType, DLDataTypeCode)):
                value = value.label
            d[field] = value
        return d


class DLDeviceType(ctypes.c_int32):

    @property
    def label(self):
        return {1:'DLCPU', 2:'DLCUDA', 3:'DLCUDAHost', 4:'DLOpenCL', 7:'DLVulkan', 8: 'DLMetal', 9: 'DLVPI',
                10: 'DLROCM', 11: 'DLROCMHost', 12: 'DLExtDev', 13: 'DLCUDAManaged'}[self.value]

    @classmethod
    def from_label(cls, label):
        return cls(dict(DLCPU=1, DLCUDA=2, DLCUDAHost=3, DLOpenCL=4, DLVulkan=7, DLMetal=8, DLVPI=9, DLROCM=10, DLROCMHost=11, DLExtDev=12, DLCUDAManaged=13)[label])

    def __repr__(self):
        return f'{type(self).__name__}({repr(self.label)})'


class DLDataTypeCode(ctypes.c_uint8):

    @property
    def label(self):
        return {0:'DLInt', 1:'DLUInt', 2: 'DLFloat', 3: 'DLOpaqueHandle', 4: 'DLBfloat', 5: 'DLComplex', 6: 'DLBool'}[self.value]

    @classmethod
    def from_label(cls, label):
        return cls(dict(DLInt=0, DLUInt=1, DLFloat=2, DLOpaqueHandle=3, DLBfloat=4, DLComplex=5, DLBool=6)[label])

    def __repr__(self):
        return f'{type(self).__name__}({repr(self.label)})'


class DLDevice(DLStructure):
    """A Device for Tensor and operator."""

    _fields_ = [
        ("device_type", DLDeviceType),
        ("device_id", ctypes.c_int32)
    ]


class DLDataType(DLStructure):
    """The data type the tensor can hold."""

    _fields_ = [
        ("code", DLDataTypeCode),
        ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
    ]

    
class DLTensor(DLStructure):
    """Plain C Tensor object, does not manage memory."""

    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", DLDevice),
        ("ndim", ctypes.c_int32),
        ("dtype", DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64)
    ]


class DLPackVersion(DLStructure):

    _fields_ = [
        ("major", ctypes.c_uint32),
        ("minor", ctypes.c_uint32),
    ]


class DLManagedTensorVersioned(DLStructure):
    pass


DLManagedTensorVersioned._fields_ = [
        ("version", DLPackVersion),
        ("manager_ctx", ctypes.c_void_p),
        ("deleter", ctypes.CFUNCTYPE(None, ctypes.POINTER(DLManagedTensorVersioned))),
        ("flags", ctypes.c_uint64),
        ("dl_tensor", DLTensor),
    ]

    
class DLManagedTensor(DLStructure):
    """C Tensor object, manage memory of DLTensor. This data structure is
    intended to facilitate the borrowing of DLTensor by another framework.

    Note: This data structure is used as Legacy DLManagedTensor in
    DLPack exchange and is deprecated after DLPack v0.8 Use
    DLManagedTensorVersioned instead.
    """

DLManagedTensor._fields_ = [
        ("dl_tensor", DLTensor),
        ("manager_ctx", ctypes.c_void_p),
        ("deleter", ctypes.CFUNCTYPE(None, ctypes.POINTER(DLManagedTensor)))
    ]


class MemoryManager:
    # cache is a dictionary {<pointer>: (<memory buffer>, <base objects>)}
    cache = {}

    align_to_bytes = 8

    @classmethod
    def align_to(cls, addr):
        if cls.align_to_bytes == 8:
            return ((addr + 7) >> 3) << 3
        elif cls.align_to_bytes == 16:
            return ((addr + 15) >> 4) << 4
        elif cls.align_to_bytes == 32:
            return ((addr + 31) >> 5) << 5
        elif cls.align_to_bytes == 64:
            return ((addr + 63) >> 6) << 6
        elif cls.align_to_bytes == 128:
            return ((addr + 127) >> 7) << 7
        raise NotImplementedError(cls.align_to_bytes)
    
    @atexit.register
    def _sanity_check():
        if len(MemoryManager.cache) != 0:
            print(f'dlpack.MemoryManager: {len(MemoryManager.cache)} objects have leaked!!!')

    @classmethod
    def allocate(cls, nbytes):
        nbytes = cls.align_to(nbytes + cls.align_to_bytes - 1)
        buf = ctypes.create_string_buffer(nbytes)  # this could be any writable buffer object
        pointer = ctypes.cast(buf, ctypes.c_void_p).value
        pointer = cls.align_to(pointer)
        assert pointer & (cls.align_to_bytes -1) == 0, pointer
        assert pointer not in cls.cache
        objects = dict(use_count=0)
        cls.cache[pointer] = buf, objects
        return pointer, objects

    @classmethod
    def deallocate(cls, pointer):
        buf, objects = cls.cache[pointer]
        objects['use_count'] -= 1
        if objects['use_count'] == 0:
            del cls.cache[pointer]




class DLManagedTensorImpl:

    def __init__(self, impl_cls, base, pointer, shape, strides, typelabel, nbits, lanes, devicelabel, device_id, offset, flags, version):
        """Allocate and initialize memory for a DLManagedTensor object.

        DLManagedTensorImpl holds a pointer to the underlying memory
        of a DLManagedTensor object that raw buffer can be acquired
        from MemoryManager.cache.
        """
        ndim = len(shape)
        align_to_bytes = MemoryManager.align_to_bytes
        align_to = MemoryManager.align_to
        sizeof_impl = align_to(ctypes.sizeof(impl_cls))
        sizeof_shape = align_to(ndim * ctypes.sizeof(ctypes.c_int64))
        managed_nbytes = sizeof_impl + 2 * sizeof_shape
        managed_pointer, managed_objects = MemoryManager.allocate(managed_nbytes)
        assert managed_pointer % align_to_bytes == 0
        managed_ptr = ctypes.cast(managed_pointer, ctypes.POINTER(impl_cls))
        shape_pointer = managed_pointer + sizeof_impl
        strides_pointer = managed_pointer + sizeof_impl + sizeof_shape
        assert shape_pointer % align_to_bytes == 0
        assert strides_pointer % align_to_bytes == 0
        shape_buf = ctypes.cast(shape_pointer, ctypes.POINTER(ctypes.c_int64))
        strides_buf = ctypes.cast(strides_pointer, ctypes.POINTER(ctypes.c_int64))

        for i in range(ndim):
            shape_buf[i] = shape[i]
            strides_buf[i] = strides[i]    
        dtype = DLDataType(DLDataTypeCode.from_label(typelabel), nbits, lanes)
        device = DLDevice(DLDeviceType.from_label(devicelabel), device_id)
        dl_tensor = DLTensor(pointer, device, ndim, dtype, shape_buf, strides_buf, offset)

        @ctypes.CFUNCTYPE(None, ctypes.POINTER(impl_cls))
        def DLManagedTensor_deleter(managed_ptr):
            managed = ctypes.cast(managed_ptr, ctypes.POINTER(impl_cls))
            managed_pointer = ctypes.cast(managed_ptr, ctypes.c_void_p).value
            MemoryManager.deallocate(managed_pointer)

        if impl_cls is DLManagedTensor:
            tmpl = DLManagedTensor(dl_tensor, None, DLManagedTensor_deleter)
        elif impl_cls is DLManagedTensorVersioned:
            tmpl = DLManagedTensorVersioned(version, None, DLManagedTensor_deleter, flags, dl_tensor)
        else:
            raise TypeError(f'{impl_cls}')

        r = ctypes.memmove(managed_ptr, ctypes.pointer(tmpl), ctypes.sizeof(impl_cls))

        # base must outlive DLManagedTensor object
        managed_objects['base'] = base
        # DLManagedTensor_deleter must outlive DLManagedTensor object
        managed_objects['deleter'] = DLManagedTensor_deleter

        self._managed_pointer = managed_pointer
        self.impl_cls = impl_cls

    @property
    def managed_pointer(self):
        MemoryManager.cache[self._managed_pointer][1]['use_count'] += 1
        return self._managed_pointer

    def get_dlpack_device(self):
        buf = MemoryManager.cache[self._managed_pointer][0]
        managed_ptr = ctypes.cast(buf, ctypes.POINTER(self.impl_cls))
        device = managed_ptr.contents.dl_tensor.device
        return (device.device_type.value, device.device_id)

    def get_dlpack_data(self):
        buf = MemoryManager.cache[self._managed_pointer][0]
        managed_ptr = ctypes.cast(buf, ctypes.POINTER(self.impl_cls))
        return managed_ptr.contents.dl_tensor.data
        
class DLPackObject:
    """A DLPack object holding a pointer to DLManagedTensor object memory.

    A DLPackObject object can be used as an input to from_dlpack
    functions of DLPack compliant array libraries.
    """

    def __init__(self, impl):
        self.impl = impl

    def __dlpack__(self, stream=None):
        # capsule destructor is needed for deallocating
        # DLManagedTensor objects that are never used.
        def DLManagedTensor_capsule_destructor(capsule):
            if capsule.is_valid(DLPACK_CAPSULE_VALID_NAME):
                pointer = capsule.get_pointer(DLPACK_CAPSULE_VALID_NAME)
                MemoryManager.deallocate(pointer)

        capsule = Capsule.new(self.impl.managed_pointer,
                              DLPACK_CAPSULE_VALID_NAME,
                              DLManagedTensor_capsule_destructor)
        return capsule.capsule

    def __dlpack_device__(self):
        return self.impl.get_dlpack_device()

def _get_strides(shape):
    strides = [1] if shape else []
    for s in reversed(shape[1:]):
        strides.insert(0, strides[0] * s)
    return tuple(strides)


def asdlpack_array_interface_v3(obj, impl_cls=None):
    """asdlpack for objects that implement array interface version 3.
    """
    if impl_cls is None:
        impl_cls = DLManagedTensor

    d = obj.__array_interface__
    
    shape = d['shape']
    typestr = d['typestr']
    interface_version = d['version']
    assert interface_version == 3, interface_version

    endian, typechr, nbytes = typestr[0], typestr[1], int(typestr[2:])
    assert endian in '<>|', endian

    typelabel = dict(t='Bit', b='DLBool', i='DLInt', u='DLUInt', f='DLFloat', c='DLComplex',
                     m='Timedelta', M='Datetime', O='Object', S='String', U='Unicode', V='VoidP')[typechr]

    nbits = nbytes if typelabel == 'Bit' else nbytes * 8

    if typelabel[:2] != 'DL':
        # TODO: remap String to 8-bit DLUint
        # TODO: remap Unicode to 32-bit DLUint
        # TODO: remap Timedelta, Datetime to DLInt
        raise ValueError(f'DLPack does not support `{typelabel}` type')

    data = d.get('data')
    if data is not None:
        if isinstance(data, tuple):
            pointer, readonly = data
            # TODO: readonly is unused
            offset = d.get('offset', 0)
            assert offset == 0, offset  # only for buffer object
        else:
            offset = d.get('offset', 0)
            readonly = True
            raise NotImplementedError(f'buffer object {data}')
    else:
        offset = d.get('offset', 0)
        readonly = True
        raise NotImplementedError(f'buffer object {obj}')

    strides = d.get('strides')
    if strides is None:  # C-style contiguous array
        strides = _get_strides(shape)
    else:
        # DL strides are not in bytes but in nof items
        strides = tuple(s // nbytes for s in strides)

    mask = d.get('mask')
    if mask is not None:
        # TODO: mask is an object that exposes array interface
        warnings.warn('export_array_interface_v3 ignores mask')

    lanes = 1
    devicelabel = 'DLCPU'
    device_id = 0
    flags = 1 if readonly else 0
    version = DLPackVersion(1, 0)
    impl = DLManagedTensorImpl(impl_cls, obj, pointer, shape, strides, typelabel, nbits, lanes, devicelabel, device_id, offset, flags, version)
    return DLPackObject(impl)


def asdlpack_dlpack(obj, impl_cls=None):
    """asdlpack for objects that implement DLPack latest or legacy protocols.
    """
    if impl_cls is None:
        impl_cls = DLManagedTensor  # legacy
    d = todict(obj.__dlpack__())

    version = d.get('version', DLPackVersion(1, 0))
    flags = d.get('flags', 0)
    dl_tensor = d['dl_tensor']
    pointer = dl_tensor['data']
    shape = dl_tensor['shape'] or ()
    strides = dl_tensor['strides']
    if strides is None:
        strides = _get_strides(shape)
    typelabel = dl_tensor['dtype']['code']
    nbits = dl_tensor['dtype']['bits']
    lanes = dl_tensor['dtype']['lanes']
    offset = dl_tensor['byte_offset']
    devicelabel = dl_tensor['device']['device_type']
    device_id = dl_tensor['device']['device_id']
    impl = DLManagedTensorImpl(impl_cls, obj, pointer, shape, strides, typelabel, nbits, lanes, devicelabel, device_id, offset, flags, version)
    return DLPackObject(impl)


def asdlpack_array_interface_v2(obj, impl_cls=None):

    if impl_cls is None:
        impl_cls = DLManagedTensor
    
    shape = obj.__array_shape__
    typestr = obj.__array_typestr__
    raise NotImplementedError(f'{type(obj).__name__}')


def asdlpack_cuda_array_interface(obj, impl_cls=None):

    if impl_cls is None:
        impl_cls = DLManagedTensor

    d = obj.__cuda_array_interface__

    shape = d['shape']
    typestr = d['typestr']
    interface_version = d['version']
    assert interface_version >= 2, interface_version

    endian, typechr, nbytes = typestr[0], typestr[1], int(typestr[2:])
    assert endian in '<>|', endian

    typelabel = dict(t='Bit', b='DLBool', i='DLInt', u='DLUInt', f='DLFloat', c='DLComplex',
                     m='Timedelta', M='Datetime', O='Object', S='String', U='Unicode', V='VoidP')[typechr]

    nbits = nbytes if typelabel == 'Bit' else nbytes * 8

    if typelabel[:2] != 'DL':
        # TODO: remap String to 8-bit DLUint
        # TODO: remap Unicode to 32-bit DLUint
        # TODO: remap Timedelta, Datetime to DLInt
        raise ValueError(f'DLPack does not support `{typelabel}` type')

    pointer, readonly = d['data']
    offset = 0

    strides = d.get('strides')
    if strides is None:  # C-style contiguous array
        strides = _get_strides(shape)
    else:
        # DL strides are not in bytes but in nof items
        strides = tuple(s // nbytes for s in strides)

    mask = d.get('mask')
    if mask is not None:
        # TODO: mask is an object that exposes array interface
        warnings.warn('asdlpack_cuda_array_interface ignores mask')

    stream = d.get('stream')
    if stream is not None:
        warnings.warn('asdlpack_cuda_array_interface ignores stream')
        
    lanes = 1
    devicelabel = 'DLCUDA'
    # Use cuPointerGetAttribute with
    # CU_POINTER_ATTRIBUTE_DEVICE_POINTER to obtain the correct
    # device_id:
    device_id = 0
    flags = 1 if readonly else 0
    version = DLPackVersion(1, 0)
    impl = DLManagedTensorImpl(impl_cls, obj, pointer, shape, strides, typelabel, nbits, lanes, devicelabel, device_id, offset, flags, version)
    return DLPackObject(impl)


def asdlpack_buffer(obj, impl_cls=None):
    """asdlpack for objects that implement DLPack latest or legacy protocols.
    """
    if impl_cls is None:
        impl_cls = DLManagedTensor  # legacy
    buf = Buffer(obj, flags=PyBUF_RECORDS_RO)
    pointer = buf.pointer
    shape = buf.shape or ()
    nbytes = buf.itemsize
    nbits = 8 * nbytes
    strides = buf.strides
    if strides is None:  # C-style contiguous array
        strides = _get_strides(shape)
    else:
        # DL strides are not in bytes but in nof items
        strides = tuple(s // nbytes for s in strides)
    fmt = buf.format
    machine_format = '@'
    if fmt and fmt[0] in '@=<>!':
        # see https://docs.python.org/3/library/struct.html#byte-order-size-and-alignment
        machine_format = fmt[0]
        fmt = fmt[1:]
    if fmt:
        if len(fmt) == 1:
            char = fmt[0]
            if char in 'cbhilq':
                typelabel = 'DLInt'
            elif char in 'BHILQ':
                typelabel = 'DLUInt'
            elif char == '?':
                typelabel = 'DLBool'
            elif char in 'efd':
                typelabel = 'DLFloat'
            else:
                assert 0, char
        elif len(fmt) == 2 and fmt[0] == 'Z':
            char = fmt[1]
            if char in 'efd':
                typelabel = 'DLComplex'
            else:
                assert 0, char
        else:
            assert 0, fmt
    else:
        assert 0, fmt
    lanes = 1
    devicelabel = 'DLCPU'
    device_id = 0
    flags = 1  # readonly
    version = DLPackVersion(1, 0)
    offset = 0
    impl = DLManagedTensorImpl(impl_cls, obj, pointer, shape, strides, typelabel, nbits, lanes, devicelabel, device_id, offset, flags, version)
    return DLPackObject(impl)


def asdlpack(obj, impl_cls=None):
    """Expose an object for buffer exchange with DLPack-compatible array libraries.

    The object must support one of the following array exchange protocols:

    - DLPack
    - NumPy array interface protocol, version 3
    - Numba CUDA array interface protocol

    The impl_cls can be either DLManagedTensor or DLManagedTensorVersioned.

    Returns DLPackObject instance that can be consumed from_dlpack
    functions of array libraries that implement the DLPack importer
    protocol.
    """
    if hasattr(obj, "__dlpack__"):
        return asdlpack_dlpack(obj, impl_cls=impl_cls)
    elif hasattr(obj, "__array_interface__"):
        return asdlpack_array_interface_v3(obj, impl_cls=impl_cls)
    elif hasattr(obj, "__cuda_array_interface__"):
        return asdlpack_cuda_array_interface(obj, impl_cls=impl_cls)
    elif hasattr(obj, "__array_shape__") and hasattr(obj, "__array_typestr__"):
        return asdlpack_array_interface_v2(obj, impl_cls=impl_cls)
    else:
        return asdlpack_buffer(obj, impl_cls=impl_cls)


def todict(capsule):
    """Convert a DLPack capsule object to a dictionary representing a DLPack interface structure.
    """
    capsule = Capsule(capsule)
    if not capsule.is_valid(DLPACK_CAPSULE_VALID_NAME):
        raise ValueError(f"{capsule.get_name()}-capsule is not a valid dltensor-capsule")
    pointer = capsule.get_pointer(DLPACK_CAPSULE_VALID_NAME)
    # Check if capsule contains a versioned DLPack managed tensor
    # buffer.  If so, its first two 32-bit integers correspond to
    # major and minor versions of the DLPack protocol. Howewver, when
    # these values are unreasonable large, the buffer must contain the
    # legacy DLPack managed tensor.
    version_ptr = ctypes.cast(pointer, ctypes.POINTER(DLPackVersion))
    version = version_ptr.contents

    if max(version.major, version.minor) > 10000:
        # the buffer most likely contains the legacy DLPack managed tensor
        managed_ptr = ctypes.cast(pointer, ctypes.POINTER(DLManagedTensor))
    else:
        managed_ptr = ctypes.cast(pointer, ctypes.POINTER(DLManagedTensorVersioned))

    return managed_ptr.contents.todict()
