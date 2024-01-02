"""Provides ctypes-based interface to Python buffer C/API
"""
# Created: December 2023
# Author: Pearu Peterson

__all__ = ["Buffer"]

import ctypes

#
# See https://docs.python.org/3/c-api/buffer.html for Python C/API of
# a buffer object.
#

Py_ssize_t = ctypes.c_ssize_t


class Py_Buffer(ctypes.Structure):
    _fields_ = [
        ("buf", ctypes.c_void_p),
        ("obj", ctypes.py_object),
        ("len", Py_ssize_t),
        ("itemsize", Py_ssize_t),
        ("readonly", ctypes.c_int),
        ("ndim", ctypes.c_int),
        ("format", ctypes.c_char_p),
        ("shape", ctypes.POINTER(Py_ssize_t)),
        ("strides", ctypes.POINTER(Py_ssize_t)),
        ("suboffsets", ctypes.POINTER(Py_ssize_t)),
        ("internal", ctypes.c_void_p),
    ]

    def __repr__(self):
        args = []
        for field, typ in self._fields_:
            value = getattr(self, field)
            if typ.__name__.startswith("LP_") and hasattr(self, "ndim"):
                if ctypes.cast(value, ctypes.c_void_p).value is None:
                    value = None
                else:
                    value = tuple(value[i] for i in range(self.ndim))
            args.append(f"{field}={value}")
        args = ", ".join(args)
        return f"{type(self).__name__}({args})"


#
# int PyObject_GetBuffer(PyObject *exporter, Py_buffer *view, int flags)
#
ctypes.pythonapi.PyObject_GetBuffer.restype = ctypes.c_int
ctypes.pythonapi.PyObject_GetBuffer.argtypes = [ctypes.py_object, ctypes.POINTER(Py_Buffer), ctypes.c_int]

#
# void PyBuffer_Release(Py_buffer *view)
#
ctypes.pythonapi.PyBuffer_Release.restype = None
ctypes.pythonapi.PyBuffer_Release.argtypes = [ctypes.POINTER(Py_Buffer)]

# TODO:
# Py_ssize_t PyBuffer_SizeFromFormat(const char *format)
# int PyBuffer_IsContiguous(const Py_buffer *view, char order)

#
# void *PyBuffer_GetPointer(const Py_buffer *view, const Py_ssize_t *indices)
#

ctypes.pythonapi.PyBuffer_GetPointer.restype = ctypes.c_void_p
ctypes.pythonapi.PyBuffer_GetPointer.argtypes = [ctypes.POINTER(Py_Buffer), ctypes.POINTER(Py_ssize_t)]

# int PyBuffer_FromContiguous(const Py_buffer *view, const void *buf, Py_ssize_t len, char fort)
# int PyBuffer_ToContiguous(void *buf, const Py_buffer *src, Py_ssize_t len, char order)
# int PyObject_CopyData(PyObject *dest, PyObject *src)
# void PyBuffer_FillContiguousStrides(int ndims, Py_ssize_t *shape, Py_ssize_t *strides, int itemsize, char order)
# int PyBuffer_FillInfo(Py_buffer *view, PyObject *exporter, void *buf, Py_ssize_t len, int readonly, int flags)

# See pybuffer.h:
PyBUF_WRITABLE = 0x0001
PyBUF_FORMAT = 0x0004
PyBUF_SIMPLE = 0
PyBUF_ND = 0x0008
PyBUF_STRIDES = 0x0010 | PyBUF_ND
PyBUF_INDIRECT = 0x0100 | PyBUF_STRIDES
PyBUF_C_CONTIGUOUS = 0x0020 | PyBUF_STRIDES
PyBUF_F_CONTIGUOUS = 0x0040 | PyBUF_STRIDES
PyBUF_ANY_CONTIGUOUS = 0x0080 | PyBUF_STRIDES
PyBUF_FULL = PyBUF_INDIRECT | PyBUF_WRITABLE | PyBUF_FORMAT
PyBUF_FULL_RO = PyBUF_INDIRECT | PyBUF_FORMAT
PyBUF_RECORDS = PyBUF_STRIDES | PyBUF_WRITABLE | PyBUF_FORMAT
PyBUF_RECORDS_RO = PyBUF_STRIDES | PyBUF_FORMAT
PyBUF_STRIDED = PyBUF_STRIDES | PyBUF_WRITABLE
PyBUF_STRIDED_RO = PyBUF_STRIDES
PyBUF_CONTIG = PyBUF_ND | PyBUF_WRITABLE
PyBUF_CONTIG_RO = PyBUF_ND


class Buffer:
    """A buffer view of an object that implements the Python buffer
    protocol.
    """

    def __init__(self, obj, flags=PyBUF_SIMPLE):
        """View object via buffer protocol using specified flags."""
        buf = Py_Buffer()
        status = ctypes.pythonapi.PyObject_GetBuffer(obj, ctypes.pointer(buf), flags)
        if status != 0:
            raise BufferError(f"requesting a buffer view with {flags=}")
        # buf has initialized obj, buf, len, itemsize, ndim
        self.py_buffer = buf

    def __del__(self):
        """Release the buffer view and release the strong reference
        (i.e. decrement the reference count) to the view’s supporting
        object, view.obj. This function MUST be called when the
        buffer is no longer being used, otherwise reference leaks may
        occur.
        """
        ctypes.pythonapi.PyBuffer_Release(self.py_buffer)

    @property
    def ndim(self):
        return self.py_buffer.ndim

    @property
    def shape(self):
        if ctypes.cast(self.py_buffer.shape, ctypes.c_void_p).value is None:
            return
        return tuple(self.py_buffer.shape[i] for i in range(self.ndim))

    @property
    def strides(self):
        if ctypes.cast(self.py_buffer.strides, ctypes.c_void_p).value is None:
            return
        return tuple(self.py_buffer.strides[i] for i in range(self.ndim))

    @property
    def suboffsets(self):
        if ctypes.cast(self.py_buffer.suboffsets, ctypes.c_void_p).value is None:
            return
        return tuple(self.py_buffer.suboffsets[i] for i in range(self.ndim))

    @property
    def itemsize(self):
        return self.py_buffer.itemsize

    @property
    def pointer(self):
        return self.py_buffer.buf

    @property
    def format(self):
        if self.py_buffer.format is None:
            return
        return self.py_buffer.format.decode()
