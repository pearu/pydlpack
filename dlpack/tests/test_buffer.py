from dlpack import buffer

import pytest

def test_from_ndarray():
    numpy = pytest.importorskip('numpy')

    arr = numpy.array([[1, 2], [3, 4]], dtype=numpy.int32)
    buf = buffer.Buffer(arr, flags=buffer.PyBUF_RECORDS_RO)
    assert buf.ndim == arr.ndim
    assert buf.shape == arr.shape
    assert buf.strides == arr.strides
    assert buf.itemsize == arr.dtype.itemsize
    assert buf.format == arr.dtype.char
    d = arr.__array_interface__
    assert buf.pointer == d['data'][0]


def test_from_bytes():
    obj = b"123"
    buf = buffer.Buffer(obj, flags=buffer.PyBUF_RECORDS_RO)
    assert buf.ndim == 1
    assert buf.shape == (3,)
    assert buf.strides == (1,)
    assert buf.itemsize == 1
    assert buf.format == 'B'
