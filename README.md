pydlpack
========

This package provides tools for exchanging data buffers of Python
objects to any DLPack consumer. The producers of Python objects must
support one (or more) of the following protocols:

- DLPack
- NumPy Array Interface version 3
- Numba CUDA Array Interface versions 2 and 3
- Python Buffer protocol

NumPy Array Interface version 2 support can be provided on request.

Currently, the package is tested with the following consumers

- NumPy
- PyTorch
- CuPy
- Numba
- Jax
- Tensorflow
- Cudf

using the following provider objects with devices:

- Numpy ndarray, CPU
- Torch Tensor, CPU and CUDA
- CuPy ndarray, CUDA
- Numba DeviceNDArray, CUDA
- Jax numpy.ndarray, CPU and CUDA
- Tensorflow Tensor, CPU
- Python bytes, CPU
- Python bytearray, CPU
- Python array.array, CPU

Basic usage
-----------

```python
>>> from dlpack import asdlpack
>>> import torch
>>> dl = asdlpack(b"Hello!")
>>> torch.from_dlpack(dl)
tensor([ 72, 101, 108, 108, 111,  33], dtype=torch.uint8)
```

that is, the Python package `dlpack` provides a function `asdlpack`
that input can be any object that implements one of the above
mentioned protocols and it will return a light-weight `DLPackObject`
instance which implements the DLPack protocol methods
`__dlpack__(stream=None)` and `__dlpack_device__()`. This
`DLPackObject` instance can be used as an argument to a
`consumer.from_dlpack(obj)` function of any DLPack-compatible consumer
library (a partial list of such libraries is listed above). For
example:

```python
>>> from dlpack import asdlpack
>>> import numba.cuda
>>> import numpy
>>> arr = numba.cuda.to_device(numpy.array([[1, 2], [3, 4]]))
>>> arr
<numba.cuda.cudadrv.devicearray.DeviceNDArray object at 0x7fbed9c548b0>
>>> dl = asdlpack(arr)
>>> import torch
>>> torch.from_dlpack(dl)
tensor([[1, 2],
        [3, 4]], device='cuda:0')
>>> import jax
>>> jax.numpy.from_dlpack(dl)
Array([[1, 2],
       [3, 4]], dtype=int32)
>>> import cupy
>>> cupy.from_dlpack(dl)
array([[1, 2],
       [3, 4]])
```

that is, the `DLPackObject` instance can be efficiently used for
exchanging the CUDA buffer created using Numba `to_device`
functionality with different consumer objects such as `torch.Tensor`,
`jax.Array`, and `cupy.ndarray` while all these array objects share the
same CUDA memory.

Testing
-------

It is a non-trivial task to install all dlpack-compatible libraries
into the same environment. Therefore, `dlpack` tests are included the
`dlpack` package so that one can import `dlpack` and run the tests on
DLPack-compatible objects that are available in a particular
environment. For example:

```python
>>> import dlpack, os, pytest 
>>> pytest.main([os.path.dirname(dlpack.__file__), "-xq"])
```
