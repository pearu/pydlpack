[![Python package](https://github.com/pearu/pydlpack/actions/workflows/python-package.yml/badge.svg)](https://github.com/pearu/pydlpack/actions/workflows/python-package.yml) [![Conda Version](https://img.shields.io/conda/vn/conda-forge/pydlpack.svg)](https://anaconda.org/conda-forge/pydlpack)

pydlpack
========

PyDLPack is a Python library for exchanging data between different
array libraries using [DLPack: Open In Memory Tensor
Structure]((https://github.com/dmlc/dlpack/). The provider library
does not need to implement the DLPack support, it will be sufficent if
the provider library implements one of the following protocols:

- [dmlc/dlpack](https://github.com/dmlc/dlpack/)
- [Array Interface Protocol, version 3](https://numpy.org/doc/stable/reference/arrays.interface.html)
- [CUDA Array Interface, version 3](https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html)
- [Python Buffer protocol](https://docs.python.org/3/c-api/buffer.html)

[Array Interface Protocol, version 2](https://numpy.org/doc/stable/reference/arrays.interface.html#differences-with-array-interface-version-2) support can be provided on request.

Currently, the package is tested with the following consumers

- [NumPy](https://numpy.org/), [`numpy.from_dlpack`](https://numpy.org/devdocs/reference/generated/numpy.from_dlpack.html)
- [PyTorch](https://pytorch.org/), [`torch.from_dlpack`](https://pytorch.org/docs/stable/generated/torch.from_dlpack.html)
- [CuPy](https://cupy.dev/), [`cupy.from_dlpack`](https://docs.cupy.dev/en/stable/reference/generated/cupy.from_dlpack.html)
- [Jax](https://github.com/google/jax), [`jax.numpy.from_dlpack`](https://jax.readthedocs.io/en/latest/_autosummary/jax.dlpack.from_dlpack.html)
- [Tensorflow](https://www.tensorflow.org/), [`tf.experimental.dlpack.from_dlpack`](https://www.tensorflow.org/api_docs/python/tf/experimental/dlpack/from_dlpack)
- [cuDF](https://github.com/rapidsai/cudf), [`cudf.from_dlpack`](https://docs.rapids.ai/api/cudf/latest/user_guide/api_docs/api/cudf.from_dlpack/)

using the following provider objects with devices:

- Numpy [`ndarray`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html), CPU
- Torch [`Tensor`](https://pytorch.org/docs/stable/tensors.html), CPU and CUDA
- CuPy [`ndarray`](https://docs.cupy.dev/en/stable/reference/generated/cupy.ndarray.html), CUDA
- [Numba](https://numba.pydata.org/) [`DeviceNDArray`](https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html#numba.cuda.as_cuda_array), CUDA
- Jax [`Array`](https://jax.readthedocs.io/en/latest/_autosummary/jax.Array.html), CPU and CUDA
- Tensorflow [`Tensor`](https://www.tensorflow.org/api_docs/python/tf/Tensor), CPU
- Python [`bytes`](https://docs.python.org/3/library/stdtypes.html#bytes), CPU
- Python [`bytearray`](https://docs.python.org/3/library/stdtypes.html#bytearray), CPU
- Python [`array`](https://docs.python.org/3/library/array.html), CPU
- Numpy [`memmap`](https://numpy.org/doc/stable/reference/generated/numpy.memmap.html), CPU
- Python [`mmap`](https://docs.python.org/3/library/mmap.html), CPU

Install
-------

- [pydlpack in conda-forge](https://github.com/conda-forge/pydlpack-feedstock)
  ```sh
  conda install pydlpack
  ```

- [pydlpack in PyPi](https://pypi.org/project/pydlpack/):

  ```sh
  pip install pydlpack
  ```

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
>>> import dlpack.tests
>>> dlpack.tests.run()
```
