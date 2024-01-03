from dlpack import Capsule, DLManagedTensor
import dlpack
import pytest
import os

backends = ["numpy", "torch", "numba", "jax", "cupy", "tensorflow", "cudf"]
protocols = ["dlpack", "array_interface_v3", "cuda_array_interface", "buffer"]
devices = ["cpu", "cuda"]


def importorskip(backend, protocol, device, require_from_dlpack=False, use_buffer_via_dlpack=False):
    backend = pytest.importorskip(backend)
    major, minor = map(int, backend.__version__.split(".")[:2])
    if backend.__name__ == "jax":
        backend = backend.numpy
    if require_from_dlpack:
        if backend.__name__ == "tensorflow":
            backend.experimental.dlpack.from_dlpack
            pass
        elif not hasattr(backend, "from_dlpack"):
            pytest.skip(f"test requires {backend.__name__}.from_dlpack [{backend.__name__} v{backend.__version__}]")
    if device == "cpu" and protocol == "cuda_array_interface":
        pytest.skip(f"{protocol} is not available for {device}")
    if backend.__name__ == "numpy":
        if device == "cuda":
            pytest.skip(f"{backend.__name__} does not support {device} arrays")
        if protocol not in {"array_interface_v3", "dlpack", "buffer"}:
            pytest.skip(f"{backend.__name__} does not support {protocol}")
        if protocol == "dlpack" and (major, minor) < (1, 22):
            pytest.skip(f"{protocol} requires {backend.__name__} version 1.22 or newer but got {backend.__version__}")
    elif backend.__name__ == "numba":
        if device == "cpu":
            pytest.skip(f"{backend.__name__} on {device} is equivalent to numpy")
        if device == "cuda":
            import numba.cuda

            if not numba.cuda.is_available():
                pytest.skip(f"{device} is not available for {backend.__name__}")
        if protocol not in {"cuda_array_interface"}:
            pytest.skip(f"{backend.__name__} does not support {protocol}")
    elif backend.__name__ == "torch":
        if device == "cuda":
            import torch.cuda

            if not torch.cuda.is_available():
                pytest.skip(f"{device} is not available for {backend.__name__}")
        if use_buffer_via_dlpack and protocol == "buffer":
            pass
        elif protocol not in {"cuda_array_interface", "dlpack"}:
            pytest.skip(f"{backend.__name__} does not support {protocol}")
    elif backend.__name__ == "jax.numpy":
        import jax

        if device not in jax._src.xla_bridge.backends():
            pytest.skip(f"{device} is not available for {backend.__name__}")
        jax.config.update("jax_enable_x64", True)
        if use_buffer_via_dlpack and protocol == "buffer":
            pass
        elif protocol not in {"dlpack", "cuda_array_interface"}:
            pytest.skip(f"{backend.__name__} does not support {protocol}")
        if require_from_dlpack and device == "cpu":
            pytest.skip("jax bug: https://github.com/google/jax/issues/19134")
    elif backend.__name__ == "cupy":
        if device == "cpu":
            pytest.skip(f"{backend.__name__} on {device} is equivalent to numpy")
        if protocol not in {"dlpack", "cuda_array_interface"}:
            pytest.skip(f"{backend.__name__} does not support {protocol}")
    elif backend.__name__ == "tensorflow":
        import tensorflow as tf

        for d in tf.config.list_physical_devices():
            if d.device_type.lower() == device:
                break
        else:
            pytest.skip(f"{device} is not available for {backend.__name__}")
        if use_buffer_via_dlpack and protocol == "buffer":
            pass
        elif protocol not in {"dlpack"}:
            pytest.skip(f"{backend.__name__} does not support {protocol}")
    elif backend.__name__ == "cudf":
        if device == "cpu":
            pytest.skip(f"{backend.__name__} on {device} is equivalent to numpy")
        if protocol not in {"dlpack"}:
            pytest.skip(f"{backend.__name__} does not support {protocol}")
    else:
        raise NotImplementedError(backend.__name__)
    return backend


def numpy_sample(shape, dtype=None):
    import math, numpy

    s = numpy.arange(1, math.prod(shape) + 1)
    s = s.reshape(shape).astype(dtype)
    if dtype in {numpy.complex64, numpy.complex128}:
        s -= 1j * s
    return s


def torch_sample(shape, dtype=None, device=None):
    import math, torch

    s = torch.arange(1, math.prod(shape) + 1, device=device)
    s = s.reshape(shape).to(dtype)
    if dtype in {torch.complex64, torch.complex128}:
        s -= 1j * s
    return s.contiguous()


def sample(backend_name, shape, dtype=None, device=None, slice=None):
    if backend_name == "numpy":
        assert device in {None, "cpu"}, device
        s = numpy_sample(shape, dtype=dtype)
        if slice is not None:
            s = s[slice]
        return s
    elif backend_name == "torch":
        s = torch_sample(shape, dtype=dtype, device=device)
        if slice is not None:
            s = s[slice]
        return s
    elif backend_name == "numba":
        assert device == "cuda", device
        import numba.cuda, numpy

        dtype = getattr(numpy, str(dtype))
        s = numba.cuda.to_device(numpy_sample(shape, dtype=dtype))
        if slice is not None:
            s = s[slice]
        return s
    elif backend_name == "jax.numpy":
        import jax
        import jax.numpy as jnp
        import math

        with jax.default_device(jax.devices(device)[0]):
            s = jnp.arange(1, math.prod(shape) + 1)
            s = s.reshape(shape).astype(dtype)
            if dtype in {jnp.complex64, jnp.complex128}:
                s -= 1j * s
            if slice is not None:
                s = s[slice]
        return s
    elif backend_name == "cupy":
        import cupy, math

        s = cupy.arange(1, math.prod(shape) + 1)
        s = s.reshape(shape).astype(dtype)
        if dtype in {cupy.complex64, cupy.complex128}:
            s -= 1j * s
        if slice is not None:
            return s[slice]
        return s
    elif backend_name == "tensorflow":
        import tensorflow as tf

        for d in tf.config.list_logical_devices():
            if d.device_type.lower() == device:
                with tf.device(d.name):
                    t = tf.ones((3, 4), dtype=dtype)
                    if dtype in {tf.complex64, tf.complex128}:
                        t -= 1j * t
                    return t
    elif backend_name == "cudf":
        import cudf, cupy, math

        shape = shape[:2]
        s = cupy.arange(1, math.prod(shape) + 1)
        s = s.reshape(shape).astype(dtype)
        if dtype in {cupy.complex64, cupy.complex128}:
            s -= 1j * s
        s = s.T
        return s
    else:
        raise NotImplementedError(backend_name)


def buffer_samples():
    yield "1234ABCD123489".encode()
    yield bytearray([1, 2, 3, 4] * 8)
    import array

    yield array.array("d", [1.0, 2.0, 3.14])
    yield array.array("l", [1, 2, 3, 4, 5])
    try:
        import numpy, tempfile

        tmp = tempfile.NamedTemporaryFile(delete=False)
        fname = tmp.name
        fp = numpy.memmap(fname, shape=(2, 3), dtype=numpy.int32, mode="w+")
        fp[0] = [1, 2, 3]
        fp[1] = [4, 5, 6]
        yield fp
        fp.flush()
        fp._mmap.close()
        tmp.file.close()
        import mmap

        f = open(fname, "r")
        fp = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
        yield fp
        fp.close()
        import os

        os.remove(fname)
    except ImportError:
        pass


def dtypes(backend):
    dtype_names = [
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "bfloat16",
        "float16",
        "float32",
        "float64",
        "complex64",
        "complex128",
        "bool",
    ]
    skip_float16 = False
    skip_complex = False
    if backend == "cudf":
        backend = "cupy"
        skip_float16 = True
        skip_complex = True
    backend = pytest.importorskip(backend)
    count = 0
    for name in dtype_names:
        if skip_float16 and name == "float16":
            # RuntimeError: CUDF failure at:/opt/conda/conda-bld/work/cpp/src/interop/dlpack.cpp:83: Unsupported bitsize for kDLFloat
            continue
        if skip_complex and name in {"complex64", "complex128"}:
            # RuntimeError: CUDF failure at:/opt/conda/conda-bld/work/cpp/src/interop/dlpack.cpp:86: Invalid DLPack type code
            continue
        if backend.__name__ == "numpy" and name == "bool":
            name = "bool_"
        dtype = getattr(backend, name, None)
        if dtype is not None:
            count += 1
            yield dtype
    assert count, backend


def samples(backend_name, device=None):
    shapes_slices = [
        ((), None),
        ((2,), None),
        ((4, 6, 5), None),
        ((5, 6, 5), (slice(1, 3), slice(None, None, 2), slice(None, None, (None if backend_name == "torch" else -1)))),
    ]
    for dtype in dtypes(backend_name):
        for shape, slice_ in shapes_slices:
            if backend_name == "cudf":
                if len(shape) < 2:
                    continue
                else:
                    shape = shape[:2]
                    if slice_ is not None:
                        slice_ = slice_[:2]
            arr = sample(backend_name, shape, dtype=dtype, device=device, slice=slice_)
            yield arr


@pytest.mark.parametrize("backend", backends)
@pytest.mark.parametrize("protocol", protocols)
@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("impl_cls", ["DLManagedTensor", "DLManagedTensorVersioned"])
def test_asdlpack_impl(backend, protocol, device, impl_cls):
    backend = importorskip(backend, protocol, device)
    method = getattr(dlpack, f"asdlpack_{protocol}")
    impl_cls = getattr(dlpack, impl_cls)

    if backend.__name__ == "tensorflow":
        to_dlpack = backend.experimental.dlpack.to_dlpack

        class W:
            def __init__(self, obj):
                self.obj = obj

            def __dlpack__(self, stream=None):
                return to_dlpack(self.obj)

    else:

        def W(obj):
            return obj

    def get_itemsize(a):
        if backend.__name__ == "tensorflow":
            return a.dtype.size
        return a.dtype.itemsize

    def get_strides(a):
        if backend.__name__ == "torch":
            return a.stride()
        if backend.__name__ in {"jax.numpy", "tensorflow"}:
            return dlpack._get_strides(a.shape)
        if a.strides is None:
            return
        return tuple(s // get_itemsize(a) for s in a.strides)

    def test(a):
        dl = method(W(a), impl_cls=impl_cls)
        c = dl.__dlpack__()
        d = dlpack.todict(c)
        strides = get_strides(a)
        assert d["dl_tensor"]["ndim"] == a.ndim
        assert d["dl_tensor"]["shape"] == a.shape
        assert d["dl_tensor"]["strides"] in {None, strides}
        assert d["dl_tensor"]["device"]["device_type"] == dict(cpu="DLCPU", cuda="DLCUDA")[device]
        assert d["dl_tensor"]["device"]["device_id"] == 0
        assert d["dl_tensor"]["byte_offset"] == 0
        assert d["dl_tensor"]["dtype"]["bits"] == get_itemsize(a) * 8
        if protocol == "array_interface_v3":
            assert d["dl_tensor"]["data"] == a.__array_interface__["data"][0]
        elif protocol == "cuda_array_interface":
            assert d["dl_tensor"]["data"] == a.__cuda_array_interface__["data"][0]

        if impl_cls is dlpack.DLManagedTensorVersioned:
            if protocol == "array_interface_v3":
                assert d["flags"] == a.__array_interface__["data"][1]
            elif protocol == "cuda_array_interface":
                assert d["flags"] == a.__cuda_array_interface__["data"][1]
            assert d.get("version") is not None

    for arr in samples(backend.__name__, device=device):
        if backend.__name__ in {"torch", "jax.numpy"} and protocol == "cuda_array_interface" and arr.dtype == backend.bfloat16:
            # cuda array interface does not support bfloat16
            continue
        if backend.__name__ == "torch" and protocol == "cuda_array_interface" and arr.dtype == backend.bool:
            # cuda array interface does not support bool
            continue
        if backend.__name__ == "numpy" and protocol == "dlpack" and arr.dtype == backend.bool_:
            major, minor = map(int, backend.__version__.split(".")[:2])
            if (major, minor) < (1, 26):  # minor may not be exact
                # BufferError: DLPack only supports signed/unsigned integers, float and complex dtypes.
                continue
        test(arr)


@pytest.mark.parametrize("backend", backends)
@pytest.mark.parametrize("protocol", protocols)
@pytest.mark.parametrize("device", devices)
def test_asdlpack_function(backend, protocol, device):
    backend = importorskip(backend, protocol, device, require_from_dlpack=True)
    method = getattr(dlpack, f"asdlpack_{protocol}")

    if backend.__name__ == "tensorflow":
        from_dlpack = lambda obj: backend.experimental.dlpack.from_dlpack(obj.__dlpack__())
        to_dlpack = backend.experimental.dlpack.to_dlpack
    else:
        from_dlpack = backend.from_dlpack

        if backend.__name__ == "cudf":
            from_dlpack = lambda obj: backend.from_dlpack(obj.__dlpack__())
            to_dlpack = lambda obj: obj.__dlpack__()

    if backend.__name__ in {"tensorflow"}:

        class W:
            def __init__(self, obj):
                self.obj = obj

            def __dlpack__(self, stream=None):
                return to_dlpack(self.obj)

    else:

        def W(obj):
            return obj

    def assert_equal(x, y):
        if backend.__name__ == "tensorflow":
            assert backend.reduce_all(x == y)
        elif backend.__name__ == "cudf":
            assert (x == y.to_cupy()).all()
        else:
            assert (x == y).all()

    def get_copy(a):
        if backend.__name__ == "tensorflow":
            return a + 0
        if backend.__name__ == "cudf":
            return (a.T + 0).T
        if backend.__name__ == "torch":
            return a.clone()
        return a.copy()

    def test_simple(a):
        dl = method(W(a))
        a1 = from_dlpack(dl)
        assert_equal(a, a1)

    def test_simple_del_dl(a):
        dl = method(W(a))
        a1 = from_dlpack(dl)
        del dl
        assert_equal(a, a1)

    def test_simple_del_array(a):
        b = get_copy(a)
        dl = method(W(b))
        del b
        a1 = from_dlpack(dl)
        assert_equal(a, a1)

    def test_local_dl(a):
        def array_from_local_dl():
            dl = method(W(a))
            return from_dlpack(dl)

        a1 = array_from_local_dl()
        assert_equal(a, a1)

    def test_local_array_and_dl(a):
        def array_from_local_array_and_dl():
            b = get_copy(a)
            dl = method(W(b))
            return from_dlpack(dl)

        a1 = array_from_local_array_and_dl()
        assert_equal(a, a1)

    def test_local_array(a):
        def dl_from_local_array():
            b = get_copy(a)
            return method(W(b))

        a1 = from_dlpack(dl_from_local_array())
        assert_equal(a, a1)

    def test_unused(a):
        dl = method(W(a))
        c1 = dl.__dlpack__()
        c2 = dl.__dlpack__()

    def test_multiple_use(a):
        dl = method(W(a))
        a1 = from_dlpack(dl)
        assert_equal(a, a1)
        a2 = from_dlpack(dl)
        assert_equal(a, a2)

    for arr in samples(backend.__name__, device=device):
        if backend.__name__ in {"torch", "jax.numpy"} and protocol == "cuda_array_interface" and arr.dtype == backend.bfloat16:
            # cuda array interface does not support bfloat16
            continue
        if backend.__name__ in {"jax.numpy"} and protocol == "dlpack" and arr.dtype == backend.bfloat16:
            # likely a bug in jax
            continue
        if backend.__name__ == "torch" and protocol == "cuda_array_interface" and arr.dtype == backend.bool:
            # cuda array interface does not support bool
            continue
        if backend.__name__ == "tensorflow" and protocol == "dlpack" and arr.dtype == backend.bool:
            # tensorflow.python.framework.errors_impl.InvalidArgumentError: cannot compute Equal as input #1(zero-based) was expected to be a bool tensor but is a uint8 tensor [Op:Equal] name:
            continue
        if (
            backend.__name__ == "numpy"
            and protocol in {"dlpack", "array_interface_v3", "buffer"}
            and arr.dtype == backend.bool_
        ):
            major, minor = map(int, backend.__version__.split(".")[:2])
            if (major, minor) < (1, 26):  # minor may not be exact
                # BufferError: DLPack only supports signed/unsigned integers, float and complex dtypes.
                # SystemError: <built-in function from_dlpack> returned NULL without setting an exception
                continue
        for test in [
            test_simple,
            test_simple_del_dl,
            test_simple_del_array,
            test_local_dl,
            test_local_array_and_dl,
            test_local_array,
            test_unused,
            test_multiple_use,
        ]:
            if backend.__name__ == "cudf" and test is test_multiple_use:
                # Avoid KeyError in managed_pointer. It looks like
                # cudf dlpack support calls DLManagedTensor destructor
                # directly rather that via capsule destructor.
                continue
            test(arr)


@pytest.mark.parametrize("backend", backends)
def test_asdlpack_buffer(backend):
    backend = importorskip(backend, "buffer", "cpu", require_from_dlpack=True, use_buffer_via_dlpack=True)
    method = getattr(dlpack, f"asdlpack_buffer")
    import numpy

    if backend.__name__ == "tensorflow":
        from_dlpack = lambda obj: backend.experimental.dlpack.from_dlpack(obj.__dlpack__())
    elif backend.__name__ == "cudf":
        from_dlpack = lambda obj: backend.from_dlpack(obj.__dlpack__())
    else:
        from_dlpack = backend.from_dlpack

    def assert_equal(x, y):
        if backend.__name__ == "tensorflow":
            assert backend.reduce_all(x == y)
        elif backend.__name__ == "cudf":
            assert (x == y.to_cupy()).all()
        else:
            assert (x == y).all()

    count = 0
    for obj in buffer_samples():
        dl = method(obj)
        if backend.__name__ == "tensorflow":
            # tf requires 128-bit alignment
            p = dl.impl.get_dlpack_data()
            if p & 127 != 0:
                # tensorflow/core/framework/tensor.cc:847] Check failed: IsAligned() ptr = 0x5578b5e717f0
                continue
            # todo: detect misalignment in shapes, strides as well,
            # also check aligment of created structures and the
            # addresses of destructors
            if type(obj).__name__ in {"memmap", "mmap"}:
                # Check failed: IsAligned()
                continue
        count += 1
        a = from_dlpack(dl)
        a1 = numpy.from_dlpack(dl)  # just to get dtype and shape estimates
        a2 = numpy.frombuffer(obj, dtype=a1.dtype).reshape(a1.shape)
        a3 = from_dlpack(a2.copy())
        assert_equal(a, a3)
        del a2  # make that buffer is released so that mmap can close
    if count == 0:
        pytest.skip("NO SAMPLES!")
