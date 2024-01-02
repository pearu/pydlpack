"""Provides ctypes-based interface to Python capsule C/API
"""
# Created: December 2023
# Author: Pearu Peterson

__all__ = ['Capsule']

import ctypes

#
# See https://docs.python.org/3/c-api/capsule.html for Python C/API of
# a capsule object.
#

PyCapsule = ctypes.py_object

#
# typedef void (*PyCapsule_Destructor)(PyObject *);
#

PyCapsule_Destructor = ctypes.CFUNCTYPE(None, ctypes.c_void_p)

#
# PyObject *PyCapsule_New(void *pointer, const char *name, PyCapsule_Destructor destructor)
#
ctypes.pythonapi.PyCapsule_New.restype = PyCapsule
ctypes.pythonapi.PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, PyCapsule_Destructor]

#
# PyCapsule_Destructor PyCapsule_GetDestructor(PyObject *capsule)
#
ctypes.pythonapi.PyCapsule_GetDestructor.restype = PyCapsule_Destructor
ctypes.pythonapi.PyCapsule_GetDestructor.argtypes = [PyCapsule]

#
# int PyCapsule_SetDestructor(PyObject *capsule, PyCapsule_Destructor destructor)
#
ctypes.pythonapi.PyCapsule_SetDestructor.restype = ctypes.c_int
ctypes.pythonapi.PyCapsule_SetDestructor.argtypes = [PyCapsule, PyCapsule_Destructor]

#
# void *PyCapsule_GetPointer(PyObject *capsule, const char *name)
#
ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [PyCapsule, ctypes.c_char_p]

#
# int PyCapsule_SetPointer(PyObject *capsule, void *pointer)
#
ctypes.pythonapi.PyCapsule_SetPointer.restype = ctypes.c_int
ctypes.pythonapi.PyCapsule_SetPointer.argtypes = [PyCapsule, ctypes.c_void_p]

#
# const char *PyCapsule_GetName(PyObject *capsule)
#
ctypes.pythonapi.PyCapsule_GetName.restype = ctypes.c_char_p
ctypes.pythonapi.PyCapsule_GetName.argtypes = [PyCapsule]

#
# int PyCapsule_SetName(PyObject *capsule, const char *name)
#
ctypes.pythonapi.PyCapsule_SetName.restype = ctypes.c_int
ctypes.pythonapi.PyCapsule_SetName.argtypes = [PyCapsule, ctypes.c_char_p]

#
# int PyCapsule_IsValid(PyObject *capsule, const char *name)
#
ctypes.pythonapi.PyCapsule_IsValid.restype = ctypes.c_int
ctypes.pythonapi.PyCapsule_IsValid.argtypes = [PyCapsule, ctypes.c_char_p]

#
# void *PyCapsule_GetContext(PyObject *capsule)
#
ctypes.pythonapi.PyCapsule_GetContext.restype = ctypes.c_void_p
ctypes.pythonapi.PyCapsule_GetContext.argtypes = [PyCapsule]

#
# int PyCapsule_SetContext(PyObject *capsule, void *context)
#
ctypes.pythonapi.PyCapsule_SetContext.restype = ctypes.c_int
ctypes.pythonapi.PyCapsule_SetContext.argtypes = [PyCapsule, ctypes.c_void_p]

#
# void *PyCapsule_Import(const char *name, int no_block)
#
ctypes.pythonapi.PyCapsule_Import.restype = ctypes.c_void_p
ctypes.pythonapi.PyCapsule_Import.argtypes = [PyCapsule, ctypes.c_int]


class Capsule:
    """OOP wrapper of a PyCapsule object
    """

    @classmethod
    def new(cls, pointer: int, name: str, destructor=None):
        """Create a Capsule encapsulating the pointer. The pointer argument may not be 0.

        destructor is None or a Python function with signature:

          def destructor(capsule: Capsule) -> None:
              # code to clean up memory held by capsule pointer...
              return
        """
        if isinstance(name, str):
            name = name.encode()

        null_destructor = ctypes.cast(None, PyCapsule_Destructor)
        capsule = ctypes.pythonapi.PyCapsule_New(pointer, name, null_destructor)
        obj = cls(capsule)
        if destructor is not None:
            obj.set_destructor(destructor)
        # name must outlive the capsule object:
        obj.__name = name
        return obj

    def __init__(self, capsule):
        """Wrapper of an existing PyCapsule object.
        """
        if not (type(capsule).__name__ == 'PyCapsule' and type(capsule).__module__ == 'builtins'
                or isinstance(capsule, PyCapsule)): 
            # ctypes does not expose PyCapsule_CheckExact
            raise TypeError(f'expected capsule object but got {type(capsule).__module__}.{type(capsule).__name__}')
        self.capsule = capsule

    def is_valid(self, name: str) -> bool:
        """Determines whether or not capsule is a valid capsule. A valid
        capsule is a PyCapsule object, has a non-None pointer stored
        in it, and its internal name matches the name parameter.
        """
        if isinstance(name, str):
            name = name.encode()
        return bool(ctypes.pythonapi.PyCapsule_IsValid(self.capsule, name))

    def get_pointer(self, name: str) -> int:
        """Retrieve the pointer stored in the capsule.
        """
        if isinstance(name, str):
            name = name.encode()
        return ctypes.pythonapi.PyCapsule_GetPointer(self.capsule, name)

    def set_pointer(self, pointer: int):
        """Set the void pointer inside capsule to pointer. The pointer may not be 0.
        """
        status = ctypes.pythonapi.PyCapsule_SetPointer(self.capsule, pointer)
        assert status == 0

    def get_name(self) -> str:
        """Return the current name stored in capsule.

        It is legal for a capsule to have a None name.
        """
        name = ctypes.pythonapi.PyCapsule_GetName(self.capsule)
        if isinstance(name, bytes):
            name = name.decode()
        return name

    def set_name(self, name: str):
        """Set the name inside capsule to name. The name can be None.
        """
        if isinstance(name, str):
            name = name.encode()
        status = ctypes.pythonapi.PyCapsule_SetName(self.capsule, name)
        assert status == 0
        # name must outlive the capsule object:
        self.__name = name

    def get_context(self) -> int:
        """Return the current context stored in the capsule.
        """
        context = ctypes.pythonapi.PyCapsule_GetContext(self.capsule)
        return context

    def set_context(self, context: int):
        """Set the context pointer inside capsule to context. The context can be 0.
        """
        status = ctypes.pythonapi.PyCapsule_SetContext(self.capsule, context)
        assert status == 0

    def get_destructor(self) -> PyCapsule_Destructor:
        """Return the current destructor stored in the capsule.
        """
        return ctypes.pythonapi.PyCapsule_GetDestructor(self.capsule)

    # User-provided destructors must outlive capsules.
    registered_destructors = dict()

    def set_destructor(self, destructor: PyCapsule_Destructor):
        """Set the destructor inside capsule to destructor. The destructor can be 0.
        """
        if isinstance(destructor, PyCapsule_Destructor):
            capsule_destructor = destructor
        elif destructor is None:
            capsule_destructor = ctypes.cast(None, PyCapsule_Destructor)            
        else:
            @PyCapsule_Destructor
            def capsule_destructor(capsule_ptr):
                # Warning: raw_capsule is py_object that value must
                # not be accessed to avoid recurssive
                # capsule_destructor calls: accessing value
                # creates a new instance.
                py_capsule = ctypes.cast(capsule_ptr, PyCapsule)
                # using destructor in the body of capsule_destructor
                # ensures that destructor outlives capsule_destructor
                destructor(Capsule(py_capsule))
                Capsule.registered_destructors.pop(capsule_ptr)

            self.registered_destructors[id(self.capsule)] = capsule_destructor

        status = ctypes.pythonapi.PyCapsule_SetDestructor(self.capsule, capsule_destructor)
        assert status == 0


# Sanity check:
import atexit
import gc
@atexit.register
def _registered_destructors_atexit():
    gc.collect()
    if len(Capsule.registered_destructors) > 0:
        print(f'dlpack: {len(Capsule.registered_destructors)} Capsule destructors have leaked!!!')
