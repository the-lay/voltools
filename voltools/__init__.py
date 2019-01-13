from .version import __version__

# Exports
__all__ = [
    'Volume'
]

# Imports
from .volume import Volume

# PyCUDA initialization
try:
    from pycuda import autoinit as __cuda

    print('voltools {} uses CUDA on {} with {}.{} CC.'.format(
        __version__, __cuda.device.name(), *__cuda.device.compute_capability()))

except Exception as e:
    print(e)
    raise e
