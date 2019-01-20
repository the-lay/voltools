from .version import __version__

# Exports
__all__ = [
    'Volume',
    'create_projections'
]

# Imports
from .volume import Volume
from .scripts import *

# PyCUDA initialization
try:
    from pycuda import autoinit as __c

    print('voltools {} uses CUDA on {} with {}.{} CC.'.format(
        __version__, __c.device.name(), *__c.device.compute_capability()))

except Exception as e:
    print(e)
    raise e
