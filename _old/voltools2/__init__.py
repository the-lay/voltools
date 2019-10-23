__version__ = '0.1.0'

# Exports
__all__ = [
    'Volume',
]

# Imports
from .volume import Volume


# PyCUDA initialization
try:
    from pycuda import autoinit as __c

    print('voltools {} running on {} ({}.{} CC)'.format(
        __version__, __c.device.name(), *__c.device.compute_capatibility()
    ))

except Exception as e:
    print(e)
    raise e
