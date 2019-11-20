__version__ = '0.2.2'

try:
    import cupy
except ImportError:
    print('cupy not found. Please install cupy>=7.0.0b4:\npip install cupy>=7.0.0b4')

from .transforms import Interpolations, scale, shear, rotate, translate, transform, affine
from .volume import StaticVolume
from . import utils
