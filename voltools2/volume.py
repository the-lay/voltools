import numpy as np

from pycuda import gpuarray as gu
from pycuda import driver
from pycuda.compiler import SourceModule

# from .transforms import
from utils.kernels import get_volume_kernel, fits_on_gpu, get_transform_kernel

class Volume:

    def __init__(self, data):

        # Validation
        if not (isinstance(data, np.ndarray) and data.ndim == 3):
            raise ValueError('Volume class expects a 3-dimensional numpy array as input')

        # Check size
        if not fits_on_gpu(data.nbytes):
            raise ValueError('OOM: require {} GB for this volume, but GPU has {} GB'.format(
                data.nbytes / (1000**3),
                driver.Context.get_device().total_memory() / (1000**3)))

        # Volume attributes
        self.shape = data.shape
        self.size = np.prod(data.shape)
        self.nbytes = data.nbytes
        self.strides = data.strides
        self.dtype = data.dtype

        # GPU
        self._kernel = get_volume_kernel(self.dtype)
        self.shape_d = gu.to_gpu(np.array(self.shape[::-1], dtype=np.int32))
        self.data_d = gu.to_gpu(data)

#
# class ProjectingVolume(Volume):
#
#     def __init__(self, data):
#         super(ProjectingVolume, self).__init__(data)



# development
d = np.random.rand(500, 500, 500)
a = Volume(d)
b = get_transform_kernel(a.dtype)

print('pause')





