import numpy as np

from pycuda import gpuarray as gu
from pycuda import driver
from pycuda.compiler import SourceModule

from .utils import fits_on_gpu, get_transform_kernel, gpuarray_to_texture, get_correlation_kernels


class Volume:

    def __init__(self, data: np.array, prefilter: bool = False, interpolation: str = 'linear'):

        # Check size
        oom_check = fits_on_gpu(data.nbytes)
        if not oom_check[0]:
            raise ValueError('OOM: require {} GB for this volume, but GPU has {} GB'.format(
                data.nbytes / (1000**3),
                oom_check[1] / (1000**3)
            ))

        # Volume attributes
        self.shape = data.shape
        self.size = np.prod(data.shape)
        self.nbytes = data.nbytes
        self.strides = data.strides
        self.dtype = data.dtype
        self.prefiltered = False
        self.interpolation = interpolation

        # GPU
        # self._kernel = get_volume_kernel(self.dtype)
        #self._affine_kernel, self._affine_tex = get_transform_kernel(dtype=self.dtype, interpolation=self.interpolation)
        self.d_shape = gu.to_gpu(np.array(self.shape, dtype=np.int32))
        self.d_data = gu.to_gpu(data)

        # Load and compile kernels
        self._kernels = {
            'cor_num': get_correlation_kernels(dtype=self.dtype)[0],
            'cor_den': get_correlation_kernels(dtype=self.dtype)[1],
            'affine': get_transform_kernel(dtype=self.dtype, interpolation=self.interpolation)
        }

        if prefilter:
            self.prefilter()
        else:
            gpuarray_to_texture(self.d_data, self.kernel('affine').texture)

    def kernel(self, name: str):
        try:
            return self._kernels[name]
        except KeyError as e:
            raise e

    def mean(self, cpu: bool = True):
        if cpu:
            return np.mean(self.d_data.get())
        else:
            return (gu.sum(self.d_data) / self.size).get().item()

    # TODO
    def prefilter(self):
        self.prefiltered = True

    def affine_transform(self, transform_m: np.ndarray, profile: bool = False, return_cpu: bool = True):
        # upload transform matrix
        transform_m_t = transform_m.transpose().copy()
        d_transform = gu.to_gpu(transform_m_t)

        # call kernel
        self.kernel('affine')(self.d_data, self.d_shape, d_transform, profile=profile)

        # cleanup
        d_transform.gpudata.free()

        if return_cpu:
            result = self.d_data.get()
            return result
        else:
            return True





