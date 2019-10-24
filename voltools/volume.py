import numpy as np
import cupy as cp
import time
from typing import Tuple, Union
from .transforms import Interpolations, _get_transform_kernel, _bspline_prefilter
from .utils import compute_elementwise_launch_dims,\
    scale_matrix, shear_matrix, rotation_matrix, translation_matrix, transform_matrix

class StaticVolume:

    def __init__(self, data: cp.ndarray, interpolation: Interpolations = Interpolations.LINEAR):

        if data.ndim != 3:
            raise ValueError('Expected a 3D array')

        self.shape = data.shape
        self.d_shape = cp.asarray(data.shape, dtype=cp.uint32)
        self.d_type = data.dtype
        self.affine_kernel = _get_transform_kernel(interpolation)

        # init texture
        ch = cp.cuda.texture.ChannelFormatDescriptor(32, 0, 0, 0, cp.cuda.runtime.cudaChannelFormatKindFloat)
        arr = cp.cuda.texture.CUDAarray(ch, *data.shape[::-1])
        self.res = cp.cuda.texture.ResourceDescriptor(cp.cuda.runtime.cudaResourceTypeArray, cuArr=arr)
        self.tex = cp.cuda.texture.TextureDescriptor((cp.cuda.runtime.cudaAddressModeBorder,
                                                     cp.cuda.runtime.cudaAddressModeBorder,
                                                     cp.cuda.runtime.cudaAddressModeBorder),
                                                     cp.cuda.runtime.cudaFilterModeLinear,
                                                     cp.cuda.runtime.cudaReadModeElementType)
        self.tex_obj = cp.cuda.texture.TextureObject(self.res, self.tex)

        # prefilter if required and upload to texture
        if interpolation == Interpolations.FILT_BSPLINE or interpolation == Interpolations.FILT_BSPLINE_SIMPLE:
            _bspline_prefilter(data)
        arr.copy_from(data)

        # workgroup dims
        self.dim_grid, self.dim_blocks = compute_elementwise_launch_dims(data.shape)

    def affine(self, transform_m: np.ndarray, profile: bool = False) -> cp.ndarray:

        t_start = time.perf_counter()

        # kernel setup
        xform = cp.asarray(transform_m)
        output = cp.zeros(tuple(self.d_shape.get().tolist()), dtype=self.d_type)

        # launch
        self.affine_kernel(self.dim_grid, self.dim_blocks, (output, self.tex_obj, xform, self.d_shape))

        if profile:
            cp.cuda.get_current_stream().synchronize()
            time_took = (time.perf_counter() - t_start) * 1000
            print(f'transform finished in {time_took:.3f}ms')

        del xform
        return output

    def transform(self, scale: Union[Tuple[float, float, float], np.ndarray] = None,
                  shear: Union[Tuple[float, float, float], np.ndarray] = None,
                  rotation: Union[Tuple[float, float, float], np.ndarray] = None,
                  rotation_units: str = 'deg', rotation_order: str = 'rzxz',
                  translation: Union[Tuple[float, float, float], np.ndarray] = None,
                  center: Union[Tuple[float, float, float], np.ndarray] = None,
                  profile: bool = False) -> cp.ndarray:

        if center is None:
            center = np.divide(self.shape, 2, dtype=np.float32)

        m = transform_matrix(scale, shear, rotation, rotation_units, rotation_order,
                             translation, center)
        return self.affine(m, profile)

    def translate(self,
                  translation: Tuple[float, float, float],
                  profile: bool = False) -> cp.ndarray:

        m = translation_matrix(translation)
        return self.affine(m, profile)

    def shear(self,
              coefficients: Union[float, Tuple[float, float, float]],
              profile: bool = False) -> cp.ndarray:

        # passing just one float is uniform scaling
        if isinstance(coefficients, float):
            coefficients = (coefficients, coefficients, coefficients)

        m = shear_matrix(coefficients)
        return self.affine(m, profile)

    def scale(self,
              coefficients: Union[float, Tuple[float, float, float]],
              profile: bool = False) -> cp.ndarray:

        # passing just one float is uniform scaling
        if isinstance(coefficients, float):
            coefficients = (coefficients, coefficients, coefficients)

        m = scale_matrix(coefficients)
        return self.affine(m, profile)

    def rotate(self,
               rotation: Tuple[float, float, float],
               rotation_units: str = 'deg',
               rotation_order: str = 'rzxz',
               profile: bool = False) -> cp.ndarray:

        m = rotation_matrix(rotation=rotation, rotation_units=rotation_units, rotation_order=rotation_order)
        return self.affine(m, profile)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    depth, height, width = 200, 200, 200
    d = cp.random.random((depth, height, width), dtype=cp.float32) * 1000

    volume = StaticVolume(d, interpolation=Interpolations.FILT_BSPLINE)

    for i in range(0, 180, 3):
        transformed_vol = volume.transform(rotation=(0, i, 0), rotation_units='deg', rotation_order='rzxz', profile=True)
        plt.imshow(transformed_vol[depth//2].get())
        plt.show()
