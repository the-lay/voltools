import numpy as np
import cupy as cp
from typing import Tuple, Union
from .transforms import Interpolations, _get_transform_kernel, _bspline_prefilter
from .utils import compute_elementwise_launch_dims,\
    scale_matrix, shear_matrix, rotation_matrix, translation_matrix, transform_matrix

class StaticVolume:

    def __init__(self, data: cp.ndarray, interpolation: Interpolations = Interpolations.LINEAR,
                 device_id: int = 0):

        if data.ndim != 3:
            raise ValueError('Expected a 3D array')

        cp.cuda.Device(device_id).use()
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
        if interpolation.name.startswith('FILT_BSPLINE'):
            prefiltered_volume = _bspline_prefilter(data.copy())  # copy to avoid modifying existing volume
            arr.copy_from(prefiltered_volume)
        else:
            arr.copy_from(data)

        # workgroup dims
        self.dim_grid, self.dim_blocks = compute_elementwise_launch_dims(data.shape)

    def affine(self, transform_m: np.ndarray, profile: bool = False, output: cp.ndarray = None) -> \
            Union[cp.ndarray, None]:

        if profile:
            stream = cp.cuda.Stream.null
            t_start = stream.record()

        # kernel setup
        xform = cp.asarray(transform_m)

        if output is None:
            output_vol = cp.zeros(tuple(self.d_shape.get().tolist()), dtype=self.d_type)
        else:
            output_vol = output

        # launch
        self.affine_kernel(self.dim_grid, self.dim_blocks, (output_vol, self.tex_obj, xform, self.d_shape))

        if profile:
            t_end = stream.record()
            t_end.synchronize()

            time_took = cp.cuda.get_elapsed_time(t_start, t_end)
            print(f'transform finished in {time_took:.3f}ms')

        del xform
        if output is None:
            return output_vol
        else:
            return None

    def transform(self, scale: Union[Tuple[float, float, float], np.ndarray] = None,
                  shear: Union[Tuple[float, float, float], np.ndarray] = None,
                  rotation: Union[Tuple[float, float, float], np.ndarray] = None,
                  rotation_units: str = 'deg', rotation_order: str = 'rzxz',
                  translation: Union[Tuple[float, float, float], np.ndarray] = None,
                  center: Union[Tuple[float, float, float], np.ndarray] = None,
                  profile: bool = False, output: cp.ndarray = None) -> Union[cp.ndarray, None]:

        if center is None:
            center = np.divide(self.shape, 2, dtype=np.float32)

        m = transform_matrix(scale, shear, rotation, rotation_units, rotation_order,
                             translation, center)
        return self.affine(m, profile, output)

    def translate(self,
                  translation: Tuple[float, float, float],
                  profile: bool = False, output: cp.ndarray = None) -> Union[cp.ndarray, None]:

        m = translation_matrix(translation)
        return self.affine(m, profile, output)

    def shear(self,
              coefficients: Union[float, Tuple[float, float, float]],
              profile: bool = False, output: cp.ndarray = None) -> Union[cp.ndarray, None]:

        # passing just one float is uniform scaling
        if isinstance(coefficients, float):
            coefficients = (coefficients, coefficients, coefficients)

        m = shear_matrix(coefficients)
        return self.affine(m, profile, output)

    def scale(self,
              coefficients: Union[float, Tuple[float, float, float]],
              profile: bool = False, output: cp.ndarray = None) -> Union[cp.ndarray, None]:

        # passing just one float is uniform scaling
        if isinstance(coefficients, float):
            coefficients = (coefficients, coefficients, coefficients)

        m = scale_matrix(coefficients)
        return self.affine(m, profile, output)

    def rotate(self,
               rotation: Tuple[float, float, float],
               rotation_units: str = 'deg',
               rotation_order: str = 'rzxz',
               profile: bool = False, output: cp.ndarray = None) -> Union[cp.ndarray, None]:

        m = rotation_matrix(rotation=rotation, rotation_units=rotation_units, rotation_order=rotation_order)
        return self.affine(m, profile, output)
