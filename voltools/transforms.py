import time
import aenum

import numpy as np
import cupy as cp
from typing import Union, Tuple
from pathlib import Path

from utils import scale_matrix, shear_matrix, rotation_matrix, translation_matrix, transform_matrix, \
    compute_prefilter_workgroup_dims, compute_elementwise_launch_dims


class Interpolations(aenum.Enum):
    _settings_ = aenum.NoAlias

    LINEAR = 'linearTex3D'
    BSPLINE = 'cubicTex3D'
    BSPLINE_SIMPLE = 'cubicTex3DSimple'
    FILT_BSPLINE = 'cubicTex3D'
    FILT_BSPLINE_SIMPLE = 'cubicTex3DSimple'


def transform(volume: Union[np.ndarray, cp.ndarray],
              scale: Union[Tuple[float, float, float], np.ndarray] = None,
              shear: Union[Tuple[float, float, float], np.ndarray] = None,
              rotation: Union[Tuple[float, float, float], np.ndarray] = None,
              rotation_units: str = 'deg', rotation_order: str = 'rzxz',
              translation: Union[Tuple[float, float, float], np.ndarray] = None,
              center: Union[Tuple[float, float, float], np.ndarray] = None,
              interpolation: Interpolations = Interpolations.LINEAR,
              profile: bool = False, inplace: bool = True) -> Union[cp.ndarray, None]:

    m = transform_matrix(scale, shear, rotation, rotation_units, rotation_order,
                         translation, center)
    return affine(volume, m, interpolation, profile, inplace)


def translate(volume: Union[np.ndarray, cp.ndarray],
              translation: Tuple[float, float, float],
              interpolation: Interpolations = Interpolations.LINEAR,
              profile: bool = False, inplace: bool = True) -> Union[cp.ndarray, None]:

    m = translation_matrix(translation)
    return affine(volume, m, interpolation, profile, inplace)


def shear(volume: Union[np.ndarray, cp.ndarray],
          coefficients: Union[float, Tuple[float, float, float]],
          interpolation: Interpolations = Interpolations.LINEAR,
          profile: bool = False, inplace: bool = True) -> Union[cp.ndarray, None]:

    # passing just one float is uniform scaling
    if isinstance(coefficients, float):
        coefficients = (coefficients, coefficients, coefficients)

    m = shear_matrix(coefficients)
    return affine(volume, m, interpolation, profile, inplace)


def scale(volume: Union[np.ndarray, cp.ndarray],
          coefficients: Union[float, Tuple[float, float, float]],
          interpolation: Interpolations = Interpolations.LINEAR,
          profile: bool = False, inplace: bool = True) -> Union[cp.ndarray, None]:

    # passing just one float is uniform scaling
    if isinstance(coefficients, float):
        coefficients = (coefficients, coefficients, coefficients)

    m = scale_matrix(coefficients)
    return affine(volume, m, interpolation, profile, inplace)


def rotate(volume: Union[np.ndarray, cp.ndarray],
           rotation: Tuple[float, float, float],
           rotation_units: str = 'deg',
           rotation_order: str = 'rzxz',
           interpolation: Interpolations = Interpolations.LINEAR,
           profile: bool = False, inplace: bool = True) -> Union[cp.ndarray, None]:

    m = rotation_matrix(rotation=rotation, rotation_units=rotation_units, rotation_order=rotation_order)
    return affine(volume, m, interpolation, profile, inplace)


def affine(volume: Union[np.ndarray, cp.ndarray],
           transform_m: np.ndarray,
           interpolation: Interpolations = Interpolations.LINEAR,
           profile: bool = False, inplace: bool = True) -> Union[cp.ndarray, None]:

    t_start = time.perf_counter()

    if isinstance(volume, np.ndarray):
        volume = cp.asarray(volume)

    # texture setup
    ch = cp.cuda.texture.ChannelFormatDescriptor(32, 0, 0, 0, cp.cuda.runtime.cudaChannelFormatKindFloat)
    arr = cp.cuda.texture.CUDAarray(ch, *volume.shape[::-1]) # CUDAArray: last dimension = fastest changing dimension
    res = cp.cuda.texture.ResourceDescriptor(cp.cuda.runtime.cudaResourceTypeArray, cuArr=arr)
    tex = cp.cuda.texture.TextureDescriptor((cp.cuda.runtime.cudaAddressModeBorder,
                                             cp.cuda.runtime.cudaAddressModeBorder,
                                             cp.cuda.runtime.cudaAddressModeBorder),
                                            cp.cuda.runtime.cudaFilterModeLinear,
                                            cp.cuda.runtime.cudaReadModeElementType)
    texobj = cp.cuda.texture.TextureObject(res, tex)

    # prefilter if required and upload to texture
    if interpolation == Interpolations.FILT_BSPLINE or interpolation == Interpolations.FILT_BSPLINE_SIMPLE:
        if inplace:
            volume = _bspline_prefilter(volume)
            arr.copy_from(volume)
        else:
            prefiltered_volume = _bspline_prefilter(volume)
            arr.copy_from(prefiltered_volume)
    else:
        arr.copy_from(volume)

    # kernel setup
    kernel = _get_transform_kernel(interpolation)
    dims = cp.asarray(volume.shape, dtype=cp.uint32)
    xform = cp.asarray(transform_m)
    dim_grid, dim_blocks = compute_elementwise_launch_dims(volume.shape)

    if inplace:
        volume.fill(0)
        kernel(dim_grid, dim_blocks, (volume, texobj, xform, dims))
    else:
        output = cp.zeros_like(volume)
        kernel(dim_grid, dim_blocks, (output, texobj, xform, dims))

    if profile:
        cp.cuda.get_current_stream().synchronize()
        time_took = (time.perf_counter() - t_start) * 1000
        print(f'transform finished in {time_took:.3f}ms')

    if inplace:
        del texobj, xform, dims
        return volume
    else:
        del texobj, xform, dims
        return output

@cp.memoize()
def _get_transform_kernel(interpolation: Interpolations = Interpolations.LINEAR) -> cp.RawKernel:

    code = f'''
        #include "helper_math.h"
        #include "bspline.h"
        #include "helper_interpolation.h"
        
        extern "C" {{
            inline __device__ int get_z_idx(const int i, const uint4* const dims) {{
                return i / (dims[0].y * dims[0].z);
            }}
            inline __device__ int get_y_idx(const int i, const uint4* const dims) {{
                return (i % (dims[0].y * dims[0].z)) / dims[0].z;
            }}
            inline __device__ int get_x_idx(const int i, const uint4* const dims) {{
                return (i % (dims[0].y * dims[0].z)) % dims[0].z;
            }}

            __global__ void transform(float* const volume,
                                      cudaTextureObject_t texture,
                                      const float4* const xform,
                                      const uint4* const dims) {{
            
                unsigned tid = threadIdx.x;
                unsigned total_threads = gridDim.x*blockDim.x;
                unsigned cta_start = blockDim.x*blockIdx.x;
                unsigned i;
                unsigned n = dims[0].x * dims[0].y * dims[0].z;
                
                for (i = cta_start + tid; i < n; i += total_threads) {{
                    int z = get_x_idx(i, dims);
                    int y = get_y_idx(i, dims);
                    int x = get_z_idx(i, dims);
                    
                    float4 voxf = make_float4(((float)x), ((float)y), ((float)z), 1.0f);
                    
                    float3 ndx;
                    ndx.z = dot(voxf, xform[0]) + .5f;
                    ndx.y = dot(voxf, xform[1]) + .5f;
                    ndx.x = dot(voxf, xform[2]) + .5f;
                    
                    if (ndx.x < 0 || ndx.y < 0 || ndx.z < 0 || ndx.x >= dims[0].z || ndx.y >= dims[0].y || ndx.z >= dims[0].x) {{
                        continue;
                    }}
                    
                    volume[i] = {interpolation.value}(texture, ndx);
                }}
            }}
        }}
    '''
    incl_path = str((Path(__file__).parent / 'kernels').absolute())
    kernel = cp.RawKernel(code=code, name='transform', options=('-I', incl_path))
    return kernel

def _bspline_prefilter(volume: cp.ndarray):

    code = f'''
        #include "helper_math.h"
        #include "bspline.h"
    '''
    incl_path = str((Path(__file__).parent / 'kernels').absolute())
    prefilter_x = cp.RawKernel(code=code, name='SamplesToCoefficients3DX', options=('-I', incl_path))
    prefilter_y = cp.RawKernel(code=code, name='SamplesToCoefficients3DY', options=('-I', incl_path))
    prefilter_z = cp.RawKernel(code=code, name='SamplesToCoefficients3DZ', options=('-I', incl_path))

    slice_stride = volume.strides[1]
    dim_grid, dim_block = compute_prefilter_workgroup_dims(volume.shape)#[::-1])
    dims = cp.asarray(volume.shape[::-1], dtype=cp.int32)

    prefilter_x(dim_grid[0], dim_block[0], (volume, slice_stride, dims))
    prefilter_y(dim_grid[1], dim_block[1], (volume, slice_stride, dims))
    prefilter_z(dim_grid[2], dim_block[2], (volume, slice_stride, dims))

    return volume





if __name__ == '__main__':

    import matplotlib.pyplot as plt
    plt.ion()

    #depth, height, width = 5, 6, 7
    depth, height, width = 100, 200, 300

    #vol = cp.arange(depth * height * width, dtype=cp.float32).reshape(depth, height, width)
    vol = cp.random.random((depth, height, width), dtype=cp.float32) * 1000
    #vol = cp.ones((depth, height, width), dtype=cp.float32)

    fig, ax = plt.subplots(2, 3, sharex='col', sharey='col')
    ax[0,0].set_ylabel('cpu')
    ax[1,0].set_ylabel('gpu')
    ax[0,0].set_title('slice 0')
    ax[0,1].set_title('slice 50')
    ax[0,2].set_title('slice 99')

    m = transform_matrix(translation=(0, 0, 0),
                         rotation=(25, 0, 0), rotation_order='rzxz',
                         center=np.divide(vol.shape, 2, dtype=np.float32))

    # cpu reference
    from scipy.ndimage import affine_transform
    cpu = affine_transform(vol.get(), m)
    ax[0, 0].imshow(cpu[0])
    ax[0, 1].imshow(cpu[50])
    ax[0, 2].imshow(cpu[99])

    # gpu output
    gpu = affine(vol, m, inplace=False, profile=False, interpolation=Interpolations.FILT_BSPLINE)
    ax[1, 0].imshow(gpu[0].get())
    ax[1, 1].imshow(gpu[50].get())
    ax[1, 2].imshow(gpu[99].get())

    fig.show()
    plt.pause(2)
