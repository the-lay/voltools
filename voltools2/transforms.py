import numpy as np

from pycuda import gpuarray, driver
from pycuda.tools import dtype_to_ctype, context_dependent_memoize

from .volume import Volume
from .utils import scale_matrix, shear_matrix, rotation_matrix, translation_matrix
from .utils import VoltoolsElementwiseKernel, gpuarray_to_texture, _kernels_folder

from typing import Union, Tuple
from enum import Enum

class Interpolations(Enum):
    LINEAR = 'linear'
    # TODO


def rotate(data: Union[np.ndarray, Volume],
           rotation: Tuple[float, float, float],
           rotation_units: str = 'deg',
           rotation_order: str = 'rzxz',
           center: Union[None, Tuple[float, float, float], np.ndarray] = None,
           interpolation: str = 'linear',
           profile: bool = False, return_cpu: bool = True):

    # compute center of rotation
    if center is None:
        center = tuple([shape // 2 for shape in data.shape])
    elif len(center) != 3:
        raise ValueError('Center argument must be length of 3')

    # combine transformation matrix
    pret_m = translation_matrix(center)
    rot_m = rotation_matrix(rotation, rotation_units=rotation_units, rotation_order=rotation_order)
    post_m = translation_matrix([-1 * x for x in center])
    transform_m = np.dot(post_m, np.dot(rot_m, pret_m))

    return affine(data, transform_m, interpolation=interpolation, profile=profile, return_cpu=return_cpu)


def scale(data: Union[np.ndarray, Volume],
          coefficients: Union[float, Tuple[float, float, float]],
          interpolation: str = 'linear',
          profile: bool = False, return_cpu: bool = True):

    # passing just one float is uniform scaling
    if isinstance(coefficients, float):
        coefficients = (coefficients, coefficients, coefficients)

    transform_m = scale_matrix(coefficients)
    return affine(data, transform_m, interpolation=interpolation, profile=profile, return_cpu=return_cpu)


def shear(data: Union[np.ndarray, Volume],
          coefficients: Union[float, Tuple[float, float, float]],
          interpolation: str = 'linear',
          profile: bool = False, return_cpu: bool = True):

    # passing just one float is uniform shearing
    if isinstance(coefficients, float):
        coefficients = (coefficients, coefficients, coefficients)

    transform_m = shear_matrix(coefficients)
    return affine(data, transform_m, interpolation=interpolation, profile=profile, return_cpu=return_cpu)


def translate(data: Union[np.ndarray, Volume],
              translation: Tuple[float, float, float],
              interpolation: str = 'linear',
              profile: bool = False, return_cpu: bool = True):

    transform_m = translation_matrix(translation)
    return affine(data, transform_m, interpolation=interpolation, profile=profile, return_cpu=return_cpu)


# A generic routine for multiple transformations
# Order: PreTrans -> Scale -> Shear -> Rotate -> PostTrans -> Trans
def transform(data: Union[np.ndarray, Volume],
              scale: Union[None, float, Tuple[float, float, float]] = None,
              shear: Union[None, float, Tuple[float, float, float]] = None,
              rotation: Union[None, Tuple[float, float, float]] = None,
              rotation_units: str = 'deg', rotation_order: str = 'rzxz',
              translation: Union[None, Tuple[float, float, float]] = None,
              center: Union[None, Tuple[float, float, float], np.ndarray] = None,
              interpolation: str = 'linear',
              profile: bool = False, return_cpu: bool = True):

    if center is None:
        center = tuple([shape // 2 for shape in data.shape[::-1]])
    elif len(center) != 3:
        raise ValueError('Center argument must be a tuple or np.ndarray with length of 3')

    transform_m = np.identity(4, dtype=np.float32)

    # Matrix multiplication is right sided, so we start with the last transformations
    # Translation
    if translation is not None:
        transform_m = np.dot(transform_m, translation_matrix(translation))

    # Post-translation
    transform_m = np.dot(transform_m, translation_matrix([-1 * x for x in center]))

    # Rotation
    if rotation is not None:
        transform_m = np.dot(transform_m, rotation_matrix(rotation,
                                                          rotation_units=rotation_units,
                                                          rotation_order=rotation_order))

    # Shear
    if shear is not None:
        transform_m = np.dot(transform_m, shear_matrix(shear))

    # Scale
    if scale is not None:
        transform_m = np.dot(transform_m, scale_matrix(scale))

    # Pre-translation
    transform_m = np.dot(transform_m, translation_matrix(center))

    # Homogeneous matrix
    transform_m /= transform_m[3, 3]

    # Call the affine transformation routine with constructed matrix
    return affine(data, transform_m, interpolation=interpolation, profile=profile, return_cpu=return_cpu)


# generic method for any transformation from given transform_m
def affine(data: Union[np.ndarray, Volume], transform_m: np.ndarray, interpolation: str = 'linear',
           profile: bool = False, return_cpu: bool = True):

    # Validate inputs
    __validate_transform_m(transform_m)
    __validate_input_data(data)
    __validate_interpolation(interpolation)

    # numpy route
    if isinstance(data, np.ndarray):
        # get kernel and transform texture
        kernel = get_transform_kernel(data.dtype, interpolation)
        # upload data and move it to transform texture
        d_data = gpuarray.to_gpu(data)
        gpuarray_to_texture(d_data, kernel.texture)

        # populate affine transform arguments
        d_shape = gpuarray.to_gpu(np.array(data.shape, dtype=np.int32))
        transform_m_t = transform_m.transpose().copy()
        d_transform = gpuarray.to_gpu(transform_m_t)
        d_data.fill(0)

        # call kernel
        kernel(d_data, d_shape, d_transform, profile=profile)

        d_shape.gpudata.free()
        d_transform.gpudata.free()

        if return_cpu:
            result = d_data.get()
            d_data.gpudata.free()
            return result
        else:
            return d_data

    # Volume route
    elif isinstance(data, Volume):

        if interpolation != data.interpolation:
            print(f'Warning: Transformation will be done with {data.interpolation} interpolation.')

        return data.affine_transform(transform_m=transform_m, return_cpu=return_cpu, profile=profile)


# Affine transform elementwise kernel
@context_dependent_memoize
def get_transform_kernel(dtype, interpolation: str = 'linear', warm_up: bool = True):
    kernel = VoltoolsElementwiseKernel(
        args='{}* const volume, const int4* const dims, const float4* const xform'
            .format(dtype_to_ctype(dtype)),

        preamble="""
            }} // helper_math should be added without C extern
            #include "helper_math.h"
            #include "helper_indexing.h"
            #include "helper_textures.h"
            extern "C" {{
            texture<{}, 3, cudaReadModeElementType> coeff_tex;
        """.format(dtype_to_ctype(dtype)),

        body="""
            // indices
            int x = get_x_idx(i, dims);
            int y = get_y_idx(i, dims);
            int z = get_z_idx(i, dims);

            // thread idx to texels (adding + .5f to be in the center of texel)
            float4 voxf = make_float4(((float)x) + .5f, ((float)y) + .5f, ((float)z) + .5f, 1.0f);

            // apply matrix
            float4 ndx;
            ndx.x = dot(voxf, xform[0]);
            ndx.y = dot(voxf, xform[1]);
            ndx.z = dot(voxf, xform[2]);

            // get interpolated value
            float v = {}(coeff_tex, ndx);
            volume[i] = v;
        """.format('linearTex3D' if interpolation == 'linear' else
                                'cubicTex3D' if interpolation == 'bspline' else
                                'cubicTex3DSimple' if interpolation == 'bsplinehq' else
                                'WRONG_INTERPOLATION'),

        name='transform3d',
        include_dirs=[str(_kernels_folder)],
        keep=True
    )

    kernel.texture = kernel.get_texref('coeff_tex')
    kernel.texture.set_filter_mode(driver.filter_mode.LINEAR)
    kernel.texture.set_address_mode(0, driver.address_mode.BORDER)
    kernel.texture.set_address_mode(1, driver.address_mode.BORDER)
    kernel.texture.set_address_mode(2, driver.address_mode.BORDER)
    kernel.texture.set_flags(driver.TRSF_READ_AS_INTEGER)

    if warm_up:
        vol = gpuarray.zeros(shape=(15, 15, 15), dtype=dtype)
        vol.fill(1)
        shape = gpuarray.to_gpu(np.array([15, 15, 15], dtype=np.int32))
        xform = gpuarray.to_gpu(np.eye(4, dtype=np.float32))
        kernel(vol, shape, xform)

        del vol, shape, xform

    return kernel












def __validate_transform_m(transform_m: np.ndarray):
    if transform_m.ndim != 2:
        raise ValueError('Transform matrix must be a 2D numpy array')

    if transform_m.shape != (4, 4):
        raise ValueError('Transformation matrix must be a homogeneous 4x4 matrix.')

def __validate_input_data(data: Union[np.ndarray, Volume]):
    if isinstance(data, np.ndarray) and data.ndim != 3:
        raise ValueError('Data must have 3 dimensions')

def __validate_interpolation(interpolation: str):
    supported_modes = ['linear', 'bspline', 'bsplinehq']
    if interpolation not in supported_modes:
        raise ValueError(f'Interpolation mode must one of these: {supported_modes}')


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # plt.ion()

    d = np.random.rand(500, 500, 500).astype(np.float32)
    a = Volume(d)

    for i in range(0, 360, 17):
        rot_d = rotate(a, rotation=(i, i, i), rotation_units='deg', rotation_order='rzxz', profile=True, return_cpu=True)
        plt.imshow(rot_d[250], vmin=d[250].min(), vmax=d[250].max())
        plt.show()

    print('pause')
    # b[0](a.data_d, a.shape_d, gu.to_gpu(translation_matrix((50, 50, 50))))
    #
    # ### DEVELOPMENT
    # import matplotlib.pyplot as plt
    # plt.ion()
    #
    # d = np.random.rand(280, 920, 920).astype(np.float32) * 1000
    # # d = np.arange(0, 157 * 197 * 271, dtype=np.float32).reshape((157, 197, 271))
    # plt.imshow(d[0])
    # plt.show()
    #
    # rot_d = transform(d, rotation=(1, 0, 0), rotation_order='rzxz', profile=True)
    # # rot_d = affine(d, np.identity(4, dtype=np.float32))
    # plt.imshow(rot_d[0], vmin=d[0].min(), vmax=d[0].max())
    # plt.show()
    #
    # rot_d = transform(d, rotation=(5, 0, 0), rotation_order='rzxz', profile=True)
    # plt.imshow(rot_d[0], vmin=d[0].min(), vmax=d[0].max())
    # plt.show()
    #
    # rot_d = transform(d, rotation=(15, 0, 0), rotation_order='rzxz', profile=True)
    # plt.imshow(rot_d[0], vmin=d[0].min(), vmax=d[0].max())
    # plt.show()
    #
    # rot_d = transform(d, rotation=(25, 0, 0), rotation_order='rzxz', profile=True)
    # plt.imshow(rot_d[0], vmin=d[0].min(), vmax=d[0].max())
    # plt.show()
    #
    # rot_d = transform(d, rotation=(35, 0, 0), rotation_order='rzxz', profile=True)
    # plt.imshow(rot_d[0], vmin=d[0].min(), vmax=d[0].max())
    # plt.show()
    #
    # rot_d = transform(d, rotation=(45, 0, 0), rotation_order='rzxz', profile=True)
    # plt.imshow(rot_d[0], vmin=d[0].min(), vmax=d[0].max())
    # plt.show()
    #
    # # rot_d = affine(d, np.identity(4, dtype=np.float32))
    # # driver.Context.synchronize()
    # # plt.imshow(rot_d[0], vmin=d[0].min(), vmax=d[0].max())
    # # plt.show()
    #
    # # print(np.allclose(d, rot_d))
    # print('stop hammer time')
