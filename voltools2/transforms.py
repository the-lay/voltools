import numpy as np
from pycuda import autoinit
#from volume import Volume

from utils.matrices import scale_matrix, shear_matrix, rotation_matrix, translation_matrix
from utils.kernels import get_transform_kernel, gpuarray_to_texture
from pycuda import gpuarray as gu
from pycuda import driver

def rotate(data,
           rotation,
           rotation_units='deg',
           rotation_order='rzxz',
           center=None,
           interpolation=''):
    """

    :param data:
    :type data:
    :param rotation:
    :type rotation:
    :param rotation_units:
    :param rotation_order:
    :param center:
    :param interpolation:
    :return:
    """

    # Validate input
    __validate_input_data(data)

    if center is None:
        center = tuple([shape // 2 for shape in data.shape])
    elif len(center) != 3:
        raise ValueError('Center argument must be a tuple or np.ndarray with length of 3')

    pret_m = translation_matrix(center)
    rot_m = rotation_matrix(rotation, rotation_units=rotation_units, rotation_order=rotation_order)
    post_m = translation_matrix([-1 * x for x in center])

    transform_m = np.dot(post_m, np.dot(rot_m, pret_m))

    return affine(data, transform_m, interpolation=interpolation)


def scale(data,
          scale_coefficients=(1, 1, 1),
          interpolation=''):

    __validate_input_data(data)

    # Uniform scaling
    if isinstance(scale_coefficients, int):
        scale_coefficients = (scale_coefficients, scale_coefficients, scale_coefficients)

    transform_m = scale_matrix(scale_coefficients)

    return affine(data, transform_m, interpolation=interpolation)


def shear(data,
          shear_coefficients=(1, 1, 1),
          interpolation='linear'):

    # In case uniform shearing was intended
    if isinstance(shear_coefficients, int):
        shear_coefficients = (shear_coefficients, shear_coefficients, shear_coefficients)

    transform_m = shear_matrix(shear_coefficients)

    return affine(data, transform_m, interpolation=interpolation)


def translate(data,
              translation=(0, 0, 0),
              interpolation='linear'):

    transform_m = translation_matrix(translation)

    return affine(data, transform_m, interpolation=interpolation)


def transform(data,
              scale=None,
              shear=None,
              rotation=None, rotation_units='deg', rotation_order='rzxz',
              translation=None,
              center=None,
              interpolation='linear',
              profile=False):
    """
    A generic routine for multiple transformations.
    Order: PreTrans -> Scale -> Shear -> Rotate -> PostTrans -> Trans

    :param data:
    :param scale:
    :param shear:
    :param rotation:
    :param rotation_units:
    :param rotation_order:
    :param translation:
    :param center:
    :param interpolation:
    :param profile:
    :return:
    """

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
    return affine(data, transform_m, interpolation=interpolation, profile=profile)


# generic method for any transformation from given transform_m
def affine(data, transform_m, profile=False, interpolation='linear'):

    # Validate inputs
    __validate_transform_m(transform_m)
    # __validate_input_data(data)

    if isinstance(data, np.ndarray):
        # get kernel and texture
        kernel, texture = get_transform_kernel(data.dtype, interpolation)

        # populate texture
        d_data = gu.to_gpu(data)
        gpuarray_to_texture(d_data, texture)

        # populate affine transform arguments
        d_shape = gu.to_gpu(np.array(data.shape, dtype=np.int32))
        transform_m_t = transform_m.transpose().copy()
        d_transform = gu.to_gpu(transform_m_t)
        d_data.fill(0)

        # call kernel
        kernel(d_data, d_shape, d_transform, profile=profile)

        # get data back
        result = d_data.get()

        # free up data
        d_shape.gpudata.free()
        d_transform.gpudata.free()
        d_data.gpudata.free()

        return result


def __validate_border_interpolation(border, interpolation):
    """
    Validates border modes and interpolation modes
    :param mode: str
    :param interpolation: str
    :return: None
    """
    borders_list = ['constant', 'edge', 'wrap']
    if border not in borders_list:
        raise ValueError('mode must be one of {}'.format(borders_list))

    interpolations_list = ['linear', 'nearest', 'bspline', 'bsplinehq']
    if interpolation not in interpolations_list:
        raise ValueError('interpolation must be one of {}'.format(interpolations_list))

    return


def __validate_transform_m(transform_m):
    """
    Validates transformation matrix
    :param transform_m: np.ndarray
    :return: None
    """

    if not isinstance(transform_m, np.ndarray) or\
            (hasattr(transform_m, 'ndim') and transform_m.ndim != 2):
        raise ValueError('Transform matrix must be a 2D numpy array')

    if transform_m.shape != (4, 4):
        raise ValueError('Transformation matrix must be a homogeneous 4x4 matrix.')


def __validate_input_data(data):
    """
    Validates input data
    :param data: np.ndarray, Volume
    :return: None
    :raises: ValueError
    """

    if not isinstance(data, np.ndarray) or not isinstance(data, Volume):
        raise ValueError('Data should be either a numpy array or a Volume')

    if isinstance(data, np.ndarray) and data.ndim != 3:
        raise ValueError('Data must have 3 dimensions')


### DEVELOPMENT
import matplotlib.pyplot as plt
plt.ion()

d = np.random.rand(280, 920, 920).astype(np.float32) * 1000
# d = np.arange(0, 157 * 197 * 271, dtype=np.float32).reshape((157, 197, 271))
plt.imshow(d[0])
plt.show()

rot_d = transform(d, rotation=(1, 0, 0), rotation_order='rzxz', profile=True)
# rot_d = affine(d, np.identity(4, dtype=np.float32))
plt.imshow(rot_d[0], vmin=d[0].min(), vmax=d[0].max())
plt.show()

rot_d = transform(d, rotation=(5, 0, 0), rotation_order='rzxz', profile=True)
plt.imshow(rot_d[0], vmin=d[0].min(), vmax=d[0].max())
plt.show()

rot_d = transform(d, rotation=(15, 0, 0), rotation_order='rzxz', profile=True)
plt.imshow(rot_d[0], vmin=d[0].min(), vmax=d[0].max())
plt.show()

rot_d = transform(d, rotation=(25, 0, 0), rotation_order='rzxz', profile=True)
plt.imshow(rot_d[0], vmin=d[0].min(), vmax=d[0].max())
plt.show()

rot_d = transform(d, rotation=(35, 0, 0), rotation_order='rzxz', profile=True)
plt.imshow(rot_d[0], vmin=d[0].min(), vmax=d[0].max())
plt.show()

rot_d = transform(d, rotation=(45, 0, 0), rotation_order='rzxz', profile=True)
plt.imshow(rot_d[0], vmin=d[0].min(), vmax=d[0].max())
plt.show()

# rot_d = affine(d, np.identity(4, dtype=np.float32))
# driver.Context.synchronize()
# plt.imshow(rot_d[0], vmin=d[0].min(), vmax=d[0].max())
# plt.show()

# print(np.allclose(d, rot_d))
print('stop hammer time')
