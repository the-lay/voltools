import numpy as np
from pycuda import driver
from pycuda.tools import context_dependent_memoize, dtype_to_ctype
from pycuda.reduction import ReductionKernel
from pycuda.elementwise import ElementwiseKernel
from transforms3d.euler import euler2mat

# Helper function to find closest power of two divider
def _pow_two_divider(n):
    if n == 0:
        return 0

    divider = 1
    while (n & divider) == 0:
        divider <<= 1

    return divider

# Computes grid and block workgroup dimensions for prefiltering
def compute_prefilter_dims(data_shape):
    depth, height, width = data_shape

    dim_x = min(min(_pow_two_divider(width), _pow_two_divider(height)), 64)
    dim_y = min(min(_pow_two_divider(depth), _pow_two_divider(height)), 512 // dim_x)

    dim_grid = ((height // dim_x, depth // dim_y),
                (width // dim_x,  depth // dim_y),
                (width // dim_x,  height // dim_y))

    dim_blocks = ((dim_x, dim_y, 1),
                  (dim_x, dim_y, 1),
                  (dim_x, dim_y, 1))

    return dim_grid, dim_blocks

# Computes grid and block workgroup dimensions for pervoxel 3D grids/blocks
def compute_per_voxel_dims(data_shape):
    depth, height, width = data_shape

    dim_grid = (width // 8 + 1 * (width % 8 != 0),
                height // 8 + 1 * (height % 8 != 0),
                depth // 8 + 1 * (depth % 8 != 0))

    dim_blocks = (8, 8, 8)

    return dim_grid, dim_blocks

# Allocates pitched memory and copies from linear d_array, and then sets texture to that memory
def gpuarray_to_texture(d_array, texture):
    # Texture memory is pitched, not linear
    # So we have to allocate new (pitched) memory and copy from linear
    # TODO do prefiltering also on pitched memory, to avoid this extra conversion
    descr = driver.ArrayDescriptor3D()
    descr.depth, descr.height, descr.width = d_array.shape
    descr.format = driver.dtype_to_array_format(d_array.dtype)
    descr.num_channels = 1
    descr.flags = 0

    ary = driver.Array(descr)

    copy = driver.Memcpy3D()
    copy.set_src_device(d_array.gpudata)
    copy.set_dst_array(ary)
    copy.width_in_bytes = copy.src_pitch = d_array.strides[1]
    copy.src_height = copy.height = descr.height
    copy.depth = descr.depth

    copy()

    # setting the texture data
    texture.set_array(ary)

# Creates a reduction kernel for equal sum
@context_dependent_memoize
def get_equal_sum_kernel(dtype_a, dtype_b):
    return ReductionKernel(np.uint32, neutral='0',
                           reduce_expr='a+b',
                           map_expr='int(a[i]==b[i])',
                           arguments='const {} *a, const {} *b'.format(
                               dtype_to_ctype(dtype_a), dtype_to_ctype(dtype_b)
                           ),
                           keep=True)

# Creates transformation matrix
def get_transform_matrix(dtype=np.float32,
                         scale=None,
                         rotation=None, rotation_units='deg', rotation_order='rzxz',
                         translation=None,
                         around_center=False, shape=None):
    """
    Returns composed transformation matrix according to the passed arguments.
    Transformation order: S->R->T or if around_center: PreT->S->R->PostT->T

    :param dtype: numpy.dtype of the matrix
    :param scale: tuple of scale coefficients to each dimension
    :param rotation: tuple of angles
    :param rotation_units: str of 'deg' or 'rad'
    :param rotation_order: str of one of 24 axis rotation combinations
    :param translation: tuple of translation
    :param around_center: bool, if True add Pre-translation and Post-translation
    :param shape: tuple of volume shape, used only if around_center is True
    :return: np.ndarray, 2d matrix of 4x4
    """


    M = np.identity(4, dtype=dtype)
    center_point = np.divide(shape, 2)

    # Translation
    if translation is not None:
        T = np.identity(4, dtype=dtype)
        T[3, :3] = translation[:3]
        M = np.dot(M, T)

    # Post-translation
    if around_center:
        post_T = np.identity(4, dtype=dtype)
        post_T[3, :3] = (-1 * center_point)[:3]
        M = np.dot(M, post_T)

    # Rotation
    if rotation is not None:
        # Reverse
        rotation = tuple([-1 * j for j in rotation])
        if rotation_units not in ['deg', 'rad']:
            raise TypeError('Rotation units should be either \'deg\' or \'rad\'.')
        if rotation_units == 'deg':
            rotation = np.deg2rad(rotation)
        R = np.identity(4, dtype=dtype)
        R[0:3, 0:3] = euler2mat(*rotation, axes=rotation_order)
        M = np.dot(M, R)

    # Scale
    if scale is not None:
        S = np.identity(4, dtype=dtype)
        S[0, 0] = scale[0]
        S[1, 1] = scale[1]
        S[2, 2] = scale[2]
        M = np.dot(M, S)

    # Pre-translation
    if around_center:
        pre_T = np.identity(4, dtype=dtype)
        pre_T[3, :3] = center_point[:3]
        M = np.dot(M, pre_T)

    # Homogeneous matrix
    M /= M[3, 3]

    return M
