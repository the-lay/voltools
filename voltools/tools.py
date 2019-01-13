import numpy as np
from pycuda import driver

# Helper function to find closest power of two divider
def _pow_two_divider(n: int):
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


# Computes grid and block workgroup dimensions for transform
def compute_per_voxel_dims(data_shape):
    depth, height, width = data_shape

    dim_grid = (width // 8 + 1 * (width % 8 != 0),
                height // 8 + 1 * (height % 8 != 0),
                depth // 8 + 1 * (depth % 8 != 0))

    dim_blocks = (8, 8, 8)

    return dim_grid, dim_blocks



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
