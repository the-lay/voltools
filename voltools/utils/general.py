import string
import random
import sys
import numpy as np
from typing import Tuple
import GPUtil
try:
    import cupy
except ImportError:
    pass


def compute_prefilter_workgroup_dims(shape: Tuple[int, int, int]) -> Tuple[Tuple, Tuple]:
    def _pow_two_divider(n):
        if n == 0:
            return 0

        divider = 1
        while (n & divider) == 0:
            divider <<= 1

        return divider

    depth, height, width = shape

    dim_x = min(min(_pow_two_divider(width), _pow_two_divider(height)), 64)
    dim_y = min(min(_pow_two_divider(depth), _pow_two_divider(height)), 512 // dim_x)

    dim_grid = ((height // dim_x, depth // dim_y),
                (width // dim_x,  depth // dim_y),
                (width // dim_x,  height // dim_y))

    dim_blocks = ((dim_x, dim_y, 1),
                  (dim_x, dim_y, 1),
                  (dim_x, dim_y, 1))

    return dim_grid, dim_blocks


def compute_pervoxel_workgroup_dims(shape: Tuple[int, int, int]) -> Tuple[Tuple, Tuple]:
    dim_grid = (shape[0] // 8 + 1 * (shape[0] % 8 != 0),
                shape[1] // 8 + 1 * (shape[1] % 8 != 0),
                shape[2] // 8 + 1 * (shape[2] % 8 != 0))
    dim_blocks = (8, 8, 8)
    return dim_grid, dim_blocks


# @cupy.memoize()
def compute_elementwise_launch_dims(shape: Tuple[int, int, int]) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    device = cupy.cuda.device.Device()
    min_threads = device.attributes['WarpSize']
    max_threads = 128
    thread_blocks_per_mp = 8
    max_blocks = 4 * thread_blocks_per_mp * device.attributes['MultiProcessorCount']
    n = np.prod(shape).item()

    if n < min_threads:
        block_count = 1
        threads_per_block = min_threads
    elif n < (max_blocks * min_threads):
        block_count = (n + min_threads - 1) // min_threads
        threads_per_block = min_threads
    elif n < (max_blocks * max_threads):
        block_count = max_blocks
        grp = (n + min_threads - 1) // min_threads
        threads_per_block = ((grp + max_blocks - 1) // max_blocks) * min_threads
    else:
        block_count = max_blocks
        threads_per_block = max_threads

    return (block_count, 1, 1), (threads_per_block, 1, 1)


def get_available_devices():

    available_devices = ['cpu']

    # get all available gpus
    gpu_ids = GPUtil.getAvailable()

    # check if cupy is installed
    try:
        import cupy

        # add auto gpu
        available_devices.append('gpu')

        # add gpus to list of available devices
        for i in gpu_ids:
            available_devices.append(f'gpu:{i}')

    except ImportError:
        print('Warning: cupy is not found. Therefore, the only available device is "cpu".\n'
              'Please install cupy>=7.0.0b4:\npip install cupy>=7.0.0b4')

    return available_devices


# parses device string and switches cupy to specific id
def switch_to_device(device: str):

    # if id provided
    if device[4:]:
        cupy.cuda.Device(int(device[4:])).use()


# computes required padding before, after and new volume dimensions after transformation
def compute_post_transform_dimensions(shape: Tuple[int, int, int], transform_m: np.ndarray) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    # constructing volume bbox vertices matrix
    a, b, c = shape # not using xyz to avoid confusion with axes
    boundaries = [[0, a, 0, a, 0, a, 0, a],
                  [0, 0, b, b, 0, 0, b, b],
                  [0, 0, 0, 0, c, c, c, c],
                  [1, 1, 1, 1, 1, 1, 1, 1]]

    # inverting transformation matrix to map from output to input
    try:
        inv_transform_m = np.linalg.inv(transform_m)
    except np.linalg.LinAlgError as e:
        print('Something went wrong. Transform matrix should have been affine but still couldnt inverse...')
        raise e

    # applying the matrix to get transformed coordinates
    new_boundaries = np.round(inv_transform_m @ boundaries).astype(int)

    # calculating how much to pad before volume
    pad_before = np.min(new_boundaries * (new_boundaries < 0), axis=1) * -1

    # calculating how much to pad after volume
    dims = np.asarray(shape + (1, ))
    extends_dims = np.tile(np.vstack(dims), len(boundaries[0]))
    extends_after = (new_boundaries - extends_dims) * (new_boundaries > extends_dims)
    pad_after = np.max(extends_after, axis=1)

    new_dims = pad_before + dims + pad_after

    return pad_before[:3], pad_after[:3], new_dims[:3]
