import numpy as np
import time
from pyrr import Matrix44
from transforms3d.euler import euler2mat

# TODO remove later
np.random.seed(1)

# init PyCUDA
try:
    from pycuda import autoinit, driver, compiler, gpuarray
    from pycuda.compiler import DynamicSourceModule
    import warnings
    print('Using CUDA on {} with {}.{} CC.'.format(autoinit.device.name(),
                                                   *autoinit.device.compute_capability()))

    from pathlib import Path as __Path
    __kernels_folder = __Path(__file__).resolve().parent / 'kernels'
    __kernels_file = __kernels_folder / 'kernels.cu'
    with __kernels_file.open('r') as f:
        __kernels_code = f.read()

    # TODO think of a better way to check/add to path
    import os
    os.environ['PATH'] += ';' + r'C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin'

    # TODO: replace kernel code templates if needed here: for different interpolation types
    # compile kernels
    kernels = DynamicSourceModule(__kernels_code, no_extern_c=True,
                                  options=['-O3', '--compiler-options', '-Wall', '-rdc=true', '-lcudadevrt'],
                                  include_dirs=[str(__kernels_folder)])

    # functions setup
    kernels.functions = {
        'prefilterX': kernels.get_function('SamplesToCoefficients3DX').prepare('PIIII'),
        'prefilterY': kernels.get_function('SamplesToCoefficients3DY').prepare('PIIII'),
        'prefilterZ': kernels.get_function('SamplesToCoefficients3DZ').prepare('PIIII'),
        'transform':  kernels.get_function('transform').prepare('PPP')
    }

    # textures setup
    kernels.textures = {
        'coeff_tex':  kernels.get_texref('coeff_tex')
    }
    kernels.textures['coeff_tex'].set_filter_mode(driver.filter_mode.LINEAR)
    kernels.textures['coeff_tex'].set_address_mode(0, driver.address_mode.BORDER)
    kernels.textures['coeff_tex'].set_address_mode(1, driver.address_mode.BORDER)
    kernels.textures['coeff_tex'].set_address_mode(2, driver.address_mode.BORDER)

except Exception as e:
    print(e)
    raise e


# Helper function to find closest power of two divider
def _pow_two_divider(n: int):
    if n == 0:
        return 0

    divider = 1
    while (n & divider) == 0:
        divider <<= 1

    return divider


# Helpers that can be used
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


class Volume:

    def __init__(self, data, prefilter=True, interpolation='bspline', gpu=True):

        # Conversion
        if data.dtype != np.float32:
            data = data.astype(np.float32)

        # Mandatory stuff
        self.data = data
        self.shape = self.data.shape
        self.dtype = self.data.dtype
        self.interpolation = interpolation
        self.prefilter = prefilter

        # GPU stuff
        self.pervoxel_dims = compute_per_voxel_dims(self.shape)  # "optimal" workgroup dimensions

        self.d_shape = None
        self._buffer = None
        self._texture_uploaded = False
        if gpu:
            self.to_gpu()

    # Protected methods
    def _upload(self, prefilter=True):

        # Upload data
        d_data = gpuarray.to_gpu(self.data)
        depth, height, width = np.uint32(self._buffer.shape)
        slice_stride = np.uint32(d_data.strides[1])

        # Optional prefiltering
        if prefilter:

            dim_grid, dim_blocks = compute_prefilter_dims(self.shape)

            kernels.functions['prefilterX'].prepared_call(dim_grid[0], dim_blocks[0],
                                                          d_data.gpudata, slice_stride,
                                                          width, height, depth)

            kernels.functions['prefilterY'].prepared_call(dim_grid[1], dim_blocks[1],
                                                          d_data.gpudata, slice_stride,
                                                          width, height, depth)

            kernels.functions['prefilterZ'].prepared_call(dim_grid[2], dim_blocks[2],
                                                          d_data.gpudata, slice_stride,
                                                          width, height, depth)

        # Upload d_data to texture
        gpuarray_to_texture(d_data, kernels.textures['coeff_tex'])
        self._texture_uploaded = True

        # Free up d_data
        d_data.gpudata.free()

    # Memory API
    def to_cpu(self):
        driver.Context.synchronize()
        return self._buffer.get()

    def free_gpu(self):
        self.d_shape.gpudata.free()
        self._buffer.gpudata.free()
        # TODO: free texture?

    def to_gpu(self):
        self.d_shape = gpuarray.to_gpu(np.array(self.data.shape[::-1], dtype=np.int32))

        # Working buffer
        self._buffer = gpuarray.to_gpu(self.data)

        # Optionally prefilter and allocate the original texture
        self._upload(self.prefilter)

    # Low-level API: core transform method
    def apply_transform_m(self, transform_m, profile=False):

        if profile:
            t_start = time.perf_counter()

        # if transform_m.ndim != 2:
        #     raise RuntimeError('Transform method expects a 2D array as transformation matrix.')
        #
        # # expand to 4x4
        # if transform_m.shape == (3, 3):
        #     mat4 = np.identity(4, dtype=transform_m.dtype)
        #     mat4[0:3, 0:3] = transform_m
        #     transform_m = mat4
        # # not 3x3 or 4x4
        # elif transform_m.shape != (4, 4):
        #     raise RuntimeError('Transformation matrix should have either 3x3 or 4x4 shape.')
        #
        # # convert to float32
        # if transform_m.dtype != np.float32:
        #     transform_m = transform_m.astype(np.float32)

        # clear buffer
        self._buffer.fill(0)

        # invoke cuda kernel
        transform_m = gpuarray.to_gpu(transform_m)
        kernels.functions['transform'].prepared_call(self.pervoxel_dims[0], self.pervoxel_dims[1],
                                                     self.d_shape.gpudata, transform_m.gpudata,
                                                     self._buffer.gpudata)

        if profile:
            driver.Context.synchronize()
            t_end = time.perf_counter() - t_start
            print('Transform fininshed in {:.4f} s'.format(t_end))

        return self

    # High-level multiple actions API
    # Order of transformations: scale->rotation->translation
    def transform(self,
                  scale=None,
                  rotation=None, rotation_units='rad', rotation_order='rzxz',
                  translation=None,
                  around_center=True, profile=False):

        if scale is None:
            scale = Matrix44.from_scale([1, 1, 1], dtype=np.float32)
        else:
            scale = Matrix44.from_scale(scale, dtype=np.float32)

        if rotation is None:
            rotation_m = np.identity(4, dtype=np.float32)
        else:
            if rotation_units not in ['deg', 'rad']:
                raise TypeError('Rotation units should be either \'deg\' or \'rad\'.')
            if rotation_units == 'deg':
                rotation = np.deg2rad(rotation)
            rotation_m = np.identity(4, dtype=np.float32)
            rotation_m[0:3, 0:3] = euler2mat(*(-1 * rotation), axes=rotation_order)

        if translation is None:
            translation = Matrix44.from_translation([0, 0, 0], dtype=np.float32)
        else:
            translation = Matrix44.from_translation(translation, dtype=np.float32)

        if around_center:
            center_point = np.divide(self.shape[::-1], 2)
            pretrans_m = Matrix44.from_translation(center_point, dtype=np.float32)
            posttrans_m = Matrix44.from_translation(-1 * center_point, dtype=np.float32)

            transform_m = pretrans_m * scale * rotation_m * posttrans_m * translation

        else:
            transform_m = scale * rotation_m * translation

        return self.apply_transform_m(transform_m, profile)


#############
# d = np.random.rand(50, 50, 50).astype(np.float32)
# a = Volume(d, prefilter=False)
# a.apply_transform_m(np.identity(4, np.float32))
# print(np.allclose(d, a.to_cpu()))
#
# print('adfgafdg')

