import numpy as np
import time
from pathlib import Path

from pycuda import driver, compiler, gpuarray
from pycuda.compiler import SourceModule

from .tools import *


class Volume:

    # Common volume things
    _kernels_folder = Path(__file__).resolve().parent / 'kernels'
    _kernels_file = _kernels_folder / 'kernels.cu'
    with _kernels_file.open('r') as f:
        _kernels_code = f.read()

    ### Initialization
    def __init__(self, data, prefilter=True, interpolation='bspline', cuda_warmup=True):

        # Check data
        if data.ndim != 3:
            raise ValueError('Volume class expects a 3-dimensional input.')

        # Conversion
        if data.dtype != np.float32:
            data = data.astype(np.float32)

        # Volume attributes
        self.initial_data = data
        self.shape = data.shape
        self.size = np.prod(data.shape)
        self.strides = data.strides
        self.dtype = data.dtype
        self.interpolation = interpolation
        self.prefilter = prefilter

        # GPU initialization
        self._init_gpu()
        self.pervoxel_dims = compute_per_voxel_dims(self.shape)   # "optimal" workgroup dimensions
        self._init_gpu_data()

        # CUDA "warmup" to avoid first-time run performance dip
        if cuda_warmup:
            self.transform_m(np.identity(4))
            assert(self == self)
            assert(self + self)

    def _init_gpu(self):
        self._kernels_code = Volume._kernels_code.replace('INTERPOLATION_FETCH',
                                                          'linearTex3D' if self.interpolation == 'linear' else
                                                          'cubicTex3D' if self.interpolation == 'bspline' else
                                                          'cubicTex3DSimple' if self.interpolation == 'bsplinehq' else
                                                          'WRONG_INTERPOLATION')

        # import os
        # os.environ['PATH'] += ';' + r'C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin'
        self.cuda = SourceModule(self._kernels_code, no_extern_c=True,
                                 options=['-O3', '--compiler-options', '-Wall'],
                                 include_dirs=[str(Volume._kernels_folder)])

        # functions setup
        self.cuda.functions = {
            'prefilterX': self.cuda.get_function('SamplesToCoefficients3DX').prepare('PIP'),
            'prefilterY': self.cuda.get_function('SamplesToCoefficients3DY').prepare('PIP'),
            'prefilterZ': self.cuda.get_function('SamplesToCoefficients3DZ').prepare('PIP'),
            'transform': self.cuda.get_function('transform').prepare('PPP')
        }

        # textures setup
        self.cuda.textures = {
            'coeff_tex': self.cuda.get_texref('coeff_tex')
        }
        self.cuda.textures['coeff_tex'].set_filter_mode(driver.filter_mode.LINEAR)
        self.cuda.textures['coeff_tex'].set_address_mode(0, driver.address_mode.BORDER)
        self.cuda.textures['coeff_tex'].set_address_mode(1, driver.address_mode.BORDER)
        self.cuda.textures['coeff_tex'].set_address_mode(2, driver.address_mode.BORDER)

    def _init_gpu_data(self):
        self.d_shape = gpuarray.to_gpu(np.array(self.shape[::-1], dtype=np.int32))  # shape array on gpu
        self.d_data = gpuarray.to_gpu(self.initial_data)

        # B-Spline interpolation setups
        if self.interpolation == 'bspline':

            if self.prefilter:
                dim_grid, dim_blocks = compute_prefilter_dims(self.shape)
                slice_stride = np.uint32(self.d_data.strides[1])

                self.cuda.functions['prefilterX'].prepared_call(dim_grid[0], dim_blocks[0],
                                                                self.d_data.gpudata, slice_stride,
                                                                self.d_shape.gpudata)

                self.cuda.functions['prefilterY'].prepared_call(dim_grid[1], dim_blocks[1],
                                                                self.d_data.gpudata, slice_stride,
                                                                self.d_shape.gpudata)

                self.cuda.functions['prefilterZ'].prepared_call(dim_grid[2], dim_blocks[2],
                                                                self.d_data.gpudata, slice_stride,
                                                                self.d_shape.gpudata)

        # Copy d_data to texture
        gpuarray_to_texture(self.d_data, self.cuda.textures['coeff_tex'])

    ### Memory API
    def to_cpu(self):
        return self.d_data.get()

    ### Geometry API
    # Affine transform with a matrix
    def transform_m(self, transform_m, profile=False):

        if profile:
            t_start = time.perf_counter()

        if transform_m.ndim != 2:
            raise RuntimeError('Transform method expects a 2D array as transformation matrix.')

        # expand to 4x4
        if transform_m.shape == (3, 3):
            mat4 = np.identity(4, dtype=transform_m.dtype)
            mat4[0:3, 0:3] = transform_m
            transform_m = mat4
        # not 3x3 or 4x4
        elif transform_m.shape != (4, 4):
            raise RuntimeError('Transformation matrix should have either 3x3 or 4x4 shape.')

        # convert to float32
        if transform_m.dtype != np.float32:
            transform_m = transform_m.astype(np.float32)

        # clear buffer
        self.d_data.fill(0)

        # invoke cuda kernel
        transform_m = gpuarray.to_gpu(transform_m)
        self.cuda.functions['transform'].prepared_call(self.pervoxel_dims[0], self.pervoxel_dims[1],
                                                       self.d_shape.gpudata, transform_m.gpudata,
                                                       self.d_data.gpudata)

        if profile:
            driver.Context.synchronize()
            t_end = time.perf_counter() - t_start # TODO: use cuda events instead
            print('Transform fininshed in {:.4f} s'.format(t_end))

        return self

    def transform(self,
                  scale=None,
                  rotation=None, rotation_units='deg', rotation_order='rzxz',
                  translation=None,
                  around_center=True, profile=False):

        # Order: S->R->T or if around_center: PreT->S->R->PostT->T
        tr_m = get_transform_matrix(dtype=self.dtype,
                                    scale=scale,
                                    rotation=rotation,
                                    rotation_units=rotation_units,
                                    rotation_order=rotation_order,
                                    translation=translation,
                                    around_center=around_center,
                                    shape=self.shape[::-1]) # shape inverted to match X-Y-Z

        return self.transform_m(tr_m, profile)

    ### Built-ins API
    def __eq__(self, other):

        if not isinstance(other, Volume):
            # TODO add comparison with numpy ndarray
            # TODO add comparison with pycuda gpuarray
            return False

        if (self.shape != other.shape) or (self.size != other.size):
            return False

        krnl = get_equal_sum_kernel(self.dtype, other.dtype)
        # krnl(A.d_data, B.d_data).get() will return a number of "true"s for elementwise comparison
        # now we check if the number of true comparisons is the same as number of voxels
        return int(krnl(self.d_data, other.d_data).get()) == self.size

    def __add__(self, other):
        """
        Returns elementwise addition of two volumes. Volumes must be of same shape, size and dtype
        :param other: Volume or gpuarray.GPUArray
        :return: gpuarray.GPUArray
        """

        if isinstance(other, Volume):

            if (self.shape != other.shape) or (self.size != other.size) or (self.dtype != other.dtype):
                raise ValueError('Shape, size and dtype of the volumes should match.')

            return self.d_data + other.d_data

        elif isinstance(other, gpuarray.GPUArray):

            if (self.shape != other.shape) or (self.size != other.size) or (self.dtype != other.dtype):
                raise ValueError('Shape, size and dtype of the Volume and GPUArray should match.')

            return self.d_data + other

        else:
            raise NotImplementedError('Addition is currently supported only with another Volume or GPUArray')

    def __sub__(self, other):
        """
        Returns elementwise substraction of two volumes. Volumes must be of same shape, size and dtype
        :param other: Volume or gpuarray.GPUArray
        :return: gpuarray.GPUArray
        """

        if isinstance(other, Volume):
            if (self.shape != other.shape) or (self.size != other.size) or (self.dtype != other.dtype):
                raise ValueError('Shape, size and dtype of the volumes should match.')

            return self.d_data - other.d_data

        elif isinstance(other, gpuarray.GPUArray):

            if (self.shape != other.shape) or (self.size != other.size) or (self.dtype != other.dtype):
                raise ValueError('Shape, size and dtype of the Volume and GPUArray should match.')

            return self.d_data - other

        else:
            raise NotImplementedError('Substraction is currently supported only with another Volume or GPUArray')


    ### Custom stuff
    def project(self, cpu=True):
        """
        Sum in 0 axis direction (np.sum(axis=0) on GPU)
        :return: np.ndarray (if CPU) / GPUArray (if not CPU) with projections
        """

        total_sum = gpuarray.zeros(self.shape[1:], dtype=np.float32)

        for i in range(self.shape[0]):
            total_sum += self.d_data[i]

        if cpu:
            return total_sum.get()
        else:
            return total_sum

    def sum(self):
        """
        Scalar sum of all voxels
        :return:
        """
        return self.dtype.type(gpuarray.sum(self.d_data).get())

