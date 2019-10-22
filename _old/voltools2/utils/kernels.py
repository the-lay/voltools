import numpy as np
from pathlib import Path
from string import Template
import os
from enum import Enum

from pycuda import autoinit as __cuda
from pycuda import driver, compiler, gpuarray
from pycuda.compiler import SourceModule
from pycuda.tools import dtype_to_ctype, context_dependent_memoize, parse_c_arg
from pycuda.reduction import ReductionKernel
from pycuda.elementwise import ElementwiseKernel, get_elwise_module, VectorArg, ScalarArg
from typing import Tuple, Union

# Load all kernels on init
_kernels = {}
_kernels_folder = Path(__file__).resolve().parent.parent / 'kernels'
for kernel in [x for x in _kernels_folder.glob('**/*') if x.suffix == '.cu']:
    with kernel.open('r') as f:
        _kernels[kernel.name] = f.read()

# OOM naive checks
def fits_on_gpu(nbytes: int) -> Tuple[bool, int]:
    return nbytes < __cuda.device.total_memory(), __cuda.device.total_memory()

# Elementwise kernel
class VoltoolsElementwiseKernel:

    def __init__(self, args, preamble, body, name, include_dirs, keep=True):

        self.arguments = [parse_c_arg(arg) for arg in args.split(',')]
        self.arguments.append(ScalarArg(np.uintp, "n"))

        options = [_flag.strip() for _flag in
                   os.environ.get("PYCUDA_DEFAULT_NVCC_FLAGS", "").split() if _flag.strip()]

        # include directories
        if include_dirs:
            for include_dir in include_dirs:
                options.append('-I {}'.format(str(include_dir)))

        # custom preamble
        preamble = '#include <pycuda-helpers.hpp>'\
                   + preamble

        self.name = name

        self.mod = get_elwise_module(arguments=self.arguments, operation=body, name=self.name,
                                     preamble=preamble, options=options, keep=keep)
        self.func = self.mod.get_function(name)
        self.func.prepare(''.join(arg.struct_char for arg in self.arguments))

    def get_texref(self, name):
        return self.mod.get_texref(name)

    def __call__(self, *args, **kwargs):

        invocation_args = []

        for arg, arg_descr in zip(args, self.arguments):
            if isinstance(arg_descr, VectorArg):
                if not arg.flags.forc:
                    raise RuntimeError('elementwise kernel cannot deal with non-contiguous arrays')

                invocation_args.append(arg.gpudata)
            else:
                invocation_args.append(arg)

        # using gpuarray computed block/grid for grid-stride patterns
        block = args[0]._block
        grid = args[0]._grid
        invocation_args.append(args[0].mem_size)

        if kwargs.get('profile', False):
            print('Calling kernel', self.name, 'with block', block, 'and grid', grid)
            timing = self.func.prepared_timed_call(grid, block, *invocation_args)
            print('Kernel {} took {:.4f} seconds ({:.2f}ms) to execute'.format(self.name, timing(), timing() * 1000))
        else:
            self.func.prepared_call(grid, block, *invocation_args)

# Correlation kernels
@context_dependent_memoize
def get_correlation_kernels(dtype, warm_up: bool = True):
    num = ReductionKernel(dtype, neutral='0',
                          map_expr='(x[i]-xm)*(y[i]-ym)', reduce_expr='a+b',
                          arguments='float *x, float *y, float xm, float ym', keep=True)

    den = ReductionKernel(dtype, neutral='0',
                          map_expr='(x[i] - xm) * (x[i] - xm)', reduce_expr='a+b',
                          arguments='float *x, float xm', keep=True)

    if warm_up:
        emp = gpuarray.zeros((20, 20), dtype=dtype)
        emp.fill(1)
        a = num(emp, emp, 0, 0)
        b = den(emp, 0)
        del emp, a, b

    return num, den

# Allocates pitched memory and copies from linear d_array, and then sets texture to that memory
def gpuarray_to_texture(d_array, texture):
    # Texture memory is pitched, not linear
    # So we have to allocate new (pitched) memory and copy from linear
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

    return ary

# TODO
# def get_prefilter_kernel(dtype):
#
#     ctype = dtype_to_ctype(dtype)
#     if ctype != 'float':
#         # TODO
#         raise ValueError('Only float arrays can be prefiltered for bspline interpolation')


