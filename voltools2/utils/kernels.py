import numpy as np
from pathlib import Path
from string import Template
import os

from pycuda import autoinit as __c
from pycuda import driver, compiler, gpuarray
from pycuda.compiler import SourceModule
from pycuda.tools import dtype_to_ctype, context_dependent_memoize, parse_c_arg
from pycuda.reduction import ReductionKernel
from pycuda.elementwise import ElementwiseKernel, get_elwise_module, VectorArg, ScalarArg


# Load all kernels on init
_kernels = {
    'volume.cu': '',
    'prefilter.cu': ''
}
_kernels_folder = Path(__file__).resolve().parent.parent / 'kernels'
for kernel in _kernels:
    with (_kernels_folder / kernel).open('r') as f:
        _kernels[kernel] = f.read()


# OOM naive checks
def fits_on_gpu(nbytes):
    return nbytes < driver.Context.get_device().total_memory()


# Volume kernels
def get_volume_kernel(dtype):

    # Load template and replace
    code = _kernels['volume.cu']
    code = code.replace('DTYPE', dtype_to_ctype(dtype))

    # Compile
    kernel = SourceModule(code, no_extern_c=True, options=['-O3', '--compiler-options', '-Wall'],
                          include_dirs=[str(_kernels_folder)])

    # Texture params
    kernel.texture = kernel.get_texref('data_tex')
    kernel.texture.set_filter_mode(driver.filter_mode.LINEAR)
    kernel.texture.set_address_mode(0, driver.address_mode.BORDER)
    kernel.texture.set_address_mode(1, driver.address_mode.BORDER)
    kernel.texture.set_address_mode(2, driver.address_mode.BORDER)

    return kernel


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

        block = args[0]._block
        grid = args[0]._grid
        invocation_args.append(args[0].mem_size)

        print('Calling kernel', self.name, 'with block', block, 'and grid', grid)

        self.func.prepared_call(grid, block, *invocation_args)


# Affine transform kernel
@context_dependent_memoize
def get_transform_kernel(dtype):

    kernel = VoltoolsElementwiseKernel(
        args='{}* const volume, const int4* const dims, const float4* const xform'
             .format(dtype_to_ctype(dtype)),

        preamble="""
}}
// helper_math should be added without C extern
#include "helper_math.h"
extern "C" {{
texture<{}, 3, cudaReadModeElementType> coeff_tex;
        """.format(dtype_to_ctype(dtype)),

        body="""
        //printf("%llu, %u, %u, %u, %u, %u \t\t", n, tid, total_threads, cta_start, i);
        
          // indices
          int a = i / (dims[0].y * dims[0].z);
          int b = (i % (dims[0].y * dims[0].z)) / dims[0].z;
          int c = (i % (dims[0].y * dims[0].z)) % dims[0].z;
                    
          //printf("%f %f %f %f \t %f %f %f %f \t %f %f %f %f \t %f %f %f %f \t\t\t",
          //xform[0].x, xform[0].y, xform[0].z, xform[0].w,
          //xform[1].x, xform[1].y, xform[1].z, xform[1].w, 
          //xform[2].x, xform[2].y, xform[2].z, xform[2].w,
          //xform[3].x, xform[3].y, xform[3].z, xform[3].w);
          
          // thread idx to texels
          float4 voxf = make_float4(((float)c) + .5f, ((float)b) + .5f, ((float)a) + .5f, 1.0f);
          
          // apply matrix
          float4 ndx;
          ndx.x = dot(voxf, xform[0]);
          ndx.y = dot(voxf, xform[1]);
          ndx.z = dot(voxf, xform[2]);
          //ndx.w = dot(voxf, xform[3]);
          
          //printf("(%i, %i, %i) got converted into (%i, %i, %i)", 
          
          // skip if transformed voxel is now outside of volume
          if (ndx.x < 0 || ndx.y < 0 || ndx.z < 0 || ndx.x >= dims[0].x || ndx.y >= dims[0].y || ndx.z >= dims[0].z) {
            //printf("WOOOOOOT");
            return;
          }
          
          // get interpolated value
          float v = tex3D(coeff_tex, ndx.x, ndx.y, ndx.z);
          volume[i] = v;
        """,

        name='transform3d',
        include_dirs=[str(_kernels_folder)],
        keep=True
    )

    texture = kernel.get_texref('coeff_tex')
    texture.set_filter_mode(driver.filter_mode.LINEAR)
    texture.set_address_mode(0, driver.address_mode.BORDER)
    texture.set_address_mode(1, driver.address_mode.BORDER)
    texture.set_address_mode(2, driver.address_mode.BORDER)
    texture.set_flags(driver.TRSF_READ_AS_INTEGER)

    return kernel, texture

    #
    # # For 3D shapes, the subscripts of the element `data[a, b, c]` where
    # # `data.shape == (A, B, C)` can be computed as
    # # `a = i/(B*C)`
    # # `b = mod(i, B*C)/C`
    # # `c = mod(mod(i, B*C), C)`.
    #
    # preamble = """
    # //}}
    # //#include "helper_math.h"
    # //extern "C" {{
    # texture<{}, 3, cudaReadModeElementType> coeff_tex;
    # """.format(dtype_to_ctype(dtype))
    # operation = """
    # // indices
    # int a = i / (500 * 500);
    # int b = (i % (500 * 500)) / 500;
    # int c = (i % (500 * 500)) % 500;
    #
    # //float3 voxf = make_float3(a, b, c);
    #
    # // texture memory is reversed
    # float v = tex3D(coeff_tex, ((float)c) +.5f, ((float)b) +.5f, ((float)a) +.5f);
    # volume[i] = v;
    #
    # //printf("%i (%i, %i, %i) = %f\t", i, a, b, c, v);"""
    # name = 'transform3d'
    #
    # mod = get_elwise_module(args, operation, name, True, options, preamble=preamble)
    # func = mod.get_function(name)
    # tex_src = mod.get_texref('coeff_tex')
    # func.prepare('P'+np.dtype(np.uintp).char)#, texrefs=[tex_src])
    # return func, tex_src

#     return ElementwiseKernel(
#         arguments='{}* const volume, const int4* const dims, const float4* const xform'.format(dtype_to_ctype(dtype)),
#         preamble="""
# //}}
# //#include "helper_math.h"
# //extern "C" {{
# texture<{}, 3, cudaReadModeElementType> coeff_tex;
# """.format(dtype_to_ctype(dtype)),
#         operation="""
# // indices
# int a = i / (dims[0].y * dims[0].z);
# int b = (i % (dims[0].y * dims[0].z)) / dims[0].z;
# int c = (i % (dims[0].y * dims[0].z)) % dims[0].z;
#
# //float3 voxf = make_float3(a, b, c);
#
# float v = tex3D(coeff_tex, a, b, c);
# volume[i] = v;
#
# //printf("%i (%i, %i, %i) = %f\t", i, a, b, c, v);
#         """,
#         name="transform3d",
#         keep=True,
#         options=options
#     )


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


def get_prefilter_kernel(dtype):

    ctype = dtype_to_ctype(dtype)
    if ctype != 'float':
        # TODO
        raise ValueError('Only float arrays can be prefiltered for bspline interpolation')


