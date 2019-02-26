import numpy as np
from pathlib import Path
from string import Template
import os

from pycuda import driver, compiler, gpuarray
from pycuda.compiler import SourceModule
from pycuda.tools import dtype_to_ctype, context_dependent_memoize
from pycuda.reduction import ReductionKernel
from pycuda.elementwise import ElementwiseKernel


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


# Affine transform kernel
@context_dependent_memoize
def get_transform_kernel(dtype):

    options = [
        _flag.strip() for _flag in
        os.environ.get("PYCUDA_DEFAULT_NVCC_FLAGS", "").split()
        if _flag.strip()]
    options.append('-I {}'.format(str(_kernels_folder)))

    return ElementwiseKernel(
        arguments='const int4* const dims, const float4* const xform, {}* const volume'.format(dtype_to_ctype(dtype)),
        preamble='}}\n#include "helper_math.h"\ntexture<{}, 3, cudaReadModeElementType> coeff_tex;\nextern "C" {{'.format(dtype_to_ctype(dtype)),
        operation="""
            // indices
            int3 vox = make_int3((i / (dims[0].x * dims[0].y)),
                                    ((i % (dims[0].x * dims[0].y)) / dims[0].z),
                                    ((i % (dims[0].x * dims[0].y)) % dims[0].z));
                        
            // early out
            if (vox.x >= dims[0].x || vox.y >= dims[0].y || vox.z >= dims[0].z) return;
            
            // match center of texels
            float3 voxf = make_float3(vox) + .5f;
            
            // apply transformation matrix
            float3 ndx;
            ndx.x = dot(voxf, make_float4(xform[0].x, xform[1].x, xform[2].x, xform[3].x));
            ndx.y = dot(voxf, make_float4(xform[0].y, xform[1].y, xform[2].y, xform[3].y));
            ndx.z = dot(voxf, make_float4(xform[0].z, xform[1].z, xform[2].z, xform[3].z));
            
            // skip if outside of volume dimensions
            if (ndx.x < 0 || ndx.y < 0 || ndx.z < 0 || ndx.x >= dims[0].x || ndx.y >= dims[0].y || ndx.z >= dims[0].z)
                return;
            
            // get interpolated value and put it into destination buffer
            float v = tex3D(coeff_tex, ndx.x, ndx.y, ndx.z);
            int index = vox.x + dims[0].x*vox.y + dims[0].x*dims[0].y*vox.z;
            volume[index] = v;
            printf("Index %i got converted into (%i, %i, %i) which got transformed into (%f, %f, %f) with value %f",
            index, vox.x, vox.y, vox.z, ndx.x, ndx.y, ndx.z, v);
        """,
        keep=True,
        options=options
    )


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


def get_prefilter_kernel(dtype):

    ctype = dtype_to_ctype(dtype)
    if ctype != 'float':
        # TODO
        raise ValueError('Only float arrays can be prefiltered for bspline interpolation')


