import numpy as np
from pathlib import Path
from string import Template

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

    return ElementwiseKernel(
        arguments='const int4* const dims, const float4* const xform, {} const volume'.format(dtype_to_ctype(dtype)),
        operation="""
            // indices
            float3 vox = make_float3(i / (dim[0].x * dim[0].y),
                                    (i % (dim[0].x * dim[0].y)) / dim[0].z,
                                    (i % (dim[0].x * dim[0].y)) % dim[0].z)
                        
            // early out
            if (vox.x >= dims[0].x || vox.y >= dims[0].y || vox.z >= dims[0].z) return;
            
            // match center of texels
            vox += .5f
            
            // apply transformation matrix
            float3 ndx;
            ndx.x = dot(vox, make_float4(xform[0].x, xform[1].x, xform[2].x, xform[3].x));
            ndx.y = dot(vox, make_float4(xform[0].y, xform[1].y, xform[2].y, xform[3].y));
            ndx.z = dot(vox, make_float4(xform[0].z, xform[1].z, xform[2].z, xform[3].z));
            
            // skip if outside of volume dimensions
            if (ndx.x < 0 || ndx.y < 0 || ndx.z < 0 || ndx.x >= dims[0].x || ndx.y >= dims[0].y || ndx.z >= dims[0].z)
                return;
            
            // get interpolated value and put it into destination buffer
            float v = tex3D(coeff_tex, ndx.x, ndx.y, ndx.z);
            volume[vox.x + dims[0].x*vox.y + dims[0].x*dims[0].y*vox.z] = v;            
        """,
        keep=True
    )



def get_prefilter_kernel(dtype):

    ctype = dtype_to_ctype(dtype)
    if ctype != 'float':
        # TODO
        raise ValueError('Only float arrays can be prefiltered for bspline interpolation')


