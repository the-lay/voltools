#include "math_functions.h"
#include "vector_functions.h"
#include "vector_types.h"
#include "helper_math.h"
typedef unsigned int uint;
typedef unsigned short ushort;
typedef unsigned char uchar;
typedef signed char schar;

inline __host__ __device__ float3 floor(const float3 v)
{
    return make_float3(floor(v.x), floor(v.y), floor(v.z));
}

inline __host__ __device__ float dot(float3 a, float4 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + b.w;
}

//////////////////////////

#define Pole (sqrt(3.0f)-2.0f)  //pole for cubic b-spline

inline __device__ __host__ uint UMIN(uint a, uint b)
{
	return a < b ? a : b;
}

extern "C" {
    __host__ __device__ float InitialCausalCoefficient(
        float* c,			// coefficients
        uint DataLength,	// number of coefficients
        int step)			// element interleave in bytes
    {
        const uint Horizon = UMIN(12, DataLength);

        // this initialization corresponds to clamping boundaries
        // accelerated loop
        float zn = Pole;
        float Sum = *c;
        for (uint n = 0; n < Horizon; n++) {
            Sum += zn * *c;
            zn *= Pole;
            c = (float*)((uchar*)c + step);
        }
        return(Sum);
    }

    __host__ __device__ float InitialAntiCausalCoefficient(
        float* c,			// last coefficient
        uint DataLength,	// number of samples or coefficients
        int step)			// element interleave in bytes
    {
        // this initialization corresponds to clamping boundaries
        return((Pole / (Pole - 1.0f)) * *c);
    }

    __host__ __device__ void ConvertToInterpolationCoefficients(
        float* coeffs,		// input samples --> output coefficients
        uint DataLength,	// number of samples or coefficients
        int step)			// element interleave in bytes
    {
        // compute the overall gain
        const float Lambda = (1.0f - Pole) * (1.0f - 1.0f / Pole);

        // causal initialization
        float* c = coeffs;
        float previous_c;  //cache the previously calculated c rather than look it up again (faster!)
        *c = previous_c = Lambda * InitialCausalCoefficient(c, DataLength, step);
        // causal recursion
        for (uint n = 1; n < DataLength; n++) {
            c = (float*)((uchar*)c + step);
            *c = previous_c = Lambda * *c + Pole * previous_c;
        }
        // anticausal initialization
        *c = previous_c = InitialAntiCausalCoefficient(c, DataLength, step);
        // anticausal recursion
        for (int n = DataLength - 2; 0 <= n; n--) {
            c = (float*)((uchar*)c - step);
            *c = previous_c = Pole * (previous_c - *c);
        }
    }

    ///////////////////////////// cubicPrefilter3D.cu

    __global__ void SamplesToCoefficients3DX(
        float* volume,		// in-place processing
        uint pitch,			// width in bytes
        uint width,			// width of the volume
        uint height,		// height of the volume
        uint depth)			// depth of the volume
    {
        // process lines in x-direction
        const uint y = blockIdx.x * blockDim.x + threadIdx.x;
        const uint z = blockIdx.y * blockDim.y + threadIdx.y;
        const uint startIdx = (z * height + y) * pitch;

        float* ptr = (float*)((uchar*)volume + startIdx);
        ConvertToInterpolationCoefficients(ptr, width, sizeof(float));
    }

    __global__ void SamplesToCoefficients3DY(
        float* volume,		// in-place processing
        uint pitch,			// width in bytes
        uint width,			// width of the volume
        uint height,		// height of the volume
        uint depth)			// depth of the volume
    {
        // process lines in y-direction
        const uint x = blockIdx.x * blockDim.x + threadIdx.x;
        const uint z = blockIdx.y * blockDim.y + threadIdx.y;
        const uint startIdx = z * height * pitch;

        float* ptr = (float*)((uchar*)volume + startIdx);
        ConvertToInterpolationCoefficients(ptr + x, height, pitch);
    }

    __global__ void SamplesToCoefficients3DZ(
        float* volume,		// in-place processing
        uint pitch,			// width in bytes
        uint width,			// width of the volume
        uint height,		// height of the volume
        uint depth)			// depth of the volume
    {
        // process lines in z-direction
        const uint x = blockIdx.x * blockDim.x + threadIdx.x;
        const uint y = blockIdx.y * blockDim.y + threadIdx.y;
        const uint startIdx = y * pitch;
        const uint slice = height * pitch;

        float* ptr = (float*)((uchar*)volume + startIdx);
        ConvertToInterpolationCoefficients(ptr + x, depth, slice);
    }

    ///////////////////////////// cubicTex3D.cu
    __device__ float linearTex3D(texture<float, 3, cudaReadModeElementType> tex, float3 coord)
    {
        return tex3D(tex, coord.x, coord.y, coord.z);
    }

    ///////////////////////////// bspline_kernel.cu
    inline __device__ void bspline_weights(float3 fraction, float3& w0, float3& w1, float3& w2, float3& w3)
    {
        const float3 one_frac = 1.0f - fraction;
        const float3 squared = fraction * fraction;
        const float3 one_sqd = one_frac * one_frac;

        w0 = 1.0f/6.0f * one_sqd * one_frac;
        w1 = 2.0f/3.0f - 0.5f * squared * (2.0f-fraction);
        w2 = 2.0f/3.0f - 0.5f * one_sqd * (2.0f-one_frac);
        w3 = 1.0f/6.0f * squared * fraction;
    }

    inline __host__ __device__ float bspline(float t)
    {
        t = fabs(t);
        const float a = 2.0f - t;

        if (t < 1.0f) return 2.0f/3.0f - 0.5f*t*t*a;
        else if (t < 2.0f) return a*a*a / 6.0f;
        else return 0.0f;
    }

    ///////////////////////////// cubicTex3D_kernel.cu
    __device__ float cubicTex3D(texture<float, 3, cudaReadModeElementType> tex, float3 coord)
    {
        // shift the coordinate from [0,extent] to [-0.5, extent-0.5]
        const float3 coord_grid = coord - 0.5f;
        const float3 index = floor(coord_grid);
        const float3 fraction = coord_grid - index;
        float3 w0, w1, w2, w3;
        bspline_weights(fraction, w0, w1, w2, w3);

        const float3 g0 = w0 + w1;
        const float3 g1 = w2 + w3;
        const float3 h0 = (w1 / g0) - 0.5f + index;  //h0 = w1/g0 - 1, move from [-0.5, extent-0.5] to [0, extent]
        const float3 h1 = (w3 / g1) + 1.5f + index;  //h1 = w3/g1 + 1, move from [-0.5, extent-0.5] to [0, extent]

        // fetch the eight linear interpolations
        // weighting and fetching is interleaved for performance and stability reasons
        float tex000 = tex3D(tex, h0.x, h0.y, h0.z);
        float tex100 = tex3D(tex, h1.x, h0.y, h0.z);
        tex000 = g0.x * tex000 + g1.x * tex100;  //weigh along the x-direction
        float tex010 = tex3D(tex, h0.x, h1.y, h0.z);
        float tex110 = tex3D(tex, h1.x, h1.y, h0.z);
        tex010 = g0.x * tex010 + g1.x * tex110;  //weigh along the x-direction
        tex000 = g0.y * tex000 + g1.y * tex010;  //weigh along the y-direction
        float tex001 = tex3D(tex, h0.x, h0.y, h1.z);
        float tex101 = tex3D(tex, h1.x, h0.y, h1.z);
        tex001 = g0.x * tex001 + g1.x * tex101;  //weigh along the x-direction
        float tex011 = tex3D(tex, h0.x, h1.y, h1.z);
        float tex111 = tex3D(tex, h1.x, h1.y, h1.z);
        tex011 = g0.x * tex011 + g1.x * tex111;  //weigh along the x-direction
        tex001 = g0.y * tex001 + g1.y * tex011;  //weigh along the y-direction

        return (g0.z * tex000 + g1.z * tex001);  //weigh along the z-direction
    }

    __device__ float cubicTex3DSimple(texture<float, 3, cudaReadModeElementType> tex, float3 coord)
    {
        // transform the coordinate from [0,extent] to [-0.5, extent-0.5]
        const float3 coord_grid = coord - 0.5f;
        float3 index = floor(coord_grid);
        const float3 fraction = coord_grid - index;
        index = index + 0.5f;  //move from [-0.5, extent-0.5] to [0, extent]

        float result = 0.0f;
        for (float z=-1; z < 2.5f; z++)  //range [-1, 2]
        {
            float bsplineZ = bspline(z-fraction.z);
            float w = index.z + z;
            for (float y=-1; y < 2.5f; y++)
            {
                float bsplineYZ = bspline(y-fraction.y) * bsplineZ;
                float v = index.y + y;
                for (float x=-1; x < 2.5f; x++)
                {
                    float bsplineXYZ = bspline(x-fraction.x) * bsplineYZ;
                    float u = index.x + x;
                    result += bsplineXYZ * tex3D(tex, u, v, w);
                }
            }
        }
        return result;
    }

    texture<float, 3, cudaReadModeElementType> coeff_tex;

    __global__ void transform(const int4* dims, const float4* xform, float* volume)
    {
        // get voxel coordinates
        // in CUDA arrays X is the fastest changing dimension, so we need to swap x with z
        // NOTE: but not textures, they are saved exactly like passed
        uint3 vox = blockIdx * make_uint3(blockDim.x, blockDim.y, blockDim.z) + threadIdx;

        // more threads are created than needed, early outing the outside of bounds
        if (vox.x >= dims[0].x || vox.y >= dims[0].y || vox.z >= dims[0].z)
            return;

        // adding .5f to match center of texels
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
        float v = linearTex3D(coeff_tex, ndx);
        uint flat_index = vox.x + dims[0].x*vox.y + dims[0].x*dims[0].y*vox.z;
        volume[flat_index] = v;
    }

}
