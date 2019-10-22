#include "math_functions.h"
#include "vector_functions.h"
#include "vector_types.h"
#include "helper_math.h"
typedef unsigned int uint;
typedef unsigned short ushort;
typedef unsigned char uchar;
typedef signed char schar;





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
        const uint3* shape)
    {
        // process lines in x-direction
        const uint y = blockIdx.x * blockDim.x + threadIdx.x;
        const uint z = blockIdx.y * blockDim.y + threadIdx.y;
        const uint startIdx = (z * shape[0].y + y) * pitch;

        float* ptr = (float*)((uchar*)volume + startIdx);
        ConvertToInterpolationCoefficients(ptr, shape[0].x, sizeof(float));
    }

    __global__ void SamplesToCoefficients3DY(
        float* volume,		// in-place processing
        uint pitch,			// width in bytes
        const uint3* shape)
    {
        // process lines in y-direction
        const uint x = blockIdx.x * blockDim.x + threadIdx.x;
        const uint z = blockIdx.y * blockDim.y + threadIdx.y;
        const uint startIdx = z * shape[0].y * pitch;

        float* ptr = (float*)((uchar*)volume + startIdx);
        ConvertToInterpolationCoefficients(ptr + x, shape[0].y, pitch);
    }

    __global__ void SamplesToCoefficients3DZ(
        float* volume,		// in-place processing
        uint pitch,			// width in bytes
        const uint3* shape)
    {
        // process lines in z-direction
        const uint x = blockIdx.x * blockDim.x + threadIdx.x;
        const uint y = blockIdx.y * blockDim.y + threadIdx.y;
        const uint startIdx = y * pitch;
        const uint slice = shape[0].y * pitch;

        float* ptr = (float*)((uchar*)volume + startIdx);
        ConvertToInterpolationCoefficients(ptr + x, shape[0].z, slice);
    }
}