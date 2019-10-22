__device__ float linearTex3D(texture<float, 3, cudaReadModeElementType> tex, float3 coord)
{
    return tex3D(tex, coord.x, coord.y, coord.z);
}
__device__ float linearTex3D(texture<float, 3, cudaReadModeElementType> tex, float4 coord)
{
    return tex3D(tex, coord.x, coord.y, coord.z);
}
/* TODO: bspline, bsplinehq */