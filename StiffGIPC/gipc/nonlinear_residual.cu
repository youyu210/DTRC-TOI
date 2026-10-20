#include <GIPC.cuh>
#include <abd_system/abd_system.h>
#include <cuda_tools/cuda_tools.h>
#include <cmath>
#include <math_constants.h>

namespace
{
// Exactly the coordinate-space gradient used by the original ABD and FEM
// RHS assemblers: ABD system_gradient, FEM fb + shape_grads. Fixed ABD
// gradients are already zeroed by setup; prescribed FEM DOFs are excluded.
__global__ void residual_blocks(double*        blocks,
                                const double*  abd,
                                int            abd_dofs,
                                const double3* contact,
                                const double3* shape,
                                const int*     boundary,
                                int            fem_offset,
                                int            fem_count)
{
    __shared__ double values[256];
    int               i     = blockIdx.x * blockDim.x + threadIdx.x;
    double            value = 0;
    if(i < abd_dofs)
        value = abd[i];
    else if(i < abd_dofs + 3 * fem_count)
    {
        int local  = i - abd_dofs;
        int vertex = fem_offset + local / 3;
        if(boundary[vertex] == 0)
        {
            double3 a = contact[vertex], b = shape[vertex];
            switch(local % 3)
            {
                case 0:
                    value = a.x + b.x;
                    break;
                case 1:
                    value = a.y + b.y;
                    break;
                default:
                    value = a.z + b.z;
                    break;
            }
        }
    }
    // Never let fmax silently discard NaN and validate a bad residual.
    values[threadIdx.x] = isfinite(value) ? fabs(value) : CUDART_INF;
    __syncthreads();
    for(int stride = 128; stride; stride /= 2)
    {
        if(threadIdx.x < stride)
            values[threadIdx.x] = fmax(values[threadIdx.x], values[threadIdx.x + stride]);
        __syncthreads();
    }
    if(threadIdx.x == 0)
        blocks[blockIdx.x + 1] = values[0];
}

__global__ void residual_finish(double* blocks, int n)
{
    __shared__ double values[256];
    double            value = 0;
    for(int i = threadIdx.x; i < n; i += 256)
        value = fmax(value, blocks[i + 1]);
    values[threadIdx.x] = value;
    __syncthreads();
    for(int stride = 128; stride; stride /= 2)
    {
        if(threadIdx.x < stride)
            values[threadIdx.x] = fmax(values[threadIdx.x], values[threadIdx.x + stride]);
        __syncthreads();
    }
    if(threadIdx.x == 0)
        blocks[0] = values[0];
}
}  // namespace

double GIPC::nonlinear_free_residual(device_TetraData& tet)
{
    int abd_dofs  = static_cast<int>(m_abd_system->system_gradient.size());
    int fem_count = abd_fem_count_info.fem_point_num;
    int n         = abd_dofs + 3 * fem_count;
    if(n == 0)
        return 0;
    int blocks = (n + 255) / 256;
    m_residual_scratch.resize(blocks + 1);
    residual_blocks<<<blocks, 256>>>(m_residual_scratch.data(),
                                     m_abd_system->system_gradient.data(),
                                     abd_dofs,
                                     tet.fb,
                                     tet.shape_grads,
                                     tet.BoundaryType,
                                     abd_fem_count_info.fem_point_offset,
                                     fem_count);
    residual_finish<<<1, 256>>>(m_residual_scratch.data(), blocks);
    double result = 0;
    CUDA_SAFE_CALL(cudaMemcpy(&result, m_residual_scratch.data(), sizeof(double), cudaMemcpyDeviceToHost));
    return result;
}
