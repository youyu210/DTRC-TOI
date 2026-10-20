#include <linear_system/solver/pcg_solver.h>
#include <linear_system/linear_system/global_matrix.h>
#include <gipc/utils/benchmark.h>
#include <cuda_tools/cuda_tools.h>
#include <cmath>
#include <cstdio>

// Reuse the original reductions, including their tree and scratch aliases.
__global__ void PCG_vdv_Reduction(double*, const double*, const double*, int);
__global__ void add_reduction(double*, int);
double My_PCG_General_v_v_Reduction_Algorithm(double*, double*, double*, int);

namespace
{
void enqueue_dot(double* scratch, const double* a, const double* b, int n)
{
    int blocks = (n + 255) / 256;
    PCG_vdv_Reduction<<<blocks, 256, sizeof(double) * 8>>>(scratch, a, b, n);
    n = blocks;
    while(n > 1)
    {
        blocks = (n + 255) / 256;
        add_reduction<<<blocks, 256, sizeof(double) * 8>>>(scratch, n);
        n = blocks;
    }
}

__global__ void alpha_kernel(double* scalars, const double* dot)
{
    scalars[1] = __ddiv_rn(scalars[0], dot[0]);
}

__global__ void dx_r_kernel(
    double* x, double* r, const double* p, const double* Ap, const double* scalars, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n)
        return;
    const double alpha = scalars[1];
    x[i]               = x[i] + alpha * p[i];
    r[i]               = r[i] - alpha * Ap[i];
}

__global__ void beta_kernel(double* scalars, const double* dot)
{
    scalars[2] = __ddiv_rn(dot[0], scalars[0]);
    scalars[0] = dot[0];
}

__global__ void p_kernel(double* p, const double* z, const double* scalars, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n)
        return;
    const double beta = scalars[2];
    p[i]              = z[i] + beta * p[i];
}

uint64_t address(const void* pointer)
{
    return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(pointer));
}

// Capture does not execute a segment. On preparation failure, submit that
// segment eagerly once. Never retry a partially executed graph launch.
template <size_t N, typename Submit>
bool prepare(cudaGraphExec_t&               exec,
             std::array<uint64_t, N>&       previous,
             const std::array<uint64_t, N>& key,
             Submit                         submit)
{
    if(exec && previous == key)
        return true;
    auto status = cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal);
    if(status != cudaSuccess)
    {
        ++gipc::benchmark::frame.fallbacks;
        std::fprintf(stderr, "Graph capture begin: %s\n", cudaGetErrorString(status));
        return false;
    }
    submit();
    cudaGraph_t graph = nullptr;
    status            = cudaStreamEndCapture(cudaStreamPerThread, &graph);
    if(status != cudaSuccess || !graph)
    {
        if(graph)
            cudaGraphDestroy(graph);
        ++gipc::benchmark::frame.fallbacks;
        std::fprintf(stderr, "Graph capture end: %s\n", cudaGetErrorString(status));
        return false;
    }
    ++gipc::benchmark::frame.captures;
    if(exec)
    {
        cudaGraphExecUpdateResultInfo info{};
        status = cudaGraphExecUpdate(exec, graph, &info);
        if(status == cudaSuccess && info.result == cudaGraphExecUpdateSuccess)
        {
            previous = key;
            ++gipc::benchmark::frame.updates;
            CUDA_SAFE_CALL(cudaGraphDestroy(graph));
            return true;
        }
    }
    cudaGraphExec_t replacement = nullptr;
    status = cudaGraphInstantiate(&replacement, graph, nullptr, nullptr, 0);
    CUDA_SAFE_CALL(cudaGraphDestroy(graph));
    if(status != cudaSuccess)
    {
        ++gipc::benchmark::frame.fallbacks;
        std::fprintf(stderr, "Graph instantiate: %s\n", cudaGetErrorString(status));
        return false;
    }
    if(exec)
    {
        CUDA_SAFE_CALL(cudaGraphExecDestroy(exec));
        ++gipc::benchmark::frame.rebuilds;
    }
    exec     = replacement;
    previous = key;
    return true;
}
}  // namespace

namespace gipc
{
PCGSolver::~PCGSolver()
{
    if(m_direction_exec)
        cudaGraphExecDestroy(m_direction_exec);
    if(m_tail_exec)
        cudaGraphExecDestroy(m_tail_exec);
    if(m_scalars)
        cudaFree(m_scalars);
}

SizeT PCGSolver::pcg_dual_graph(cudatool::DenseVectorView<Float>  x,
                                cudatool::CDenseVectorView<Float> b,
                                SizeT                             max_iter)
{
    if(!m_scalars)
        CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&m_scalars), 3 * sizeof(Float)));
    const int n = static_cast<int>(z.size());
    // This setup is identical to pcg(), including its initial host reduction.
    r.buffer_view().copy_from(b.buffer_view());
    apply_preconditioner(z, r);
    Float rz = My_PCG_General_v_v_Reduction_Algorithm(p.data(), r.data(), z.data(), n);
    p               = z;
    const Float rz0 = rz;
    CUDA_SAFE_CALL(cudaMemcpy(m_scalars, &rz, sizeof(Float), cudaMemcpyHostToDevice));

    auto&                          A = matrix();
    const std::array<uint64_t, 14> direction_key{address(x.data()),
                                                 address(r.data()),
                                                 address(p.data()),
                                                 address(z.data()),
                                                 address(Ap.data()),
                                                 address(m_scalars),
                                                 address(A.block_values()),
                                                 address(A.block_row_indices()),
                                                 address(A.block_col_indices()),
                                                 static_cast<uint64_t>(n),
                                                 static_cast<uint64_t>(A.h_unique_key_number),
                                                 A.triplet_capacity(),
                                                 p.capacity(),
                                                 Ap.capacity()};
    const std::array<uint64_t, 9>  tail_key{address(r.data()),
                                           address(z.data()),
                                           address(p.data()),
                                           address(Ap.data()),
                                           address(m_scalars),
                                           static_cast<uint64_t>(n),
                                           r.capacity(),
                                           z.capacity(),
                                           p.capacity()};

    auto direction = [&]()
    {
        spmv(p.cview(), Ap.view());
        enqueue_dot(z.data(), p.data(), Ap.data(), n);
        alpha_kernel<<<1, 1>>>(m_scalars, z.data());
        dx_r_kernel<<<(n + 255) / 256, 256>>>(
            x.data(), r.data(), p.data(), Ap.data(), m_scalars, n);
    };
    auto tail = [&]()
    {
        enqueue_dot(Ap.data(), r.data(), z.data(), n);
        beta_kernel<<<1, 1>>>(m_scalars, Ap.data());
        p_kernel<<<(n + 255) / 256, 256>>>(p.data(), z.data(), m_scalars, n);
    };
    const auto prepare_start = benchmark::Clock::now();
    const bool direction_ready =
        prepare(m_direction_exec, m_direction_key, direction_key, direction);
    const bool tail_ready = prepare(m_tail_exec, m_tail_key, tail_key, tail);
    if(benchmark::options.profile_stages)
        benchmark::frame.graph_prepare_ms += benchmark::elapsed_ms(prepare_start);
    SizeT      k;
    for(k = 1; k < max_iter; ++k)
    {
        if(direction_ready)
        {
            CUDA_SAFE_CALL(cudaGraphLaunch(m_direction_exec, cudaStreamPerThread));
            ++benchmark::frame.direction_launches;
        }
        else
            direction();

        // Preserve upstream's old-rho check AFTER the direction update. Do not
        // add a zero-rhs shortcut, new residual rule, or iteration cap.
        if(std::abs(rz) <= m_config.global_tol_rate * rz0)
            break;

        apply_preconditioner(z, r);
        if(tail_ready)
        {
            CUDA_SAFE_CALL(cudaGraphLaunch(m_tail_exec, cudaStreamPerThread));
            ++benchmark::frame.tail_launches;
        }
        else
            tail();
        // Host termination still uses the original scalar each iteration.
        CUDA_SAFE_CALL(cudaMemcpy(&rz, m_scalars, sizeof(Float), cudaMemcpyDeviceToHost));
    }
    return k;
}
}  // namespace gipc
