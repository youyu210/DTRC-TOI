#include <linear_system/solver/pcg_solver.h>
#include <gipc/utils/timer.h>
#include <gipc/statistics.h>
#include <cuda_tools/cuda_tools.h>
#include <gipc/utils/benchmark.h>
#include <cmath>
#include <algorithm>
#include <filesystem>
#include <fstream>



__global__ void PCG_vdv_Reduction(double* squeue, const double* a, const double* b, int numbers)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];

    if(idx >= numbers)
        return;

    double temp = a[idx] * b[idx];

    int    warpTid = threadIdx.x % 32;
    int    warpId  = (threadIdx.x >> 5);
    //double nextTp;
    int    warpNum;
    if(blockIdx.x == gridDim.x - 1)
    {
        warpNum = ((numbers - idof + 31) >> 5);
    }
    else
    {
        warpNum = ((blockDim.x) >> 5);
    }
    for(int i = 1; i < 32; i = (i << 1))
    {
        temp += __shfl_down_sync(0xffffffff, temp, i);
    }
    if(warpTid == 0)
    {
        tep[warpId] = temp;
    }
    __syncthreads();
    if(threadIdx.x >= warpNum)
        return;
    if(warpNum > 1)
    {
        temp = tep[threadIdx.x];
        for(int i = 1; i < warpNum; i = (i << 1))
        {
            temp += __shfl_down_sync(0xffffffff, temp, i);
        }
    }
    if(threadIdx.x == 0)
    {
        squeue[blockIdx.x] = temp;
    }
}



__global__ void add_reduction(double* mem, int numbers)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;
    extern __shared__ double tep[];
    if(idx >= numbers)
        return;
    double temp = mem[idx];
    int    warpTid = threadIdx.x % 32;
    int    warpId  = (threadIdx.x >> 5);
    int    warpNum;
    if(blockIdx.x == gridDim.x - 1)
    {
        warpNum = ((numbers - idof + 31) >> 5);
    }
    else
    {
        warpNum = ((blockDim.x) >> 5);
    }
    for(int i = 1; i < 32; i = (i << 1))
    {
        temp += __shfl_down_sync(0xffffffff, temp, i);
    }
    if(warpTid == 0)
    {
        tep[warpId] = temp;
    }
    __syncthreads();
    if(threadIdx.x >= warpNum)
        return;
    if(warpNum > 1)
    {
        temp = tep[threadIdx.x];
        for(int i = 1; i < warpNum; i = (i << 1))
        {
            temp += __shfl_down_sync(0xffffffff, temp, i);
        }
    }
    if(threadIdx.x == 0)
    {
        mem[blockIdx.x] = temp;
    }
}



__global__ void update_vector_dx_r(
    double* dx, double* r, const double* c, const double* q, double alpha, int numbers)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= numbers)
        return;
    dx[idx] = dx[idx] + alpha * c[idx];
    r[idx]  = r[idx] - alpha * q[idx];
}

__global__ void update_vector_c(
    double* c, const double* s, double beta, int numbers)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= numbers)
        return;
    c[idx] = s[idx] + beta * c[idx];
}


double My_PCG_General_v_v_Reduction_Algorithm(double* temp, double* A, double* B, int vertexNum)
{

    int numbers = vertexNum;
    if(numbers < 1)
        return 0;
    const unsigned int threadNum = 256;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(double) * (threadNum >> 5);
    PCG_vdv_Reduction<<<blockNum, threadNum, sharedMsize>>>(temp, A, B, numbers);


    numbers  = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;

    while(numbers > 1)
    {
        add_reduction<<<blockNum, threadNum, sharedMsize>>>(temp, numbers);
        numbers  = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;
    }
    double result;
    cudaMemcpy(&result, temp, sizeof(double), cudaMemcpyDeviceToHost);
    return result;
}

namespace gipc
{
PCGSolver::PCGSolver(const PCGSolverConfig& cfg)
    : m_config(cfg)
{
}
SizeT PCGSolver::solve(cudatool::DenseVectorView<Float> x, cudatool::CDenseVectorView<Float> b)
{
    Timer timer{"pcg"};
    if(benchmark::active() && benchmark::options.trace_termination)
    {
        // Diagnostic only: independently check the scalar against the original
        // global assembler's actual RHS, before any PCG residual is formed.
        std::vector<Float> native_rhs(b.size());
        b.copy_to(native_rhs.data());
        double norm = 0;
        for(double value : native_rhs)
            norm = std::max(norm, std::abs(value));
        const double measured = benchmark::frame.exit_residual;
        const bool agrees = std::isfinite(norm) && std::isfinite(measured)
            && std::abs(norm - measured) <= 1e-13 * std::max({1.0, norm, measured});
        benchmark::audits.push_back({{"kind", "nonlinear_native_rhs"},
            {"frame", benchmark::frame_id}, {"rhs_infinity_norm", norm},
            {"measured_residual", measured}, {"agrees", agrees}});
        if(!agrees)
            throw std::runtime_error("Nonlinear residual differs from original assembled RHS");
    }
    auto start = benchmark::Clock::now();

    x.buffer_view().fill(0);
    z.resize(b.size());
    p.resize(b.size());
    r.resize(b.size());
    //temp.resize(b.size());
    Ap.resize(b.size());
    const SizeT max_iter = m_config.max_iter_ratio * b.size();
    auto iter = m_config.dual_graph ? pcg_dual_graph(x, b, max_iter)
                                    : pcg(x, b, max_iter);
    if(benchmark::active())
    {
        CUDA_SAFE_CALL(cudaStreamSynchronize(cudaStreamPerThread));
        benchmark::frame.pcg_ms += benchmark::elapsed_ms(start);
        const bool audit = benchmark::options.audit_frame == benchmark::frame_id
                           && benchmark::frame.pcg_iterations.empty();
        benchmark::frame.pcg_iterations.push_back(iter);
        if(audit)
        {
            // An independent eager solve of the exact same A and b. The actual
            // output x is retained; this diagnostic is excluded from PCG time.
            // Audit runs are separate from Newton/step speed measurements.
            DeviceDenseVector reference(b.size());
            reference.buffer_view().fill(0);
            auto reference_iter = pcg(reference.view(), b, max_iter);
            CUDA_SAFE_CALL(cudaStreamSynchronize(cudaStreamPerThread));
            std::vector<Float> actual(b.size()), expected(b.size());
            x.copy_to(actual.data());
            reference.copy_to(expected);
            double max_abs = 0, squared_diff = 0, squared_ref = 0;
            for(size_t i = 0; i < actual.size(); ++i)
            {
                double diff = actual[i] - expected[i];
                max_abs = std::max(max_abs, std::abs(diff));
                squared_diff += diff * diff;
                squared_ref += expected[i] * expected[i];
            }
            const auto output = std::filesystem::path(benchmark::options.output_dir);
            std::ofstream actual_file(output / "audit_direction.f64", std::ios::binary);
            std::ofstream eager_file(output / "audit_direction_eager.f64", std::ios::binary);
            actual_file.write(reinterpret_cast<const char*>(actual.data()),
                              actual.size() * sizeof(Float));
            eager_file.write(reinterpret_cast<const char*>(expected.data()),
                             expected.size() * sizeof(Float));
            if(!actual_file || !eager_file)
                throw std::runtime_error("Failed to write PCG audit direction");
            DeviceDenseVector product(b.size());
            std::vector<Float> rhs(b.size()), ax(b.size());
            b.copy_to(rhs.data());
            auto true_relative_residual = [&](cudatool::CDenseVectorView<Float> solution)
            {
                spmv(solution, product.view());
                product.copy_to(ax);
                double residual2 = 0, rhs2 = 0;
                for(size_t i = 0; i < rhs.size(); ++i)
                {
                    const double value = rhs[i] - ax[i];
                    residual2 += value * value;
                    rhs2 += rhs[i] * rhs[i];
                }
                return std::sqrt(residual2) / std::max(std::sqrt(rhs2), 1e-30);
            };
            const double actual_residual = true_relative_residual(
                cudatool::CDenseVectorView<Float>(x.data(), x.size()));
            const double eager_residual = true_relative_residual(reference.cview());
            benchmark::audits.push_back({{"frame", benchmark::frame_id},
                {"actual_iterations", iter}, {"eager_iterations", reference_iter},
                {"actual_true_relative_residual", actual_residual},
                {"eager_true_relative_residual", eager_residual},
                {"max_abs_direction_diff", max_abs},
                {"relative_l2_direction_diff", std::sqrt(squared_diff)
                    / std::max(std::sqrt(squared_ref), 1e-30)}});
        }
    }

    return iter;
}


SizeT PCGSolver::pcg(cudatool::DenseVectorView<Float> x, cudatool::CDenseVectorView<Float> b, SizeT max_iter)
{
    SizeT k = 0;

    r.buffer_view().copy_from(b.buffer_view());

    Float alpha, beta, rz, rz0;

    {
        //Timer timer{"preconditioner"};
        apply_preconditioner(z, r);
    }

    {
        //Timer timer{"dot"};
        rz = My_PCG_General_v_v_Reduction_Algorithm(p.buffer_view().data(),
                                                    r.buffer_view().data(),
                                                    z.buffer_view().data(),
                                                    z.size());
    }

    p   = z;
    rz0 = rz;

    for(k = 1; k < max_iter; ++k)
    {
        {
            //Timer timer{"spmv"};
            // Ap = A * p
            spmv(p.cview(), Ap.view());
        }

        {
            //Timer timer{"dot"};

            Float dot_res =
                My_PCG_General_v_v_Reduction_Algorithm(z.buffer_view().data(),
                                                       p.buffer_view().data(),
                                                       Ap.buffer_view().data(),
                                                       z.size());

            alpha = rz / dot_res;
        }

        {
            //Timer timer{"axpby"};
            LaunchCudaKernal_default(z.size(),
                                     256,
                                     0,
                                     update_vector_dx_r,
                                     x.buffer_view().data(),
                                     r.buffer_view().data(),
                                     (const double*)p.buffer_view().data(),
                                     (const double*)Ap.buffer_view().data(),
                                     alpha,
                                     (int)z.size());
        }

        if(std::abs(rz) <= m_config.global_tol_rate * rz0)
            break;

        {
            //Timer timer{"preconditioner"};
            apply_preconditioner(z, r);
        }

        Float rz_new = 0;
        {
            //Timer timer{"dot"};
            rz_new = My_PCG_General_v_v_Reduction_Algorithm(Ap.buffer_view().data(),
                                                            r.buffer_view().data(),
                                                            z.buffer_view().data(),
                                                            z.size());
        }

        beta = rz_new / rz;

        {
            //Timer timer{"axpby"};
            LaunchCudaKernal_default(z.size(),
                                     256,
                                     0,
                                     update_vector_c,
                                     p.buffer_view().data(),
                                     (const double*)z.buffer_view().data(),
                                     beta,
                                     (int)z.size());
        }

        rz = rz_new;
    }

    return k;
}

}  // namespace gipc
