#pragma once
#include <linear_system/linear_system/i_linear_system_solver.h>
#include <array>
#include <cstdint>

namespace gipc
{
class PCGSolverConfig
{
  public:
    /**
     * \brief the maximum number of iterations will be:
     *  dof * max_iter_ratio
     */
    Float max_iter_ratio  = 0.3;
    Float global_tol_rate = 1e-4;
    bool  use_bsr         = true;
    bool  dual_graph      = false;
};

class PCGSolver : public IterativeSolver
{
    using DeviceDenseVector = cudatool::DeviceDenseVector<Float>;

  public:
    PCGSolver(const PCGSolverConfig& cfg);
    ~PCGSolver() override;

    void config(const PCGSolverConfig& config) { this->m_config = config; }
    const auto& config() const { return this->m_config; }

  private:
    DeviceDenseVector z;   // preconditioned residual
    DeviceDenseVector r;   // residual
    DeviceDenseVector p;   // search direction
    DeviceDenseVector Ap;  // A*p
    PCGSolverConfig   m_config;

    // Separate keys: changing matrix storage/count only invalidates direction.
    cudaGraphExec_t          m_direction_exec = nullptr;
    cudaGraphExec_t          m_tail_exec      = nullptr;
    std::array<uint64_t, 14> m_direction_key{};
    std::array<uint64_t, 9>  m_tail_key{};
    Float* m_scalars = nullptr;  // rho, alpha, beta; stable between replays

  protected:
    SizeT solve(cudatool::DenseVectorView<Float>  x,
                cudatool::CDenseVectorView<Float> b) override;

  private:
    SizeT pcg(cudatool::DenseVectorView<Float> x, cudatool::CDenseVectorView<Float> b, SizeT max_iter);
    SizeT pcg_dual_graph(cudatool::DenseVectorView<Float>  x,
                         cudatool::CDenseVectorView<Float> b,
                         SizeT                             max_iter);
};
}  // namespace gipc
