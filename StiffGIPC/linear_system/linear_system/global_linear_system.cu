#include <linear_system/linear_system/global_linear_system.h>
#include <linear_system/linear_system/i_linear_system_solver.h>
#include <linear_system/linear_system/i_preconditioner.h>
#include <cuda_tools/cuda_tools.h>
#include <gipc/utils/timer.h>
#include <gipc/utils/benchmark.h>
#include <filesystem>
#include <fstream>
#include <vector>

namespace
{
void profile_begin(gipc::benchmark::Clock::time_point& start)
{
    if(gipc::benchmark::options.profile_stages)
    {
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        ++gipc::benchmark::frame.profiled_sync_calls;
        start = gipc::benchmark::Clock::now();
    }
}

void profile_end(gipc::benchmark::Clock::time_point start, double& total)
{
    if(gipc::benchmark::options.profile_stages)
    {
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        ++gipc::benchmark::frame.profiled_sync_calls;
        total += gipc::benchmark::elapsed_ms(start);
    }
}

template <typename T>
void write_audit_binary(const std::filesystem::path& path,
                        const std::vector<T>&        values)
{
    std::ofstream stream(path, std::ios::binary);
    if(!stream)
        throw std::runtime_error("Failed to open linear audit output");
    stream.write(reinterpret_cast<const char*>(values.data()),
                 values.size() * sizeof(T));
    if(!stream)
        throw std::runtime_error("Failed to write linear audit output");
}
}

namespace gipc
{
bool GlobalLinearSystem::build_linear_system()
{
    benchmark::Clock::time_point profile_start{};
    if(benchmark::options.profile_stages)
        ++benchmark::frame.linear_build_calls;
    profile_begin(profile_start);
    auto hessian_provider_count  = m_subsystems.size();
    auto gradient_provider_count = m_inner_subsystems.size();

    // right hand side can only be provided by both LinearSubsystem
    m_rhs_count_per_subsystem.resize(gradient_provider_count);
    m_rhs_offset_per_subsystem.resize(gradient_provider_count);

    for(auto& subsystem : m_subsystems)
        subsystem->report_subsystem_info();

    for(auto& gp : m_inner_subsystems)
    {
        auto i                       = gp->gid();
        m_rhs_count_per_subsystem[i] = gp->right_hand_side_dof();
    }

    std::exclusive_scan(m_rhs_count_per_subsystem.begin(),
                        m_rhs_count_per_subsystem.end(),
                        m_rhs_offset_per_subsystem.begin(),
                        0);

    for(auto& gp : m_inner_subsystems)
    {
        auto i = gp->gid();
        gp->dof_offset(m_rhs_offset_per_subsystem[i]);
    }

    auto total_rhs_count =
        m_rhs_offset_per_subsystem.back() + m_rhs_count_per_subsystem.back();

    if(benchmark::options.profile_stages)
        benchmark::frame.profiled_rhs_dofs += total_rhs_count;
    profile_end(profile_start, benchmark::frame.linear_info_ms);


    if(gipc_global_triplet->global_triplet_offset == 0 || total_rhs_count == 0)
    {
        std::cout << "The global linear system is empty, skip *assembling, *solving and *solution distributing phase."
                  << std::endl;
        return false;
    }


    profile_begin(profile_start);
    if(benchmark::options.profile_stages)
    {
        benchmark::frame.profiled_resize_calls += 2;
        benchmark::frame.profiled_resize_changes +=
            (m_b.size() != total_rhs_count) + (m_x.size() != total_rhs_count);
        if(total_rhs_count > m_b.capacity())
        {
            ++benchmark::frame.profiled_allocation_calls;
            benchmark::frame.profiled_allocation_bytes +=
                total_rhs_count * sizeof(Float);
        }
        if(total_rhs_count > m_x.capacity())
        {
            ++benchmark::frame.profiled_allocation_calls;
            benchmark::frame.profiled_allocation_bytes +=
                total_rhs_count * sizeof(Float);
        }
    }
    m_b.resize(total_rhs_count);
    m_x.resize(total_rhs_count);
    profile_end(profile_start, benchmark::frame.linear_resize_ms);

    auto rhs_view = m_b.view();

    profile_begin(profile_start);
    for(auto& subsystem : m_subsystems)
        subsystem->do_assemble(rhs_view);
    profile_end(profile_start, benchmark::frame.linear_subsystem_assemble_ms);

    int start_preconditioner_id = 0;
    profile_begin(profile_start);
    if(m_local_preconditioners.size() && m_local_preconditioners[0]->preconditioner_id == 0)
    {
        m_local_preconditioners[0]->assemble();
        start_preconditioner_id++;
    }
    profile_end(profile_start, benchmark::frame.linear_preconditioner_pre_ms);

    profile_begin(profile_start);
    convert_new();
    profile_end(profile_start, benchmark::frame.linear_convert_ms);

    profile_begin(profile_start);
    if(m_global_preconditioner)
        m_global_preconditioner->do_assemble(*gipc_global_triplet);
    profile_end(profile_start, benchmark::frame.linear_preconditioner_global_ms);

    profile_begin(profile_start);
    for(int i = start_preconditioner_id; i < m_local_preconditioners.size(); i++)
    {
        m_local_preconditioners[i]->assemble();
    }
    profile_end(profile_start, benchmark::frame.linear_preconditioner_post_ms);

    const bool audit = benchmark::options.audit_frame == benchmark::frame_id
                       && benchmark::frame.pcg_iterations.empty();
    if(audit)
    {
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        const auto unique_count = gipc_global_triplet->h_unique_key_number;
        std::vector<int> rows(unique_count), cols(unique_count);
        std::vector<Matrix3x3> blocks(unique_count);
        std::vector<Float> rhs(total_rhs_count);
        CUDA_SAFE_CALL(cudaMemcpy(rows.data(),
                                  gipc_global_triplet->block_row_indices(),
                                  rows.size() * sizeof(int),
                                  cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(cols.data(),
                                  gipc_global_triplet->block_col_indices(),
                                  cols.size() * sizeof(int),
                                  cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(blocks.data(),
                                  gipc_global_triplet->block_values(),
                                  blocks.size() * sizeof(Matrix3x3),
                                  cudaMemcpyDeviceToHost));
        m_b.copy_to(rhs);

        std::vector<Float> fixed(total_rhs_count);
        for(size_t i = 0; i < fixed.size(); ++i)
            fixed[i] = (static_cast<int>(i % 251) - 125) / 251.0;
        cudatool::DeviceDenseVector<Float> fixed_device;
        cudatool::DeviceDenseVector<Float> spmv_device(total_rhs_count);
        cudatool::DeviceDenseVector<Float> preconditioned_device(total_rhs_count);
        fixed_device.copy_from(fixed);
        spmv_device.buffer_view().fill(0);
        preconditioned_device.buffer_view().fill(0);
        spmv(1, fixed_device.cview(), 0, spmv_device.view());
        apply_preconditioner(preconditioned_device.view(), fixed_device.cview());
        std::vector<Float> spmv_result(total_rhs_count);
        std::vector<Float> preconditioned_result(total_rhs_count);
        spmv_device.copy_to(spmv_result);
        preconditioned_device.copy_to(preconditioned_result);

        const auto output = std::filesystem::path(benchmark::options.output_dir);
        write_audit_binary(output / "audit_matrix_rows.i32", rows);
        write_audit_binary(output / "audit_matrix_cols.i32", cols);
        write_audit_binary(output / "audit_matrix_blocks.f64", blocks);
        write_audit_binary(output / "audit_rhs.f64", rhs);
        write_audit_binary(output / "audit_fixed_input.f64", fixed);
        write_audit_binary(output / "audit_fixed_spmv.f64", spmv_result);
        write_audit_binary(output / "audit_fixed_preconditioner.f64",
                           preconditioned_result);
        benchmark::audits.push_back({{"kind", "linear_system_artifacts"},
                                     {"frame", benchmark::frame_id},
                                     {"unique_blocks", unique_count},
                                     {"rhs_dofs", total_rhs_count}});
    }

    return true;
}

void GlobalLinearSystem::distribute_solution()
{
    auto x_view = std::as_const(m_x).view();

    for(auto& subsystem : m_inner_subsystems)
        subsystem->do_retrieve_solution(x_view);

    wait_device();
}

DiagonalSubsystem& GlobalLinearSystem::_create_subsystem(U<DiagonalSubsystem>&& subsystem)
{
    auto ptr = subsystem.get();
    ptr->gid(m_inner_subsystems.size());
    m_inner_subsystems.push_back(ptr);  // push to gradient providers

    ptr->hid(m_subsystems.size());
    ptr->system(*this);
    m_subsystems.emplace_back(std::move(subsystem));  // push to hessian providers

    return *ptr;
}



IterativeSolver& GlobalLinearSystem::_create_solver(U<IterativeSolver>&& solver)
{
    m_solver = std::move(solver);
    m_solver->system(*this);
    return *m_solver;
}

GlobalLinearSystem::~GlobalLinearSystem() {}

LocalPreconditioner& GlobalLinearSystem::_create_preconditioner(U<LocalPreconditioner>&& preconditioner)
{
    preconditioner->system(*this);
    return *m_local_preconditioners.emplace_back(std::move(preconditioner));
}

GlobalPreconditioner& GlobalLinearSystem::_create_preconditioner(U<GlobalPreconditioner>&& preconditioner)
{
    CT_ASSERT(m_global_preconditioner == nullptr, "Global preconditioner already exists.");
    preconditioner->system(*this);
    m_global_preconditioner = std::move(preconditioner);
    return *m_global_preconditioner;
}

gipc::SizeT GlobalLinearSystem::solve_linear_system()
{
    bool success = build_linear_system();
    if(!success)
        return 0;
    CT_ASSERT(m_solver, "Solver is null, call create_solver() to setup a solver.");
    benchmark::Clock::time_point profile_start{};
    profile_begin(profile_start);
    auto iter = m_solver->solve(m_x, m_b);
    profile_end(profile_start, benchmark::frame.linear_solver_only_ms);
    profile_begin(profile_start);
    distribute_solution();
    profile_end(profile_start, benchmark::frame.linear_distribute_ms);
    return iter;
}

Json GlobalLinearSystem::solution_diagnostics()
{
    std::vector<Float> gradient(m_b.size());
    std::vector<Float> solution(m_x.size());
    m_b.copy_to(gradient);
    m_x.copy_to(solution);
    double gradient_sq = 0;
    double solution_sq = 0;
    double gradient_dot_solution = 0;
    for(size_t i = 0; i < gradient.size(); ++i)
    {
        gradient_sq += gradient[i] * gradient[i];
        solution_sq += solution[i] * solution[i];
        gradient_dot_solution += gradient[i] * solution[i];
    }
    // Both FEM and ABD apply state <- state - alpha * solution.
    return {{"dof_count", gradient.size()},
            {"gradient_l2", std::sqrt(gradient_sq)},
            {"solution_l2", std::sqrt(solution_sq)},
            {"gradient_dot_solution", gradient_dot_solution},
            {"gradient_dot_actual_direction", -gradient_dot_solution},
            {"actual_direction_sign", -1}};
}

Json GlobalLinearSystem::as_json() const
{
    Json j;
    j["solver"]     = typeid(*m_solver).name();
    j["subsystems"] = Json::array();
    for(auto& s : m_subsystems)
    {
        j["subsystems"].push_back(s->as_json());
    }
    j["preconditioners"] = Json::array();
    for(auto& p : m_local_preconditioners)
    {
        j["preconditioners"].push_back(p->as_json());
    }
    return j;
}

void GlobalLinearSystem::apply_preconditioner(cudatool::DenseVectorView<Float>  z,
                                              cudatool::CDenseVectorView<Float> r)
{
    // first apply global preconditioner
    if(m_global_preconditioner)
        m_global_preconditioner->do_apply(r, z);
    else  // if no global preconditioner, use identity
        z.buffer_view().copy_from(r.buffer_view());

    // then apply local preconditioners
    // it's user's choice to rewrite or reuse the global preconditioner
    for(auto& p : m_local_preconditioners)
        p->do_apply(r, z);
}



void GlobalLinearSystem::convert_new()
{
    m_converter.convert(*gipc_global_triplet,
                        0,
                        gipc_global_triplet->global_triplet_offset,
                        gipc_global_triplet->global_triplet_offset);
//#ifndef SymGH
//    m_converter.ge2sym(*gipc_global_triplet);
//#endif
}



void GlobalLinearSystem::spmv(Float                         a,
                              cudatool::CDenseVectorView<Float> x,
                              Float                         b,
                              cudatool::DenseVectorView<Float>  y)
{

    m_spmv.warp_reduce_sym_spmv(a,
                                gipc_global_triplet->block_values(),
                                gipc_global_triplet->block_row_indices(),
                                gipc_global_triplet->block_col_indices(),
                                gipc_global_triplet->h_unique_key_number,
                                x,
                                b,
                                y);
}
}  // namespace gipc
