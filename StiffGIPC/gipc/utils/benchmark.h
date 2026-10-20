#pragma once
#include <gipc/utils/json.h>
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace gipc::benchmark
{
using Clock = std::chrono::steady_clock;
struct Options
{
    int         scene                  = 1;
    int         frames                 = 0;
    bool        dual_graph             = false;
    bool        export_states          = false;
    int         audit_frame            = 0;
    std::string termination            = "original";
    int         toi_min_updates        = 6;
    double      toi_threshold          = 1e-3;
    double      residual_floor         = 1e-30;
    double      residual_threshold     = 0;
    bool        profile_stages         = false;
    bool        production_measurement = false;
    bool        converter_indexed_reduce = false;
    double      cloth_offset_y         = 0;
    bool        trace_termination      = false;
    bool        trace_newton           = false;
    double      diagnostic_max_ms      = 0;
    int         diagnostic_max_newton  = 0;
    double      newton_tolerance_scale = 1;
    double      pcg_tolerance_scale    = 1;
    std::string output_dir;
    std::string checkpoint_in;
    std::string checkpoint_out;
    std::string checkpoint_roundtrip_out;
    int         checkpoint_frame       = -1;
    std::string checkpoint_output_root;
    std::vector<int> checkpoint_frames;
};
struct Frame
{
    double                newton_ms          = 0;
    double                pcg_ms             = 0;
    double                step_ms            = 0;
    uint64_t              newton_iterations  = 0;
    uint64_t              newton_cap_hits    = 0;
    uint64_t              direction_launches = 0;
    uint64_t              tail_launches      = 0;
    uint64_t              captures           = 0;
    uint64_t              updates            = 0;
    uint64_t              rebuilds           = 0;
    uint64_t              fallbacks          = 0;
    std::vector<uint64_t> pcg_iterations;
    double                exit_residual         = 0;
    double                exit_relative_initial = 0;
    double                exit_budget           = 1;
    double                exit_beta             = 1;
    uint64_t              residual_evaluations  = 0;
    uint64_t              target_changes        = 0;
    int                   first_toi_ready_k     = -1;
    int                   exit_k                = -1;
    uint64_t              newton_solves         = 0;
    uint64_t              tail_iterations       = 0;
    double                tail_ms               = 0;
    uint64_t              residual_rejections   = 0;
    uint64_t              history_rejections    = 0;
    uint64_t              guard_rejections      = 0;
    uint64_t              diagnostic_budget_hits = 0;
    double                gradient_hessian_ms   = 0;
    double                residual_check_ms     = 0;
    double                linear_system_ms      = 0;
    double                ccd_ms                = 0;
    double                line_search_ms        = 0;
    double                graph_prepare_ms      = 0;
    // B1 detailed profiling. These fields are populated only when
    // --profile-stages is enabled; the default path only sees guarded adds.
    double gradient_zero_ms          = 0;
    double gradient_kinetic_ms       = 0;
    double gradient_barrier_ms       = 0;
    double gradient_friction_grad_ms = 0;
    double gradient_friction_hess_ms = 0;
    double gradient_ground_ms        = 0;
    double gradient_contact_classify_ms = 0;
    double gradient_abd_ms           = 0;
    double gradient_contact_index_ms = 0;
    double gradient_tet_fem_ms       = 0;
    double gradient_bending_ms       = 0;
    double gradient_triangle_fem_ms  = 0;
    double gradient_soft_boundary_mass_ms = 0;
    double linear_info_ms            = 0;
    double linear_resize_ms          = 0;
    double linear_subsystem_assemble_ms = 0;
    double linear_preconditioner_pre_ms = 0;
    double linear_convert_ms         = 0;
    double linear_preconditioner_global_ms = 0;
    double linear_preconditioner_post_ms = 0;
    double linear_solver_only_ms     = 0;
    double linear_distribute_ms      = 0;
    double convert_hash_index_ms     = 0;
    double convert_radix_sort_ms     = 0;
    double convert_gather_ms         = 0;
    double convert_partition_mark_ms = 0;
    double convert_partition_scan_ms = 0;
    double convert_row_col_ms        = 0;
    double convert_host_read_ms      = 0;
    double convert_block_zero_ms     = 0;
    double convert_reduce_ms         = 0;
    double convert_output_copy_ms    = 0;
    uint64_t gradient_hessian_calls  = 0;
    uint64_t linear_build_calls      = 0;
    uint64_t convert_calls           = 0;
    uint64_t profiled_sync_calls     = 0;
    uint64_t profiled_d2h_calls      = 0;
    uint64_t profiled_d2h_bytes      = 0;
    uint64_t profiled_resize_calls   = 0;
    uint64_t profiled_resize_changes = 0;
    uint64_t profiled_allocation_calls = 0;
    uint64_t profiled_allocation_bytes = 0;
    uint64_t profiled_contact_triplets = 0;
    uint64_t profiled_total_triplets = 0;
    uint64_t profiled_rhs_dofs       = 0;
    uint64_t profiled_convert_input_triplets = 0;
    uint64_t profiled_unique_blocks  = 0;
    uint64_t profiled_fem_tets       = 0;
    uint64_t profiled_tri_edges      = 0;
    uint64_t profiled_triangles      = 0;
    uint64_t profiled_fem_points     = 0;
    bool                  budget_valid          = false;
    bool                  budget_active         = false;
    bool                  bounds_valid          = true;
    std::string           exit_reason           = "iteration_cap";
    Json                  termination_samples   = Json::array();
    Json                  newton_samples        = Json::array();
};
inline Options options;
inline Frame   frame;
inline int     frame_id = 0;
inline Json    audits   = Json::array();
inline bool    active()
{
    return options.frames > 0;
}
inline double elapsed_ms(Clock::time_point start)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
}  // namespace gipc::benchmark
