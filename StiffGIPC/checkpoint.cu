#include <GIPC.cuh>
#include <abd_system/abd_sim_data.h>
#include <cuda_tools/cuda_tools.h>
#include <gipc/utils/json.h>

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

extern int total_Frames;

namespace
{
namespace fs = std::filesystem;

template <typename T>
void save_device(const fs::path& file, const T* device, size_t count)
{
    std::vector<T> host(count);
    if(count)
        CUDA_SAFE_CALL(cudaMemcpy(host.data(), device, count * sizeof(T), cudaMemcpyDeviceToHost));
    std::ofstream out(file, std::ios::binary);
    out.write(reinterpret_cast<const char*>(host.data()), count * sizeof(T));
    if(!out)
        throw std::runtime_error("Checkpoint write failed: " + file.string());
}

template <typename T>
void load_device(const fs::path& file, T* device, size_t count)
{
    const auto expected = static_cast<uintmax_t>(count * sizeof(T));
    if(!fs::exists(file) || fs::file_size(file) != expected)
        throw std::runtime_error("Checkpoint size mismatch: " + file.string());
    std::vector<T> host(count);
    std::ifstream in(file, std::ios::binary);
    in.read(reinterpret_cast<char*>(host.data()), expected);
    if(!in || in.peek() != std::ifstream::traits_type::eof())
        throw std::runtime_error("Checkpoint read failed: " + file.string());
    if(count)
        CUDA_SAFE_CALL(cudaMemcpy(device, host.data(), expected, cudaMemcpyHostToDevice));
}

template <typename T>
void add_file(gipc::Json& files, const std::string& name, size_t count)
{
    files.push_back({{"name", name}, {"count", count}, {"element_bytes", sizeof(T)},
                     {"bytes", count * sizeof(T)}});
}

void validate_entry(const gipc::Json& manifest,
                    const std::string& name,
                    size_t count,
                    size_t element_bytes)
{
    for(const auto& entry : manifest.at("files"))
        if(entry.at("name").get<std::string>() == name)
        {
            if(entry.at("count").get<size_t>() != count
               || entry.at("element_bytes").get<size_t>() != element_bytes
               || entry.at("bytes").get<size_t>() != count * element_bytes)
                throw std::runtime_error("Checkpoint manifest mismatch: " + name);
            return;
        }
    throw std::runtime_error("Checkpoint field missing: " + name);
}
}  // namespace

int GIPC::physical_frame() const { return total_Frames; }

void GIPC::save_checkpoint(device_TetraData& tet, const std::string& directory)
{
    const fs::path target(directory);
    const fs::path temporary(directory + ".tmp");
    if(fs::exists(target) || fs::exists(temporary))
        throw std::runtime_error("Checkpoint output already exists: " + directory);
    fs::create_directories(temporary);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    auto& abd = m_abd_sim_data->device;
    gipc::Json files = gipc::Json::array();
#define SAVE_RAW(name, ptr, type, count) \
    save_device<type>(temporary / name, ptr, count); \
    add_file<type>(files, name, count)
    SAVE_RAW("fem_vertexes.bin", tet.vertexes, double3, vertexNum);
    SAVE_RAW("fem_o_vertexes.bin", tet.o_vertexes, double3, vertexNum);
    SAVE_RAW("fem_velocities.bin", tet.velocities, double3, vertexNum);
    SAVE_RAW("fem_x_tilde.bin", tet.xTilta, double3, vertexNum);
    SAVE_RAW("fem_targets.bin", tet.targetVert, double3, softNum);
    SAVE_RAW("fem_boundary.bin", tet.BoundaryType, int, vertexNum);
    SAVE_RAW("fem_apply_gravity.bin", tet.apply_gravity, int, vertexNum);
    SAVE_RAW("contact_counts.bin", _cpNum, uint32_t, 5);
    SAVE_RAW("contact_pairs.bin", _collisonPairs, int4, h_cpNum[0]);
    SAVE_RAW("contact_matrix_indices.bin", _MatIndex, int, h_cpNum[0]);
    SAVE_RAW("ground_count.bin", _gpNum, uint32_t, 1);
    SAVE_RAW("ground_pairs.bin", _environment_collisionPair, uint32_t, h_gpNum);
    SAVE_RAW("abd_q.bin", abd.body_id_to_q.data(), gipc::Vector12, abd.body_id_to_q.size());
    SAVE_RAW("abd_q_temp.bin", abd.body_id_to_q_temp.data(), gipc::Vector12, abd.body_id_to_q_temp.size());
    SAVE_RAW("abd_q_tilde.bin", abd.body_id_to_q_tilde.data(), gipc::Vector12, abd.body_id_to_q_tilde.size());
    SAVE_RAW("abd_q_prev.bin", abd.body_id_to_q_prev.data(), gipc::Vector12, abd.body_id_to_q_prev.size());
    SAVE_RAW("abd_q_v.bin", abd.body_id_to_q_v.data(), gipc::Vector12, abd.body_id_to_q_v.size());
    SAVE_RAW("abd_dq.bin", abd.body_id_to_dq.data(), gipc::Vector12, abd.body_id_to_dq.size());
#undef SAVE_RAW

    gipc::Json manifest{{"schema", "stiffgipc-step-boundary-v3"},
                        {"complete", true},
                        {"vertex_count", vertexNum},
                        {"soft_count", softNum},
                        {"abd_body_count", abd.body_id_to_q.size()},
                        {"contact_count", h_cpNum[0]},
                        {"contact_host_counts", {h_cpNum[0], h_cpNum[1], h_cpNum[2],
                                                  h_cpNum[3], h_cpNum[4]}},
                        {"ground_contact_count", h_gpNum},
                        {"physical_frame", total_Frames},
                        {"ipc_dt", IPC_dt},
                        {"kappa", Kappa},
                        {"animation_sub_rate", animation_subRate},
                        {"animation_full_rate", animation_fullRate},
                        {"files", files},
                        {"derived_state", {{"bvh", "rebuilt"},
                                           {"contact_pairs", "restored_exactly"},
                                           {"friction_set", "rebuilt_at_next_physical_step"},
                                           {"linear_system_and_graph", "existing_initialized_instance; numeric data overwritten before use"}}}};
    std::ofstream manifest_file(temporary / "manifest.json", std::ios::binary);
    manifest_file << manifest.dump(2) << '\n';
    manifest_file.close();
    if(!manifest_file)
        throw std::runtime_error("Checkpoint manifest write failed");
    fs::rename(temporary, target);
}

void GIPC::load_checkpoint(device_TetraData& tet, const std::string& directory)
{
    const fs::path source(directory);
    std::ifstream manifest_file(source / "manifest.json", std::ios::binary);
    gipc::Json manifest;
    manifest_file >> manifest;
    if(!manifest_file || manifest.value("schema", "") != "stiffgipc-step-boundary-v3"
       || !manifest.value("complete", false))
        throw std::runtime_error("Invalid or incomplete checkpoint manifest");
    auto& abd = m_abd_sim_data->device;
    if(manifest.at("vertex_count").get<size_t>() != vertexNum
       || manifest.at("soft_count").get<size_t>() != softNum
       || manifest.at("abd_body_count").get<size_t>() != abd.body_id_to_q.size()
       || manifest.at("ipc_dt").get<double>() != IPC_dt)
        throw std::runtime_error("Checkpoint is incompatible with initialized scene");

#define LOAD_RAW(name, ptr, type, count) \
    validate_entry(manifest, name, count, sizeof(type)); \
    load_device<type>(source / name, ptr, count)
    LOAD_RAW("fem_vertexes.bin", tet.vertexes, double3, vertexNum);
    LOAD_RAW("fem_o_vertexes.bin", tet.o_vertexes, double3, vertexNum);
    LOAD_RAW("fem_velocities.bin", tet.velocities, double3, vertexNum);
    LOAD_RAW("fem_x_tilde.bin", tet.xTilta, double3, vertexNum);
    LOAD_RAW("fem_targets.bin", tet.targetVert, double3, softNum);
    LOAD_RAW("fem_boundary.bin", tet.BoundaryType, int, vertexNum);
    LOAD_RAW("fem_apply_gravity.bin", tet.apply_gravity, int, vertexNum);
    // BVH construction uses shared collision scratch storage. Rebuild it after
    // geometry is present, then restore the exact frame-boundary contact order.
    buildBVH();
    const auto host_counts = manifest.at("contact_host_counts");
    if(host_counts.size() != 5)
        throw std::runtime_error("Checkpoint host contact count mismatch");
    for(size_t i = 0; i < 5; ++i)
        h_cpNum[i] = host_counts.at(i).get<uint32_t>();
    h_gpNum = manifest.at("ground_contact_count").get<uint32_t>();
    if(h_cpNum[0] > static_cast<uint32_t>(MAX_COLLITION_PAIRS_NUM))
        throw std::runtime_error("Checkpoint contact capacity mismatch");
    LOAD_RAW("contact_counts.bin", _cpNum, uint32_t, 5);
    LOAD_RAW("contact_pairs.bin", _collisonPairs, int4, h_cpNum[0]);
    LOAD_RAW("contact_matrix_indices.bin", _MatIndex, int, h_cpNum[0]);
    LOAD_RAW("ground_count.bin", _gpNum, uint32_t, 1);
    LOAD_RAW("ground_pairs.bin", _environment_collisionPair, uint32_t, h_gpNum);
    CUDA_SAFE_CALL(cudaMemcpy(&h_gpNum, _gpNum, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    LOAD_RAW("abd_q.bin", abd.body_id_to_q.data(), gipc::Vector12, abd.body_id_to_q.size());
    LOAD_RAW("abd_q_temp.bin", abd.body_id_to_q_temp.data(), gipc::Vector12, abd.body_id_to_q_temp.size());
    LOAD_RAW("abd_q_tilde.bin", abd.body_id_to_q_tilde.data(), gipc::Vector12, abd.body_id_to_q_tilde.size());
    LOAD_RAW("abd_q_prev.bin", abd.body_id_to_q_prev.data(), gipc::Vector12, abd.body_id_to_q_prev.size());
    LOAD_RAW("abd_q_v.bin", abd.body_id_to_q_v.data(), gipc::Vector12, abd.body_id_to_q_v.size());
    LOAD_RAW("abd_dq.bin", abd.body_id_to_dq.data(), gipc::Vector12, abd.body_id_to_dq.size());
#undef LOAD_RAW

    total_Frames = manifest.at("physical_frame").get<int>();
    Kappa = manifest.at("kappa").get<double>();
    animation_subRate = manifest.at("animation_sub_rate").get<double>();
    animation_fullRate = manifest.at("animation_full_rate").get<double>();
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
}
