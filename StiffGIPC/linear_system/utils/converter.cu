#include <linear_system/utils/converter.h>
#include <cuda_tools/cuda_all.h>
#include <gipc/utils/timer.h>
#include <gipc/utils/benchmark.h>
#include <gipc/utils/parallel_algorithm/fast_segmental_reduce.h>

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

__global__ void compute_hash_and_index_kernel(int          length,
                                              int*         row_indices,
                                              int*         col_indices,
                                              uint64_t*    ij_hash_input,
                                              uint32_t*    index_input)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= length)
        return;
    ij_hash_input[i] =
        (uint64_t{row_indices[i]} << 32) + uint64_t{col_indices[i]};
    index_input[i]   = i;
}

__global__ void set_dst_val_kernel(int                   length,
                                   uint32_t*             sort_index,
                                   Eigen::Matrix3d*      src_blocks,
                                   Eigen::Matrix3d*      dst_val)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= length)
        return;
    dst_val[i] = src_blocks[sort_index[i]];
}

__global__ void set_row_col_from_unique_key_kernel(int          unique_count,
                                                   int*         row_indices,
                                                   int*         col_indices,
                                                   uint64_t*    unique_key)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= unique_count)
        return;
    row_indices[i] = unique_key[i] >> 32;
    col_indices[i] = unique_key[i] & 0xffffffff;
}

__global__ void compute_sorted_partition_kernel(int             length,
                                                uint32_t*       sorted_partition_input,
                                                uint64_t*       ij_hash)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= length)
        return;
    sorted_partition_input[i] = ij_hash[i] != ij_hash[i + 1] ? 1 : 0;
}

__global__ void set_row_col_from_partition_kernel(int             length,
                                                  int*            row_indices,
                                                  int*            col_indices,
                                                  uint64_t*       ij_hash,
                                                  uint32_t*       sorted_partition_output)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= length)
        return;
    int index = sorted_partition_output[i];
    if(i == 0)
    {
        auto key           = ij_hash[i];
        row_indices[index] = key >> 32;
        col_indices[index] = key & 0xffffffff;
    }
    else
    {
        if(index != sorted_partition_output[i - 1])
        {
            auto key           = ij_hash[i];
            row_indices[index] = key >> 32;
            col_indices[index] = key & 0xffffffff;
        }
    }
}

__global__ void setup_ge2sym_kernel(int             unique_count,
                                    int*            row_indices,
                                    int*            col_indices,
                                    uint64_t*       ij_hash,
                                    Eigen::Matrix3d* blocks,
                                    Eigen::Matrix3d* block_temp,
                                    uint32_t*       counts)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= unique_count)
        return;
    counts[i]    = row_indices[i] <= col_indices[i] ? 1 : 0;
    ij_hash[i]   =
        (uint64_t{row_indices[i]} << 32) + uint64_t{col_indices[i]};
    block_temp[i] = blocks[i];
}

__global__ void finalize_ge2sym_kernel(int             unique_count,
                                       Eigen::Matrix3d* dst_blocks,
                                       Eigen::Matrix3d* block_temp,
                                       uint64_t*       ij_hash,
                                       int*            row_indices,
                                       int*            col_indices,
                                       uint32_t*       counts,
                                       uint32_t*       offsets,
                                       int*            total_count,
                                       int             number)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= unique_count)
        return;
    auto count  = counts[i];
    auto offset = offsets[i];

    if(count != 0)
    {
        dst_blocks[offset]  = block_temp[i];
        auto ij             = ij_hash[i];
        row_indices[offset] = ij >> 32;
        col_indices[offset] = ij & 0xffffffff;
    }

    if(i == number - 1)
    {
        *total_count = offsets[i] + counts[i];
    }
}
}  // namespace

namespace gipc
{

constexpr bool UseRadixSort   = true;
constexpr bool UseReduceByKey = false;

void Converter::convert(GIPCTripletMatrix& global_triplets,
                        const int&                          start,
                        const int&                          length,
                        const int&                          out_start_id)
{
    gipc::Timer timer("convert3x3");
    if(length < 1)
        return;
    if(benchmark::options.profile_stages)
    {
        ++benchmark::frame.convert_calls;
        benchmark::frame.profiled_convert_input_triplets += length;
    }
    _radix_sort_indices_and_blocks(global_triplets, start, length, out_start_id);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());


    //_make_unique_indices(global_triplets, start, length, out_start_id);

    //CUDA_SAFE_CALL(cudaDeviceSynchronize());


    _make_unique_block_warp_reduction(global_triplets, start, length, out_start_id);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
}



void Converter::_radix_sort_indices_and_blocks(GIPCTripletMatrix& global_triplets,
                                               const int& start,
                                               const int& length,
                                               const int& out_start_id)
{
    using namespace cudatool;
    benchmark::Clock::time_point profile_start{};

    auto src_row_indices = global_triplets.block_row_indices(start);
    auto src_col_indices = global_triplets.block_col_indices(start);
    auto src_blocks      = global_triplets.block_values(start);
    auto index_input   = global_triplets.block_index();
    auto ij_hash_input = global_triplets.block_hash_value();

    profile_begin(profile_start);
    LaunchCudaKernal_default(length,
                             256,
                             0,
                             compute_hash_and_index_kernel,
                             length,
                             src_row_indices,
                             src_col_indices,
                             ij_hash_input,
                             index_input);
    profile_end(profile_start, benchmark::frame.convert_hash_index_ms);

    profile_begin(profile_start);
    DeviceRadixSort().SortPairs(ij_hash_input,
                                global_triplets.block_sort_hash_value(),
                                index_input,
                                global_triplets.block_sort_index(),
                                length);
    profile_end(profile_start, benchmark::frame.convert_radix_sort_ms);

    if(!benchmark::options.converter_indexed_reduce)
    {
        auto dst_val = global_triplets.block_values() + out_start_id;
        profile_begin(profile_start);
        LaunchCudaKernal_default(length,
                                 256,
                                 0,
                                 set_dst_val_kernel,
                                 length,
                                 global_triplets.block_sort_index(),
                                 src_blocks,
                                 dst_val);
        profile_end(profile_start, benchmark::frame.convert_gather_ms);
    }
}


void Converter::_make_unique_indices(GIPCTripletMatrix& global_triplets,
                                     const int&         start,
                                     const int&         length,
                                     const int&         out_start_id)
{
    auto row_indices = global_triplets.block_row_indices(start);
    auto col_indices = global_triplets.block_col_indices(start);

    auto unique_key = global_triplets.block_hash_value();
    auto sort_key   = global_triplets.block_sort_hash_value();

    cudatool::DeviceRunLengthEncode().Encode(sort_key,
                                         unique_key,
                                         global_triplets.block_temp_buffer(),
                                         global_triplets.d_unique_key_number,
                                         length);

    CUDA_SAFE_CALL(cudaMemcpy(&(global_triplets.h_unique_key_number),
                              global_triplets.d_unique_key_number,
                              sizeof(int),
                              cudaMemcpyDeviceToHost));

    LaunchCudaKernal_default(global_triplets.h_unique_key_number,
                             256,
                             0,
                             set_row_col_from_unique_key_kernel,
                             global_triplets.h_unique_key_number,
                             row_indices,
                             col_indices,
                             unique_key);
}





void Converter::_make_unique_block_warp_reduction(GIPCTripletMatrix& global_triplets,
                                                  const int& start, const int& length, const int& out_start_id)
{
    using namespace cudatool;
    benchmark::Clock::time_point profile_start{};

    auto sorted_partition_input = global_triplets.block_temp_buffer();
    profile_begin(profile_start);
    LaunchCudaKernal_default(length - 1,
                             256,
                             0,
                             compute_sorted_partition_kernel,
                             length - 1,
                             sorted_partition_input,
                             global_triplets.block_sort_hash_value());
    profile_end(profile_start, benchmark::frame.convert_partition_mark_ms);
    auto sorted_partition_output = global_triplets.block_index();
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    // scatter
    profile_begin(profile_start);
    DeviceScan().ExclusiveSum(sorted_partition_input, sorted_partition_output, length);
    profile_end(profile_start, benchmark::frame.convert_partition_scan_ms);

    auto row_indices = global_triplets.block_row_indices(start);
    auto col_indices = global_triplets.block_col_indices(start);


    profile_begin(profile_start);
    LaunchCudaKernal_default(length,
                             256,
                             0,
                             set_row_col_from_partition_kernel,
                             length,
                             row_indices,
                             col_indices,
                             global_triplets.block_sort_hash_value(),
                             sorted_partition_output);
    profile_end(profile_start, benchmark::frame.convert_row_col_ms);


    profile_begin(profile_start);
    if(benchmark::options.profile_stages)
    {
        ++benchmark::frame.profiled_d2h_calls;
        benchmark::frame.profiled_d2h_bytes += sizeof(int);
    }
    CUDA_SAFE_CALL(cudaMemcpy(&(global_triplets.h_unique_key_number),
                              sorted_partition_output + length - 1,
                              sizeof(int),
                              cudaMemcpyDeviceToHost));
    global_triplets.h_unique_key_number += 1;
    profile_end(profile_start, benchmark::frame.convert_host_read_ms);
    if(benchmark::options.profile_stages)
        benchmark::frame.profiled_unique_blocks += global_triplets.h_unique_key_number;

    auto reduced_blocks = benchmark::options.converter_indexed_reduce
                              ? global_triplets.block_values(out_start_id)
                              : global_triplets.block_values(start);
    profile_begin(profile_start);
    CUDA_SAFE_CALL(cudaMemset(reduced_blocks,
                              0,
                              global_triplets.h_unique_key_number * sizeof(Eigen::Matrix3d)));
    profile_end(profile_start, benchmark::frame.convert_block_zero_ms);

    profile_begin(profile_start);
    if(benchmark::options.converter_indexed_reduce)
    {
        cudatool::parallel::FastSegmentalReduce<>::reduce_indexed(
            length,
            sorted_partition_output,
            global_triplets.block_sort_index(),
            global_triplets.block_values(start),
            reduced_blocks);
    }
    else
    {
        cudatool::parallel::FastSegmentalReduce<>::reduce(
            length,
            sorted_partition_output,
            global_triplets.block_values(out_start_id),
            reduced_blocks);
    }
    profile_end(profile_start, benchmark::frame.convert_reduce_ms);

    if(benchmark::options.converter_indexed_reduce)
    {
        profile_begin(profile_start);
        CUDA_SAFE_CALL(cudaMemcpy(global_triplets.block_values(start),
                                  reduced_blocks,
                                  global_triplets.h_unique_key_number
                                      * sizeof(Eigen::Matrix3d),
                                  cudaMemcpyDeviceToDevice));
        profile_end(profile_start, benchmark::frame.convert_output_copy_ms);
    }
}

void Converter::ge2sym(GIPCTripletMatrix& global_triplets)
{
    using namespace cudatool;

    auto counts  = global_triplets.block_index();
    auto offsets = global_triplets.block_sort_index();
    auto block_temp = global_triplets.block_values(global_triplets.h_unique_key_number);
    auto blocks      = global_triplets.block_values();
    auto ij_hash     = global_triplets.block_hash_value();
    auto row_indices = global_triplets.block_row_indices();
    auto col_indices = global_triplets.block_col_indices();

    LaunchCudaKernal_default(global_triplets.h_unique_key_number,
                             256,
                             0,
                             setup_ge2sym_kernel,
                             global_triplets.h_unique_key_number,
                             row_indices,
                             col_indices,
                             ij_hash,
                             blocks,
                             block_temp,
                             counts);

    // exclusive sum
    DeviceScan().ExclusiveSum(counts, offsets, global_triplets.h_unique_key_number);

    // set the values
    auto dst_blocks = global_triplets.block_values();

    LaunchCudaKernal_default(global_triplets.h_unique_key_number,
                             256,
                             0,
                             finalize_ge2sym_kernel,
                             global_triplets.h_unique_key_number,
                             dst_blocks,
                             block_temp,
                             ij_hash,
                             row_indices,
                             col_indices,
                             counts,
                             offsets,
                             global_triplets.d_unique_key_number,
                             global_triplets.h_unique_key_number);


    CUDA_SAFE_CALL(cudaMemcpy(&(global_triplets.h_unique_key_number),
                              global_triplets.d_unique_key_number,
                              sizeof(int),
                              cudaMemcpyDeviceToHost));
}

}  // namespace gipc
