#ifndef NEXUS_CORE_CUDA_PREPROCESS_CU_H
#define NEXUS_CORE_CUDA_PREPROCESS_CU_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "nexus/common/types.hpp"

namespace nexus {
namespace cuda {

void launchPreprocessKernelFP32(
    const uint8_t* d_src_bgr,
    int src_width,
    int src_height,
    int src_pitch,
    float* d_dst_nchw,
    int dst_width,
    int dst_height,
    const PreprocessParams& params,
    cudaStream_t stream
);

void launchPreprocessKernelFP16(
    const uint8_t* d_src_bgr,
    int src_width,
    int src_height,
    int src_pitch,
    __half* d_dst_nchw_fp16,
    int dst_width,
    int dst_height,
    const PreprocessParams& params,
    cudaStream_t stream
);

} // namespace cuda
} // namespace nexus

#endif // NEXUS_CORE_CUDA_PREPROCESS_CU_H
