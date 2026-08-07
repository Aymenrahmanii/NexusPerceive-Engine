#ifndef NEXUS_CORE_CUDA_POSTPROCESS_CU_H
#define NEXUS_CORE_CUDA_POSTPROCESS_CU_H

#include <cuda_runtime.h>
#include "nexus/common/types.hpp"

namespace nexus {
namespace cuda {

void launchPostprocessNMSKernel(
    const float* d_raw_boxes_scores, // [num_anchors, 4 + num_classes] or [num_anchors, 6]
    int num_anchors,
    int num_classes,
    float score_threshold,
    float nms_threshold,
    int max_detections,
    BoundingBox* d_out_boxes,
    int* d_out_count,
    cudaStream_t stream
);

} // namespace cuda
} // namespace nexus

#endif // NEXUS_CORE_CUDA_POSTPROCESS_CU_H
