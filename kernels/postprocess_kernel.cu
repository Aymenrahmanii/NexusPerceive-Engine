#include "nexus/core/cuda_postprocess.cu.h"
#include <cuda_runtime.h>
#include <algorithm>

namespace nexus {
namespace cuda {

__device__ inline float calculate_iou(const BoundingBox& a, const BoundingBox& b) {
    float inter_x1 = fmaxf(a.x1, b.x1);
    float inter_y1 = fmaxf(a.y1, b.y1);
    float inter_x2 = fminf(a.x2, b.x2);
    float inter_y2 = fminf(a.y2, b.y2);

    float inter_w = fmaxf(0.0f, inter_x2 - inter_x1);
    float inter_h = fmaxf(0.0f, inter_y2 - inter_y1);
    float inter_area = inter_w * inter_h;

    float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    float union_area = area_a + area_b - inter_area;

    return (union_area > 0.0f) ? (inter_area / union_area) : 0.0f;
}

__global__ void decode_and_threshold_kernel(
    const float* __restrict__ d_raw,
    int num_anchors,
    int num_classes,
    float score_threshold,
    BoundingBox* __restrict__ d_candidates,
    int* __restrict__ d_candidate_count,
    int max_candidates
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_anchors) return;

    // Anchor format: [cx, cy, w, h, class0_score, class1_score, ...] or [x1, y1, x2, y2, score, class_id]
    int stride = 4 + num_classes;
    const float* anchor_ptr = d_raw + idx * stride;

    float cx = anchor_ptr[0];
    float cy = anchor_ptr[1];
    float w  = anchor_ptr[2];
    float h  = anchor_ptr[3];

    float max_score = 0.0f;
    int max_class = -1;

    for (int c = 0; c < num_classes; ++c) {
        float score = anchor_ptr[4 + c];
        if (score > max_score) {
            max_score = score;
            max_class = c;
        }
    }

    if (max_score >= score_threshold) {
        int pos = atomicAdd(d_candidate_count, 1);
        if (pos < max_candidates) {
            d_candidates[pos].x1 = cx - w * 0.5f;
            d_candidates[pos].y1 = cy - h * 0.5f;
            d_candidates[pos].x2 = cx + w * 0.5f;
            d_candidates[pos].y2 = cy + h * 0.5f;
            d_candidates[pos].confidence = max_score;
            d_candidates[pos].class_id = max_class;
        }
    }
}

__global__ void nms_gpu_suppression_kernel(
    const BoundingBox* __restrict__ d_candidates,
    int candidate_count,
    float nms_threshold,
    bool* __restrict__ d_suppressed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= candidate_count) return;

    BoundingBox curr_box = d_candidates[idx];

    for (int j = 0; j < idx; ++j) {
        if (d_suppressed[j]) continue;
        BoundingBox prev_box = d_candidates[j];
        if (curr_box.class_id == prev_box.class_id) {
            float iou = calculate_iou(curr_box, prev_box);
            if (iou >= nms_threshold) {
                d_suppressed[idx] = true;
                break;
            }
        }
    }
}

__global__ void collect_nms_results_kernel(
    const BoundingBox* __restrict__ d_candidates,
    const bool* __restrict__ d_suppressed,
    int candidate_count,
    int max_detections,
    BoundingBox* __restrict__ d_out_boxes,
    int* __restrict__ d_out_count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= candidate_count) return;

    if (!d_suppressed[idx]) {
        int pos = atomicAdd(d_out_count, 1);
        if (pos < max_detections) {
            d_out_boxes[pos] = d_candidates[idx];
        }
    }
}

void launchPostprocessNMSKernel(
    const float* d_raw_boxes_scores,
    int num_anchors,
    int num_classes,
    float score_threshold,
    float nms_threshold,
    int max_detections,
    BoundingBox* d_out_boxes,
    int* d_out_count,
    cudaStream_t stream
) {
    // Scratch memory allocation via stack/static CUDA buffers or stream allocations
    const int max_candidates = 2048;
    BoundingBox* d_candidates = nullptr;
    int* d_candidate_count = nullptr;
    bool* d_suppressed = nullptr;

    cudaMallocAsync(&d_candidates, sizeof(BoundingBox) * max_candidates, stream);
    cudaMallocAsync(&d_candidate_count, sizeof(int), stream);
    cudaMallocAsync(&d_suppressed, sizeof(bool) * max_candidates, stream);

    cudaMemsetAsync(d_candidate_count, 0, sizeof(int), stream);
    cudaMemsetAsync(d_suppressed, 0, sizeof(bool) * max_candidates, stream);
    cudaMemsetAsync(d_out_count, 0, sizeof(int), stream);

    int block_size = 256;
    int grid_size = (num_anchors + block_size - 1) / block_size;

    decode_and_threshold_kernel<<<grid_size, block_size, 0, stream>>>(
        d_raw_boxes_scores,
        num_anchors,
        num_classes,
        score_threshold,
        d_candidates,
        d_candidate_count,
        max_candidates
    );

    int candidate_grid = (max_candidates + block_size - 1) / block_size;
    nms_gpu_suppression_kernel<<<candidate_grid, block_size, 0, stream>>>(
        d_candidates,
        max_candidates,
        nms_threshold,
        d_suppressed
    );

    collect_nms_results_kernel<<<candidate_grid, block_size, 0, stream>>>(
        d_candidates,
        d_suppressed,
        max_candidates,
        max_detections,
        d_out_boxes,
        d_out_count
    );

    cudaFreeAsync(d_candidates, stream);
    cudaFreeAsync(d_candidate_count, stream);
    cudaFreeAsync(d_suppressed, stream);
}

} // namespace cuda
} // namespace nexus
