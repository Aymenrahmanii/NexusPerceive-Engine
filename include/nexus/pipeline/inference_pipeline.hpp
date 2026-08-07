#ifndef NEXUS_PIPELINE_INFERENCE_PIPELINE_HPP
#define NEXUS_PIPELINE_INFERENCE_PIPELINE_HPP

#include <vector>
#include <memory>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include "nexus/common/types.hpp"
#include "nexus/common/memory_pool.hpp"
#include "nexus/core/trt_engine.hpp"

namespace nexus {

struct StreamSlot {
    cudaStream_t stream;
    void* d_src_bgr = nullptr;
    float* d_preproc_nchw = nullptr;
    float* d_raw_output = nullptr;
    BoundingBox* d_filtered_boxes = nullptr;
    int* d_box_count = nullptr;
    BoundingBox* h_pinned_boxes = nullptr;
    int* h_pinned_count = nullptr;
};

class InferencePipeline {
public:
    explicit InferencePipeline(const EngineConfig& config);
    ~InferencePipeline();

    bool initialize();
    DetectionBatch processFrame(const cv::Mat& frame, int64_t frame_id, uint64_t timestamp_us);

private:
    EngineConfig config_;
    std::shared_ptr<TensorRTEngine> engine_;
    std::vector<StreamSlot> stream_slots_;
    size_t current_stream_idx_ = 0;
};

} // namespace nexus

#endif // NEXUS_PIPELINE_INFERENCE_PIPELINE_HPP
