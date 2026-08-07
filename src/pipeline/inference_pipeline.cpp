#include "nexus/pipeline/inference_pipeline.hpp"
#include "nexus/common/logger.hpp"
#include "nexus/core/cuda_preprocess.cu.h"
#include "nexus/core/cuda_postprocess.cu.h"
#include <chrono>

namespace nexus {

InferencePipeline::InferencePipeline(const EngineConfig& config)
    : config_(config) {
    engine_ = std::make_shared<TensorRTEngine>(config_);
}

InferencePipeline::~InferencePipeline() {
    for (auto& slot : stream_slots_) {
        if (slot.stream) cudaStreamDestroy(slot.stream);
        if (slot.d_src_bgr) cudaFree(slot.d_src_bgr);
        if (slot.d_preproc_nchw) cudaFree(slot.d_preproc_nchw);
        if (slot.d_raw_output) cudaFree(slot.d_raw_output);
        if (slot.d_filtered_boxes) cudaFree(slot.d_filtered_boxes);
        if (slot.d_box_count) cudaFree(slot.d_box_count);
        if (slot.h_pinned_boxes) cudaFreeHost(slot.h_pinned_boxes);
        if (slot.h_pinned_count) cudaFreeHost(slot.h_pinned_count);
    }
}

bool InferencePipeline::initialize() {
    NEXUS_LOG_INFO("Initializing NexusPerceive Inference Pipeline...");
    
    if (!engine_->buildOrLoadEngine(config_.model_path)) {
        NEXUS_LOG_ERROR("Failed to load TensorRT Engine.");
        return false;
    }

    size_t num_streams = config_.num_streams;
    stream_slots_.resize(num_streams);

    size_t raw_bgr_size = 1920 * 1080 * 3;
    size_t preproc_size = config_.input_width * config_.input_height * config_.input_channels * sizeof(float);
    size_t raw_output_size = 8400 * 84 * sizeof(float);
    size_t filtered_boxes_size = config_.max_detections * sizeof(BoundingBox);

    for (size_t i = 0; i < num_streams; ++i) {
        cudaStreamCreateWithFlags(&stream_slots_[i].stream, cudaStreamNonBlocking);

        cudaMalloc(&stream_slots_[i].d_src_bgr, raw_bgr_size);
        cudaMalloc(&stream_slots_[i].d_preproc_nchw, preproc_size);
        cudaMalloc(&stream_slots_[i].d_raw_output, raw_output_size);
        cudaMalloc(&stream_slots_[i].d_filtered_boxes, filtered_boxes_size);
        cudaMalloc(&stream_slots_[i].d_box_count, sizeof(int));

        cudaHostAlloc(&stream_slots_[i].h_pinned_boxes, filtered_boxes_size, cudaHostAllocMapped);
        cudaHostAlloc(&stream_slots_[i].h_pinned_count, sizeof(int), cudaHostAllocMapped);
    }

    NEXUS_LOG_INFO("Triple-buffered CUDA Stream Ring Buffer allocated successfully.");
    return true;
}

DetectionBatch InferencePipeline::processFrame(const cv::Mat& frame, int64_t frame_id, uint64_t timestamp_us) {
    auto start_time = std::chrono::high_resolution_clock::now();

    StreamSlot& slot = stream_slots_[current_stream_idx_];
    current_stream_idx_ = (current_stream_idx_ + 1) % stream_slots_.size();

    // Calculate affine parameters
    PreprocessParams params;
    params.src_width = frame.cols;
    params.src_height = frame.rows;
    params.dst_width = config_.input_width;
    params.dst_height = config_.input_height;

    float scale_w = static_cast<float>(params.dst_width) / params.src_width;
    float scale_h = static_cast<float>(params.dst_height) / params.src_height;
    params.scale = std::min(scale_w, scale_h);

    params.pad_w = static_cast<int>((params.dst_width - params.src_width * params.scale) * 0.5f);
    params.pad_h = static_cast<int>((params.dst_height - params.src_height * params.scale) * 0.5f);

    params.mean[0] = 0.0f; params.mean[1] = 0.0f; params.mean[2] = 0.0f;
    params.std[0]  = 1.0f; params.std[1]  = 1.0f; params.std[2]  = 1.0f;

    size_t frame_bytes = frame.step * frame.rows;

    // 1. Async H2D Transfer of raw BGR image over PCIe Gen4
    cudaMemcpyAsync(slot.d_src_bgr, frame.data, frame_bytes, cudaMemcpyHostToDevice, slot.stream);

    // 2. Fused CUDA Pre-processing Kernel
    cuda::launchPreprocessKernel(
        static_cast<const uint8_t*>(slot.d_src_bgr),
        params.src_width,
        params.src_height,
        static_cast<int>(frame.step),
        slot.d_preproc_nchw,
        params.dst_width,
        params.dst_height,
        params,
        slot.stream
    );

    // 3. TensorRT Execution via enqueueV3
    void* bindings[2] = { slot.d_preproc_nchw, slot.d_raw_output };
    engine_->enqueueV3(bindings, slot.stream);

    // 4. GPU Fused NMS Kernel
    cuda::launchPostprocessNMSKernel(
        slot.d_raw_output,
        8400,
        80,
        config_.score_threshold,
        config_.nms_threshold,
        config_.max_detections,
        slot.d_filtered_boxes,
        slot.d_box_count,
        slot.stream
    );

    // 5. Async D2H Transfer of Filtered Bounding Boxes (~2.4 KB)
    cudaMemcpyAsync(slot.h_pinned_count, slot.d_box_count, sizeof(int), cudaMemcpyDeviceToHost, slot.stream);
    cudaMemcpyAsync(slot.h_pinned_boxes, slot.d_filtered_boxes, config_.max_detections * sizeof(BoundingBox), cudaMemcpyDeviceToHost, slot.stream);

    // Synchronize current stream slot
    cudaStreamSynchronize(slot.stream);

    auto end_time = std::chrono::high_resolution_clock::now();
    float latency_ms = std::chrono::duration<float, std::milli>(end_time - start_time).count();

    DetectionBatch batch;
    batch.frame_id = frame_id;
    batch.timestamp_us = timestamp_us;
    batch.end_to_end_latency_ms = latency_ms;

    int num_found = std::min(*slot.h_pinned_count, config_.max_detections);
    batch.boxes.assign(slot.h_pinned_boxes, slot.h_pinned_boxes + std::max(0, num_found));

    return batch;
}

} // namespace nexus
