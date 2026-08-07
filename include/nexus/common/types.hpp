#ifndef NEXUS_COMMON_TYPES_HPP
#define NEXUS_COMMON_TYPES_HPP

#include <cstdint>
#include <vector>
#include <string>

namespace nexus {

struct BoundingBox {
    float x1;
    float y1;
    float x2;
    float y2;
    float confidence;
    int32_t class_id;
};

struct DetectionBatch {
    int64_t frame_id;
    uint64_t timestamp_us;
    float end_to_end_latency_ms;
    std::vector<BoundingBox> boxes;
};

struct PreprocessParams {
    int src_width;
    int src_height;
    int dst_width;
    int dst_height;
    float scale;
    int pad_w;
    int pad_h;
    float mean[3];
    float std[3];
};

struct EngineConfig {
    std::string model_path;
    int input_width = 640;
    int input_height = 640;
    int input_channels = 3;
    int max_batch_size = 1;
    int max_detections = 100;
    float score_threshold = 0.25f;
    float nms_threshold = 0.45f;
    bool use_fp16 = true;
    bool use_int8 = false;
    int num_streams = 3;
};

} // namespace nexus

#endif // NEXUS_COMMON_TYPES_HPP
