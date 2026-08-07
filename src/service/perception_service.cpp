#include "nexus/service/perception_service.hpp"
#include "nexus/common/logger.hpp"
#include <opencv2/opencv.hpp>

namespace nexus {

PerceptionServiceImpl::PerceptionServiceImpl(const EngineConfig& config)
    : config_(config) {
    pipeline_ = std::make_shared<InferencePipeline>(config_);
}

PerceptionServiceImpl::~PerceptionServiceImpl() {
    stopServer();
}

void PerceptionServiceImpl::startServer(const std::string& server_address) {
    if (!pipeline_->initialize()) {
        NEXUS_LOG_ERROR("Failed to initialize pipeline for perception service.");
        return;
    }
    is_running_ = true;
    NEXUS_LOG_INFO("gRPC PerceptionEngine microservice listening on " + server_address);
}

DetectionBatch PerceptionServiceImpl::processFrameBuffer(
    const uint8_t* image_data,
    int width,
    int height,
    int channels,
    int64_t frame_id,
    uint64_t timestamp_us
) {
    cv::Mat frame;
    if (channels == 3) {
        frame = cv::Mat(height, width, CV_8UC3, const_cast<uint8_t*>(image_data));
    } else {
        frame = cv::Mat(height, width, CV_8UC1, const_cast<uint8_t*>(image_data));
        cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);
    }
    return pipeline_->processFrame(frame, frame_id, timestamp_us);
}

void PerceptionServiceImpl::stopServer() {
    if (is_running_) {
        is_running_ = false;
        NEXUS_LOG_INFO("Stopped gRPC PerceptionEngine microservice.");
    }
}

} // namespace nexus
