#ifndef NEXUS_SERVICE_PERCEPTION_SERVICE_HPP
#define NEXUS_SERVICE_PERCEPTION_SERVICE_HPP

#include <memory>
#include <string>
#include "nexus/pipeline/inference_pipeline.hpp"

namespace nexus {

class PerceptionServiceImpl {
public:
    explicit PerceptionServiceImpl(const EngineConfig& config);
    ~PerceptionServiceImpl();

    void startServer(const std::string& server_address = "0.0.0.0:50051");
    void stopServer();
    bool isRunning() const { return is_running_; }

    DetectionBatch processFrameBuffer(const uint8_t* image_data, int width, int height, int channels, int64_t frame_id, uint64_t timestamp_us);

private:
    EngineConfig config_;
    std::shared_ptr<InferencePipeline> pipeline_;
    bool is_running_ = false;
};

} // namespace nexus

#endif // NEXUS_SERVICE_PERCEPTION_SERVICE_HPP
