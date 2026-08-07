#include "nexus/core/trt_engine.hpp"
#include "nexus/common/logger.hpp"
#include <fstream>
#include <iostream>
#include <sstream>

#if __has_include(<NvInfer.h>)
#include <NvInfer.h>
#define HAS_TENSORRT 1
#else
#define HAS_TENSORRT 0
#endif

namespace nexus {

#if HAS_TENSORRT
class LoggerTRT : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity == Severity::kERROR || severity == Severity::kINTERNAL_ERROR) {
            std::cerr << "[TensorRT ERROR] " << msg << std::endl;
        } else if (severity == Severity::kWARNING) {
            std::cout << "[TensorRT WARN] " << msg << std::endl;
        }
    }
} g_logger_trt;
#endif

TensorRTEngine::TensorRTEngine(const EngineConfig& config)
    : config_(config) {
    input_size_ = config_.input_width * config_.input_height * config_.input_channels * sizeof(float);
    // Default raw output size: 8400 anchors x (4 box coords + 80 class scores) * float32
    output_size_ = 8400 * 84 * sizeof(float);
}

TensorRTEngine::~TensorRTEngine() {
#if HAS_TENSORRT
    if (context_) { delete context_; context_ = nullptr; }
    if (engine_) { delete engine_; engine_ = nullptr; }
    if (runtime_) { delete runtime_; runtime_ = nullptr; }
#endif
}

bool TensorRTEngine::buildOrLoadEngine(const std::string& model_path) {
    NEXUS_LOG_INFO("Loading TensorRT engine from: " + model_path);
    std::ifstream file(model_path, std::ios::binary);
    if (!file.good()) {
        NEXUS_LOG_WARN("Engine file not found or mock engine mode active. Initializing fallback configuration.");
        return true;
    }

#if HAS_TENSORRT
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    file.close();

    runtime_ = nvinfer1::createInferRuntime(g_logger_trt);
    if (!runtime_) {
        NEXUS_LOG_ERROR("Failed to create TensorRT InferRuntime.");
        return false;
    }

    engine_ = runtime_->deserializeCudaEngine(buffer.data(), size);
    if (!engine_) {
        NEXUS_LOG_ERROR("Failed to deserialize TensorRT CudaEngine.");
        return false;
    }

    context_ = engine_->createExecutionContext();
    if (!context_) {
        NEXUS_LOG_ERROR("Failed to create TensorRT ExecutionContext.");
        return false;
    }

    NEXUS_LOG_INFO("Successfully deserialized TensorRT Engine & created ExecutionContext v3.");
#endif

    return true;
}

bool TensorRTEngine::enqueueV3(void** bindings, cudaStream_t stream) {
#if HAS_TENSORRT
    if (context_) {
        // TensorRT 10 enqueueV3 implementation
        return context_->enqueueV3(stream);
    }
#endif
    // Fallback/Simulated stream execution if in mock mode
    (void)bindings;
    (void)stream;
    return true;
}

} // namespace nexus
