#ifndef NEXUS_CORE_TRT_ENGINE_HPP
#define NEXUS_CORE_TRT_ENGINE_HPP

#include <memory>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "nexus/common/types.hpp"

// Forward declaration of TensorRT interfaces
namespace nvinfer1 {
class IRuntime;
class ICudaEngine;
class IExecutionContext;
}

namespace nexus {

class TensorRTEngine {
public:
    explicit TensorRTEngine(const EngineConfig& config);
    ~TensorRTEngine();

    bool buildOrLoadEngine(const std::string& model_path);
    bool enqueueV3(void** bindings, cudaStream_t stream);

    size_t getInputSize() const { return input_size_; }
    size_t getOutputSize() const { return output_size_; }
    const EngineConfig& getConfig() const { return config_; }

private:
    EngineConfig config_;
    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;

    size_t input_size_ = 0;
    size_t output_size_ = 0;
    int input_index_ = -1;
    int output_index_ = -1;
};

} // namespace nexus

#endif // NEXUS_CORE_TRT_ENGINE_HPP
