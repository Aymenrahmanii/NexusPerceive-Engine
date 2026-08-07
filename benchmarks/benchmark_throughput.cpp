#include "nexus/common/logger.hpp"
#include "nexus/pipeline/inference_pipeline.hpp"
#include <iostream>
#include <chrono>
#include <vector>

int main() {
    nexus::EngineConfig config;
    nexus::InferencePipeline pipeline(config);

    if (!pipeline.initialize()) {
        std::cerr << "Pipeline initialization failed." << std::endl;
        return 1;
    }

    cv::Mat frame(1080, 1920, CV_8UC3, cv::Scalar(114, 114, 114));

    // Warmup
    for (int i = 0; i < 50; ++i) {
        pipeline.processFrame(frame, i, i * 1000);
    }

    const int iterations = 500;
    std::vector<float> latencies;
    latencies.reserve(iterations);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        auto batch = pipeline.processFrame(frame, i, i * 1000);
        latencies.push_back(batch.end_to_end_latency_ms);
    }
    auto end = std::chrono::high_resolution_clock::now();

    double total_sec = std::chrono::duration<double>(end - start).count();
    double fps = iterations / total_sec;

    std::cout << "Throughput Benchmark Results:\n";
    std::cout << "  Iterations : " << iterations << "\n";
    std::cout << "  Total Time : " << total_sec << " s\n";
    std::cout << "  FPS        : " << fps << " FPS\n";

    return 0;
}
