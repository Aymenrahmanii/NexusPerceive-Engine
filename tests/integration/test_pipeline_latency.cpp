#include <gtest/gtest.h>
#include "nexus/pipeline/inference_pipeline.hpp"
#include <opencv2/opencv.hpp>

TEST(PipelineIntegrationTest, EndToEndLatencyCheck) {
    nexus::EngineConfig config;
    config.input_width = 640;
    config.input_height = 640;

    nexus::InferencePipeline pipeline(config);
    ASSERT_TRUE(pipeline.initialize());

    cv::Mat frame(1080, 1920, CV_8UC3, cv::Scalar(128, 128, 128));

    // Process warmup frame
    auto batch = pipeline.processFrame(frame, 1, 1000);
    EXPECT_GE(batch.end_to_end_latency_ms, 0.0f);
}
