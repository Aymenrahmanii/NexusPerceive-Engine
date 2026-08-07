#include <gtest/gtest.h>
#include "nexus/core/trt_engine.hpp"

TEST(TRTEngineTest, EngineInitialization) {
    nexus::EngineConfig config;
    config.input_width = 640;
    config.input_height = 640;
    config.input_channels = 3;

    nexus::TensorRTEngine engine(config);
    EXPECT_EQ(engine.getInputSize(), 640 * 640 * 3 * sizeof(float));

    bool success = engine.buildOrLoadEngine("non_existent_model.engine");
    EXPECT_TRUE(success); // Engine handles mock loading gracefully
}
