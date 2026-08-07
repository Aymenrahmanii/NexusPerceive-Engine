cmake_minimum_required(VERSION 3.28)
project(NexusPerceiveEngine VERSION 1.0.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Optimization Flags
if(MSVC)
    add_compile_options(/W4 /O2 /permissive-)
else()
    add_compile_options(-Wall -Wextra -O3 -march=native -ffast-math)
endif()

# CUDA Flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math -lineinfo")
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # Default to Ampere (80/86) and Ada (89) / Hopper (90) architectures
    set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Find Dependencies
find_package(CUDA REQUIRED)
find_package(TensorRT REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui)
find_package(Protobuf REQUIRED)
find_package(gRPC REQUIRED)

# Include Directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${Protobuf_INCLUDE_DIRS}
    ${CMAKE_CURRENT_BINARY_DIR}
)

# Protobuf / gRPC Generation
set(PROTO_FILE "${CMAKE_CURRENT_SOURCE_DIR}/proto/perception_service.proto")
get_filename_component(PROTO_NAME ${PROTO_FILE} NAME_WE)
set(PROTO_SRC "${CMAKE_CURRENT_BINARY_DIR}/${PROTO_NAME}.pb.cc")
set(PROTO_HDR "${CMAKE_CURRENT_BINARY_DIR}/${PROTO_NAME}.pb.h")
set(GRPC_SRC "${CMAKE_CURRENT_BINARY_DIR}/${PROTO_NAME}.grpc.pb.cc")
set(GRPC_HDR "${CMAKE_CURRENT_BINARY_DIR}/${PROTO_NAME}.grpc.pb.h")

add_custom_command(
    OUTPUT "${PROTO_SRC}" "${PROTO_HDR}" "${GRPC_SRC}" "${GRPC_HDR}"
    COMMAND Protobuf::protoc
    ARGS --grpc_out="${CMAKE_CURRENT_BINARY_DIR}"
         --cpp_out="${CMAKE_CURRENT_BINARY_DIR}"
         -I "${CMAKE_CURRENT_SOURCE_DIR}/proto"
         --plugin=protoc-gen-grpc=$<TARGET_FILE:gRPC::grpc_cpp_plugin>
         "${PROTO_FILE}"
    DEPENDS "${PROTO_FILE}"
)

add_library(perception_proto ${PROTO_SRC} ${PROTO_HDR} ${GRPC_SRC} ${GRPC_HDR})
target_link_libraries(perception_proto PUBLIC Protobuf::libprotobuf gRPC::grpc++)

# CUDA Kernels Library
add_library(nexus_cuda_kernels STATIC
    kernels/preprocess_kernel.cu
    kernels/postprocess_kernel.cu
)
set_target_properties(nexus_cuda_kernels PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(nexus_cuda_kernels PRIVATE ${CUDA_LIBRARIES})

# Core Nexus Engine Library
add_library(nexus_engine STATIC
    src/common/memory_pool.cpp
    src/core/trt_engine.cpp
    src/pipeline/inference_pipeline.cpp
    src/service/perception_service.cpp
)
target_link_libraries(nexus_engine PUBLIC
    nexus_cuda_kernels
    perception_proto
    ${TENSORRT_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${OpenCV_LIBS}
    gRPC::grpc++
    Threads::Threads
)

# Executables
add_executable(nexus_perceive_app src/main.cpp)
target_link_libraries(nexus_perceive_app PRIVATE nexus_engine)

# Benchmarks
add_executable(benchmark_throughput benchmarks/benchmark_throughput.cpp)
target_link_libraries(benchmark_throughput PRIVATE nexus_engine)

# Testing (GoogleTest)
enable_testing()
find_package(GTest QUIET)
if(GTest_FOUND)
    add_executable(test_memory_pool tests/unit/test_memory_pool.cpp)
    target_link_libraries(test_memory_pool PRIVATE nexus_engine GTest::gtest_main)

    add_executable(test_trt_engine tests/unit/test_trt_engine.cpp)
    target_link_libraries(test_trt_engine PRIVATE nexus_engine GTest::gtest_main)

    add_executable(test_pipeline_latency tests/integration/test_pipeline_latency.cpp)
    target_link_libraries(test_pipeline_latency PRIVATE nexus_engine GTest::gtest_main)

    add_test(NAME MemoryPoolTest COMMAND test_memory_pool)
    add_test(NAME TRTEngineTest COMMAND test_trt_engine)
    add_test(NAME PipelineLatencyTest COMMAND test_pipeline_latency)
endif()
