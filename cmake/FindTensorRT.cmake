# FindTensorRT.cmake
# Find NVIDIA TensorRT headers and libraries

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS
        ${TENSORRT_ROOT}
        $ENV{TENSORRT_ROOT}
        /usr/include
        /usr/include/x86_64-linux-gnu
        /usr/local/cuda/include
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/include"
    PATH_SUFFIXES include
)

find_library(TENSORRT_NVINFER_LIBRARY nvinfer
    HINTS
        ${TENSORRT_ROOT}
        $ENV{TENSORRT_ROOT}
        /usr/lib
        /usr/lib/x86_64-linux-gnu
        /usr/local/cuda/lib64
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/lib/x64"
    PATH_SUFFIXES lib lib64 lib/x64
)

find_library(TENSORRT_NVONNXPARSER_LIBRARY nvonnxparser
    HINTS
        ${TENSORRT_ROOT}
        $ENV{TENSORRT_ROOT}
        /usr/lib
        /usr/lib/x86_64-linux-gnu
        /usr/local/cuda/lib64
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/lib/x64"
    PATH_SUFFIXES lib lib64 lib/x64
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TensorRT
    DEFAULT_MSG
    TENSORRT_NVINFER_LIBRARY
    TENSORRT_INCLUDE_DIR
)

if(TensorRT_FOUND)
    set(TENSORRT_INCLUDE_DIRS ${TENSORRT_INCLUDE_DIR})
    set(TENSORRT_LIBRARIES ${TENSORRT_NVINFER_LIBRARY} ${TENSORRT_NVONNXPARSER_LIBRARY})
    message(STATUS "Found TensorRT include: ${TENSORRT_INCLUDE_DIRS}")
    message(STATUS "Found TensorRT libs:    ${TENSORRT_LIBRARIES}")
endif()
