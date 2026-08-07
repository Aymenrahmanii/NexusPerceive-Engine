# FindCUDA.cmake
# Auxiliary helper for CUDA toolkit path resolution

find_path(CUDA_INCLUDE_DIR cuda_runtime.h
    HINTS
        $ENV{CUDA_PATH}
        /usr/local/cuda
        /usr/include
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4"
    PATH_SUFFIXES include
)

find_library(CUDA_CUDART_LIBRARY cudart
    HINTS
        $ENV{CUDA_PATH}
        /usr/local/cuda
        /usr/lib/x86_64-linux-gnu
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4"
    PATH_SUFFIXES lib lib64 lib/x64
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CUDA
    DEFAULT_MSG
    CUDA_CUDART_LIBRARY
    CUDA_INCLUDE_DIR
)

if(CUDA_FOUND)
    set(CUDA_INCLUDE_DIRS ${CUDA_INCLUDE_DIR})
    set(CUDA_LIBRARIES ${CUDA_CUDART_LIBRARY})
endif()
