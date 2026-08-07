# FindOpenCV.cmake fallback module
find_package(OpenCV CONFIG QUIET)
if(NOT OpenCV_FOUND)
    find_path(OpenCV_INCLUDE_DIRS opencv2/opencv.hpp
        HINTS /usr/include /usr/local/include "C:/opencv/build/include"
    )
    find_library(OpenCV_LIBS opencv_core
        HINTS /usr/lib /usr/local/lib "C:/opencv/build/x64/vc16/lib"
    )
    include(FindPackageHandleStandardArgs)
    find_package_handle_standard_args(OpenCV DEFAULT_MSG OpenCV_INCLUDE_DIRS OpenCV_LIBS)
endif()
