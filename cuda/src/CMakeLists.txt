# Find libraries
find_package(OpenCV REQUIRED)
if (NOT "${OpenCV_opencv_core_FOUND}"
        OR NOT "${OpenCV_opencv_imgcodecs_FOUND}"
        OR NOT "${OpenCV_opencv_highgui_FOUND}")
    message(SEND_ERROR "Missing opencv `core`|`imgcodecs`|`highgui` library !")
endif ()

# ============================ CUDA SOURCE ====================================
add_library(SOURCE_LIB_CUDA SHARED
    image/splice.cu image/splice.hh
    image/matrix.cu image/matrix.hh
    lbp/lbpGpu.cu lbp/lbpCpu.cc
    kmeans/kmeansTransformGpu.cu kmeans/kmeansTransformGpu.hh
    )
target_compile_features(SOURCE_LIB_CUDA PRIVATE
    cuda_std_17
    )
target_compile_options(SOURCE_LIB_CUDA PRIVATE
    "-Wall"
    "-Wextra"
    )
set_target_properties(SOURCE_LIB_CUDA
        PROPERTIES CUDA_SEPARABLE_COMPILATION ON
    )
target_link_libraries(SOURCE_LIB_CUDA
    )
target_include_directories(SOURCE_LIB_CUDA PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    )

# ============================ CXX SOURCE =====================================
add_library(SOURCE_LIB STATIC
    entryPoint.cc entryPoint.hh
    cuda/mycuda.hh

    kmeans/KmeansTransform.cc kmeans/KmeansTransform.hh

    lbp/lbpAlgorithm.hh
    lbp/lbpCpu.hh lbp/lbpGpu.hh
    )
target_compile_features(SOURCE_LIB PRIVATE
    cxx_std_17
    )
target_compile_options(SOURCE_LIB PRIVATE
    "-Wall"
    "-Wextra"
    $<$<CONFIG:Release>:-Werror -O3>
    $<$<CONFIG:Debug>:-O0 -ggdb>
    "-pedantic"
    )
target_link_libraries(SOURCE_LIB
    "opencv_core;opencv_imgcodecs;opencv_highgui"
    SOURCE_LIB_CUDA
    )
target_include_directories(SOURCE_LIB PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    )
