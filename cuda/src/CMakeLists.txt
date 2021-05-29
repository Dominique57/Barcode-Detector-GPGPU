# Find libraries
find_package(OpenCV REQUIRED)

# Add your source files here
add_library(SOURCE_LIB STATIC
    # TODO: add your source file here
    entryPoint.cc entryPoint.hh

    image/splice.cu image/splice.hh
    image/matrix.cu image/matrix.hh

    lbp/lbp.cc lbp/lbp.hh
    lbp/lbpSplices.cu lbp/lbpSplices.hh

    kmeans/KmeansTransform.cc kmeans/KmeansTransform.hh cuda/mycuda.hh)

# add important compile features
target_compile_features(SOURCE_LIB PRIVATE
    cxx_std_17
    )

target_compile_options(SOURCE_LIB PRIVATE
    "-Wall"
    "-Wextra"
    $<$<COMPILE_LANGUAGE:CXX>:
        $<$<CONFIG:Release>:-Werror -O3>
        $<$<CONFIG:Debug>:-O0 -ggdb>
        "-pedantic"
    >
    )

# link library to my files (ex: -lm for math utilities)
if (NOT ${OpenCV_opencv_core_FOUND} OR NOT ${OpenCV_opencv_imgcodecs_FOUND} OR NOT ${OpenCV_opencv_highgui_FOUND})
    message( SEND_ERROR "Missing opencv `core` or `imgcodecs` or `highgui` library !" )
endif ()

target_link_libraries(SOURCE_LIB
    #    ${OpenCV_LIBS}
    "opencv_core;opencv_imgcodecs;opencv_highgui"
    )

# This is gcc's -I option for simple #include<> directive
# we also want to do it for libraries (better system compatibilities)
target_include_directories(SOURCE_LIB PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    )
