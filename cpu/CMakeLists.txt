cmake_minimum_required(VERSION 3.8)

project(BarcodeDetector
    VERSION 1.0.0
    DESCRIPTION "CPU implementation of a barcode detector"
    LANGUAGES CXX
)

if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected, default to Release")
    set(CMAKE_BUILD_TYPE Release)
endif()

# create SOURCE_LIB static library: user souce library
add_subdirectory(src)

# add test folder
enable_testing()
add_subdirectory(test)

# final binary output
add_executable(${PROJECT_NAME} src/main.cc)
target_link_libraries(${PROJECT_NAME} LINK_PUBLIC SOURCE_LIB)
