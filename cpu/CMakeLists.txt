cmake_minimum_required(VERSION 3.8)

project(BarcodeDetector
    VERSION 1.0.0
    DESCRIPTION "CPU implementation of a barcode detector"
    LANGUAGES CXX
)

set(OpenCV_DIR "C:\\Program Files\\opencv\\build\\x64\\vc15\\lib")
set(BOOST_INCLUDE_DIR "C:\\local\\boost_1_74_0\\boost")
set (BOOST_ROOT "C:\\local\\boost_1_74_0")

if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected, default to Release")
    set(CMAKE_BUILD_TYPE Release)
endif()

# create SOURCE_LIB static library: user source library
add_subdirectory(src)

# add test folder
enable_testing()
add_subdirectory(test)

set(Boost_USE_STATIC_LIBS ON)

# final binary output
find_package(Boost COMPONENTS program_options REQUIRED)

# Eventually do library handling if release or not
if (CMAKE_BUILD_TYPE STREQUAL "Release")
    set(Boost_USE_STATIC_LIBS ON)
endif()

if(CMAKE_COMPILER_IS_GNUCXX)
    add_definitions(-Wall -Wextra)
endif(CMAKE_COMPILER_IS_GNUCXX)


# add important compile features
add_executable(${PROJECT_NAME} src/main.cc)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
target_compile_options(${PROJECT_NAME} PRIVATE
        "-Wall"
        $<$<CONFIG:Release>:-Werror -Ofast>
        $<$<CONFIG:Debug>:-O0 -ggdb>
        "-pedantic"
        )
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${Boost_INCLUDE_DIR})

# link library to my files (ex: -lm for math utilities)
target_link_libraries(${PROJECT_NAME} LINK_PUBLIC ${Boost_LIBRARIES} SOURCE_LIB)