include(FetchContent)

FetchContent_Declare(
    GoogleBenchmark
    URL https://github.com/google/benchmark/archive/v1.8.3.tar.gz
)

if (NOT GoogleBenchmark_POPULATED)
    FetchContent_Populate(GoogleBenchmark)
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "From Gtest")
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "From Gtest")
    add_subdirectory(${googlebenchmark_SOURCE_DIR} ${googlebenchmark_BINARY_DIR})
endif()

add_executable(bench bench.cc)
target_compile_features(bench PUBLIC cxx_std_17)
target_link_libraries(bench PRIVATE SOURCE_LIB benchmark)
target_include_directories(bench PRIVATE ../src)
