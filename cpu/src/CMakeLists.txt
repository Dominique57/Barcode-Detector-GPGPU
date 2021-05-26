# Find libraries
find_package(Boost COMPONENTS program_options REQUIRED)

# Eventually do library handling if release or not
if (CMAKE_BUILD_TYPE STREQUAL "Release")
    set(Boost_USE_STATIC_LIBS ON)
endif()

# Add your source files here
add_library(SOURCE_LIB STATIC
    # TODO: add your source file here
    main.cc main.hh

    image/image.cc image/image.hh
        image/splice.cc image/splice.hh image/matrix.cc image/matrix.hh lbp/lbp.cc lbp/lbp.hh lbp/lbpSplices.cc lbp/lbpSplices.hh)

# add important compile features
target_compile_features(SOURCE_LIB PRIVATE
    cxx_std_17
)

target_compile_options(SOURCE_LIB PRIVATE
    "-Wall"
    "-Wextra"
    $<$<CONFIG:Release>:-Werror -Ofast>
    $<$<CONFIG:Debug>:-O0 -ggdb>
    "-pedantic"
)

# link library to my files (ex: -lm for math utilities)
target_link_libraries(SOURCE_LIB
    ${Boost_LIBRARIES}
)

# This is gcc's -I option for simple #include<> directive
# we also want to do it for libraries (better system compatibilities)
target_include_directories(SOURCE_LIB PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${Boost_INCLUDE_DIR}
)