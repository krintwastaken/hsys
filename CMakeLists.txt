cmake_minimum_required(VERSION 3.22)
project(HSysMaster LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES native)

add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=940,20012>)

find_package(CUDAToolkit REQUIRED)
find_package(Eigen3 3.3 QUIET CONFIG)

if(NOT TARGET Eigen3::Eigen)
    find_path(EIGEN3_INCLUDE_DIR
        NAMES Eigen/Dense
        PATHS
            "C:/Users/krintwastaken/My stuff/ucheba/GVS/vcpkg/installed/x64-windows/include"
            "C:/Users/krintwastaken/My stuff/ucheba/GVS/vcpkg/installed/x86-windows/include"
            "C:/Users/krintwastaken/My stuff/ucheba/GVS/vcpkg/buildtrees/eigen3/src/5.0.1-d487a628b0.clean"
    )

    if(EIGEN3_INCLUDE_DIR)
        add_library(Eigen3::Eigen INTERFACE IMPORTED)
        target_include_directories(Eigen3::Eigen INTERFACE "${EIGEN3_INCLUDE_DIR}")
        message(STATUS "Found Eigen3 headers: ${EIGEN3_INCLUDE_DIR}")
    else()
        include(FetchContent)
        FetchContent_Declare(
            Eigen3
            URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
        )
        FetchContent_MakeAvailable(Eigen3)
    endif()
endif()

include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

FetchContent_Declare(
    google_benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.8.3
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(google_benchmark)

add_subdirectory(work1)
add_subdirectory(work2)