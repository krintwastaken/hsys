add_executable(work1_benchmarks
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main_bench.cu
)

target_link_libraries(work1_benchmarks
    PRIVATE
        work1_core
        benchmark::benchmark
        Eigen3::Eigen
        CUDA::cudart
)