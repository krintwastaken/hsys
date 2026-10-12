add_executable(work2_benchmarks
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main_bench.cu
)

target_link_libraries(work2_benchmarks
    PRIVATE
        work2_core
        benchmark::benchmark
        Eigen3::Eigen
        CUDA::cudart
)