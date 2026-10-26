add_executable(work3_benchmarks
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main_bench.cu
)

target_link_libraries(work3_benchmarks
    PRIVATE
        work3_core
        benchmark::benchmark
        Eigen3::Eigen
        CUDA::cudart
)