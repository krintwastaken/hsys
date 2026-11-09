add_executable(work4_benchmarks
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main_bench.cu
)

target_link_libraries(work4_benchmarks
    PRIVATE
        work4_core
        benchmark::benchmark
        CUDA::cudart
)