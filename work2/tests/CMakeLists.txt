add_executable(work2_tests
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main_test.cu
)

target_link_libraries(work2_tests
    PRIVATE
        work2_core
        GTest::gtest
        Eigen3::Eigen
        CUDA::cudart
)