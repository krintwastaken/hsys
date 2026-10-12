add_executable(work1_tests
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main_test.cu
)

target_link_libraries(work1_tests
    PRIVATE
        work1_core
        GTest::gtest
        Eigen3::Eigen
        CUDA::cudart
)