add_executable(work5_tests
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main_test.cu
)

target_link_libraries(work5_tests
    PRIVATE
        work5_core
        GTest::gtest
        Eigen3::Eigen
        CUDA::cudart
)