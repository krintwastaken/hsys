add_executable(work4_tests
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main_test.cu
)

target_link_libraries(work4_tests
    PRIVATE
        work4_core
        GTest::gtest
        Eigen3::Eigen
        CUDA::cudart
)