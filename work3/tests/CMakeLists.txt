add_executable(work3_tests
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main_test.cu
)

target_link_libraries(work3_tests
    PRIVATE
        work3_core
        GTest::gtest
        Eigen3::Eigen
        CUDA::cudart
)