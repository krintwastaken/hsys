add_library(work2_core
    ${CMAKE_CURRENT_SOURCE_DIR}/src/MatrixOps.cu
)

target_include_directories(work2_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(work2_core
    PUBLIC
        CUDA::cudart
)