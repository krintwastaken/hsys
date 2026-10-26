add_library(work3_core
    ${CMAKE_CURRENT_SOURCE_DIR}/src/MatrixOps.cu
)

target_include_directories(work3_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/work1/core/include
        ${CMAKE_SOURCE_DIR}/work2/core/include
)

target_link_libraries(work3_core
    PUBLIC
        CUDA::cudart
)