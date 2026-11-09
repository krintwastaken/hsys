add_library(work4_core
    ${CMAKE_CURRENT_SOURCE_DIR}/src/MatrixOps.cu
)

target_include_directories(work4_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/work1/core/include
        ${CMAKE_SOURCE_DIR}/work2/core/include
        ${CMAKE_SOURCE_DIR}/work3/core/include
)

target_link_libraries(work4_core
    PUBLIC
        CUDA::cudart
)