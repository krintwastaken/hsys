add_library(work1_core
    ${CMAKE_CURRENT_SOURCE_DIR}/src/VectorOps.cu
)

target_include_directories(work1_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(work1_core
    PUBLIC
        CUDA::cudart
)