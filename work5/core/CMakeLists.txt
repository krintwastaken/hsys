add_library(work5_core
    ${CMAKE_CURRENT_SOURCE_DIR}/src/VectorOps.cu
)

target_include_directories(work5_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/work1/core/include
)

target_link_libraries(work5_core
    PUBLIC
        CUDA::cudart
)