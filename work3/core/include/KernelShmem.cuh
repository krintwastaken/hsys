#pragma once

#include <cstddef>
#include <cuda_runtime.h>
#include "../../../work2/core/include/MatrixView.cuh"

template <typename AtomT, unsigned int BLOCK_SIZE = 16>
__global__ void kernel_matmul_shmem(MatrixView<AtomT> a, MatrixView<AtomT> b, MatrixView<AtomT> c) {
    __shared__ AtomT As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ AtomT Bs[BLOCK_SIZE][BLOCK_SIZE];

    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;

    const std::size_t row = static_cast<std::size_t>(blockIdx.y) * BLOCK_SIZE + ty;
    const std::size_t col = static_cast<std::size_t>(blockIdx.x) * BLOCK_SIZE + tx;

    const std::size_t m = c.nrows();
    const std::size_t k_len = a.ncols();
    const std::size_t n = c.ncols();

    AtomT sum = static_cast<AtomT>(0);
    const std::size_t num_tiles = (k_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (std::size_t t = 0; t < num_tiles; ++t) {
        const std::size_t a_col = t * BLOCK_SIZE + tx;
        if (row < m && a_col < k_len) {
            As[ty][tx] = a(row, a_col);
        } else {
            As[ty][tx] = static_cast<AtomT>(0);
        }

        const std::size_t b_row = t * BLOCK_SIZE + ty;
        if (b_row < k_len && col < n) {
            Bs[ty][tx] = b(b_row, col);
        } else {
            Bs[ty][tx] = static_cast<AtomT>(0);
        }

        __syncthreads();

        #pragma unroll
        for (unsigned int k = 0; k < BLOCK_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < m && col < n) {
        c(row, col) = sum;
    }
}