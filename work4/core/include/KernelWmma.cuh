#pragma once

#include <cstddef>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include "../../../work2/core/include/MatrixView.cuh"

using namespace nvcuda;

constexpr unsigned int WMMA_M = 16;
constexpr unsigned int WMMA_N = 16;
constexpr unsigned int WMMA_K = 16;

constexpr unsigned int WARPS_X = 4;
constexpr unsigned int WARPS_Y = 2;

// Шаблонизация размерами тайла обеспечивает COMDAT-компоновку и предотвращает ошибку LNK2005
template <unsigned int BLOCK_ITEMS_X = 64, unsigned int BLOCK_ITEMS_Y = 32>
__global__ void kernel_matmul_wmma(MatrixView<half> a, MatrixView<half> b, MatrixView<half> c) {
    const unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const unsigned int warp_id = tid / 32;

    if (warp_id >= WARPS_X * WARPS_Y) {
        return;
    }

    // Раскладка 8 варпов внутри блока 4x2 по Рис. 1 методички:
    // Строка 0: Warp0, Warp2, Warp4, Warp6
    // Строка 1: Warp1, Warp3, Warp5, Warp7
    const unsigned int warp_row = warp_id % WARPS_Y;
    const unsigned int warp_col = warp_id / WARPS_Y;

    const std::size_t c_row = static_cast<std::size_t>(blockIdx.y) * BLOCK_ITEMS_Y + warp_row * WMMA_M;
    const std::size_t c_col = static_cast<std::size_t>(blockIdx.x) * BLOCK_ITEMS_X + warp_col * WMMA_N;

    const std::size_t m = c.nrows();
    const std::size_t n = c.ncols();
    const std::size_t k_len = a.ncols();

    if (c_row >= m || c_col >= n) {
        return;
    }

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> acc_frag;
    wmma::fill_fragment(acc_frag, __float2half(0.0f));

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;

    for (std::size_t k = 0; k < k_len; k += WMMA_K) {
        const half* a_ptr = a.data() + c_row * k_len + k;
        const half* b_ptr = b.data() + k * n + c_col;

        wmma::load_matrix_sync(a_frag, a_ptr, k_len);
        wmma::load_matrix_sync(b_frag, b_ptr, n);

        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    half* c_ptr = c.data() + c_row * n + c_col;
    wmma::store_matrix_sync(c_ptr, acc_frag, n, wmma::mem_row_major);
}