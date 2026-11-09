#pragma once

#include <cstddef>
#include <cuda_runtime.h>
#include "../../../work1/core/include/VectorView.cuh"

template <typename AtomT>
__device__ inline AtomT warp_reduce_sum(AtomT val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template <typename AtomT, unsigned int BLOCK_SIZE = 256>
__global__ void kernel_vecred_br(VectorView<AtomT> vec, AtomT* out) {
    constexpr unsigned int NUM_WARPS = BLOCK_SIZE / 32;
    __shared__ AtomT warp_sums[NUM_WARPS];

    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid % 32;
    const unsigned int wid = tid / 32;

    AtomT my_sum = static_cast<AtomT>(0);

    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + tid;
         i < vec.size();
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        my_sum += vec[i];
    }

    // 1. Быстрая редукция внутри каждого варпа на регистрах
    my_sum = warp_reduce_sum(my_sum);

    // 2. Нить lane 0 каждого варпа сохраняет частичную сумму варпа в разделяемую память
    if (lane == 0) {
        warp_sums[wid] = my_sum;
    }
    __syncthreads();

    // 3. Первый варп блока считывает суммы варпов и редуцирует их
    AtomT block_sum = (tid < NUM_WARPS) ? warp_sums[lane] : static_cast<AtomT>(0);
    if (wid == 0) {
        block_sum = warp_reduce_sum(block_sum);
        if (tid == 0) {
            atomicAdd(out, block_sum);
        }
    }
}

// Перегрузка для совместимости при передаче VectorView в качестве выходного буфера
template <typename AtomT, unsigned int BLOCK_SIZE = 256>
__global__ void kernel_vecred_br(VectorView<AtomT> vec, VectorView<AtomT> out) {
    kernel_vecred_br<AtomT, BLOCK_SIZE>(vec, out.data());
}