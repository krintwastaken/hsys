#pragma once

#include <cstddef>
#include <cuda_runtime.h>
#include "../../../work1/core/include/VectorView.cuh"

template <typename AtomT, unsigned int BLOCK_SIZE = 256>
__global__ void kernel_vecred_nobr(VectorView<AtomT> vec, AtomT* out) {
    __shared__ AtomT s_data[BLOCK_SIZE];

    const unsigned int tid = threadIdx.x;
    AtomT my_sum = static_cast<AtomT>(0);

    // Коалесцированный сеточный цикл по всему вектору
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + tid;
         i < vec.size();
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        my_sum += vec[i];
    }

    s_data[tid] = my_sum;
    __syncthreads();

    // Древовидная блочная редукция в разделяемой памяти
    #pragma unroll
    for (unsigned int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }

    // Сохранение частичной суммы блока в глобальный аккумулятор
    if (tid == 0) {
        atomicAdd(out, s_data[0]);
    }
}

// Перегрузка для совместимости при передаче VectorView в качестве выходного буфера
template <typename AtomT, unsigned int BLOCK_SIZE = 256>
__global__ void kernel_vecred_nobr(VectorView<AtomT> vec, VectorView<AtomT> out) {
    kernel_vecred_nobr<AtomT, BLOCK_SIZE>(vec, out.data());
}