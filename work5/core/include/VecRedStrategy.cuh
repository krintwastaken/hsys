#pragma once

#include <algorithm>
#include <cstddef>
#include <cuda_runtime.h>
#include "../../../work1/core/include/CudaCheck.cuh"
#include "../../../work1/core/include/VectorView.cuh"
#include "KernelNoBr.cuh"
#include "KernelBr.cuh"

template <typename AtomT>
class IVecRedStrategy {
public:
    virtual ~IVecRedStrategy() = default;
    virtual void reduce(VectorView<AtomT> vec, AtomT* out) const = 0;
};

template <typename AtomT>
class NoBrVecRedStrategy : public IVecRedStrategy<AtomT> {
public:
    void reduce(VectorView<AtomT> vec, AtomT* out) const override {
        constexpr unsigned int BLOCK_SIZE = 256;
        const unsigned int num_blocks = static_cast<unsigned int>(std::min<std::size_t>(
            (vec.size() + BLOCK_SIZE - 1) / BLOCK_SIZE,
            1024
        ));
        const unsigned int grid_size = (num_blocks == 0) ? 1 : num_blocks;

        check_cuda(cudaMemsetAsync(out, 0, sizeof(AtomT)));
        kernel_vecred_nobr<AtomT, BLOCK_SIZE><<<grid_size, BLOCK_SIZE>>>(vec, out);
        check_cuda(cudaGetLastError());
        check_cuda(cudaDeviceSynchronize());
    }
};

template <typename AtomT>
class BrVecRedStrategy : public IVecRedStrategy<AtomT> {
public:
    void reduce(VectorView<AtomT> vec, AtomT* out) const override {
        constexpr unsigned int BLOCK_SIZE = 256;
        const unsigned int num_blocks = static_cast<unsigned int>(std::min<std::size_t>(
            (vec.size() + BLOCK_SIZE - 1) / BLOCK_SIZE,
            1024
        ));
        const unsigned int grid_size = (num_blocks == 0) ? 1 : num_blocks;

        check_cuda(cudaMemsetAsync(out, 0, sizeof(AtomT)));
        kernel_vecred_br<AtomT, BLOCK_SIZE><<<grid_size, BLOCK_SIZE>>>(vec, out);
        check_cuda(cudaGetLastError());
        check_cuda(cudaDeviceSynchronize());
    }
};