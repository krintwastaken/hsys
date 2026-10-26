#pragma once

#include <cstddef>
#include <cuda_runtime.h>
#include "../../../work1/core/include/CudaCheck.cuh"
#include "../../../work2/core/include/MatrixView.cuh"
#include "../../../work2/core/include/Kernel.cuh"
#include "KernelShmem.cuh"

template <typename AtomT>
class IMatMulStrategy {
public:
    virtual ~IMatMulStrategy() = default;
    virtual void multiply(MatrixView<AtomT> a, MatrixView<AtomT> b, MatrixView<AtomT> c) const = 0;
};

template <typename AtomT>
class SharedMemoryMatMulStrategy : public IMatMulStrategy<AtomT> {
public:
    void multiply(MatrixView<AtomT> a, MatrixView<AtomT> b, MatrixView<AtomT> c) const override {
        constexpr unsigned int BLOCK_SIZE = 16;
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid(
            static_cast<unsigned int>((c.ncols() + BLOCK_SIZE - 1) / BLOCK_SIZE),
            static_cast<unsigned int>((c.nrows() + BLOCK_SIZE - 1) / BLOCK_SIZE)
        );

        kernel_matmul_shmem<AtomT, BLOCK_SIZE><<<grid, block>>>(a, b, c);
        check_cuda(cudaGetLastError());
        check_cuda(cudaDeviceSynchronize());
    }
};

template <typename AtomT>
class NaiveMatMulStrategy : public IMatMulStrategy<AtomT> {
public:
    void multiply(MatrixView<AtomT> a, MatrixView<AtomT> b, MatrixView<AtomT> c) const override {
        constexpr unsigned int block_dim_x = 16;
        constexpr unsigned int block_dim_y = 16;
        dim3 block(block_dim_x, block_dim_y);
        dim3 grid(
            static_cast<unsigned int>((c.ncols() + block_dim_x - 1) / block_dim_x),
            static_cast<unsigned int>((c.nrows() + block_dim_y - 1) / block_dim_y)
        );

        kernel_matmul_naive<AtomT><<<grid, block>>>(a, b, c);
        check_cuda(cudaGetLastError());
        check_cuda(cudaDeviceSynchronize());
    }
};