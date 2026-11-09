#pragma once

#include <cstddef>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../../../work1/core/include/CudaCheck.cuh"
#include "../../../work2/core/include/MatrixView.cuh"
#include "../../../work3/core/include/KernelShmem.cuh"
#include "KernelWmma.cuh"

template <typename AtomT>
class IMatMulStrategy {
public:
    virtual ~IMatMulStrategy() = default;
    virtual void multiply(MatrixView<AtomT> a, MatrixView<AtomT> b, MatrixView<AtomT> c) const = 0;
};

class WmmaMatMulStrategy : public IMatMulStrategy<half> {
public:
    void multiply(MatrixView<half> a, MatrixView<half> b, MatrixView<half> c) const override {
        dim3 block(32, 8); // 8 варпов (256 нитей)
        dim3 grid(
            static_cast<unsigned int>((c.ncols() + BLOCK_ITEMS_X - 1) / BLOCK_ITEMS_X),
            static_cast<unsigned int>((c.nrows() + BLOCK_ITEMS_Y - 1) / BLOCK_ITEMS_Y)
        );

        kernel_matmul_wmma<<<grid, block>>>(a, b, c);
        check_cuda(cudaGetLastError());
        check_cuda(cudaDeviceSynchronize());
    }
};

class SharedMemoryMatMulStrategy : public IMatMulStrategy<half> {
public:
    void multiply(MatrixView<half> a, MatrixView<half> b, MatrixView<half> c) const override {
        constexpr unsigned int BLOCK_SIZE = 16;
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid(
            static_cast<unsigned int>((c.ncols() + BLOCK_SIZE - 1) / BLOCK_SIZE),
            static_cast<unsigned int>((c.nrows() + BLOCK_SIZE - 1) / BLOCK_SIZE)
        );

        kernel_matmul_shmem<half, BLOCK_SIZE><<<grid, block>>>(a, b, c);
        check_cuda(cudaGetLastError());
        check_cuda(cudaDeviceSynchronize());
    }
};