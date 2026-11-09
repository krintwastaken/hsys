#include <benchmark/benchmark.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "Matrix.cuh"
#include "KernelWmma.cuh"
#include "../../../work1/core/include/CudaCheck.cuh"
#include "../../../work3/core/include/KernelShmem.cuh"

// Выделенные функции для запуска вычислений в соответствии с п. 4 задания
static void run_matmul_wmma(MatrixView<half> a, MatrixView<half> b, MatrixView<half> c) {
    constexpr unsigned int BLOCK_ITEMS_X = 64;
    constexpr unsigned int BLOCK_ITEMS_Y = 32;
    dim3 block(32, 8);
    dim3 grid(
        static_cast<unsigned int>((c.ncols() + BLOCK_ITEMS_X - 1) / BLOCK_ITEMS_X),
        static_cast<unsigned int>((c.nrows() + BLOCK_ITEMS_Y - 1) / BLOCK_ITEMS_Y)
    );
    kernel_matmul_wmma<BLOCK_ITEMS_X, BLOCK_ITEMS_Y><<<grid, block>>>(a, b, c);
}

static void run_matmul_shmem(MatrixView<half> a, MatrixView<half> b, MatrixView<half> c) {
    constexpr unsigned int BLOCK_SIZE = 16;
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(
        static_cast<unsigned int>((c.ncols() + BLOCK_SIZE - 1) / BLOCK_SIZE),
        static_cast<unsigned int>((c.nrows() + BLOCK_SIZE - 1) / BLOCK_SIZE)
    );
    kernel_matmul_shmem<half, BLOCK_SIZE><<<grid, block>>>(a, b, c);
}

static void BM_MatMul_CUDA_Shmem(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));

    Matrix<half> a(n, n);
    Matrix<half> b(n, n);
    Matrix<half> c(n, n);

    cudaEvent_t start, stop;
    check_cuda(cudaEventCreate(&start));
    check_cuda(cudaEventCreate(&stop));

    for (auto _ : state) {
        check_cuda(cudaEventRecord(start, 0));
        run_matmul_shmem(a.view(), b.view(), c.view());
        check_cuda(cudaEventRecord(stop, 0));
        check_cuda(cudaEventSynchronize(stop));

        float milliseconds = 0.0f;
        check_cuda(cudaEventElapsedTime(&milliseconds, start, stop));
        state.SetIterationTime(static_cast<double>(milliseconds) / 1000.0);
    }

    check_cuda(cudaEventDestroy(start));
    check_cuda(cudaEventDestroy(stop));

    state.SetComplexityN(static_cast<int64_t>(n));
}

static void BM_MatMul_CUDA_WMMA(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));

    Matrix<half> a(n, n);
    Matrix<half> b(n, n);
    Matrix<half> c(n, n);

    cudaEvent_t start, stop;
    check_cuda(cudaEventCreate(&start));
    check_cuda(cudaEventCreate(&stop));

    for (auto _ : state) {
        check_cuda(cudaEventRecord(start, 0));
        run_matmul_wmma(a.view(), b.view(), c.view());
        check_cuda(cudaEventRecord(stop, 0));
        check_cuda(cudaEventSynchronize(stop));

        float milliseconds = 0.0f;
        check_cuda(cudaEventElapsedTime(&milliseconds, start, stop));
        state.SetIterationTime(static_cast<double>(milliseconds) / 1000.0);
    }

    check_cuda(cudaEventDestroy(start));
    check_cuda(cudaEventDestroy(stop));

    state.SetComplexityN(static_cast<int64_t>(n));
}

static void CustomArguments(benchmark::internal::Benchmark* b) {
    for (int n : {16, 32, 64, 128, 256, 512, 1024}) {
        b->Arg(n);
    }
}

BENCHMARK(BM_MatMul_CUDA_Shmem)
    ->Name("CUDA Matrix Multiplication (GPU, SharedMemory, half)")
    ->Apply(CustomArguments)
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_MatMul_CUDA_WMMA)
    ->Name("CUDA Matrix Multiplication (GPU, WMMA, half)")
    ->Apply(CustomArguments)
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();