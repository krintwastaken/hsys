#include <benchmark/benchmark.h>

#include <complex>
using std::arg;

#include <Eigen/Dense>
#include <cuda_runtime.h>
#include "Matrix.cuh"
#include "MatMulStrategy.cuh"
#include "../../../work1/core/include/CudaCheck.cuh"

static void BM_MatMul_CUDA_Naive(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));

    Matrix<float> a(n, n);
    Matrix<float> b(n, n);
    Matrix<float> c(n, n);

    NaiveMatMulStrategy<float> strategy;

    cudaEvent_t start, stop;
    check_cuda(cudaEventCreate(&start));
    check_cuda(cudaEventCreate(&stop));

    for (auto _ : state) {
        check_cuda(cudaEventRecord(start, 0));
        strategy.multiply(a.view(), b.view(), c.view());
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

static void BM_MatMul_CUDA_Shmem(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));

    Matrix<float> a(n, n);
    Matrix<float> b(n, n);
    Matrix<float> c(n, n);

    SharedMemoryMatMulStrategy<float> strategy;

    cudaEvent_t start, stop;
    check_cuda(cudaEventCreate(&start));
    check_cuda(cudaEventCreate(&stop));

    for (auto _ : state) {
        check_cuda(cudaEventRecord(start, 0));
        strategy.multiply(a.view(), b.view(), c.view());
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

BENCHMARK(BM_MatMul_CUDA_Naive)
    ->Name("CUDA Matrix Multiplication (GPU, Naive, float)")
    ->Apply(CustomArguments)
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_MatMul_CUDA_Shmem)
    ->Name("CUDA Matrix Multiplication (GPU, SharedMemory, float)")
    ->Apply(CustomArguments)
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();