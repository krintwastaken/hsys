#include <benchmark/benchmark.h>

#include <complex>
using std::arg;

#include <Eigen/Dense>
#include <cuda_runtime.h>
#include "Matrix.cuh"
#include "Kernel.cuh"
#include "../../../work1/core/include/CudaCheck.cuh"

using RowMatrixXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

static void BM_MatMul_Eigen(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));

    RowMatrixXf a = RowMatrixXf::Random(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));
    RowMatrixXf b = RowMatrixXf::Random(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));
    RowMatrixXf c(n, n);

    for (auto _ : state) {
        c.noalias() = a * b;
        benchmark::DoNotOptimize(c.data());
    }

    state.SetComplexityN(static_cast<int64_t>(n));
}

static void BM_MatMul_CUDA(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));

    Matrix<float> a(n, n);
    Matrix<float> b(n, n);
    Matrix<float> c(n, n);

    constexpr unsigned int block_dim_x = 16;
    constexpr unsigned int block_dim_y = 16;
    dim3 block(block_dim_x, block_dim_y);
    dim3 grid(
        static_cast<unsigned int>((n + block_dim_x - 1) / block_dim_x),
        static_cast<unsigned int>((n + block_dim_y - 1) / block_dim_y)
    );

    cudaEvent_t start, stop;
    check_cuda(cudaEventCreate(&start));
    check_cuda(cudaEventCreate(&stop));

    for (auto _ : state) {
        check_cuda(cudaEventRecord(start, 0));
        kernel_matmul_naive<float><<<grid, block>>>(a.view(), b.view(), c.view());
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

BENCHMARK(BM_MatMul_Eigen)
    ->Name("Eigen Matrix Multiplication (CPU, float)")
    ->Apply(CustomArguments)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_MatMul_CUDA)
    ->Name("CUDA Matrix Multiplication (GPU, Naive, float)")
    ->Apply(CustomArguments)
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();