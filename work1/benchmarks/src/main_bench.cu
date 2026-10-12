#include <benchmark/benchmark.h>
#include <Eigen/Dense>
#include <cuda_runtime.h>
#include "Vector.cuh"
#include "Kernel.cuh"
#include "CudaCheck.cuh"

static void BM_VecAdd_Eigen(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));

    Eigen::VectorXf a = Eigen::VectorXf::Random(static_cast<Eigen::Index>(n));
    Eigen::VectorXf b = Eigen::VectorXf::Random(static_cast<Eigen::Index>(n));
    Eigen::VectorXf c(n);

    for (auto _ : state) {
        c = a + b;
        benchmark::DoNotOptimize(c.data());
    }

    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n * sizeof(float) * 3));
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n));
}

static void BM_VecAdd_CUDA(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));

    Vector<float> a(n);
    Vector<float> b(n);
    Vector<float> c(n);

    constexpr unsigned int block_size = 256;
    const unsigned int grid_size = static_cast<unsigned int>((n + block_size - 1) / block_size);

    cudaEvent_t start, stop;
    check_cuda(cudaEventCreate(&start));
    check_cuda(cudaEventCreate(&stop));

    for (auto _ : state) {
        check_cuda(cudaEventRecord(start, 0));
        kernel_vecadd<float><<<grid_size, block_size>>>(a.view(), b.view(), c.view());
        check_cuda(cudaEventRecord(stop, 0));
        check_cuda(cudaEventSynchronize(stop));

        float milliseconds = 0.0f;
        check_cuda(cudaEventElapsedTime(&milliseconds, start, stop));
        state.SetIterationTime(static_cast<double>(milliseconds) / 1000.0);
    }

    check_cuda(cudaEventDestroy(start));
    check_cuda(cudaEventDestroy(stop));

    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n * sizeof(float) * 3));
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n));
}

BENCHMARK(BM_VecAdd_Eigen)
    ->RangeMultiplier(8)
    ->Range(8, 16777216)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_VecAdd_CUDA)
    ->RangeMultiplier(8)
    ->Range(8, 16777216)
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();