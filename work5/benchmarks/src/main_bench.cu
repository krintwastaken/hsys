#include <benchmark/benchmark.h>
#include <cuda_runtime.h>
#include "Vector.cuh"
#include "../../../work1/core/include/CudaCheck.cuh"

static void BM_VecRed_CUDA_NoBr(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));

    Vector<float> vec(n);
    Data<float> d_out(1);

    cudaEvent_t start, stop;
    check_cuda(cudaEventCreate(&start));
    check_cuda(cudaEventCreate(&stop));

    for (auto _ : state) {
        check_cuda(cudaEventRecord(start, 0));
        run_vecred_nobr(vec.view(), d_out.data());
        check_cuda(cudaEventRecord(stop, 0));
        check_cuda(cudaEventSynchronize(stop));

        float milliseconds = 0.0f;
        check_cuda(cudaEventElapsedTime(&milliseconds, start, stop));
        state.SetIterationTime(static_cast<double>(milliseconds) / 1000.0);
    }

    check_cuda(cudaEventDestroy(start));
    check_cuda(cudaEventDestroy(stop));

    state.SetComplexityN(static_cast<int64_t>(n));
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n * sizeof(float)));
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n));
}

static void BM_VecRed_CUDA_Br(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));

    Vector<float> vec(n);
    Data<float> d_out(1);

    cudaEvent_t start, stop;
    check_cuda(cudaEventCreate(&start));
    check_cuda(cudaEventCreate(&stop));

    for (auto _ : state) {
        check_cuda(cudaEventRecord(start, 0));
        run_vecred_br(vec.view(), d_out.data());
        check_cuda(cudaEventRecord(stop, 0));
        check_cuda(cudaEventSynchronize(stop));

        float milliseconds = 0.0f;
        check_cuda(cudaEventElapsedTime(&milliseconds, start, stop));
        state.SetIterationTime(static_cast<double>(milliseconds) / 1000.0);
    }

    check_cuda(cudaEventDestroy(start));
    check_cuda(cudaEventDestroy(stop));

    state.SetComplexityN(static_cast<int64_t>(n));
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n * sizeof(float)));
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n));
}

static void CustomArguments(benchmark::internal::Benchmark* b) {
    for (int i = 0; i <= 28; ++i) {
        // Ограничение до i=25 (~268 млн эл-тов, 1.07 ГБ), чтобы гарантированно
        // уложиться в лимит VRAM 4 ГБ видеокарты RTX 3050 Laptop без сбоев аллокации
        if (i <= 25) {
            b->Arg(8ULL << i);
        }
    }
}

BENCHMARK(BM_VecRed_CUDA_NoBr)
    ->Name("CUDA Vector Reduction (GPU, NoBr, float)")
    ->Apply(CustomArguments)
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_VecRed_CUDA_Br)
    ->Name("CUDA Vector Reduction (GPU, Br, float)")
    ->Apply(CustomArguments)
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();