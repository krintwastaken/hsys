#pragma once

#include <cstddef>
#include <cuda_runtime.h>
#include "VectorView.cuh"

template <typename AtomT>
__global__ void kernel_vecadd(VectorView<AtomT> a, VectorView<AtomT> b, VectorView<AtomT> c) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < a.size()) {
        c[idx] = a[idx] + b[idx];
    }
}