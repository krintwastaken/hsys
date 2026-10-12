#include <stdexcept>
#include "Vector.cuh"
#include "Kernel.cuh"
#include "CudaCheck.cuh"

template <typename AtomT>
Vector<AtomT> operator+(const Vector<AtomT>& a, const Vector<AtomT>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vector dimensions mismatch in operator+");
    }

    const std::size_t n = a.size();
    Vector<AtomT> res(n);

    if (n == 0) {
        return res;
    }

    constexpr unsigned int block_size = 256;
    const unsigned int grid_size = static_cast<unsigned int>((n + block_size - 1) / block_size);

    kernel_vecadd<AtomT><<<grid_size, block_size>>>(a.view(), b.view(), res.view());
    check_cuda(cudaGetLastError());
    check_cuda(cudaDeviceSynchronize());

    return res;
}

template Vector<float> operator+(const Vector<float>&, const Vector<float>&);
template Vector<double> operator+(const Vector<double>&, const Vector<double>&);
template Vector<int> operator+(const Vector<int>&, const Vector<int>&);