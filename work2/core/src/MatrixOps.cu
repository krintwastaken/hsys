#include <stdexcept>
#include "Matrix.cuh"
#include "Kernel.cuh"
#include "CudaCheck.cuh"

template <typename AtomT>
Matrix<AtomT> operator*(const Matrix<AtomT>& a, const Matrix<AtomT>& b) {
    if (a.ncols() != b.nrows()) {
        throw std::invalid_argument("Matrix dimension mismatch: a.ncols != b.nrows");
    }

    const std::size_t m = a.nrows();
    const std::size_t n = b.ncols();
    Matrix<AtomT> c(m, n);

    if (m == 0 || n == 0 || a.ncols() == 0) {
        return c;
    }

    constexpr unsigned int block_dim_x = 16;
    constexpr unsigned int block_dim_y = 16;
    dim3 block(block_dim_x, block_dim_y);
    dim3 grid(
        static_cast<unsigned int>((n + block_dim_x - 1) / block_dim_x),
        static_cast<unsigned int>((m + block_dim_y - 1) / block_dim_y)
    );

    kernel_matmul_naive<AtomT><<<grid, block>>>(a.view(), b.view(), c.view());
    check_cuda(cudaGetLastError());
    check_cuda(cudaDeviceSynchronize());

    return c;
}

template Matrix<float> operator*(const Matrix<float>&, const Matrix<float>&);
template Matrix<double> operator*(const Matrix<double>&, const Matrix<double>&);
template Matrix<int> operator*(const Matrix<int>&, const Matrix<int>&);