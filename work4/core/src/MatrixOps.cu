#include <stdexcept>
#include <cuda_fp16.h>
#include "Matrix.cuh"

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

    auto strategy = Matrix<AtomT>::get_strategy();
    if (!strategy) {
        throw std::runtime_error("No valid matrix multiplication strategy selected for this type");
    }

    strategy->multiply(a.view(), b.view(), c.view());
    return c;
}

template Matrix<half> operator*(const Matrix<half>&, const Matrix<half>&);