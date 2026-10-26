#include <stdexcept>
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

    Matrix<AtomT>::get_strategy()->multiply(a.view(), b.view(), c.view());
    return c;
}

template Matrix<float> operator*(const Matrix<float>&, const Matrix<float>&);
template Matrix<double> operator*(const Matrix<double>&, const Matrix<double>&);
template Matrix<int> operator*(const Matrix<int>&, const Matrix<int>&);