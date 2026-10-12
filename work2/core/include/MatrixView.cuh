#pragma once

#include <cstddef>
#include <type_traits>
#include <cuda_runtime.h>

template <typename AtomT>
class MatrixView {
private:
    AtomT* data_{nullptr};
    std::size_t nrows_{0};
    std::size_t ncols_{0};

public:
    MatrixView() = default;

    __host__ __device__ MatrixView(AtomT* data, std::size_t nrows, std::size_t ncols)
        : data_(data), nrows_(nrows), ncols_(ncols) {}

    __host__ __device__ [[nodiscard]] std::size_t size() const noexcept {
        return nrows_ * ncols_;
    }

    __host__ __device__ [[nodiscard]] std::size_t nrows() const noexcept {
        return nrows_;
    }

    __host__ __device__ [[nodiscard]] std::size_t ncols() const noexcept {
        return ncols_;
    }

    __host__ __device__ [[nodiscard]] AtomT* data() noexcept {
        return data_;
    }

    __host__ __device__ [[nodiscard]] const AtomT* data() const noexcept {
        return data_;
    }

    __host__ __device__ AtomT& operator[](std::size_t n) noexcept {
        return data_[n];
    }

    __host__ __device__ const AtomT& operator[](std::size_t n) const noexcept {
        return data_[n];
    }

    __host__ __device__ AtomT& operator()(std::size_t i, std::size_t j) noexcept {
        return data_[i * ncols_ + j];
    }

    __host__ __device__ const AtomT& operator()(std::size_t i, std::size_t j) const noexcept {
        return data_[i * ncols_ + j];
    }
};

static_assert(std::is_trivially_copyable_v<MatrixView<float>>);