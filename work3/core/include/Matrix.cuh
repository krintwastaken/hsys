#pragma once

#include <cstddef>
#include <memory>
#include <utility>
#include "../../../work1/core/include/Data.cuh"
#include "../../../work2/core/include/MatrixView.cuh"
#include "MatMulStrategy.cuh"

template <typename AtomT>
class Matrix {
private:
    std::shared_ptr<Data<AtomT>> data_;
    MatrixView<AtomT> view_;
    inline static std::shared_ptr<IMatMulStrategy<AtomT>> strategy_ = nullptr;

public:
    explicit Matrix(std::size_t nrows, std::size_t ncols)
        : data_(std::make_shared<Data<AtomT>>(nrows * ncols)),
          view_(data_->data(), nrows, ncols) {}

    [[nodiscard]] std::size_t size() const noexcept { return view_.size(); }
    [[nodiscard]] std::size_t nrows() const noexcept { return view_.nrows(); }
    [[nodiscard]] std::size_t ncols() const noexcept { return view_.ncols(); }

    [[nodiscard]] Data<AtomT>& data() noexcept { return *data_; }
    [[nodiscard]] const Data<AtomT>& data() const noexcept { return *data_; }

    [[nodiscard]] MatrixView<AtomT>& view() noexcept { return view_; }
    [[nodiscard]] const MatrixView<AtomT>& view() const noexcept { return view_; }

    void copy_to_host(AtomT* host_ptr) const { data_->copy_to_host(host_ptr); }
    void copy_from_host(const AtomT* host_ptr) { data_->copy_from_host(host_ptr); }

    static void set_strategy(std::shared_ptr<IMatMulStrategy<AtomT>> strategy) {
        strategy_ = std::move(strategy);
    }

    static std::shared_ptr<IMatMulStrategy<AtomT>> get_strategy() {
        if (!strategy_) {
            strategy_ = std::make_shared<SharedMemoryMatMulStrategy<AtomT>>();
        }
        return strategy_;
    }
};

template <typename AtomT>
Matrix<AtomT> operator*(const Matrix<AtomT>& a, const Matrix<AtomT>& b);