#pragma once

#include <cstddef>
#include <memory>
#include <utility>
#include "../../../work1/core/include/Data.cuh"
#include "../../../work1/core/include/VectorView.cuh"
#include "VecRedStrategy.cuh"

template <typename AtomT>
class Vector {
private:
    std::shared_ptr<Data<AtomT>> data_;
    VectorView<AtomT> view_;
    inline static std::shared_ptr<IVecRedStrategy<AtomT>> strategy_ = nullptr;

public:
    explicit Vector(std::size_t size)
        : data_(std::make_shared<Data<AtomT>>(size)),
          view_(data_->data(), size) {}

    [[nodiscard]] std::size_t size() const noexcept { return view_.size(); }

    [[nodiscard]] Data<AtomT>& data() noexcept { return *data_; }
    [[nodiscard]] const Data<AtomT>& data() const noexcept { return *data_; }

    [[nodiscard]] VectorView<AtomT>& view() noexcept { return view_; }
    [[nodiscard]] const VectorView<AtomT>& view() const noexcept { return view_; }

    void copy_to_host(AtomT* host_ptr) const { data_->copy_to_host(host_ptr); }
    void copy_from_host(const AtomT* host_ptr) { data_->copy_from_host(host_ptr); }

    static void set_strategy(std::shared_ptr<IVecRedStrategy<AtomT>> strategy) {
        strategy_ = std::move(strategy);
    }

    static std::shared_ptr<IVecRedStrategy<AtomT>> get_strategy() {
        if (!strategy_) {
            strategy_ = std::make_shared<BrVecRedStrategy<AtomT>>();
        }
        return strategy_;
    }

    [[nodiscard]] AtomT sum() const;
};

// Отдельные изолированные функции вычислений в соответствии с п. 4 задания
template <typename AtomT>
void run_vecred_nobr(VectorView<AtomT> vec, AtomT* d_out);

template <typename AtomT>
void run_vecred_br(VectorView<AtomT> vec, AtomT* d_out);

// Вспомогательные функции редукции полного вектора с возвратом скаляра
template <typename AtomT>
AtomT vecred_nobr(const Vector<AtomT>& vec);

template <typename AtomT>
AtomT vecred_br(const Vector<AtomT>& vec);