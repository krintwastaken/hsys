#pragma once

#include <cstddef>
#include <memory>
#include "Data.cuh"
#include "VectorView.cuh"

template <typename AtomT>
class Vector {
private:
    std::shared_ptr<Data<AtomT>> data_;
    VectorView<AtomT> view_;

public:
    explicit Vector(std::size_t size)
        : data_(std::make_shared<Data<AtomT>>(size)),
          view_(data_->data(), size) {}

    std::size_t size() const noexcept {
        return view_.size();
    }

    Data<AtomT>& data() noexcept {
        return *data_;
    }

    const Data<AtomT>& data() const noexcept {
        return *data_;
    }

    VectorView<AtomT>& view() noexcept {
        return view_;
    }

    const VectorView<AtomT>& view() const noexcept {
        return view_;
    }

    void copy_to_host(AtomT* host_ptr) const {
        data_->copy_to_host(host_ptr);
    }

    void copy_from_host(const AtomT* host_ptr) {
        data_->copy_from_host(host_ptr);
    }
};

template <typename AtomT>
Vector<AtomT> operator+(const Vector<AtomT>& a, const Vector<AtomT>& b);