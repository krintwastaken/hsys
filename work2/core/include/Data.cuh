#pragma once

#include <cstddef>
#include <utility>
#include <cuda_runtime.h>
#include "CudaCheck.cuh"

template <typename AtomT>
class Data {
private:
    std::size_t size_{0};
    AtomT* data_{nullptr};

public:
    explicit Data(std::size_t size) : size_(size) {
        if (size_ > 0) {
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&data_), size_ * sizeof(AtomT)));
        }
    }

    ~Data() {
        if (data_) {
            cudaFree(data_);
            data_ = nullptr;
        }
    }

    Data(const Data& other) : size_(other.size_) {
        if (size_ > 0) {
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&data_), size_ * sizeof(AtomT)));
            check_cuda(cudaMemcpy(data_, other.data_, size_ * sizeof(AtomT), cudaMemcpyDeviceToDevice));
        }
    }

    Data& operator=(const Data& other) {
        if (this != &other) {
            if (data_) {
                check_cuda(cudaFree(data_));
                data_ = nullptr;
            }
            size_ = other.size_;
            if (size_ > 0) {
                check_cuda(cudaMalloc(reinterpret_cast<void**>(&data_), size_ * sizeof(AtomT)));
                check_cuda(cudaMemcpy(data_, other.data_, size_ * sizeof(AtomT), cudaMemcpyDeviceToDevice));
            }
        }
        return *this;
    }

    Data(Data&& other) noexcept : size_(other.size_), data_(other.data_) {
        other.size_ = 0;
        other.data_ = nullptr;
    }

    Data& operator=(Data&& other) noexcept {
        if (this != &other) {
            if (data_) {
                cudaFree(data_);
            }
            size_ = other.size_;
            data_ = other.data_;
            other.size_ = 0;
            other.data_ = nullptr;
        }
        return *this;
    }

    [[nodiscard]] AtomT* data() noexcept {
        return data_;
    }

    [[nodiscard]] const AtomT* data() const noexcept {
        return data_;
    }

    [[nodiscard]] std::size_t size() const noexcept {
        return size_;
    }

    void copy_to_host(AtomT* host_ptr) const {
        if (size_ > 0 && host_ptr != nullptr) {
            check_cuda(cudaMemcpy(host_ptr, data_, size_ * sizeof(AtomT), cudaMemcpyDeviceToHost));
        }
    }

    void copy_from_host(const AtomT* host_ptr) {
        if (size_ > 0 && host_ptr != nullptr) {
            check_cuda(cudaMemcpy(data_, host_ptr, size_ * sizeof(AtomT), cudaMemcpyHostToDevice));
        }
    }
};