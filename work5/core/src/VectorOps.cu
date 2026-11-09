#include <algorithm>
#include "Vector.cuh"
#include "../../../work1/core/include/CudaCheck.cuh"

template <typename AtomT>
void run_vecred_nobr(VectorView<AtomT> vec, AtomT* d_out) {
    constexpr unsigned int BLOCK_SIZE = 256;
    const unsigned int num_blocks = static_cast<unsigned int>(std::min<std::size_t>(
        (vec.size() + BLOCK_SIZE - 1) / BLOCK_SIZE,
        1024
    ));
    const unsigned int grid_size = (num_blocks == 0) ? 1 : num_blocks;

    check_cuda(cudaMemsetAsync(d_out, 0, sizeof(AtomT)));
    kernel_vecred_nobr<AtomT, BLOCK_SIZE><<<grid_size, BLOCK_SIZE>>>(vec, d_out);
}

template <typename AtomT>
void run_vecred_br(VectorView<AtomT> vec, AtomT* d_out) {
    constexpr unsigned int BLOCK_SIZE = 256;
    const unsigned int num_blocks = static_cast<unsigned int>(std::min<std::size_t>(
        (vec.size() + BLOCK_SIZE - 1) / BLOCK_SIZE,
        1024
    ));
    const unsigned int grid_size = (num_blocks == 0) ? 1 : num_blocks;

    check_cuda(cudaMemsetAsync(d_out, 0, sizeof(AtomT)));
    kernel_vecred_br<AtomT, BLOCK_SIZE><<<grid_size, BLOCK_SIZE>>>(vec, d_out);
}

template <typename AtomT>
AtomT vecred_nobr(const Vector<AtomT>& vec) {
    if (vec.size() == 0) {
        return static_cast<AtomT>(0);
    }
    Data<AtomT> d_out(1);
    run_vecred_nobr(vec.view(), d_out.data());
    check_cuda(cudaDeviceSynchronize());
    AtomT res{};
    d_out.copy_to_host(&res);
    return res;
}

template <typename AtomT>
AtomT vecred_br(const Vector<AtomT>& vec) {
    if (vec.size() == 0) {
        return static_cast<AtomT>(0);
    }
    Data<AtomT> d_out(1);
    run_vecred_br(vec.view(), d_out.data());
    check_cuda(cudaDeviceSynchronize());
    AtomT res{};
    d_out.copy_to_host(&res);
    return res;
}

template <typename AtomT>
AtomT Vector<AtomT>::sum() const {
    if (size() == 0) {
        return static_cast<AtomT>(0);
    }
    Data<AtomT> d_out(1);
    get_strategy()->reduce(view(), d_out.data());
    AtomT res{};
    d_out.copy_to_host(&res);
    return res;
}

// Явные инстанцирования шаблонов
template void run_vecred_nobr<float>(VectorView<float>, float*);
template void run_vecred_nobr<double>(VectorView<double>, double*);
template void run_vecred_nobr<int>(VectorView<int>, int*);

template void run_vecred_br<float>(VectorView<float>, float*);
template void run_vecred_br<double>(VectorView<double>, double*);
template void run_vecred_br<int>(VectorView<int>, int*);

template float vecred_nobr<float>(const Vector<float>&);
template double vecred_nobr<double>(const Vector<double>&);
template int vecred_nobr<int>(const Vector<int>&);

template float vecred_br<float>(const Vector<float>&);
template double vecred_br<double>(const Vector<double>&);
template int vecred_br<int>(const Vector<int>&);

template float Vector<float>::sum() const;
template double Vector<double>::sum() const;
template int Vector<int>::sum() const;