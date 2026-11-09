#include <gtest/gtest.h>

#include <complex>
using std::arg;

// Полное отключение CUDA-расширений и SIMD в Eigen для хостового кода
#define EIGEN_NO_CUDA
#define EIGEN_DONT_VECTORIZE
#include <Eigen/Dense>

#include <cuda_fp16.h>
#include <random>
#include <tuple>
#include <vector>
#include "Matrix.cuh"

using RowMatrixXhalf = Eigen::Matrix<Eigen::half, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

class MatMulWmmaTest : public ::testing::TestWithParam<std::tuple<std::size_t, std::size_t, std::size_t>> {};

TEST_P(MatMulWmmaTest, MatchesEigenResult) {
    const auto [m, k, n] = GetParam();

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    RowMatrixXhalf eigen_a(m, k);
    RowMatrixXhalf eigen_b(k, n);

    std::vector<half> host_a(m * k);
    std::vector<half> host_b(k * n);

    for (std::size_t r = 0; r < m; ++r) {
        for (std::size_t c = 0; c < k; ++c) {
            float val = dist(gen);
            eigen_a(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(c)) = static_cast<Eigen::half>(val);
            host_a[r * k + c] = __float2half(val);
        }
    }

    for (std::size_t r = 0; r < k; ++r) {
        for (std::size_t c = 0; c < n; ++c) {
            float val = dist(gen);
            eigen_b(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(c)) = static_cast<Eigen::half>(val);
            host_b[r * n + c] = __float2half(val);
        }
    }

    Matrix<half> gpu_a(m, k);
    Matrix<half> gpu_b(k, n);

    gpu_a.copy_from_host(host_a.data());
    gpu_b.copy_from_host(host_b.data());

    Matrix<half> gpu_c = gpu_a * gpu_b;

    // Эталонное перемножение в библиотеке Eigen на CPU
    RowMatrixXhalf eigen_res = eigen_a * eigen_b;
    Eigen::MatrixXf ref_f = eigen_res.cast<float>();

    std::vector<half> host_c(m * n);
    gpu_c.copy_to_host(host_c.data());

    Eigen::MatrixXf gpu_f(m, n);
    for (std::size_t r = 0; r < m; ++r) {
        for (std::size_t c = 0; c < n; ++c) {
            gpu_f(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(c)) =
                __half2float(host_c[r * n + c]);
        }
    }

    EXPECT_TRUE(gpu_f.isApprox(ref_f, 1e-2f));
}

INSTANTIATE_TEST_SUITE_P(
    ParametrizedDimensions,
    MatMulWmmaTest,
    ::testing::Combine(
        ::testing::Values(std::size_t{16}, std::size_t{32}, std::size_t{64}, std::size_t{128}, std::size_t{256}, std::size_t{512}),
        ::testing::Values(std::size_t{16}, std::size_t{32}, std::size_t{64}, std::size_t{128}, std::size_t{256}, std::size_t{512}),
        ::testing::Values(std::size_t{16}, std::size_t{32}, std::size_t{64}, std::size_t{128}, std::size_t{256}, std::size_t{512})
    )
);

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}