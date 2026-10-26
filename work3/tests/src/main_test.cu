#include <gtest/gtest.h>

#include <complex>
using std::arg;

#include <Eigen/Dense>
#include <random>
#include <tuple>
#include "Matrix.cuh"

using RowMatrixXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

class MatMulShmemTest : public ::testing::TestWithParam<std::tuple<std::size_t, std::size_t, std::size_t>> {};

TEST_P(MatMulShmemTest, MatchesEigenResult) {
    const auto [m, k, n] = GetParam();

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    RowMatrixXf eigen_a(m, k);
    RowMatrixXf eigen_b(k, n);

    for (std::size_t r = 0; r < m; ++r) {
        for (std::size_t c = 0; c < k; ++c) {
            eigen_a(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(c)) = dist(gen);
        }
    }

    for (std::size_t r = 0; r < k; ++r) {
        for (std::size_t c = 0; c < n; ++c) {
            eigen_b(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(c)) = dist(gen);
        }
    }

    Matrix<float> gpu_a(m, k);
    Matrix<float> gpu_b(k, n);

    gpu_a.copy_from_host(eigen_a.data());
    gpu_b.copy_from_host(eigen_b.data());

    Matrix<float> gpu_c = gpu_a * gpu_b;

    Eigen::MatrixXf ref_a = eigen_a;
    Eigen::MatrixXf ref_b = eigen_b;
    Eigen::MatrixXf ref_c = ref_a * ref_b;

    RowMatrixXf gpu_c_host(m, n);
    gpu_c.copy_to_host(gpu_c_host.data());

    EXPECT_TRUE(gpu_c_host.isApprox(ref_c, 1e-5f));
}

INSTANTIATE_TEST_SUITE_P(
    ParametrizedDimensions,
    MatMulShmemTest,
    ::testing::Combine(
        ::testing::Values(std::size_t{1}, std::size_t{2}, std::size_t{3}, std::size_t{127}, std::size_t{128}, std::size_t{129}, std::size_t{512}),
        ::testing::Values(std::size_t{1}, std::size_t{2}, std::size_t{3}, std::size_t{127}, std::size_t{128}, std::size_t{129}, std::size_t{512}),
        ::testing::Values(std::size_t{1}, std::size_t{2}, std::size_t{3}, std::size_t{127}, std::size_t{128}, std::size_t{129}, std::size_t{512})
    )
);

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}