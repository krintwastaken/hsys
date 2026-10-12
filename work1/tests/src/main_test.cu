#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <vector>
#include <random>
#include "Vector.cuh"

class VecAddTest : public ::testing::TestWithParam<std::size_t> {};

TEST_P(VecAddTest, MatchesEigenResult) {
    const std::size_t n = GetParam();

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-100.0f, 100.0f);

    Eigen::VectorXf eigen_a(n);
    Eigen::VectorXf eigen_b(n);

    for (std::size_t i = 0; i < n; ++i) {
        eigen_a[static_cast<Eigen::Index>(i)] = dist(gen);
        eigen_b[static_cast<Eigen::Index>(i)] = dist(gen);
    }

    Vector<float> gpu_a(n);
    Vector<float> gpu_b(n);

    gpu_a.copy_from_host(eigen_a.data());
    gpu_b.copy_from_host(eigen_b.data());

    Vector<float> gpu_res = gpu_a + gpu_b;

    Eigen::VectorXf eigen_res = eigen_a + eigen_b;

    Eigen::VectorXf gpu_res_host(n);
    gpu_res.copy_to_host(gpu_res_host.data());

    EXPECT_TRUE(gpu_res_host.isApprox(eigen_res, 1e-6f));
}

INSTANTIATE_TEST_SUITE_P(
    VaryingSizes,
    VecAddTest,
    ::testing::Values(
        std::size_t{1},
        std::size_t{2},
        std::size_t{3},
        std::size_t{127},
        std::size_t{128},
        std::size_t{129},
        std::size_t{512},
        std::size_t{1024},
        std::size_t{1029}
    )
);

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}