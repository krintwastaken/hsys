#include <gtest/gtest.h>

#include <complex>
using std::arg;

#define EIGEN_NO_CUDA
#define EIGEN_DONT_VECTORIZE
#include <Eigen/Dense>

#include <random>
#include "Vector.cuh"

class VecRedTest : public ::testing::TestWithParam<std::size_t> {};

TEST_P(VecRedTest, MatchesEigenResult) {
    const std::size_t n = GetParam();

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    Eigen::VectorXf eigen_vec(n);
    for (std::size_t i = 0; i < n; ++i) {
        eigen_vec[static_cast<Eigen::Index>(i)] = dist(gen);
    }

    Vector<float> gpu_vec(n);
    gpu_vec.copy_from_host(eigen_vec.data());

    const float expected_sum = eigen_vec.sum();

    // Проверка кёрнела (1) nobr
    const float res_nobr = vecred_nobr(gpu_vec);
    EXPECT_NEAR(res_nobr, expected_sum, 1e-4f);

    // Проверка кёрнела (2) br
    const float res_br = vecred_br(gpu_vec);
    EXPECT_NEAR(res_br, expected_sum, 1e-4f);
}

INSTANTIATE_TEST_SUITE_P(
    SpecifiedDimensions,
    VecRedTest,
    ::testing::Values(
        std::size_t{1},
        std::size_t{2},
        std::size_t{3},
        std::size_t{127},
        std::size_t{129},
        std::size_t{512},
        std::size_t{541},
        std::size_t{1037}
    )
);

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}