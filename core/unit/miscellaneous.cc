#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "jiminy/core/fwd.h"
#include "jiminy/core/utilities/helpers.h"
#include "jiminy/core/utilities/random.h"


using namespace jiminy;


TEST(Miscellaneous, swapMatrixRows)
{
    {
        Eigen::MatrixXi matrixIn(2, 9);
        Eigen::MatrixXi matrixOut(2, 9);
        // clang-format off
        matrixIn << 0, 1, 2, 3, 4, 5, 6, 7, 8,
                    0, 1, 2, 3, 4, 5, 6, 7, 8;
        matrixOut << 5, 6, 7, 8, 3, 4, 0, 1, 2,
                     5, 6, 7, 8, 3, 4, 0, 1, 2;
        // clang-format on
        swapMatrixRows(matrixIn.transpose(), 0, 3, 5, 4);
        ASSERT_PRED2([](const Eigen::MatrixXi & lhs, const Eigen::MatrixXi & rhs)
                     { return lhs.isApprox(rhs); },
                     matrixIn,
                     matrixOut);
    }
    {
        Eigen::MatrixXi matrixIn(2, 9);
        Eigen::MatrixXi matrixOut(2, 9);
        // clang-format off
        matrixIn << 0, 1, 2, 3, 4, 5, 6, 7, 8,
                    0, 1, 2, 3, 4, 5, 6, 7, 8;
        matrixOut << 0, 1, 2, 8, 6, 7, 3, 4, 5,
                     0, 1, 2, 8, 6, 7, 3, 4, 5;
        // clang-format on
        swapMatrixRows(matrixIn.transpose(), 3, 3, 8, 1);
        ASSERT_PRED2([](const Eigen::MatrixXi & lhs, const Eigen::MatrixXi & rhs)
                     { return lhs.isApprox(rhs); },
                     matrixIn,
                     matrixOut);
    }
    {
        Eigen::MatrixXi matrixIn(2, 9);
        Eigen::MatrixXi matrixOut(2, 9);
        // clang-format off
        matrixIn << 0, 1, 2, 3, 4, 5, 6, 7, 8,
                    0, 1, 2, 3, 4, 5, 6, 7, 8;
        matrixOut << 0, 4, 5, 6, 7, 3, 1, 2, 8,
                     0, 4, 5, 6, 7, 3, 1, 2, 8;
        // clang-format on
        swapMatrixRows(matrixIn.transpose(), 1, 2, 4, 4);
        ASSERT_PRED2([](const Eigen::MatrixXi & lhs, const Eigen::MatrixXi & rhs)
                     { return lhs.isApprox(rhs); },
                     matrixIn,
                     matrixOut);
    }
}


TEST(Miscellaneous, MatrixRandom)
{
    std::mt19937 gen32{0};
    uniform_random_bit_generator_ref<uint32_t> gen32_ref = gen32;

    float mean = 5.0;
    float stddev = 2.0;

    auto mean_vec = Eigen::MatrixXf::Constant(1, 2, mean);
    auto stddev_vec = Eigen::MatrixXf::Constant(1, 2, stddev);
    float value1 = normal(gen32, mean, stddev);
    float value2 = normal(gen32, mean, stddev);

    {
        gen32.seed(0);
        scalar_random_op<float(std::mt19937 &, float, float)> op{
            [](auto & g, float mean, float stddev) -> float { return normal(g, mean, stddev); },
            gen32,
            mean_vec,
            stddev_vec};
        ASSERT_FLOAT_EQ(op(0, 0), value1);
        ASSERT_FLOAT_EQ(op(0, 1), value2);
        ASSERT_THAT(op(0, 0), testing::Not(testing::FloatEq(value1)));
    }
    {
        gen32.seed(0);
        scalar_random_op<float(const uniform_random_bit_generator_ref<uint32_t> &, float, float),
                         uniform_random_bit_generator_ref<uint32_t>,
                         float,
                         float>
            op{normal, gen32, mean, stddev};
        ASSERT_FLOAT_EQ(op(0, 0), value1);
        ASSERT_FLOAT_EQ(op(10, 10), value2);
        ASSERT_THAT(op(0, 0), testing::Not(testing::FloatEq(value1)));
    }
    {
        gen32.seed(0);
        auto mat_expr = normal(gen32_ref, mean_vec, stddev_vec);
        ASSERT_FLOAT_EQ(mat_expr(0, 0), value1);
        ASSERT_FLOAT_EQ(mat_expr(0, 1), value2);
        ASSERT_THAT(mat_expr(0, 0), testing::Not(testing::FloatEq(value1)));
    }
    {
        gen32.seed(0);
        auto mat_expr = normal(1, 2, gen32, mean, stddev);
        ASSERT_FLOAT_EQ(mat_expr(0, 0), value1);
        ASSERT_FLOAT_EQ(mat_expr(0, 1), value2);
        ASSERT_THAT(mat_expr(0, 0), testing::Not(testing::FloatEq(value1)));
    }
    {
        gen32.seed(0);
        auto mat_expr = normal(gen32_ref, mean, stddev_vec);
        ASSERT_FLOAT_EQ(mat_expr(0, 0), value1);
        ASSERT_FLOAT_EQ(mat_expr(1, 0), value2);
        ASSERT_THAT(mat_expr(0, 0), testing::Not(testing::FloatEq(value1)));
    }
}
