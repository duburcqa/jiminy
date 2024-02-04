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
        Eigen::VectorXi matrixIn(9);
        Eigen::VectorXi matrixOut(9);
        matrixIn << 0, 1, 2, 3, 4, 5, 6, 7, 8;
        matrixOut << 0, 6, 7, 8, 2, 3, 4, 5, 1;
        swapMatrixRows(matrixIn, 1, 1, 6, 3);
        ASSERT_PRED2([](const Eigen::VectorXi & lhs, const Eigen::VectorXi & rhs)
                     { return lhs.isApprox(rhs); },
                     matrixIn,
                     matrixOut);
    }
    {
        Eigen::VectorXi matrixIn(9);
        Eigen::VectorXi matrixOut(9);
        matrixIn << 0, 1, 2, 3, 4, 5, 6, 7, 8;
        matrixOut << 6, 7, 8, 1, 2, 3, 4, 5, 0;
        swapMatrixRows(matrixIn, 0, 1, 6, 3);
        ASSERT_PRED2([](const Eigen::VectorXi & lhs, const Eigen::VectorXi & rhs)
                     { return lhs.isApprox(rhs); },
                     matrixIn,
                     matrixOut);
    }
    {
        Eigen::VectorXi matrixIn(9);
        Eigen::VectorXi matrixOut(9);
        matrixIn << 0, 1, 2, 3, 4, 5, 6, 7, 8;
        matrixOut << 0, 1, 2, 8, 6, 7, 3, 4, 5;
        swapMatrixRows(matrixIn, 3, 3, 8, 1);
        ASSERT_PRED2([](const Eigen::VectorXi & lhs, const Eigen::VectorXi & rhs)
                     { return lhs.isApprox(rhs); },
                     matrixIn,
                     matrixOut);
    }
    {
        Eigen::VectorXi matrixIn(9);
        Eigen::VectorXi matrixOut(9);
        matrixIn << 0, 1, 2, 3, 4, 5, 6, 7, 8;
        matrixOut << 0, 4, 5, 6, 7, 3, 1, 2, 8;
        swapMatrixRows(matrixIn, 1, 2, 4, 4);
        ASSERT_PRED2([](const Eigen::VectorXi & lhs, const Eigen::VectorXi & rhs)
                     { return lhs.isApprox(rhs); },
                     matrixIn,
                     matrixOut);
    }
}


TEST(Miscellaneous, MatrixRandom)
{
    using generator_t =
        std::independent_bits_engine<std::mt19937, std::numeric_limits<uint32_t>::digits, uint32_t>;

    generator_t gen32{0};
    uniform_random_bit_generator_ref<uint32_t> gen32_ref = gen32;

    float mean = 5.0;
    float stddev = 2.0;

    auto mean_vec = Eigen::MatrixXf::Constant(1, 2, mean);
    auto stddev_vec = Eigen::MatrixXf::Constant(1, 2, stddev);
    float value1 = normal(gen32, mean, stddev);
    float value2 = normal(gen32, mean, stddev);

    {
        gen32.seed(0);
        scalar_random_op<float(generator_t &, float, float)> op{
            [](auto & g, float _mean, float _stddev) -> float
            { return normal(g, _mean, _stddev); },
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
        auto mat_expr = normal(gen32_ref, mean, stddev_vec.transpose());
        ASSERT_FLOAT_EQ(mat_expr(0, 0), value1);
        ASSERT_FLOAT_EQ(mat_expr(1, 0), value2);
        ASSERT_THAT(mat_expr(0, 0), testing::Not(testing::FloatEq(value1)));
    }
}
