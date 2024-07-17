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
