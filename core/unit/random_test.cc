#include <gmock/gmock.h>  // `testing::*`
#include <gtest/gtest.h>

#include "jiminy/core/utilities/random.h"


using namespace jiminy;

static inline constexpr double DELTA{1e-6};
static inline constexpr double TOL{1e-4};


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


TEST(PerlinNoiseTest, RandomPerlinNoiseOctaveInitialization)
{
    double wavelength = 10.0;
    RandomPerlinNoiseOctave<1> octave(wavelength);
    octave.reset(PCG32{std::seed_seq{0}});

    EXPECT_DOUBLE_EQ(octave.getWavelength(), wavelength);
}

TEST(PerlinNoiseTest, PeriodicPerlinNoiseOctaveInitialization)
{
    double wavelength = 10.0;
    double period = 20.0;
    PeriodicPerlinNoiseOctave<1> octave(wavelength, period);
    octave.reset(PCG32{std::seed_seq{0}});

    EXPECT_DOUBLE_EQ(octave.getWavelength(), wavelength);
    EXPECT_DOUBLE_EQ(octave.getPeriod(), period);
}

TEST(PerlinNoiseTest, RandomGradientCalculation1D)
{
    {
        double wavelength = 10.0;
        RandomPerlinNoiseOctave<1> octave(wavelength);
        octave.reset(PCG32{std::seed_seq{0}});

        Eigen::Array<double, 1, 1> t{5.43};
        Eigen::Matrix<double, 1, 1> finiteDiffGrad{(octave(t + DELTA) - octave(t - DELTA)) /
                                                   (2 * DELTA)};
        ASSERT_TRUE(finiteDiffGrad.isApprox(octave.grad(t), TOL));
    }
    {
        double wavelength = 3.41;
        RandomPerlinNoiseOctave<1> octave(wavelength);
        octave.reset(PCG32{std::seed_seq{0}});

        Eigen::Array<double, 1, 1> t{17.0};
        Eigen::Matrix<double, 1, 1> finiteDiffGrad{(octave(t + DELTA) - octave(t - DELTA)) /
                                                   (2 * DELTA)};
        ASSERT_TRUE(finiteDiffGrad.isApprox(octave.grad(t), TOL));
    }
}

TEST(PerlinNoiseTest, PeriodicGradientCalculation1D)
{
    double wavelength = 10.0;
    double period = 30.0;
    PeriodicPerlinNoiseOctave<1> octave(wavelength, period);
    octave.reset(PCG32{std::seed_seq{0}});

    Eigen::Array<double, 1, 1> t{5.43};
    Eigen::Matrix<double, 1, 1> finiteDiffGrad{(octave(t + DELTA) - octave(t - DELTA)) /
                                               (2 * DELTA)};
    ASSERT_TRUE(finiteDiffGrad.isApprox(octave.grad(t), TOL));
}


TEST(PerlinNoiseTest, RandomGradientCalculation2D)
{
    double wavelength = 10.0;
    RandomPerlinNoiseOctave<2> octave(wavelength);
    octave.reset(PCG32{std::seed_seq{0}});

    Eigen::Vector2d pos{5.43, 7.12};
    Eigen::Vector2d finiteDiffGrad{(octave(pos + DELTA * Eigen::Vector2d::UnitX()) -
                                    octave(pos - DELTA * Eigen::Vector2d::UnitX())) /
                                       (2 * DELTA),
                                   (octave(pos + DELTA * Eigen::Vector2d::UnitY()) -
                                    octave(pos - DELTA * Eigen::Vector2d::UnitY())) /
                                       (2 * DELTA)};
    ASSERT_TRUE(finiteDiffGrad.isApprox(octave.grad(pos), TOL));
}


TEST(PerlinNoiseTest, PeriodicGradientCalculation2D)
{
    double wavelength = 10.0;
    double period = 30.0;
    PeriodicPerlinNoiseOctave<2> octave(wavelength, period);
    octave.reset(PCG32{std::seed_seq{0}});

    Eigen::Vector2d pos{5.43, 7.12};
    Eigen::Vector2d finiteDiffGrad{(octave(pos + DELTA * Eigen::Vector2d::UnitX()) -
                                    octave(pos - DELTA * Eigen::Vector2d::UnitX())) /
                                       (2 * DELTA),
                                   (octave(pos + DELTA * Eigen::Vector2d::UnitY()) -
                                    octave(pos - DELTA * Eigen::Vector2d::UnitY())) /
                                       (2 * DELTA)};
    ASSERT_TRUE(finiteDiffGrad.isApprox(octave.grad(pos), TOL));
}
