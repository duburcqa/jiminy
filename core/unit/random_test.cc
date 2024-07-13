#include <gtest/gtest.h>

#include "jiminy/core/utilities/random.h"


using namespace jiminy;

static inline constexpr double DELTA{1e-6};
static inline constexpr double TOL{1e-3};


TEST(PerlinNoiseTest, RandomPerlinNoiseOctaveInitialization)
{
    double wavelength = 10.0;
    RandomPerlinNoiseOctave<1> randomOctave(wavelength);

    EXPECT_DOUBLE_EQ(randomOctave.getWavelength(), wavelength);
}

TEST(PerlinNoiseTest, PeriodicPerlinNoiseOctaveInitialization)
{
    double wavelength = 10.0;
    double period = 20.0;
    PeriodicPerlinNoiseOctave<1> periodicOctave(wavelength, period);

    EXPECT_DOUBLE_EQ(periodicOctave.getWavelength(), wavelength);
    EXPECT_DOUBLE_EQ(periodicOctave.getPeriod(), period);
}

TEST(PerlinNoiseTest, RandomGradientCalculation1D)
{
    {
        double wavelength = 10.0;
        RandomPerlinNoiseOctave<1> octave(wavelength);

        Eigen::Array<double, 1, 1> t{5.43};
        Eigen::Matrix<double, 1, 1> finiteDiffGrad{(octave(t + DELTA) - octave(t - DELTA)) /
                                                   (2 * DELTA)};
        ASSERT_TRUE(finiteDiffGrad.isApprox(octave.grad(t), TOL));
    }
    {
        double wavelength = 3.41;
        RandomPerlinNoiseOctave<1> octave(wavelength);

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

    Eigen::Array<double, 1, 1> t{5.43};
    Eigen::Matrix<double, 1, 1> finiteDiffGrad{(octave(t + DELTA) - octave(t - DELTA)) /
                                               (2 * DELTA)};
    ASSERT_TRUE(finiteDiffGrad.isApprox(octave.grad(t), TOL));
}


TEST(PerlinNoiseTest, RandomGradientCalculation2D)
{
    double wavelength = 10.0;
    RandomPerlinNoiseOctave<2> octave(wavelength);

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

    Eigen::Vector2d pos{5.43, 7.12};
    Eigen::Vector2d finiteDiffGrad{(octave(pos + DELTA * Eigen::Vector2d::UnitX()) -
                                    octave(pos - DELTA * Eigen::Vector2d::UnitX())) /
                                       (2 * DELTA),
                                   (octave(pos + DELTA * Eigen::Vector2d::UnitY()) -
                                    octave(pos - DELTA * Eigen::Vector2d::UnitY())) /
                                       (2 * DELTA)};
    ASSERT_TRUE(finiteDiffGrad.isApprox(octave.grad(pos), TOL));
}
