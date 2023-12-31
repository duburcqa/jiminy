#ifndef JIMINY_RANDOM_H
#define JIMINY_RANDOM_H

#include <optional>

#include "jiminy/core/fwd.h"


namespace jiminy
{
    // ************ Random number generator utilities ***************

    void JIMINY_DLLAPI resetRandomGenerators(
        const std::optional<uint32_t> & seed = std::nullopt) noexcept;

    hresult_t JIMINY_DLLAPI getRandomSeed(uint32_t & seed);

    double JIMINY_DLLAPI randUniform(double lo = 0.0, double hi = 1.0);

    double JIMINY_DLLAPI randNormal(double mean = 0.0, double std = 1.0);

    Eigen::VectorXd JIMINY_DLLAPI randVectorNormal(uint32_t size, double mean, double std);

    Eigen::VectorXd JIMINY_DLLAPI randVectorNormal(uint32_t size, double std);

    Eigen::VectorXd JIMINY_DLLAPI randVectorNormal(const Eigen::VectorXd & std);

    Eigen::VectorXd JIMINY_DLLAPI randVectorNormal(const Eigen::VectorXd & mean,
                                                   const Eigen::VectorXd & std);

    void JIMINY_DLLAPI shuffleIndices(std::vector<uint32_t> & vector);

    // ************ Continuous 1D Perlin processes ***************

    namespace details
    {
        /// \brief Lower Cholesky factor of a Toeplitz positive semi-definite matrix having 1.0 on
        /// its
        ///        main diagonal.
        ///
        /// \details In practice, it is advisable to combine this algorithm with Tikhonov
        ///          regularization of relative magnitude 1e-9 to avoid numerical instabilities
        ///          because of double machine precision.
        ///
        /// \see Michael Stewart, Cholesky factorization of semi-definite Toeplitz matrices. Linear
        ///      Algebra and its Applications, Volume 254, pages 497-525, 1997.
        ///
        /// \see
        /// https://people.sc.fsu.edu/~jburkardt/cpp_src/toeplitz_cholesky/toeplitz_cholesky.html
        ///
        /// \param[in] a First row of the matrix to decompose.
        template<typename DerivedType>
        MatrixX<typename DerivedType::Scalar>
        standardToeplitzCholeskyLower(const Eigen::MatrixBase<DerivedType> & coeffs)
        {
            using Scalar = typename DerivedType::Scalar;

            // Initialize lower Cholesky factor
            const Eigen::Index n = coeffs.size();
            MatrixX<Scalar> l{n, n};

            /* Compute compressed representation of the matrix.
               It coincides with the Schur generator for Toepliz matrices. */
            Eigen::Matrix<Scalar, 2, Eigen::Dynamic> g{2, n};
            g.rowwise() = coeffs.transpose();

            // Run progressive Schur algorithm, adapted to Toepliz matrices
            l.col(0) = g.row(0);
            g.row(0).tail(n - 1) = g.row(0).head(n - 1).eval();
            Eigen::Matrix<Scalar, 2, 2> H{Eigen::Matrix<Scalar, 2, 2>::Ones()};
            for (Eigen::Index i = 1; i < n; ++i)
            {
                const double rho = -g(1, i) / g(0, i);
                // H << 1.0, rho,
                //      rho, 1.0;
                Eigen::Map<Eigen::Vector4d>(H.data()).template segment<2>(1).fill(rho);
                g.rightCols(n - i) = H * g.rightCols(n - i) / std::sqrt((1.0 - rho) * (1.0 + rho));
                l.col(i).tail(n - i) = g.row(0).tail(n - i);
                g.row(0).tail(n - i - 1) = g.row(0).segment(i, n - i - 1).eval();
            }

            return l;
        }
    }

    class JIMINY_DLLAPI PeriodicGaussianProcess
    {
    public:
        DISABLE_COPY(PeriodicGaussianProcess)

    public:
        explicit PeriodicGaussianProcess(
            double wavelength, double period, double scale = 1.0) noexcept;

        void reset();

        double operator()(const float & t);

        double getWavelength() const;
        double getPeriod() const;
        double getDt() const;

    private:
        const double wavelength_;
        const double period_;
        const double scale_;
        const double dt_{0.02 * wavelength_};
        const int numTimes_{static_cast<int>(std::ceil(period_ / dt_))};

        Eigen::VectorXd values_{numTimes_};
        /// \brief Cholesky decomposition (LLT) of the covariance matrix.
        ///
        /// \details All decompositions are equivalent as the covariance matrix is symmetric,
        ///          namely eigen-values, singular-values, Cholesky and Schur decompositions.
        ///          Yet, Cholesky is by far the most efficient one.
        ///          See: https://math.stackexchange.com/q/22825/375496
        ///          Moreover, the covariance is a positive semi-definite Toepliz matrix, which
        ///          means that the computational complexity can be reduced even further using an
        ///          specialized Cholesky decomposition algorithm. */
        Eigen::MatrixXd covSqrtRoot_{
            details::standardToeplitzCholeskyLower(Eigen::VectorXd::NullaryExpr(
                numTimes_,
                [numTimes = numTimes_, wavelength = wavelength_](double i) {
                    return std::exp(-2.0 *
                                    std::pow(std::sin(M_PI / numTimes * i) / wavelength, 2));
                }))};
    };


    /// \see Based on "Smooth random functions, random ODEs, and Gaussian processes":
    ///      https://hal.inria.fr/hal-01944992/file/random_revision2.pdf */
    class JIMINY_DLLAPI PeriodicFourierProcess
    {
    public:
        DISABLE_COPY(PeriodicFourierProcess)

    public:
        explicit PeriodicFourierProcess(
            double wavelength, double period, double scale = 1.0) noexcept;

        void reset();

        double operator()(const float & t);

        double getWavelength() const;
        double getPeriod() const;
        int getNumHarmonics() const;
        double getDt() const;

    private:
        const double wavelength_;
        const double period_;
        const double scale_;
        const double dt_{0.02 * wavelength_};
        const int numTimes_{static_cast<int>(std::ceil(period_ / dt_))};
        const int numHarmonics_{static_cast<int>(std::ceil(period_ / wavelength_))};

        Eigen::VectorXd values_{numTimes_};
        const Eigen::MatrixXd cosMat_{
            Eigen::MatrixXd::NullaryExpr(numTimes_,
                                         numHarmonics_,
                                         [numTimes = numTimes_](double i, double j)
                                         { return std::cos(2 * M_PI / numTimes * i * j); })};
        const Eigen::MatrixXd sinMat_{
            Eigen::MatrixXd::NullaryExpr(numTimes_,
                                         numHarmonics_,
                                         [numTimes = numTimes_](double i, double j)
                                         { return std::cos(2 * M_PI / numTimes * i * j); })};
    };

    class JIMINY_DLLAPI AbstractPerlinNoiseOctave
    {
    public:
        explicit AbstractPerlinNoiseOctave(double wavelength, double scale) noexcept;
        virtual ~AbstractPerlinNoiseOctave() = default;

        virtual void reset();

        double operator()(double t) const;

        double getWavelength() const;
        double getScale() const;

    protected:
        // Copy on purpose
        virtual double grad(int32_t knot, double delta) const = 0;

        double fade(double delta) const;

        double lerp(double ratio, double yLeft, double yRight) const;

    protected:
        const double wavelength_;
        const double scale_;

        double shift_{0.0};
    };

    class JIMINY_DLLAPI RandomPerlinNoiseOctave : public AbstractPerlinNoiseOctave
    {
    public:
        using AbstractPerlinNoiseOctave::AbstractPerlinNoiseOctave;
        virtual ~RandomPerlinNoiseOctave() = default;

        virtual void reset() override final;

    protected:
        virtual double grad(int32_t knot, double delta) const override final;

    private:
        uint32_t seed_{0};
    };

    class JIMINY_DLLAPI PeriodicPerlinNoiseOctave : public AbstractPerlinNoiseOctave
    {
    public:
        explicit PeriodicPerlinNoiseOctave(double wavelength, double period, double scale);
        virtual ~PeriodicPerlinNoiseOctave() = default;

        virtual void reset() override final;

    protected:
        virtual double grad(int32_t knot, double delta) const override final;

    private:
        const double period_;

        std::array<uint8_t, 256> perm_{};
    };

    /// \brief  Sum of Perlin noise octaves.
    ///
    /// \details The original implementation uses fixed size permutation table to generate random
    ///          gradient directions. As a result, the generated process is inherently periodic,
    ///          which must be avoided. To circumvent this limitation, MurmurHash3 algorithm is
    ///          used to get random gradients at every point in time, without any periodicity, but
    ///          deterministically for a given seed. It is computationally more depending but not
    ///          critically slower.
    ///
    /// \sa  For technical references:
    ///      https://www.scratchapixel.com/lessons/procedural-generation-virtual-worlds/perlin-noise-part-2
    ///      https://adrianb.io/2014/08/09/perlinnoise.html
    ///      https://gamedev.stackexchange.com/a/23705/148509
    ///      https://gamedev.stackexchange.com/q/161923/148509
    ///      https://gamedev.stackexchange.com/q/134561/148509
    ///
    /// \sa  For reference about the implementation:
    ///      https://github.com/bradykieffer/SimplexNoise/blob/master/simplexnoise/noise.py
    ///      https://github.com/sol-prog/Perlin_Noise/blob/master/PerlinNoise.cpp
    ///      https://github.com/ashima/webgl-noise/blob/master/src/classicnoise2D.glsl
    class JIMINY_DLLAPI AbstractPerlinProcess
    {
    public:
        DISABLE_COPY(AbstractPerlinProcess)

    public:
        void reset();

        double operator()(const float & t);

        double getWavelength() const noexcept;
        std::size_t getNumOctaves() const noexcept;
        double getScale() const noexcept;

    protected:
        explicit AbstractPerlinProcess(
            double scale,
            std::vector<std::unique_ptr<AbstractPerlinNoiseOctave>> && octaves) noexcept;

    protected:
        const double scale_;

        std::vector<std::unique_ptr<AbstractPerlinNoiseOctave>> octaves_;
        double amplitude_{0.0};

        double grad_{0.0};
    };

    class JIMINY_DLLAPI RandomPerlinProcess : public AbstractPerlinProcess
    {
    public:
        explicit RandomPerlinProcess(
            double wavelength, double scale = 1.0, std::size_t numOctaves = 6U);
    };

    class PeriodicPerlinProcess : public AbstractPerlinProcess
    {
    public:
        explicit PeriodicPerlinProcess(
            double wavelength, double period, double scale = 1.0, std::size_t numOctaves = 6U);

        double getPeriod() const noexcept;

    private:
        const double period_;
    };

    // ************ Random terrain generators ***************

    HeightmapFunctor JIMINY_DLLAPI randomTileGround(const Eigen::Vector2d & size,
                                                    double heightMax,
                                                    const Eigen::Vector2d & interpDelta,
                                                    uint32_t sparsity,
                                                    double orientation,
                                                    uint32_t seed);

    HeightmapFunctor JIMINY_DLLAPI sumHeightmap(const std::vector<HeightmapFunctor> & heightmaps);
    HeightmapFunctor JIMINY_DLLAPI mergeHeightmap(
        const std::vector<HeightmapFunctor> & heightmaps);

    Eigen::MatrixXd JIMINY_DLLAPI discretizeHeightmap(
        const HeightmapFunctor & heightmap, double gridSize, double gridUnit);
}

#endif  // JIMINY_RANDOM_H
