#ifndef JIMINY_RANDOM_HXX
#define JIMINY_RANDOM_HXX

#include <utility>  // `std::forward`

#include "jiminy/core/fwd.h"


namespace jiminy
{
    static inline constexpr double PERLIN_NOISE_PERSISTENCE{1.50};
    static inline constexpr double PERLIN_NOISE_LACUNARITY{0.85};

    // ***************************** Uniform random bit generators ***************************** //

    namespace internal
    {
        template<std::size_t I = 0UL, typename SeedSeq>
        inline uint64_t generateState(SeedSeq && seedSeq) noexcept
        {
            constexpr std::size_t kN = I + 1UL;
            constexpr std::size_t kDestSize = sizeof(uint64_t);
            constexpr std::size_t kGenSize = sizeof(uint32_t);
            constexpr std::size_t kSrcBits = kGenSize * 8;
            constexpr std::size_t kScale = (kDestSize + kGenSize - 1) / kGenSize;
            constexpr std::size_t kFromElems = kN * kScale;

            std::array<uint64_t, kN> result;
            std::array<uint32_t, kFromElems> buffer;
            seedSeq.generate(buffer.begin(), buffer.end());
            auto bufferIt = buffer.cbegin();
            for (uint64_t & dest : result)
            {
                uint64_t value{0UL};
                uint32_t shift{0U};
                for (std::size_t j = 0; j < kScale; ++j)
                {
                    value |= static_cast<uint64_t>(*(bufferIt++)) << shift;
                    shift += static_cast<uint32_t>(kSrcBits);
                }
                dest = value;
            }
            return result[I];
        }
    }

    template<typename SeedSeq, typename, typename>
    PCG32::PCG32(SeedSeq && seedSeq) noexcept :
    PCG32(internal::generateState(seedSeq))
    {
    }

    template<typename SeedSeq>
    void PCG32::seed(SeedSeq && seedSeq) noexcept
    {
        new (this) PCG32(std::forward<SeedSeq>(seedSeq));
    }

    // ****************************** Random number distributions ****************************** //

    template<typename Generator, typename Derived1, typename Derived2>
    std::enable_if_t<
        (is_eigen_any_v<Derived1> || is_eigen_any_v<Derived2>) &&
            (!std::is_arithmetic_v<std::decay_t<Derived1>> ||
             !std::is_arithmetic_v<std::decay_t<Derived2>>),
        Eigen::CwiseNullaryOp<
            scalar_random_op<float(const uniform_random_bit_generator_ref<uint32_t> &, float, float),
                             Generator &,
                             Derived1,
                             Derived2>,
            Eigen::MatrixXf>>
    uniform(Generator & g, Derived1 && lo, Derived2 && hi)
    {
        Eigen::Index nrows, ncols;
        if constexpr (is_eigen_any_v<Derived1>)
        {
            nrows = lo.rows();
            ncols = lo.cols();
        }
        else
        {
            nrows = hi.rows();
            ncols = hi.cols();
        }
        scalar_random_op<float(const uniform_random_bit_generator_ref<uint32_t> &, float, float),
                         Generator &,
                         Derived1,
                         Derived2>
            op{uniform, g, lo, hi};
        return Eigen::MatrixXf::NullaryExpr(nrows, ncols, op);
    }

    template<typename Generator>
    Eigen::CwiseNullaryOp<
        scalar_random_op<float(const uniform_random_bit_generator_ref<uint32_t> &, float, float),
                         Generator &,
                         const Eigen::MatrixXf::ConstantReturnType,
                         const Eigen::MatrixXf::ConstantReturnType>,
        Eigen::MatrixXf>
    uniform(Eigen::Index nrows, Eigen::Index ncols, Generator & g, float lo, float hi)
    {
        return uniform(g,
                       Eigen::MatrixXf::Constant(nrows, ncols, lo),
                       Eigen::MatrixXf::Constant(nrows, ncols, hi));
    }

    template<typename Generator, typename Derived1, typename Derived2>
    std::enable_if_t<
        (is_eigen_any_v<Derived1> || is_eigen_any_v<Derived2>) &&
            (!std::is_arithmetic_v<std::decay_t<Derived1>> ||
             !std::is_arithmetic_v<std::decay_t<Derived2>>),
        Eigen::CwiseNullaryOp<
            scalar_random_op<float(const uniform_random_bit_generator_ref<uint32_t> &, float, float),
                             Generator &,
                             Derived1,
                             Derived2>,
            Eigen::MatrixXf>>
    normal(Generator & g, Derived1 && mean, Derived2 && stddev)
    {
        Eigen::Index nrows, ncols;
        if constexpr (is_eigen_any_v<Derived1>)
        {
            nrows = mean.rows();
            ncols = mean.cols();
        }
        else
        {
            nrows = stddev.rows();
            ncols = stddev.cols();
        }
        scalar_random_op<float(const uniform_random_bit_generator_ref<uint32_t> &, float, float),
                         Generator &,
                         Derived1,
                         Derived2>
            op{normal, g, mean, stddev};
        return Eigen::MatrixXf::NullaryExpr(nrows, ncols, op);
    }

    template<typename Generator>
    Eigen::CwiseNullaryOp<
        scalar_random_op<float(const uniform_random_bit_generator_ref<uint32_t> &, float, float),
                         Generator &,
                         const Eigen::MatrixXf::ConstantReturnType,
                         const Eigen::MatrixXf::ConstantReturnType>,
        Eigen::MatrixXf>
    normal(Eigen::Index nrows, Eigen::Index ncols, Generator & g, float mean, float stddev)
    {
        return normal(g,
                      Eigen::MatrixXf::Constant(nrows, ncols, mean),
                      Eigen::MatrixXf::Constant(nrows, ncols, stddev));
    }

    // **************************** Continuous 1D Gaussian processes *************************** //

    namespace internal
    {
        template<typename Derived>
        MatrixX<typename Derived::Scalar>
        standardToeplitzCholeskyLower(const Eigen::MatrixBase<Derived> & coeffs, double reg)
        {
            using Scalar = typename Derived::Scalar;

            // Initialize lower Cholesky factor
            const Eigen::Index n = coeffs.size();
            MatrixX<Scalar> l{n, n};

            /* Compute compressed representation of the matrix.
               It coincides with the Schur generator for Toepliz matrices. */
            Eigen::Matrix<Scalar, 2, Eigen::Dynamic> g{2, n};
            g.rowwise() = coeffs.transpose();
            g(0, 0) += reg;

            // Run progressive Schur algorithm, adapted to Toepliz matrices
            l.col(0) = g.row(0);
            g.row(0).tail(n - 1) = g.row(0).head(n - 1).eval();
            Eigen::Matrix<Scalar, 2, 2> H{Eigen::Matrix<Scalar, 2, 2>::Ones()};
            for (Eigen::Index i = 1; i < n; ++i)
            {
                const double rho = -g(1, i) / g(0, i);
                // H << 1.0, rho,
                //      rho, 1.0;
                Eigen::Vector4d::Map(H.data()).template segment<2>(1).fill(rho);
                g.rightCols(n - i) = H * g.rightCols(n - i) / std::sqrt((1.0 - rho) * (1.0 + rho));
                l.col(i).tail(n - i) = g.row(0).tail(n - i);
                g.row(0).tail(n - i - 1) = g.row(0).segment(i, n - i - 1).eval();
            }

            return l;
        }
    }

    // ****************************** Continuous Perlin processes ****************************** //

    /// \brief Improved Smoothstep function by Ken Perlin (aka Smootherstep).
    ///
    /// \details It has zero 1st and 2nd-order derivatives at dt = 0.0, and 1.0.
    ///
    /// \sa For reference, see:
    ///     https://en.wikipedia.org/wiki/Smoothstep#Variations
    static inline double fade(double delta) noexcept
    {
        return delta * delta * delta * (delta * (delta * 6.0 - 15.0) + 10.0);
    }

    static inline double derivativeFade(double delta) noexcept
    {
        return 30.0 * delta * delta * (delta * (delta - 2.0) + 1.0);
    }

    template<typename T1, typename T2>
    static std::common_type_t<std::decay_t<T1>, std::decay_t<T2>>
    lerp(double ratio, T1 && yLeft, T2 && yRight) noexcept
    {
        return yLeft + ratio * (yRight - yLeft);
    }

    template<typename T1, typename T2>
    static std::common_type_t<std::decay_t<T1>, std::decay_t<T2>>
    derivativeLerp(double dratio, T1 && yLeft, T2 && yRight) noexcept
    {
        return dratio * (yRight - yLeft);
    }

    template<template<unsigned int> class DerivedPerlinNoiseOctave, unsigned int N>
    AbstractPerlinNoiseOctave<DerivedPerlinNoiseOctave, N>::AbstractPerlinNoiseOctave(
        double wavelength) :
    wavelength_{wavelength}
    {
        if (wavelength_ <= 0.0)
        {
            JIMINY_THROW(std::invalid_argument, "'wavelength' must be strictly larger than 0.0.");
        }
        reset(std::random_device{});
    }

    template<template<unsigned int> class DerivedPerlinNoiseOctave, unsigned int N>
    void AbstractPerlinNoiseOctave<DerivedPerlinNoiseOctave, N>::reset(
        const uniform_random_bit_generator_ref<uint32_t> & g) noexcept
    {
        // Sample random cell shift
        shift_ = uniform(N, 1, g).cast<double>();

        // Clear cache index
        cellIndex_.setConstant(std::numeric_limits<int32_t>::max());
    }

    template<template<unsigned int> class DerivedPerlinNoiseOctave, unsigned int N>
    double AbstractPerlinNoiseOctave<DerivedPerlinNoiseOctave, N>::getWavelength() const noexcept
    {
        return wavelength_;
    }

    template<template<unsigned int> class DerivedPerlinNoiseOctave, unsigned int N>
    template<bool isGradient>
    std::conditional_t<
        isGradient,
        typename AbstractPerlinNoiseOctave<DerivedPerlinNoiseOctave, N>::template VectorN<double>,
        double>
    AbstractPerlinNoiseOctave<DerivedPerlinNoiseOctave, N>::evaluate(
        const VectorN<double> & x) const
    {
        // Get current cell
        const VectorN<double> cell = x / wavelength_ + shift_;

        // Compute the bottom left corner knot
        const VectorN<int32_t> cellIndexLeft = cell.array().floor().template cast<int32_t>();
        const VectorN<int32_t> cellIndexRight = cellIndexLeft.array() + 1;

        // Compute smoothed ratio of query point wrt to the bottom left corner knot
        const VectorN<double> deltaLeft = cell - cellIndexLeft.template cast<double>();
        const VectorN<double> deltaRight = deltaLeft.array() - 1.0;

        // Compute gradients at knots (on a meshgrid), then corresponding offsets at query point
        bool isCacheValid = (cellIndexLeft.array() == cellIndex_.array()).all();
        std::array<double, (1U << N)> offsets;
        if (isCacheValid)
        {
            VectorN<double> delta;
            for (uint32_t k = 0; k < (1U << N); k++)
            {
                // Mapping from index to knot
                for (uint32_t i = 0; i < N; i++)
                {
                    if (k & (1U << i))
                    {
                        delta[i] = deltaRight[i];
                    }
                    else
                    {
                        delta[i] = deltaLeft[i];
                    }
                }

                // Compute the offset at query point
                offsets[k] = gradKnots_[k].dot(delta);
            }
        }
        else
        {
            VectorN<int32_t> knot;
            VectorN<double> delta;
            const auto & derived = static_cast<const DerivedPerlinNoiseOctave<N> &>(*this);
            for (uint32_t k = 0; k < (1U << N); k++)
            {
                // Mapping from index to knot
                for (uint32_t i = 0; i < N; i++)
                {
                    if (k & (1U << i))
                    {
                        knot[i] = cellIndexRight[i];
                        delta[i] = deltaRight[i];
                    }
                    else
                    {
                        knot[i] = cellIndexLeft[i];
                        delta[i] = deltaLeft[i];
                    }
                }

                // Evaluate the gradient at knot
                gradKnots_[k] = derived.gradKnot(knot);

                // Compute the offset at query point
                offsets[k] = gradKnots_[k].dot(delta);
            }
        }

        // Update cache index
        cellIndex_ = cellIndexLeft;

        // Compute the derivative along each axis
        const VectorN<double> ratio = deltaLeft.array().unaryExpr(std::ref(fade));
        if constexpr (isGradient)
        {
            const VectorN<double> dratio = deltaLeft.array().unaryExpr(std::ref(derivativeFade));
            std::array<VectorN<double>, (1U << N)> _interpGrads = gradKnots_;
            for (int32_t i = N - 1; i >= 0; --i)
            {
                for (uint32_t k = 0; k < (1U << i); k++)
                {
                    VectorN<double> & gradLeft = _interpGrads[k];
                    const VectorN<double> gradRight = _interpGrads[k | (1U << i)];
                    gradLeft = lerp(ratio[i], gradLeft, gradRight);
                }
            }
            for (int32_t j = 0; j < static_cast<int32_t>(N); ++j)
            {
                std::array<double, (1U << N)> _interpOffsets = offsets;
                for (int32_t i = N - 1; i >= 0; --i)
                {
                    for (uint32_t k = 0; k < (1U << i); k++)
                    {
                        double & offsetLeft = _interpOffsets[k];
                        const double offsetRight = _interpOffsets[k | (1U << i)];
                        if (i == j)
                        {
                            offsetLeft = derivativeLerp(dratio[i], offsetLeft, offsetRight);
                        }
                        else
                        {
                            offsetLeft = lerp(ratio[i], offsetLeft, offsetRight);
                        }
                    }
                }
                _interpGrads[0][j] += _interpOffsets[0];
            }
            return _interpGrads[0] / wavelength_;
        }
        else
        {
            // Perform linear interpolation on each dimension recursively until to get a scalar
            for (int32_t i = N - 1; i >= 0; --i)
            {
                for (uint32_t k = 0; k < (1U << i); k++)
                {
                    double & offsetLeft = offsets[k];
                    const double offsetRight = offsets[k | (1U << i)];
                    offsetLeft = lerp(ratio[i], offsetLeft, offsetRight);
                }
            }
            return offsets[0];
        }
    }

    template<template<unsigned int> class DerivedPerlinNoiseOctave, unsigned int N>
    double AbstractPerlinNoiseOctave<DerivedPerlinNoiseOctave, N>::operator()(
        const VectorN<double> & x) const
    {
        return evaluate<false>(x);
    }

    template<template<unsigned int> class DerivedPerlinNoiseOctave, unsigned int N>
    typename AbstractPerlinNoiseOctave<DerivedPerlinNoiseOctave, N>::template VectorN<double>
    AbstractPerlinNoiseOctave<DerivedPerlinNoiseOctave, N>::grad(const VectorN<double> & x) const
    {
        return evaluate<true>(x);
    }

    template<unsigned int N>
    RandomPerlinNoiseOctave<N>::RandomPerlinNoiseOctave(double wavelength) :
    AbstractPerlinNoiseOctave<RandomPerlinNoiseOctave, N>(wavelength)
    {
        reset(std::random_device{});
    }

    template<unsigned int N>
    void RandomPerlinNoiseOctave<N>::reset(
        const uniform_random_bit_generator_ref<uint32_t> & g) noexcept
    {
        // Call base implementation
        AbstractPerlinNoiseOctave<RandomPerlinNoiseOctave, N>::reset(g);

        // Sample new random seed
        seed_ = g();
    }

    template<unsigned int N>
    typename RandomPerlinNoiseOctave<N>::template VectorN<double>
    RandomPerlinNoiseOctave<N>::gradKnot(const VectorN<int32_t> & knot) const noexcept
    {
        constexpr float fHashMax = static_cast<float>(std::numeric_limits<uint32_t>::max());

        // Compute knot hash
        uint32_t hash = xxHash(knot.data(), static_cast<int32_t>(sizeof(int32_t) * N), seed_);

        /* Generate random gradient uniformly distributed on n-ball.
           For technical reference, see:
           https://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/
        */
        if constexpr (N == 1)
        {
            // Sample random scalar in [0.0, 1.0)
            const float s = static_cast<float>(hash) / fHashMax;

            // Compute rescaled gradient between [-1.0, 1.0)
            return VectorN<double>{2.0 * s - 1.0};
        }
        else if constexpr (N == 2)
        {
            // Sample random vector on a 2-ball (disk) using
            // const double theta = 2 * M_PI * static_cast<float>(hash) / fHashMax;
            // hash = xxHash(&hash, sizeof(uint32_t), seed_);
            // const float radius = std::sqrt(static_cast<float>(hash) / fHashMax);
            // return VectorN<double>{radius * std::cos(theta), radius * std::sin(theta)};

            /* The rejection method is much fast in 2d because it does not involve complex math
               (sqrt, sincos) and the acceptance rate is high (~78%) compared to the cost of
               sampling random numbers using `xxHash`. */
            while (true)
            {
                const float x = 2 * static_cast<float>(hash) / fHashMax - 1.0F;
                hash = xxHash(&hash, sizeof(uint32_t), seed_);
                const float y = 2 * static_cast<float>(hash) / fHashMax - 1.0F;
                if (x * x + y * y <= 1.0F)
                {
                    return VectorN<double>{x, y};
                }
            }
        }
        else
        {
            // Generate a uniformly distributed random vector on n-sphere
            VectorN<double> dir;
            for (uint32_t i = 0; i < N; i += 2)
            {
                // Generate 2 uniformly distributed random variables
                const float u1 = static_cast<float>(hash) / fHashMax;
                hash = xxHash(&hash, sizeof(uint32_t), seed_);
                const float u2 = static_cast<float>(hash) / fHashMax;
                hash = xxHash(&hash, sizeof(uint32_t), seed_);

                // Apply Box-Mueller algorithm to deduce 2 normally distributed random variables
                const double theta = 2 * M_PI * u2;
                const float radius = std::sqrt(-2 * std::log(u1));
                dir[i] = radius * std::cos(theta);
                if (i + 1 < N)
                {
                    dir[i + 1] = radius * std::sin(theta);
                }
            }
            dir.normalize();

            // Sample radius
            const double radius = std::pow(static_cast<float>(hash) / fHashMax, 1.0 / N);

            // Return the resulting random vector on n-ball using Muller method
            return radius * dir;
        }
    }

    template<unsigned int N>
    PeriodicPerlinNoiseOctave<N>::PeriodicPerlinNoiseOctave(double wavelength, double period) :
    AbstractPerlinNoiseOctave<PeriodicPerlinNoiseOctave, N>(
        period / std::max(std::round(period / wavelength), 1.0)),
    period_{period}
    {
        // Make sure the period is larger than the wavelength
        if (period < wavelength)
        {
            JIMINY_THROW(std::invalid_argument, "'period' must be larger than 'wavelength'.");
        }

        // Initialize the pre-computed hash table
        reset(std::random_device{});
    }

    template<unsigned int N>
    void PeriodicPerlinNoiseOctave<N>::reset(
        const uniform_random_bit_generator_ref<uint32_t> & g) noexcept
    {
        // Call base implementation
        AbstractPerlinNoiseOctave<PeriodicPerlinNoiseOctave, N>::reset(g);

        // Re-initialize the pre-computed hash table
        for (auto & grad : grads_)
        {
            if constexpr (N == 1)
            {
                grad = VectorN<double>{uniform(g, -1.0F, 1.0F)};
            }
            else if constexpr (N == 2)
            {
                const double theta = 2 * M_PI * uniform(g);
                const float radius = std::sqrt(uniform(g));
                grad = VectorN<double>{radius * std::cos(theta), radius * std::sin(theta)};
            }
            else
            {
                const VectorN<double> dir = normal(N, 1, g).cast<double>().normalized();
                const double radius = std::pow(uniform(g), 1.0 / N);
                grad = radius * dir;
            }
        }
    }

    template<unsigned int N>
    typename PeriodicPerlinNoiseOctave<N>::template VectorN<double>
    PeriodicPerlinNoiseOctave<N>::gradKnot(const VectorN<int32_t> & knot) const noexcept
    {
        // Wrap knot is period interval
        int32_t index = 0;
        int32_t shift = 1;
        for (uint_fast8_t i = 0; i < N; ++i)
        {
            int32_t coord = knot[i] % size_;
            if (coord < 0)
            {
                coord += size_;
            }
            index += coord * shift;
            shift *= size_;
        }

        // Return the gradient
        return grads_[index];
    }

    template<unsigned int N>
    double PeriodicPerlinNoiseOctave<N>::getPeriod() const noexcept
    {
        return period_;
    }

    template<template<unsigned int> class DerivedPerlinNoiseOctave, unsigned int N>
    AbstractPerlinProcess<DerivedPerlinNoiseOctave, N>::AbstractPerlinProcess(
        std::vector<OctaveScalePair> && octaveScalePairs) noexcept :
    octaveScalePairs_(std::move(octaveScalePairs))
    {
        // Compute the scaling factor to keep values within range [-1.0, 1.0]
        double amplitudeSquared = 0.0;
        for (const OctaveScalePair & octaveScale : octaveScalePairs_)
        {
            // FIXME: replaced `std::get<N>` by placeholder `_` when moving to C++26 (P2169R4)
            amplitudeSquared += std::pow(std::get<1>(octaveScale), 2);
        }
        amplitude_ = std::sqrt(amplitudeSquared);
    }

    template<template<unsigned int> class DerivedPerlinNoiseOctave, unsigned int N>
    void AbstractPerlinProcess<DerivedPerlinNoiseOctave, N>::reset(
        const uniform_random_bit_generator_ref<uint32_t> & g) noexcept
    {
        // Reset octaves
        for (OctaveScalePair & octaveScale : octaveScalePairs_)
        {
            // FIXME: replaced `std::get<N>` by placeholder `_` when moving to C++26 (P2169R4)
            std::get<0>(octaveScale).reset(g);
        }
    }

    template<template<unsigned int> class DerivedPerlinNoiseOctave, unsigned int N>
    double
    AbstractPerlinProcess<DerivedPerlinNoiseOctave, N>::operator()(const VectorN<double> & x) const
    {
        // Compute sum of octaves' values
        double value = 0.0;
        for (const auto & [octave, scale] : octaveScalePairs_)
        {
            value += scale * octave(x);
        }

        // Scale sum by maximum amplitude
        return value / amplitude_;
    }

    template<template<unsigned int> class DerivedPerlinNoiseOctave, unsigned int N>
    typename AbstractPerlinProcess<DerivedPerlinNoiseOctave, N>::template VectorN<double>
    AbstractPerlinProcess<DerivedPerlinNoiseOctave, N>::grad(const VectorN<double> & x) const
    {
        // Compute sum of octaves' values
        VectorN<double> value = VectorN<double>::Zero();
        for (const auto & [octave, scale] : octaveScalePairs_)
        {
            value += scale * octave.grad(x);
        }

        // Scale sum by maximum amplitude
        return value / amplitude_;
    }

    template<template<unsigned int> class DerivedPerlinNoiseOctave, unsigned int N>
    double AbstractPerlinProcess<DerivedPerlinNoiseOctave, N>::getWavelength() const noexcept
    {
        double wavelength = INF;
        for (const OctaveScalePair & octaveScale : octaveScalePairs_)
        {
            // FIXME: replaced `std::get<N>` by placeholder `_` when moving to C++26 (P2169R4)
            wavelength = std::min(wavelength, std::get<0>(octaveScale).getWavelength());
        }
        return wavelength;
    }

    template<template<unsigned int> class DerivedPerlinNoiseOctave, unsigned int N>
    std::size_t AbstractPerlinProcess<DerivedPerlinNoiseOctave, N>::getNumOctaves() const noexcept
    {
        return octaveScalePairs_.size();
    }

    template<template<unsigned int> class DerivedPerlinNoiseOctave,
             unsigned int N,
             typename... ExtraArgs>
    static std::vector<std::pair<DerivedPerlinNoiseOctave<N>, const double>>
    buildPerlinNoiseOctaves(double wavelength, std::size_t numOctaves, ExtraArgs &&... args)
    {
        // Make sure that at least one octave has been requested
        if (numOctaves < 1)
        {
            JIMINY_THROW(std::invalid_argument, "'numOctaves' must at least 1.");
        }

        // Make sure that wavelength of all the octaves is consistent with period if application
        if constexpr (std::is_base_of_v<PeriodicPerlinNoiseOctave<N>, DerivedPerlinNoiseOctave<N>>)
        {
            const double period = std::get<0>(std::tuple{std::forward<ExtraArgs>(args)...});
            const double wavelengthFinal =
                wavelength / std::pow(PERLIN_NOISE_LACUNARITY, numOctaves - 1);
            if (period < std::max(wavelength, wavelengthFinal))
            {
                JIMINY_THROW(std::invalid_argument,
                             "'period' must be larger than the wavelength of all the octaves (",
                             std::max(wavelength, wavelengthFinal),
                             "), ie 'wavelength' / ",
                             PERLIN_NOISE_LACUNARITY,
                             "^i for i in [1, ..., 'numOctaves'].");
            }
        }

        std::vector<std::pair<DerivedPerlinNoiseOctave<N>, const double>> octaveScalePairs;
        octaveScalePairs.reserve(numOctaves);
        double scale = 1.0;
        for (std::size_t i = 0; i < numOctaves; ++i)
        {
            octaveScalePairs.emplace_back(DerivedPerlinNoiseOctave<N>(wavelength, args...), scale);
            wavelength /= PERLIN_NOISE_LACUNARITY;
            scale *= PERLIN_NOISE_PERSISTENCE;
        }
        return octaveScalePairs;
    }

    template<unsigned int N>
    RandomPerlinProcess<N>::RandomPerlinProcess(double wavelength, std::size_t numOctaves) :
    AbstractPerlinProcess<RandomPerlinNoiseOctave, N>(
        buildPerlinNoiseOctaves<RandomPerlinNoiseOctave, N>(wavelength, numOctaves))
    {
    }

    template<unsigned int N>
    PeriodicPerlinProcess<N>::PeriodicPerlinProcess(
        double wavelength, double period, std::size_t numOctaves) :
    AbstractPerlinProcess<PeriodicPerlinNoiseOctave, N>(
        buildPerlinNoiseOctaves<PeriodicPerlinNoiseOctave, N>(wavelength, numOctaves, period)),
    period_{period}
    {
    }

    template<unsigned int N>
    double PeriodicPerlinProcess<N>::getPeriod() const noexcept
    {
        return period_;
    }
}

#endif  // JIMINY_RANDOM_HXX
