#ifndef JIMINY_RANDOM_H
#define JIMINY_RANDOM_H

#include <array>     // `std::array`
#include <memory>    // `std::unique_ptr`
#include <optional>  // `std::optional`
#include <utility>   // `std::pair`, `std::declval`
#include <vector>    // `std::vector`

#include <type_traits>  // `std::enable_if_t`, `std::is_same_v`, `std::is_convertible_v`, `std::is_unsigned_v`

#include "jiminy/core/fwd.h"  // `remove_cvref_t`


namespace jiminy
{
    // ***************************** Uniform random bit generators ***************************** //

    /// @brief 32-bits Permuted Congruential Generator (PCG) random number generator. This
    ///        generator has excellent statistical quality while being much faster than Mersenne
    ///        Twister (std::mt19937) and having acceptable period (2^62).
    ///
    /// @details The PCG random number generation scheme has been developed by Melissa O'Neill. The
    ///          technical details can be found in his origin paper, "PCG: A Family of Simple Fast
    ///          Space-Efficient Statistically Good Algorithms for Random Number Generation.",
    ///          2014. For additional information, please visit: http://www.pcg-random.org
    ///
    /// @warning This implementation has not been vectorized by leveraging SIMD instructions. As
    ///          such, all platforms are supported out-of-the-box and yield reproducible results,
    ///          at the cost of running significantly slower to when it comes to sampling enough
    ///          data at once (e.g. AVX512 operates on 16 floats at once). This use-case is largely
    ///          irrelevant at the time being for this project, and as such, is out of its scope.
    ///          For completeness, here is a another fully vectorized implementation:
    ///          https://github.com/lemire/simdpcg
    ///
    /// @sa The proposed implementation is a minimal version of `pcg32_fast` released under the
    ///     Apache License, Version 2.0: https://github.com/imneme/pcg-cpp
    class JIMINY_DLLAPI PCG32
    {
    public:
        using result_type = uint32_t;

        explicit PCG32(uint64_t state = 0xcafef00dd15ea5e5ULL) noexcept;
        template<typename SeedSeq,
                 typename = decltype(std::declval<SeedSeq &>().generate(
                     std::declval<uint_least32_t *>(), std::declval<uint_least32_t *>())),
                 typename = std::enable_if_t<!std::is_same_v<remove_cvref_t<SeedSeq>, PCG32> &&
                                             !std::is_convertible_v<SeedSeq, uint64_t> &&
                                             std::is_unsigned_v<typename SeedSeq::result_type>>>
        explicit PCG32(SeedSeq && seedSeq) noexcept;

        explicit PCG32(PCG32 && other) = default;
        explicit PCG32(PCG32 & other) = default;
        PCG32 & operator=(const PCG32 & other) = default;

        template<typename SeedSeq>
        void seed(SeedSeq && seedSeq) noexcept;

        result_type operator()() noexcept;

        constexpr static result_type min() noexcept
        {
            return std::numeric_limits<result_type>::min();
        }
        constexpr static result_type max() noexcept
        {
            return std::numeric_limits<result_type>::max();
        }

    private:
        uint64_t state_;
    };

    /// \brief Lightweight non-owning wrapper around a random generator without seeding capability.
    ///
    /// \details It allows for type erasure of the underlying random generator, at the cost of
    ///          having to enforce return_type and min/max bounds at compile time. This enables
    ///          supporting passing any random generator to virtual methods of polymorphic classes.
    ///          For convenience, the exposed 'operator()' is const-qualified, which implies that
    ///          sampling from a 'const uniform_random_bit_generator_ref<T> &' is permitted.
    ///
    /// \note The user is responsible to adapt the return type of a random generator if it does not
    ///       match the requirement of this wrapper. `std::independent_bits_engine` adaptor is the
    ///       right way to do so if copying the original generator is not an issue.
    ///
    /// \sa For technical reference about type-erasure for random generators:
    ///     https://stackoverflow.com/a/77809228/4820605
    template<typename ResultType,
             ResultType min_ = std::numeric_limits<ResultType>::min(),
             ResultType max_ = std::numeric_limits<ResultType>::max()>
    class uniform_random_bit_generator_ref : private function_ref<ResultType()>
    {
    public:
        using result_type = ResultType;

        template<typename F
                 //  ,typename = std::enable_if_t<std::bool_constant<
                 //      std::decay_t<F>::min() == min_ && std::decay_t<F>::max() == max_>::value>
                 >
        constexpr uniform_random_bit_generator_ref(F && f) noexcept :
        function_ref<result_type()>(f)
        {
        }

        using function_ref<result_type()>::operator();

        static constexpr result_type min() noexcept { return min_; }
        static constexpr result_type max() noexcept { return max_; }
    };

    // ****************************** Random number distributions ****************************** //

    /// \brief Lightweight wrapper around random number distributions and generators to make them
    ///        appear as functors compatible with `Eigen::CwiseNullaryOp` for generating random
    ///        expressions procedurally.
    ///
    /// \details It extends the capability of the original `Eigen::internal::scalar_random_op`
    ///          functor to support specifying a custom distribution along with the accompanying
    ///          random number generator.
    ///
    /// \sa For technical reference about saving packed arguments for later use:
    ///     https://stackoverflow.com/q/7858817/4820605
    template<typename F, typename... StackedArgs>
    class scalar_random_op;

    template<typename R, typename... DerivedArgs, typename... StackedArgs>
    class scalar_random_op<R(DerivedArgs...), StackedArgs...>
    {
    public:
        // FIXME: Add 'noexcept' specifier once Eigen>=3.4.0 is enforced as strict requirement
        constexpr scalar_random_op(
            const scalar_random_op<R(DerivedArgs...), StackedArgs...> & rhs) = default;

        template<typename... StackedArgsIn>
        constexpr scalar_random_op(R (*f)(DerivedArgs...), StackedArgsIn &&... args) :
        fun_(f),
        args_(std::forward<StackedArgsIn>(args)...)
        {
        }

        constexpr scalar_random_op<R(DerivedArgs...), StackedArgs...> & operator=(
            const scalar_random_op<R(DerivedArgs...), StackedArgs...> & rhs) = default;

        template<typename F,
                 typename = std::enable_if_t<!std::is_same_v<std::decay_t<F>, scalar_random_op>>>
        scalar_random_op<R(DerivedArgs...), StackedArgs...> & operator=(F && f) = delete;

        R operator()(Eigen::Index i, Eigen::Index j = 0) const
        {
            return coeff(i, j, std::index_sequence_for<DerivedArgs...>{});
        }

    private:
        template<std::size_t... Is>
        R coeff(Eigen::Index i, Eigen::Index j, std::integer_sequence<std::size_t, Is...>) const
        {
            // Arguments bound by value are passed with const qualifier. The const qualifier is not
            // removed and therefore must honored when being forwarded to the callable, which must
            // accept such arguments either by value or const reference. This behavior is desirable
            // as it would be error-prone to cast-away constness, otherwise the user may expect the
            // original object to be modified while actually only the copy will be.
            // FIXME: Replacing `std::integral_constant` by template-lambda when moving to C++20
            return fun_(
                [&](auto I) -> decltype(auto)
                {
                    // Extract the right stacked argument
                    auto & arg = std::get<I>(args_);

                    // Coeff-wise dispatching when forwarding stacked arguments to the callable for
                    // those that are not implicitly convertible to the original type.
                    using StackedArg = std::tuple_element_t<I, decltype(args_)>;
                    using DerivedArg = std::tuple_element_t<I, std::tuple<DerivedArgs...>>;
                    if constexpr (std::is_convertible_v<StackedArg, DerivedArg>)
                    {
                        // if constexpr (!std::is_lvalue_reference_v<StackedArg>)
                        // {
                        //     return const_cast<std::add_lvalue_reference_t<StackedArg>>(arg);
                        // }
                        return arg;
                    }
                    else
                    {
                        return arg(i, j);
                    }
                }(std::integral_constant<std::size_t, Is>{})...);
        }

    private:
        R (*fun_)(DerivedArgs...);
        std::conditional_t<
            (sizeof...(StackedArgs) > 0),
            std::tuple<StackedArgs...>,
            std::tuple<std::conditional_t<std::is_arithmetic_v<std::decay_t<DerivedArgs>>,
                                          MatrixX<std::decay_t<DerivedArgs>>,
                                          DerivedArgs>...>>
            args_;
    };
}

namespace Eigen::internal
{
    template<typename R, typename Generator, typename... DerivedArgs, typename... StackedArgs>
    struct functor_traits<jiminy::scalar_random_op<R(DerivedArgs...), Generator, StackedArgs...>>
    {
        enum
        {
            Cost = 5 * Eigen::NumTraits<std::decay_t<R>>::MulCost,
            PacketAccess = false,
            IsRepeatable = false
        };
    };
}

namespace jiminy
{
    float JIMINY_DLLAPI uniform(const uniform_random_bit_generator_ref<uint32_t> & g);

    float JIMINY_DLLAPI uniform(
        const uniform_random_bit_generator_ref<uint32_t> & g, float lo, float hi);

    template<typename Generator, typename Derived1, typename Derived2>
    std::enable_if_t<
        (is_eigen_any_v<Derived1> ||
         is_eigen_any_v<Derived2>)&&(!std::is_arithmetic_v<std::decay_t<Derived1>> ||
                                     !std::is_arithmetic_v<std::decay_t<Derived2>>),
        Eigen::CwiseNullaryOp<
            scalar_random_op<float(const uniform_random_bit_generator_ref<uint32_t> &, float, float),
                             Generator &,
                             Derived1,
                             Derived2>,
            Eigen::MatrixXf>>
    uniform(Generator & g, Derived1 && lo, Derived2 && hi);

    template<typename Generator>
    Eigen::CwiseNullaryOp<
        scalar_random_op<float(const uniform_random_bit_generator_ref<uint32_t> &, float, float),
                         Generator &,
                         const Eigen::MatrixXf::ConstantReturnType,
                         const Eigen::MatrixXf::ConstantReturnType>,
        Eigen::MatrixXf>
    uniform(
        Eigen::Index nrows, Eigen::Index ncols, Generator & g, float lo = 0.0F, float hi = 1.0F);

    /// @details The original Ziggurat algorithm for single-precision floating-point scalars is
    ///          used internally. This method is known to be about x4 times faster than the
    ///          Box–Muller transform but significantly more complex to implement and more
    ///          notably to vectorize using SIMD instructions. For details about these methods:
    ///          https://en.wikipedia.org/wiki/Ziggurat_algorithm
    ///          https://en.wikipedia.org/wiki/Box–Muller_transform
    ///
    /// @warning This implementation has been optimized for sampling individual scalar here and
    ///          there, rather than all at once in a vector. The proposed PCG32 random number
    ///          generator is consistent with this design choice. Fully vectorized implementations
    ///          of Ziggurat algorithm and Box-Muller transform that supports both x86 and
    ///          Arm64-Neon intrinsics are publicly available here:
    ///          https://github.com/lfarizav/pseudorandomnumbergenerators
    ///          It you are looking for fully vectorized implementation of some other statistical
    ///          distributions, have a look to this project: https://github.com/bab2min/EigenRand
    ///
    /// @sa Based on the original implementation by Marsaglia and Tsang (JSS, 2000):
    ///     https://people.sc.fsu.edu/~jburkardt/cpp_src/ziggurat/ziggurat.html
    ///     This implementation is known to fail some standard statistical tests as described
    ///     by Doornik (2005). This is not an issue for most applications. For reference:
    ///     https://www.doornik.com/research/ziggurat.pdf
    float JIMINY_DLLAPI normal(const uniform_random_bit_generator_ref<uint32_t> & g,
                               float mean = 0.0F,
                               float stddev = 1.0F);

    /// \details Enforcing a common interface to all scalar statistical distribution by means of
    /// `uniform_random_bit_generator_ref` does not incur any overhead when compiling with
    /// optimizations enabled (level 01 is enough), probably due to inlining.
    template<typename Generator, typename Derived1, typename Derived2>
    std::enable_if_t<
        (is_eigen_any_v<Derived1> ||
         is_eigen_any_v<Derived2>)&&(!std::is_arithmetic_v<std::decay_t<Derived1>> ||
                                     !std::is_arithmetic_v<std::decay_t<Derived2>>),
        Eigen::CwiseNullaryOp<
            scalar_random_op<float(const uniform_random_bit_generator_ref<uint32_t> &, float, float),
                             Generator &,
                             Derived1,
                             Derived2>,
            Eigen::MatrixXf>>
    normal(Generator & g, Derived1 && mean, Derived2 && stddev);

    template<typename Generator>
    Eigen::CwiseNullaryOp<
        scalar_random_op<float(const uniform_random_bit_generator_ref<uint32_t> &, float, float),
                         Generator &,
                         const Eigen::MatrixXf::ConstantReturnType,
                         const Eigen::MatrixXf::ConstantReturnType>,
        Eigen::MatrixXf>
    normal(Eigen::Index nrows,
           Eigen::Index ncols,
           Generator & g,
           float mean = 0.0F,
           float stddev = 1.0F);

    // **************************** Continuous 1D Gaussian processes *************************** //

    namespace internal
    {
        /// \brief Lower Cholesky factor of a Toeplitz positive semi-definite matrix having 1.0 on
        ///        its main diagonal.
        ///
        /// \details In practice, it is advisable to combine this algorithm with Tikhonov
        ///          regularization of relative magnitude 1e-9 to avoid numerical instabilities
        ///          because of double machine precision.
        ///
        /// \sa Michael Stewart, Cholesky factorization of semi-definite Toeplitz matrices. Linear
        ///     Algebra and its Applications, Volume 254, pages 497-525, 1997.
        ///     The original implementation is available here:
        ///     https://people.sc.fsu.edu/~jburkardt/cpp_src/toeplitz_cholesky/toeplitz_cholesky.html
        ///
        /// \param[in] coeffs First row of the matrix to decompose.
        template<typename Derived>
        MatrixX<typename Derived::Scalar>
        standardToeplitzCholeskyLower(const Eigen::MatrixBase<Derived> & coeffs);
    }

    class JIMINY_DLLAPI PeriodicGaussianProcess
    {
    public:
        DISABLE_COPY(PeriodicGaussianProcess)

    public:
        explicit PeriodicGaussianProcess(double wavelength, double period) noexcept;

        void reset(const uniform_random_bit_generator_ref<uint32_t> & g) noexcept;

        double operator()(float t);

        double getWavelength() const noexcept;
        double getPeriod() const noexcept;

    private:
        const double wavelength_;
        const double period_;

        const double dt_{0.02 * wavelength_};
        const Eigen::Index numTimes_{static_cast<int>(std::ceil(period_ / dt_))};

        /// \brief Cholesky decomposition (LLT) of the covariance matrix.
        ///
        /// \details All decompositions are equivalent as the covariance matrix is symmetric,
        ///          namely eigen-values, singular-values, Cholesky and Schur decompositions. Yet,
        ///          Cholesky is by far the most efficient one. Moreover, the covariance is a
        ///          positive semi-definite Toepliz matrix, which means that the computational
        ///          complexity can be reduced even further using an specialized Cholesky
        ///          decomposition algorithm. See: https://math.stackexchange.com/q/22825/375496
        Eigen::MatrixXd covSqrtRoot_{
            internal::standardToeplitzCholeskyLower(Eigen::VectorXd::NullaryExpr(
                numTimes_,
                [numTimes = static_cast<double>(numTimes_), wavelength = wavelength_](double i) {
                    return std::exp(-2.0 *
                                    std::pow(std::sin(M_PI / numTimes * i) / wavelength, 2));
                }))};
        Eigen::VectorXd values_{numTimes_};
    };

    // **************************** Continuous 1D Fourier processes **************************** //

    /// \see Based on "Smooth random functions, random ODEs, and Gaussian processes":
    ///      https://hal.inria.fr/hal-01944992/file/random_revision2.pdf */
    class JIMINY_DLLAPI PeriodicFourierProcess
    {
    public:
        DISABLE_COPY(PeriodicFourierProcess)

    public:
        explicit PeriodicFourierProcess(double wavelength, double period) noexcept;

        void reset(const uniform_random_bit_generator_ref<uint32_t> & g) noexcept;

        double operator()(float t);

        double getWavelength() const noexcept;
        double getPeriod() const noexcept;

    private:
        const double wavelength_;
        const double period_;

        const double dt_{0.02 * wavelength_};
        const Eigen::Index numTimes_{static_cast<Eigen::Index>(std::ceil(period_ / dt_))};
        const Eigen::Index numHarmonics_{
            static_cast<Eigen::Index>(std::ceil(period_ / wavelength_))};

        const Eigen::MatrixXd cosMat_{Eigen::MatrixXd::NullaryExpr(
            numTimes_,
            numHarmonics_,
            [numTimes = static_cast<double>(numTimes_)](double i, double j)
            { return std::cos(2 * M_PI / numTimes * i * j); })};
        const Eigen::MatrixXd sinMat_{Eigen::MatrixXd::NullaryExpr(
            numTimes_,
            numHarmonics_,
            [numTimes = static_cast<double>(numTimes_)](double i, double j)
            { return std::sin(2 * M_PI / numTimes * i * j); })};
        Eigen::VectorXd values_{numTimes_};
    };

    // ***************************** Continuous 1D Perlin processes **************************** //

    class JIMINY_DLLAPI AbstractPerlinNoiseOctave
    {
    public:
        explicit AbstractPerlinNoiseOctave(double wavelength);
        virtual ~AbstractPerlinNoiseOctave() = default;

        virtual void reset(const uniform_random_bit_generator_ref<uint32_t> & g) noexcept;

        double operator()(double t) const;

        double getWavelength() const noexcept;

    protected:
        virtual double grad(int32_t knot, double delta) const noexcept = 0;

        /// @brief Improved Smoothstep function by Ken Perlin (aka Smootherstep).
        ///
        /// @details It has zero 1st and 2nd-order derivatives at dt = 0.0, and 1.0.
        ///
        /// @sa For reference, see:
        ///     https://en.wikipedia.org/wiki/Smoothstep#Variations
        static double fade(double delta) noexcept;
        static double lerp(double ratio, double yLeft, double yRight) noexcept;

    protected:
        const double wavelength_;

        double shift_{0.0};
    };

    class JIMINY_DLLAPI RandomPerlinNoiseOctave : public AbstractPerlinNoiseOctave
    {
    public:
        explicit RandomPerlinNoiseOctave(double wavelength);
        ~RandomPerlinNoiseOctave() override = default;

        void reset(const uniform_random_bit_generator_ref<uint32_t> & g) noexcept override;

    protected:
        double grad(int32_t knot, double delta) const noexcept override;

    private:
        uint32_t seed_{0U};
    };

    class JIMINY_DLLAPI PeriodicPerlinNoiseOctave : public AbstractPerlinNoiseOctave
    {
    public:
        explicit PeriodicPerlinNoiseOctave(double wavelength, double period);
        ~PeriodicPerlinNoiseOctave() override = default;

        void reset(const uniform_random_bit_generator_ref<uint32_t> & g) noexcept override;

        double getPeriod() const noexcept;

    protected:
        double grad(int32_t knot, double delta) const noexcept override;

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

        using OctaveScalePair =
            std::pair<std::unique_ptr<AbstractPerlinNoiseOctave>, const double>;

    public:
        void reset(const uniform_random_bit_generator_ref<uint32_t> & g) noexcept;

        double operator()(float t);

        double getWavelength() const noexcept;
        std::size_t getNumOctaves() const noexcept;

    protected:
        explicit AbstractPerlinProcess(std::vector<OctaveScalePair> && octaveScalePairs) noexcept;

    protected:
        std::vector<OctaveScalePair> octaveScalePairs_;

    private:
        double amplitude_{0.0};
    };

    class JIMINY_DLLAPI RandomPerlinProcess : public AbstractPerlinProcess
    {
    public:
        explicit RandomPerlinProcess(double wavelength, std::size_t numOctaves = 6U);
    };

    class PeriodicPerlinProcess : public AbstractPerlinProcess
    {
    public:
        explicit PeriodicPerlinProcess(
            double wavelength, double period, std::size_t numOctaves = 6U);

        double getPeriod() const noexcept;

    private:
        const double period_;
    };

    // ******************************* Random terrain generators ******************************* //

    HeightmapFunctor JIMINY_DLLAPI tiles(const Eigen::Vector2d & size,
                                         double heightMax,
                                         const Eigen::Vector2d & interpDelta,
                                         uint32_t sparsity,
                                         double orientation,
                                         uint32_t seed);
}

#include "jiminy/core/utilities/random.hxx"

#endif  // JIMINY_RANDOM_H
