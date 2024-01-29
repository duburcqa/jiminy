#ifndef JIMINY_RANDOM_HXX
#define JIMINY_RANDOM_HXX

#include <utility>  // `std::forward`

#include "jiminy/core/fwd.h"


namespace jiminy
{
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
        (is_eigen_any_v<Derived1> ||
         is_eigen_any_v<Derived2>)&&(!std::is_arithmetic_v<std::decay_t<Derived1>> ||
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
        (is_eigen_any_v<Derived1> ||
         is_eigen_any_v<Derived2>)&&(!std::is_arithmetic_v<std::decay_t<Derived1>> ||
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
        standardToeplitzCholeskyLower(const Eigen::MatrixBase<Derived> & coeffs)
        {
            using Scalar = typename Derived::Scalar;

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
                Eigen::Vector4d::Map(H.data()).template segment<2>(1).fill(rho);
                g.rightCols(n - i) = H * g.rightCols(n - i) / std::sqrt((1.0 - rho) * (1.0 + rho));
                l.col(i).tail(n - i) = g.row(0).tail(n - i);
                g.row(0).tail(n - i - 1) = g.row(0).segment(i, n - i - 1).eval();
            }

            return l;
        }
    }
}

#endif  // JIMINY_RANDOM_HXX
