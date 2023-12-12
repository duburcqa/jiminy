#ifndef JIMINY_LCP_SOLVERS_H
#define JIMINY_LCP_SOLVERS_H

#include "jiminy/core/fwd.h"


namespace jiminy
{
    class AbstractConstraintBase;
    struct constraintsHolder_t;

    struct ConstraintBlock
    {
        float64_t lo;
        float64_t hi;
        bool_t isZero;
        Eigen::Index fIdx[3];
        std::uint_fast8_t fSize;
    };

    struct ConstraintData
    {
    public:
        DISABLE_COPY(ConstraintData)

    public:
        ConstraintData() = default;
        ConstraintData(ConstraintData && constraintData) = default;

    public:
        AbstractConstraintBase * constraint;
        Eigen::Index startIdx;
        bool_t isInactive;
        Eigen::Index dim;
        ConstraintBlock blocks[3];
        std::uint_fast8_t nBlocks;
    };

    class JIMINY_DLLAPI AbstractConstraintSolver
    {
    public:
        AbstractConstraintSolver() = default;
        virtual ~AbstractConstraintSolver() = default;

        /// \brief Compute the solution of the Nonlinear Complementary Problem:
        ///        A x + b = w,
        ///        s.t. (w[i] > 0 and x[i] = 0) or (w[i] = 0 and x[i] > 0
        ///
        ///        for non-linear boxed bounds lo(x) < x < hi(x):
        ///        s.t. if fIndices[i].size() == 0, lo[i] < x[i] < hi[i]
        ///             else, sqrt(x[i] ** 2 + sum_{j>=1}(x[fIndices[i][j]] ** 2)) < hi[i] *
        ///                   max(0.0, x[fIndices[i][0]])
        virtual bool_t SolveBoxedForwardDynamics(const float64_t & dampingInv,
                                                 const bool_t & isStateUpToDate,
                                                 const bool_t & ignoreBounds) = 0;
    };

    class JIMINY_DLLAPI PGSSolver : public AbstractConstraintSolver
    {
    public:
        DISABLE_COPY(PGSSolver)

    public:
        PGSSolver(const pinocchio::Model * model,
                  pinocchio::Data * data,
                  constraintsHolder_t * constraintsHolder,
                  const float64_t & friction,
                  const float64_t & torsion,
                  const float64_t & tolAbs,
                  const float64_t & tolRel,
                  const uint32_t & maxIter);
        virtual ~PGSSolver() = default;

        virtual bool_t SolveBoxedForwardDynamics(
            const float64_t & dampingInv,
            const bool_t & isStateUpToDate = false,
            const bool_t & ignoreBounds = false) override final;

    private:
        void ProjectedGaussSeidelIter(const Eigen::MatrixXd & A,
                                      const Eigen::VectorXd::SegmentReturnType & b,
                                      Eigen::VectorXd::SegmentReturnType & x);
        bool_t ProjectedGaussSeidelSolver(const Eigen::MatrixXd & A,
                                          const Eigen::VectorXd::SegmentReturnType & b,
                                          Eigen::VectorXd::SegmentReturnType & x);

    private:
        const pinocchio::Model * model_;
        pinocchio::Data * data_;

        uint32_t maxIter_;
        float64_t tolAbs_;
        float64_t tolRel_;

        /// \brief Matrix holding the jacobian of the constraints.
        Eigen::Matrix<float64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> J_;
        /// \brief Vector holding the drift of the constraints.
        Eigen::VectorXd gamma_;
        /// \brief Vector holding the multipliers of the constraints.
        Eigen::VectorXd lambda_;
        std::vector<ConstraintData> constraintsData_;

        Eigen::VectorXd b_;
        Eigen::VectorXd y_;
        Eigen::VectorXd yPrev_;

        bool_t isLcpFullyUpToDate_;
    };
}

#endif  // JIMINY_LCP_SOLVERS_H
