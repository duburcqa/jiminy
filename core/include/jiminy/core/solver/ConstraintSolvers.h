#ifndef JIMINY_LCP_SOLVERS_H
#define JIMINY_LCP_SOLVERS_H

#include "jiminy/core/Types.h"


namespace jiminy
{
    class AbstractConstraintBase;
    struct constraintsHolder_t;

    struct ConstraintBlock
    {
    public:
        ConstraintBlock(void):
        lo(-INF),
        hi(INF),
        isZero(false),
        fIdx(),
        fSize(0)
        {
            // Empty on purpose
        }

    public:
        float64_t lo;
        float64_t hi;
        bool_t isZero;
        Eigen::Index fIdx[3];
        std::uint_fast8_t fSize;
    };

    struct ConstraintData
    {
    public:
        ConstraintData(void):
        constraint(nullptr),
        startIdx(0),
        isBounded(true),
        isActive(true),
        dim(0),
        blocks(),
        nBlocks(0)
        {
            // Empty on purpose
        }
        ConstraintData(ConstraintData && constraintData) = default;
        ConstraintData(ConstraintData const & constraintData) = delete;
        ConstraintData & operator = (ConstraintData const & other) = delete;

    public:
        AbstractConstraintBase * constraint;
        Eigen::Index startIdx;
        bool_t isBounded;
        bool_t isActive;
        Eigen::Index dim;
        ConstraintBlock blocks[3];
        std::uint_fast8_t nBlocks;
    };

    class AbstractConstraintSolver
    {
    public:
        AbstractConstraintSolver(void) = default;
        virtual ~AbstractConstraintSolver(void) = default;

        /// \brief Compute the solution of the Nonlinear Complementary Problem:
        ///        A x + b = w,
        ///        s.t. (w[i] > 0 and x[i] = 0) or (w[i] = 0 and x[i] > 0
        ///
        ///        for non-linear boxed bounds lo(x) < x < hi(x):
        ///        s.t. if fIndices[i].size() == 0, lo[i] < x[i] < hi[i]
        ///             else, sqrt(x[i] ** 2 + sum_{j>=1}(x[fIndices[i][j]] ** 2)) < hi[i] * max(0.0, x[fIndices[i][0]])
        ///
        virtual bool_t SolveBoxedForwardDynamics(float64_t const & inv_damping) = 0;
    };

    class PGSSolver : public AbstractConstraintSolver
    {
    public:
        // Disable the copy of the class
        PGSSolver(PGSSolver const & solver) = delete;
        PGSSolver & operator = (PGSSolver const & solver) = delete;

    public:
        PGSSolver(pinocchio::Model const * model,
                  pinocchio::Data * data,
                  constraintsHolder_t * constraintsHolder,
                  float64_t const & friction,
                  float64_t const & torsion,
                  float64_t const & tolAbs,
                  float64_t const & tolRel,
                  uint32_t const & maxIter);
        virtual ~PGSSolver(void) = default;

        virtual bool_t SolveBoxedForwardDynamics(float64_t const & inv_damping) override final;

    private:
        void ProjectedGaussSeidelIter(matrixN_t const & A,
                                      vectorN_t::SegmentReturnType const & b,
                                      vectorN_t::SegmentReturnType & x);
        bool_t ProjectedGaussSeidelSolver(matrixN_t const & A,
                                          vectorN_t::SegmentReturnType const & b,
                                          vectorN_t::SegmentReturnType & x);

    private:
        pinocchio::Model const * model_;
        pinocchio::Data * data_;

        uint32_t maxIter_;
        float64_t tolAbs_;
        float64_t tolRel_;

        Eigen::Matrix<float64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> J_;  ///< Matrix holding the jacobian of the constraints
        vectorN_t gamma_;   ///< Vector holding the drift of the constraints
        vectorN_t lambda_;  ///< Vector holding the multipliers of the constraints
        std::vector<ConstraintData> constraintsData_;

        vectorN_t b_;
        vectorN_t y_;
        vectorN_t yPrev_;
    };
}

#endif  // JIMINY_LCP_SOLVERS_H
