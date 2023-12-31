#ifndef JIMINY_LCP_SOLVERS_H
#define JIMINY_LCP_SOLVERS_H

#include "jiminy/core/fwd.h"


namespace jiminy
{
    class AbstractConstraintBase;
    struct constraintsHolder_t;

    struct ConstraintBlock
    {
        double lo{0.0};
        double hi{0.0};
        bool isZero{false};
        Eigen::Index fIdx[3]{-1, -1, -1};
        std::uint_fast8_t fSize{0};
    };

    struct ConstraintData
    {
    public:
        DISABLE_COPY(ConstraintData)

    public:
        explicit ConstraintData() = default;
        ConstraintData(ConstraintData && constraintData) = default;

    public:
        AbstractConstraintBase * constraint{nullptr};
        Eigen::Index startIdx{-1};
        bool isInactive{false};
        Eigen::Index dim{-1};
        ConstraintBlock blocks[3]{};
        std::uint_fast8_t nBlocks{0};
    };

    class JIMINY_DLLAPI AbstractConstraintSolver
    {
    public:
        virtual ~AbstractConstraintSolver() = default;

        /// \brief Compute the solution of the Nonlinear Complementary Problem:
        ///        A x + b = w,
        ///        s.t. (w[i] > 0 and x[i] = 0) or (w[i] = 0 and x[i] > 0
        ///
        ///        for non-linear boxed bounds lo(x) < x < hi(x):
        ///        s.t. if fIndices[i].size() == 0, lo[i] < x[i] < hi[i]
        ///             else, sqrt(x[i] ** 2 + sum_{j>=1}(x[fIndices[i][j]] ** 2)) < hi[i] *
        ///                   max(0.0, x[fIndices[i][0]])
        virtual bool SolveBoxedForwardDynamics(
            double dampingInv, bool isStateUpToDate, bool ignoreBounds) = 0;
    };

    class JIMINY_DLLAPI PGSSolver : public AbstractConstraintSolver
    {
    public:
        DISABLE_COPY(PGSSolver)

    public:
        explicit PGSSolver(const pinocchio::Model * model,
                           pinocchio::Data * data,
                           constraintsHolder_t * constraintsHolder,
                           double friction,
                           double torsion,
                           double tolAbs,
                           double tolRel,
                           uint32_t maxIter) noexcept;
        virtual ~PGSSolver() = default;

        virtual bool SolveBoxedForwardDynamics(double dampingInv,
                                               bool isStateUpToDate = false,
                                               bool ignoreBounds = false) override final;

    private:
        void ProjectedGaussSeidelIter(const Eigen::MatrixXd & A,
                                      const Eigen::VectorXd::SegmentReturnType & b,
                                      Eigen::VectorXd::SegmentReturnType & x);
        bool ProjectedGaussSeidelSolver(const Eigen::MatrixXd & A,
                                        const Eigen::VectorXd::SegmentReturnType & b,
                                        Eigen::VectorXd::SegmentReturnType & x);

    private:
        const pinocchio::Model * model_;
        pinocchio::Data * data_;

        uint32_t maxIter_;
        double tolAbs_;
        double tolRel_;

        /// \brief Matrix holding the jacobian of the constraints.
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> J_{};
        /// \brief Vector holding the drift of the constraints.
        Eigen::VectorXd gamma_{};
        /// \brief Vector holding the multipliers of the constraints.
        Eigen::VectorXd lambda_{};
        std::vector<ConstraintData> constraintsData_{};

        Eigen::VectorXd b_{};
        Eigen::VectorXd y_{};
        Eigen::VectorXd yPrev_{};

        bool isLcpFullyUpToDate_{false};
    };
}

#endif  // JIMINY_LCP_SOLVERS_H
