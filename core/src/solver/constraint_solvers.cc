#include "pinocchio/multibody/model.hpp"     // `pinocchio::Model`
#include "pinocchio/multibody/data.hpp"      // `pinocchio::Data`
#include "pinocchio/algorithm/cholesky.hpp"  // `pinocchio::cholesky::`

#include "jiminy/core/utilities/helpers.h"
#include "jiminy/core/constraints/abstract_constraint.h"
#include "jiminy/core/robot/pinocchio_overload_algorithms.h"
#include "jiminy/core/robot/model.h"

#include "jiminy/core/solver/constraint_solvers.h"


namespace jiminy
{
    inline constexpr double MIN_REGULARIZER{1.0e-11};

    inline constexpr double RELAX_MIN{0.01};
    inline constexpr double RELAX_MAX{1.0};
    inline constexpr uint32_t RELAX_MIN_ITER_NUM{20};
    inline constexpr uint32_t RELAX_MAX_ITER_NUM{30};
    inline constexpr double RELAX_SLOPE_ORDER{2.0};

    PGSSolver::PGSSolver(const pinocchio::Model * model,
                         pinocchio::Data * data,
                         const ConstraintTree * constraints,
                         double friction,
                         double torsion,
                         double tolAbs,
                         double tolRel,
                         uint32_t iterMax) noexcept :
    model_{model},
    data_{data},
    iterMax_{iterMax},
    tolAbs_{tolAbs},
    tolRel_{tolRel}
    {
        Eigen::Index constraintsRowsMax = 0U;
        constraints->foreach(
            [&](const std::shared_ptr<AbstractConstraintBase> & constraint,
                ConstraintRegistryType type)
            {
                // Define constraint blocks
                const Eigen::Index constraintSize =
                    static_cast<Eigen::Index>(constraint->getSize());
                ConstraintBlock block{};
                ConstraintData constraintData{};
                switch (type)
                {
                case ConstraintRegistryType::BOUNDS_JOINTS:
                    // The joint is blocked in only one direction
                    block.lo = 0.0;
                    block.hi = INF;
                    block.fIndex[0] = 0;
                    block.fSize = 1;
                    constraintData.blocks[0] = block;
                    constraintData.nBlocks = 1;
                    break;
                case ConstraintRegistryType::CONTACT_FRAMES:
                case ConstraintRegistryType::COLLISION_BODIES:
                    // Non-penetration normal force
                    block.lo = 0.0;
                    block.hi = INF;
                    block.fIndex[0] = 2;
                    block.fSize = 1;
                    constraintData.blocks[0] = block;

                    // Torsional friction around normal axis
                    block.lo = qNAN;
                    block.hi = torsion;
                    block.isZero = (torsion < EPS);
                    block.fIndex[0] = 3;
                    block.fIndex[1] = 2;
                    block.fSize = 2;
                    constraintData.blocks[1] = block;

                    // Friction cone in tangential plane
                    block.lo = qNAN;
                    block.hi = friction;
                    block.isZero = (friction < EPS);
                    block.fIndex[0] = 0;
                    block.fIndex[1] = 1;
                    block.fIndex[2] = 2;
                    block.fSize = 3;
                    constraintData.blocks[2] = block;

                    constraintData.nBlocks = 3;
                    break;
                case ConstraintRegistryType::USER:
                default:
                    break;
                }
                constraintData.dim = constraintSize;
                constraintData.constraint = constraint.get();
                constraintsData_.emplace_back(std::move(constraintData));
                constraintsRowsMax += constraintSize;
            });

        // Resize buffers
        J_.resize(constraintsRowsMax, model_->nv);
        gamma_.resize(constraintsRowsMax);
        lambda_.resize(constraintsRowsMax);
        b_.resize(constraintsRowsMax);
        y_.resize(constraintsRowsMax);
        yPrev_.resize(constraintsRowsMax);
    }

    void PGSSolver::ProjectedGaussSeidelIter(const Eigen::MatrixXd & A,
                                             const Eigen::VectorXd::SegmentReturnType & b,
                                             const double w,
                                             Eigen::VectorXd::SegmentReturnType & x)
    {
        // First, loop over all unbounded constraints
        for (const ConstraintData & constraintData : constraintsData_)
        {
            // Bypass inactive and bounded constraints
            if (constraintData.isInactive || constraintData.nBlocks != 0)
            {
                continue;
            }

            // Loop over all coefficients individually
            Eigen::Index i = constraintData.startIndex;
            const Eigen::Index endIndex = i + constraintData.dim;
            for (; i < endIndex; ++i)
            {
                y_[i] = b[i] - A.col(i).dot(x);
                x[i] += y_[i] / A(i, i);
            }
        }

        /* Second, loop over all bounds constraints.
           Update breadth-first for faster convergence, knowing that the number of blocks
           per constraint cannot exceed 3. */
        for (std::size_t i = 0; i < 3; ++i)
        {
            for (const ConstraintData & constraintData : constraintsData_)
            {
                // Bypass inactive or unbounded constraints or no block left
                if (constraintData.isInactive || constraintData.nBlocks <= i)
                {
                    continue;
                }

                // Extract block data
                const ConstraintBlock & block = constraintData.blocks[i];
                const Eigen::Index * fIndex = block.fIndex;
                const std::uint_fast8_t & fSize = block.fSize;
                const Eigen::Index o = constraintData.startIndex;
                const Eigen::Index i0 = o + fIndex[0];
                const double hi = block.hi;
                const double lo = block.lo;
                double & e = x[i0];

                // Bypass zero-ed coefficients
                if (block.isZero)
                {
                    // Specialization for speed-up
                    e *= 0;
                    for (std::uint_fast8_t j = 1; j < fSize - 1; ++j)
                    {
                        x[o + fIndex[j]] *= 0;
                    }
                    continue;
                }

                // Update several coefficients at once with the same step
                double A_max = A(i0, i0);
                y_[i0] = b[i0] - A.col(i0).dot(x);
                for (std::uint_fast8_t j = 1; j < fSize - 1; ++j)
                {
                    const Eigen::Index k = o + fIndex[j];
                    y_[k] = b[k] - A.col(k).dot(x);
                    const double A_kk = A(k, k);
                    if (A_kk > A_max)
                    {
                        A_max = A_kk;
                    }
                }
                e += w * y_[i0] / A_max;
                for (std::uint_fast8_t j = 1; j < fSize - 1; ++j)
                {
                    const Eigen::Index k = o + fIndex[j];
                    x[k] += w * y_[k] / A_max;
                }

                // Project the coefficient between lower and upper bounds
                auto xConst = x.segment(o, constraintData.dim);
                if (fSize == 1)
                {
                    e = std::clamp(e, lo, hi);
                }
                else
                {
                    const double thr = hi * xConst[fIndex[fSize - 1]];
                    if (fSize == 2)
                    {
                        // Specialization for speedup and numerical stability
                        e = std::clamp(e, -thr, thr);
                    }
                    else
                    {
                        // Generic case
                        double squaredNorm = e * e;
                        for (std::uint_fast8_t j = 1; j < fSize - 1; ++j)
                        {
                            const double f = xConst[fIndex[j]];
                            squaredNorm += f * f;
                        }
                        if (squaredNorm > thr * thr)
                        {
                            const double scale = thr / std::sqrt(squaredNorm);
                            e *= scale;
                            for (std::uint_fast8_t j = 1; j < fSize - 1; ++j)
                            {
                                xConst[fIndex[j]] *= scale;
                            }
                        }
                    }
                }
            }
        }
    }

    bool PGSSolver::ProjectedGaussSeidelSolver(const Eigen::MatrixXd & A,
                                               const Eigen::VectorXd::SegmentReturnType & b,
                                               Eigen::VectorXd::SegmentReturnType & x)
    {
        /* For some reason, it is impossible to get a better accuracy than 1e-5 for the absolute
           tolerance, even if unconstrained. It seems to be related to compounding of errors, maybe
           due to the recursive computations. */

        if (b.size() == 0)
        {
            throw std::logic_error("The number of inequality constraints must be larger than 0.");
        }

        // Reset the residuals
        y_.setZero();

        // Perform multiple PGS loop until convergence or max iter reached
        for (uint32_t iter = 0; iter < iterMax_; ++iter)
        {
            // Backup previous residuals
            yPrev_ = y_;

            // Update the under-relaxation factor
            const double ratio = (static_cast<double>(iterMax_ - RELAX_MIN_ITER_NUM) - iter) /
                                 (iterMax_ - RELAX_MIN_ITER_NUM - RELAX_MAX_ITER_NUM);
            double w = RELAX_MAX;
            if (ratio < 1.0)
            {
                w = RELAX_MIN;
                if (ratio > 0.0)
                {
                    w += (RELAX_MAX - RELAX_MIN) * std::pow(ratio, RELAX_SLOPE_ORDER);
                }
            }

            // Do one iteration
            ProjectedGaussSeidelIter(A, b, w, x);

            /* Abort if stopping criteria is met.
               It is not possible to define a stopping criteria on the residuals directly because
               they can grow arbitrary large for constraints whose bounds are active. It follows
               that stagnation of residuals is the only viable criteria.
               The PGS algorithm has been modified for solving second-order cone LCP, which means
               that only the L^2-norm of the tangential forces can be expected to converge. Because
               of this, it is too restrictive to check the element-wise variation of the residuals
               over iterations. It makes more sense to look at the Linf-norm instead, but this
               criteria is very lax. A good compromise may be to look at the constraint-block-wise
               L^2-norm, which is similar to what Drake simulator is doing. For reference, see:
               https://github.com/RobotLocomotion/drake/blob/master/multibody/contact_solvers/pgs_solver.cc
            */
            const double tol = tolAbs_ + tolRel_ * y_.lpNorm<Eigen::Infinity>() + EPS;
            if (((y_ - yPrev_).array().abs() < tol).all())
            {
                return true;
            }

            // std::cout << "[" << iter << "] (" << w << "): ";
            // bool isSuccess = true;
            // for (std::size_t i = 0; i < 3; ++i)
            //{
            //     for (const ConstraintData & constraintData : constraintsData_)
            //     {
            //         if (constraintData.isInactive || constraintData.nBlocks <= i)
            //         {
            //             continue;
            //         }
            //
            //         const ConstraintBlock & block = constraintData.blocks[i];
            //         const Eigen::Index * fIndex = block.fIndex;
            //         const Eigen::Index i0 = constraintData.startIndex + fIndex[0];
            //         double yNorm = y_[i0] * y_[i0];
            //         double yPrevNorm = yPrev_[i0] * yPrev_[i0];
            //         for (std::uint_fast8_t j = 1; j < block.fSize - 1; ++j)
            //         {
            //             yNorm += y_[fIndex[j]] * y_[fIndex[j]];
            //             yPrevNorm += yPrev_[fIndex[j]] * yPrev_[fIndex[j]];
            //         }
            //         yNorm = std::sqrt(yNorm);
            //         yPrevNorm = std::sqrt(yPrevNorm);
            //
            //         const double tol = tolAbs_ + tolRel_ * yNorm + EPS;
            //         std::cout << std::abs(yNorm - yPrevNorm) << "(" << tol << "), ";
            //         if (std::abs(yNorm - yPrevNorm) > tol)
            //         {
            //             isSuccess = false;
            //             break;
            //         }
            //     }
            //     if (!isSuccess)
            //     {
            //         break;
            //     }
            // }
            // std::cout << std::endl;
            // if (isSuccess)
            //{
            //     return true;
            // }
        }

        // Impossible to converge
        return false;
    }

    bool PGSSolver::SolveBoxedForwardDynamics(
        double dampingInv, bool isStateUpToDate, bool ignoreBounds)
    {
        // Update constraints start indices, jacobian, drift and multipliers
        Eigen::Index constraintRows = 0U;
        for (auto & constraintData : constraintsData_)
        {
            AbstractConstraintBase * constraint = constraintData.constraint;
            constraintData.isInactive = !constraint->getIsEnabled();
            if (constraintData.isInactive)
            {
                continue;
            }
            const Eigen::Index constraintSize = constraintData.dim;
            if (!isStateUpToDate)
            {
                J_.middleRows(constraintRows, constraintSize) = constraint->getJacobian();
                gamma_.segment(constraintRows, constraintSize) = constraint->getDrift();
                lambda_.segment(constraintRows, constraintSize) = constraint->lambda_;
            }
            constraintData.startIndex = constraintRows;
            constraintRows += constraintSize;
        };

        // Extract active rows
        auto J = J_.topRows(constraintRows);
        auto lambda = lambda_.head(constraintRows);
        auto gamma = gamma_.head(constraintRows);
        auto b = b_.head(constraintRows);

        // Check if problem is bounded
        bool isUnbounded =
            std::all_of(constraintsData_.cbegin(),
                        constraintsData_.cend(),
                        [](const ConstraintData & constraintData)
                        { return constraintData.isInactive || constraintData.nBlocks == 0; });

        Eigen::MatrixXd & A = data_->JMinvJt;
        if (!isStateUpToDate)
        {
            // Compute JMinvJt, including cholesky decomposition of inertia matrix
            pinocchio_overload::computeJMinvJt(*model_, *data_, J);

            /* Add regularization term in case A is not invertible.

               Note that Mujoco defines an impedance function that depends on the distance instead
               of a constant value to model soft contacts.

               See: - http://mujoco.org/book/modeling.html#CSolver
                    - http://mujoco.org/book/computation.html#soParameters  */
            A.diagonal() += (A.diagonal() * dampingInv).cwiseMax(MIN_REGULARIZER);

            /* The LCP is not fully up-to-date since the upper triangular part is still missing.
               This will only be done if necessary. */
            isLcpFullyUpToDate_ = false;
        }

        // Compute the dynamic drift (control - nle)
        data_->torque_residual = data_->u - data_->nle;
        pinocchio::cholesky::solve(*model_, *data_, data_->torque_residual);

        /* Compute b.
           - TODO: Leverage sparsity of J to avoid dense matrix multiplication */
        b = -gamma;
        b.noalias() -= J * data_->torque_residual;

        // Compute resulting forces solving forward dynamics
        bool isSuccess = false;
        if (ignoreBounds || isUnbounded)
        {
            /* There is no inequality constraint, so the problem can be
               solved exactly and efficiently using cholesky decomposition.

               The implementation of this particular case is based on `pinocchio::forwardDynamics`
               methods without modification. See
               https://github.com/stack-of-tasks/pinocchio/blob/master/src/algorithm/contact-dynamics.hxx
             */

            // Compute the Lagrange Multipliers
            lambda = pinocchio_overload::solveJMinvJtv(*data_, b, true);

            // Always successful
            isSuccess = true;
        }
        else
        {
            /* Full matrix is needed to enable leveraging vectorization.
               Note that updating the lower part of the matrix is necessary even if the state is
               up-to-date because of the 'if' branching. */
            if (!isLcpFullyUpToDate_)
            {
                A.triangularView<Eigen::StrictlyUpper>() = A.transpose();
                isLcpFullyUpToDate_ = true;
            }

            // Run standard PGS algorithm
            isSuccess = ProjectedGaussSeidelSolver(A, b, lambda);
        }

        // Update lagrangian multipliers associated with the constraint
        constraintRows = 0U;
        for (const auto & constraintData : constraintsData_)
        {
            AbstractConstraintBase * constraint = constraintData.constraint;
            if (!constraint->getIsEnabled())
            {
                continue;
            }
            const Eigen::Index constraintSize = static_cast<Eigen::Index>(constraint->getSize());
            constraint->lambda_ = lambda_.segment(constraintRows, constraintSize);
            constraintRows += constraintSize;
        };

        /* Compute resulting acceleration, no matter if computing forces was successful.
           - TODO: Leverage sparsity of J to avoid dense matrix multiplication */
        data_->ddq.noalias() = J.transpose() * lambda;
        pinocchio::cholesky::solve(*model_, *data_, data_->ddq);
        data_->ddq += data_->torque_residual;

        return isSuccess;
    }
}
