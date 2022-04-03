#include "pinocchio/multibody/model.hpp"     // `pinocchio::Model`
#include "pinocchio/multibody/data.hpp"      // `pinocchio::Data`
#include "pinocchio/algorithm/cholesky.hpp"  // `pinocchio::cholesky::`

#include "jiminy/core/robot/PinocchioOverloadAlgorithms.h"
#include "jiminy/core/constraints/AbstractConstraint.h"
#include "jiminy/core/utilities/Random.h"
#include "jiminy/core/utilities/Helpers.h"
#include "jiminy/core/Constants.h"

#include "jiminy/core/solver/ConstraintSolvers.h"


namespace jiminy
{
    PGSSolver::PGSSolver(pinocchio::Model const * model,
                         pinocchio::Data * data,
                         constraintsHolder_t * constraintsHolder,
                         float64_t const & friction,
                         float64_t const & torsion,
                         float64_t const & tolAbs,
                         float64_t const & tolRel,
                         uint32_t const & maxIter) :
    model_(model),
    data_(data),
    maxIter_(maxIter),
    tolAbs_(tolAbs),
    tolRel_(tolRel),
    J_(),
    gamma_(),
    lambda_(),
    constraintsData_(),
    b_(),
    y_(),
    dy_(),
    yPrev_()
    {
        Eigen::Index constraintsRowsMax = 0U;
        constraintsHolder->foreach(
            [&](
                std::shared_ptr<AbstractConstraintBase> const & constraint,
                constraintsHolderType_t const & holderType)
            {
                // Define constraint blocks
                Eigen::Index const constraintDim = static_cast<Eigen::Index>(constraint->getDim());
                ConstraintBlock block;
                ConstraintData constraintData;
                switch (holderType)
                {
                case constraintsHolderType_t::BOUNDS_JOINTS:
                    // The joint is blocked in only one direction
                    block.lo = 0;
                    block.hi = INF;
                    block.fIndices = std::vector<int32_t> {
                        static_cast<int32_t>(0)
                    };
                    constraintData.blocks.push_back(block);
                    break;
                case constraintsHolderType_t::CONTACT_FRAMES:
                case constraintsHolderType_t::COLLISION_BODIES:
                    // Non-penetration normal force
                    block.lo = 0;
                    block.hi = INF;
                    block.fIndices = std::vector<int32_t> {
                        static_cast<int32_t>(2)
                    };
                    constraintData.blocks.push_back(block);

                    // Torsional friction around normal axis
                    block.lo = qNAN;
                    block.hi = torsion;
                    block.fIndices = std::vector<int32_t> {
                        static_cast<int32_t>(3),
                        static_cast<int32_t>(2)
                    };
                    constraintData.blocks.push_back(block);

                    // Friction cone in tangential plane
                    block.lo = qNAN;
                    block.hi = friction;
                    block.fIndices = std::vector<int32_t> {
                        static_cast<int32_t>(0),
                        static_cast<int32_t>(1),
                        static_cast<int32_t>(2)
                    };
                    constraintData.blocks.push_back(block);
                    break;
                case constraintsHolderType_t::USER:
                    for (Eigen::Index i = 0; i < constraintDim; ++i)
                    {
                        block.lo = -INF;
                        block.hi = INF;
                        block.fIndices = std::vector<int32_t> {
                            static_cast<int32_t>(i)
                        };
                        constraintData.blocks.push_back(block);
                    }
                    break;
                default:
                    break;
                }
                constraintData.constraint = constraint.get();
                constraintsData_.emplace_back(std::move(constraintData));
                constraintsRowsMax += constraintDim;
            });

        // Resize buffers
        J_.resize(constraintsRowsMax, model_->nv);
        gamma_.resize(constraintsRowsMax);
        lambda_.resize(constraintsRowsMax);
        y_.resize(constraintsRowsMax);
        b_.resize(constraintsRowsMax);
        y_.resize(constraintsRowsMax);
        dy_.resize(constraintsRowsMax);
        yPrev_.resize(constraintsRowsMax);
    }

    void PGSSolver::ProjectedGaussSeidelIter(matrixN_t const & A,
                                             vectorN_t const & b,
                                             vectorN_t::SegmentReturnType x)
    {
        /* Update width first instead of depth to converge faster. The deeper
           is it the more nested it gets, so it is better to updated those
           coefficients at the very end.
           Note that the maximum number of blocks per constraint is 3. */
        for (std::size_t i = 0; i < 3 ; ++i)
        {
            for (ConstraintData const & constraintData : constraintsData_)
            {
                // Bypass inactive constraints or no block left
                int32_t const & o = constraintData.startIdx;
                if (o < 0 || constraintData.blocks.size() <= i)
                {
                    continue;
                }

                // Extract current group of indices
                ConstraintBlock const & block = constraintData.blocks[i];
                std::vector<int32_t> const & fIdx = block.fIndices;
                float64_t const & hi = block.hi;
                float64_t const & lo = block.lo;
                int32_t const i0 = o + fIdx[0];
                float64_t & e = x[i0];

                // Update the coefficient if relevant
                int32_t const fSize = static_cast<int32_t>(fIdx.size());
                if ((fSize == 1 && (hi - lo > EPS)) || (hi > EPS))
                {
                    // Update several coefficients at once with the same step
                    float64_t scale = A(i0, i0);
                    y_[i0] = b[i0] - A.col(i0).dot(x);
                    for (int32_t j = 1; j < fSize - 1; ++j)
                    {
                        int32_t const & k = o + fIdx[j];
                        y_[k] = b[k] - A.col(k).dot(x);
                        float64_t const & A_kk = A(k, k);
                        if (A_kk > scale)
                        {
                            scale = A_kk;
                        }
                    }
                    e += y_[i0] / scale;
                    for (int32_t j = 1; j < fSize - 1; ++j)
                    {
                        int32_t const & k = o + fIdx[j];
                        x[k] += y_[k] / scale;
                    }
                }

                // Project the coefficient between lower and upper bounds
                if (fSize == 1)
                {
                    e = clamp(e, lo, hi);
                }
                else
                {
                    float64_t const thr = hi * x[o + fIdx[fSize - 1]];
                    if (thr > EPS)
                    {
                        if (fSize == 2)
                        {
                            // Specialization for speedup and numerical stability
                            e = clamp(e, -thr, thr);
                        }
                        else
                        {
                            // Generic case
                            float64_t squaredNorm = e * e;
                            for (int32_t j = 1; j < fSize - 1; ++j)
                            {
                                float64_t const f = x[o + fIdx[j]];
                                squaredNorm += f * f;
                            }
                            if (squaredNorm > thr * thr)
                            {
                                float64_t const scale = thr / std::sqrt(squaredNorm);
                                e *= scale;
                                for (int32_t j = 1; j < fSize - 1; ++j)
                                {
                                    x[o + fIdx[j]] *= scale;
                                }
                            }
                        }
                    }
                    else
                    {
                        // Specialization for speed-up
                        e *= 0;
                        for (int32_t j = 1; j < fSize - 1; ++j)
                        {
                            x[o + fIdx[j]] *= 0;
                        }
                    }
                }
            }
        }
    }

    bool_t PGSSolver::ProjectedGaussSeidelSolver(matrixN_t const & A,
                                                 vectorN_t const & b,
                                                 vectorN_t::SegmentReturnType x)
    {
        assert(b.size() > 0 && "The number of inequality constraints must be larger than 0.");

        // Reset the residuals
        y_.setZero();

        // Perform multiple PGS loop until convergence or max iter reached
        for (uint32_t iter = 0; iter < maxIter_; ++iter)
        {
            // Backup previous residuals
            yPrev_ = y_;

            // Do a single iteration
            ProjectedGaussSeidelIter(A, b, x);

            // Check if terminate conditions are satisfied
            dy_ = y_ - yPrev_;
            if ((dy_.array().abs() < tolAbs_ || (dy_.array() / y_.array()).abs() < tolRel_).all())
            {
                return true;
            }
        }

        // Impossible to converge
        return false;
    }

    bool_t PGSSolver::SolveBoxedForwardDynamics(float64_t const & inv_damping)
    {
        // Check if problem is bounded
        bool_t isBounded = false;
        for (ConstraintData const & constraintData : constraintsData_)
        {
            for (ConstraintBlock const & block : constraintData.blocks)
            {
                if (!(std::isinf(block.lo) && std::isinf(block.hi)))
                {
                    isBounded = true;
                    break;
                }
            }
        }

        // Update constraints start indices, jacobian, drift and multipliers
        Eigen::Index constraintRows = 0U;
        for (auto & constraintData : constraintsData_)
        {
            AbstractConstraintBase * constraint = constraintData.constraint;
            if (!constraint->getIsEnabled())
            {
                constraintData.startIdx = -1;
                continue;
            }
            Eigen::Index const constraintDim = static_cast<Eigen::Index>(constraint->getDim());
            J_.middleRows(constraintRows, constraintDim) = constraint->getJacobian();
            gamma_.segment(constraintRows, constraintDim) = constraint->getDrift();
            lambda_.segment(constraintRows, constraintDim) = constraint->lambda_;
            constraintData.startIdx = static_cast<int32_t>(constraintRows);
            constraintRows += constraintDim;
        };

        // Compute JMinvJt, including cholesky decomposition of inertia matrix
        matrixN_t & A = pinocchio_overload::computeJMinvJt(
            *model_, *data_, J_.topRows(constraintRows), false);

        /* Add regularization term in case A is not invertible.
           Note that Mujoco defines an impedance function that depends on
           the distance instead of a constant value to model soft contacts.
           See: - http://mujoco.org/book/modeling.html#CSolver
                - http://mujoco.org/book/computation.html#soParameters  */
        A.diagonal() += clamp(
            A.diagonal() * inv_damping,
            PGS_MIN_REGULARIZER,
            INF);

        // Compute the dynamic drift (control - nle)
        data_->torque_residual = data_->u - data_->nle;
        pinocchio::cholesky::solve(*model_, *data_, data_->torque_residual);

        // Compute b
        b_ = - gamma_;
        b_.noalias() -= J_ * data_->torque_residual;

        // Compute resulting forces solving forward dynamics
        bool_t isSuccess = false;
        if (!isBounded)
        {
            /* There is no inequality constraint, so the problem can be
               solved exactly and efficiently using cholesky decomposition.

               The implementation of this particular case is based on
               `pinocchio::forwardDynamics methods` without modification.
               See https://github.com/stack-of-tasks/pinocchio/blob/master/src/algorithm/contact-dynamics.hxx */

            // Compute the Lagrange Multipliers
            lambda_.head(constraintRows) = pinocchio_overload::solveJMinvJtv(
                *data_, b_.head(constraintRows), true);

            // Always successful
            isSuccess = true;
        }
        else
        {
            // Full matrix is needed to enable vectorization
            A.triangularView<Eigen::Upper>() = A.transpose();

            // Run standard PGS algorithm
            isSuccess = ProjectedGaussSeidelSolver(A, b_, lambda_.head(constraintRows));
        }

        // Update lagrangian multipliers associated with the constraint
        constraintRows = 0U;
        for (auto const & constraintData : constraintsData_)
        {
            AbstractConstraintBase * constraint = constraintData.constraint;
            if (!constraint->getIsEnabled())
            {
                continue;
            }
            Eigen::Index const constraintDim = static_cast<Eigen::Index>(constraint->getDim());
            constraint->lambda_ = lambda_.segment(constraintRows, constraintDim);
            constraintRows += constraintDim;
        };

        // Compute resulting acceleration, no matter if computing forces was successful
        data_->ddq.noalias() = J_.topRows(constraintRows).transpose() * lambda_.head(constraintRows);
        pinocchio::cholesky::solve(*model_, *data_, data_->ddq);
        data_->ddq += data_->torque_residual;

        return isSuccess;
    }
}
