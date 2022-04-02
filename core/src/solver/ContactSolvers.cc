#include "pinocchio/multibody/model.hpp"     // `pinocchio::Model`
#include "pinocchio/multibody/data.hpp"      // `pinocchio::Data`
#include "pinocchio/algorithm/cholesky.hpp"  // `pinocchio::cholesky::`

#include "jiminy/core/robot/PinocchioOverloadAlgorithms.h"
#include "jiminy/core/constraints/AbstractConstraint.h"
#include "jiminy/core/utilities/Random.h"
#include "jiminy/core/utilities/Helpers.h"
#include "jiminy/core/Constants.h"

#include "jiminy/core/solver/ContactSolvers.h"


namespace jiminy
{
    PGSSolver::PGSSolver(uint32_t const & maxIter,
                         float64_t const & tolAbs,
                         float64_t const & tolRel) :
    maxIter_(maxIter),
    tolAbs_(tolAbs),
    tolRel_(tolRel),
    J_(),
    gamma_(),
    lo_(),
    hi_(),
    fIndices_(),
    b_(),
    y_(),
    dy_(),
    yPrev_(),
    xPrev_()
    {
        // Empty on purpose.
    }

    void PGSSolver::ProjectedGaussSeidelIter(matrixN_t const & A,
                                             vectorN_t const & b,
                                             vectorN_t & x)
    {
        // Backup previous solution
        xPrev_ = x;

        // Update every coefficients only once
        for (uint32_t i = 0; i < fIndices_.size(); ++i)
        {
            // Extract current group of indices
            std::vector<int32_t> const & fIdx = fIndices_[i];
            std::size_t const fSize = fIdx.size();
            float64_t const & hi = hi_[i];
            float64_t const & lo = lo_[i];

            // Update the coefficient if relevant
            if ((fSize == 1 && (hi - lo > EPS)) || (hi > EPS))
            {
                if (fSize < 3)
                {
                    uint32_t const j = fIdx[0];
                    x[j] += (b[j] - A.col(j).dot(x)) / A(j, j);
                }
                else
                {
                    vectorN_t dx(fSize - 1);
                    float64_t scale = 0;
                    for (uint32_t j = 0; j < fSize - 1; ++j)
                    {
                        float64_t const A_j = A(j, j);
                        if (A_j > scale)
                        {
                            scale = A_j;
                        }
                    }
                    for (uint32_t j = 0; j < fSize - 1; ++j)
                    {
                        uint32_t const k = fIdx[j];
                        dx[j] = (b[k] - A.col(k).dot(x)) / scale;
                    }
                    for (uint32_t j = 0; j < fSize - 1; ++j)
                    {
                        x[fIdx[j]] += dx[j];
                    }
                }
            }

            // Project the coefficient between lower and upper bounds
            if (fSize == 1)
            {
                float64_t & e = x[fIdx[0]];
                e = clamp(e, lo, hi);
            }
            else
            {
                float64_t const thr = hi * x[fIdx[fSize - 1]];
                if (thr > EPS)
                {
                    if (fSize == 2)
                    {
                        // Specialization for speedup and numerical stability
                        float64_t & e = x[fIdx[0]];
                        e = clamp(e, -thr, thr);
                    }
                    else
                    {
                        // Generic case
                        float64_t squaredNorm = 0.0;
                        for (uint32_t j = 0; j < fSize - 1; ++j)
                        {
                            float64_t const f = x[fIdx[j]];
                            squaredNorm += f * f;
                        }
                        if (squaredNorm > thr * thr)
                        {
                            float64_t const scale = thr / std::sqrt(squaredNorm);
                            for (uint32_t j = 0; j < fSize - 1; ++j)
                            {
                                x[fIdx[j]] *= scale;
                            }
                        }
                    }
                }
                else
                {
                    // Specialization for speedup
                    for (uint32_t j = 0; j < fSize - 1; ++j)
                    {
                        x[fIdx[j]] *= 0;
                    }
                }
            }
        }
    }

    bool_t PGSSolver::ProjectedGaussSeidelSolver(matrixN_t const & A,
                                                 vectorN_t const & b,
                                                 vectorN_t & x)
    {
        assert(b.size() > 0 && "The number of inequality constraints must be larger than 0.");

        // Initialize the residuals
        y_ = - b;
        y_.noalias() += A * x;

        // Perform multiple PGS loop until convergence or max iter reached
        for (uint32_t iter = 0; iter < maxIter_; ++iter)
        {
            // Do a single iteration
            ProjectedGaussSeidelIter(A, b, x);

            // Check if terminate conditions are satisfied
            yPrev_ = y_;
            y_ = - b;
            y_.noalias() += A * x;
            dy_ = y_ - yPrev_;
            if ((dy_.array().abs() < tolAbs_ || (dy_.array() / y_.array()).abs() < tolRel_).all())
            {
                return true;
            }
        }

        // Impossible to converge
        return false;
    }

    bool_t PGSSolver::BoxedForwardDynamics(pinocchio::Model const & model,
                                           pinocchio::Data & data,
                                           constraintsHolder_t & constraintsHolder,
                                           std::vector<int32_t> const & boundJointsActiveDir,
                                           float64_t const & friction,
                                           float64_t const & torsion,
                                           float64_t const & inv_damping)
    {
        // Compute total active constraint and resize buffers
        bool_t isBounded = false;
        Eigen::Index constraintsRows = 0;
        constraintsHolder.foreach(
            [&constraintsRows, &isBounded](
                std::shared_ptr<AbstractConstraintBase> const & constraint,
                constraintsHolderType_t const & holderType)
            {
                if (!constraint->getIsEnabled())
                {
                    return;
                }
                // Constraints are bounded if any active constraint is not user-specified
                if (holderType != constraintsHolderType_t::USER)
                {
                    isBounded = true;
                }
                constraintsRows += static_cast<Eigen::Index>(constraint->getDim());
            });
        J_.resize(constraintsRows, model.nv);
        gamma_.resize(constraintsRows);
        data.lambda_c.resize(constraintsRows);

        // Update constraints bounds
        Eigen::Index constraintRow = 0U;
        lo_.clear();
        hi_.clear();
        fIndices_.clear();
        constraintsHolder.foreach(
            [&](
                std::shared_ptr<AbstractConstraintBase> const & constraint,
                constraintsHolderType_t const & holderType)
            {
                if (!isBounded || !constraint || !constraint->getIsEnabled())
                {
                    return;
                }

                Eigen::Index const constraintDim = static_cast<Eigen::Index>(constraint->getDim());
                switch (holderType)
                {
                case constraintsHolderType_t::BOUNDS_JOINTS:
                    // The joint is blocked in only one direction
                    if (boundJointsActiveDir[constraintRow])
                    {
                        lo_.emplace_back(-INF);
                        hi_.emplace_back(0.0);
                    }
                    else
                    {
                        lo_.emplace_back(0);
                        hi_.emplace_back(INF);
                    }
                    fIndices_.emplace_back(std::initializer_list<int32_t>{
                        static_cast<int32_t>(constraintRow)
                    });
                    break;
                case constraintsHolderType_t::CONTACT_FRAMES:
                case constraintsHolderType_t::COLLISION_BODIES:
                    // Non-penetration normal force
                    lo_.emplace_back(0);
                    hi_.emplace_back(INF);
                    fIndices_.emplace_back(std::initializer_list<int32_t>{
                        static_cast<int32_t>(constraintRow + 2)
                    });

                    // Torsional friction around normal axis
                    lo_.emplace_back(qNAN);
                    hi_.emplace_back(torsion);
                    fIndices_.emplace_back(std::initializer_list<int32_t>{
                        static_cast<int32_t>(constraintRow + 3),
                        static_cast<int32_t>(constraintRow + 2)
                    });

                    // Friction cone in tangential plane
                    lo_.emplace_back(qNAN);
                    hi_.emplace_back(friction);
                    fIndices_.emplace_back(std::initializer_list<int32_t>{
                        static_cast<int32_t>(constraintRow),
                        static_cast<int32_t>(constraintRow + 1),
                        static_cast<int32_t>(constraintRow + 2)
                    });
                    break;
                case constraintsHolderType_t::USER:
                    for (Eigen::Index i = 0; i < constraintDim; ++i)
                    {
                        lo_.emplace_back(-INF);
                        hi_.emplace_back(INF);
                        fIndices_.emplace_back(std::initializer_list<int32_t>{
                            static_cast<int32_t>(constraintRow + i)
                        });
                    }
                    break;
                default:
                    break;
                }

                // Update constraints jacobian, drift and multipliers
                J_.middleRows(constraintRow, constraintDim) = constraint->getJacobian();
                gamma_.segment(constraintRow, constraintDim) = constraint->getDrift();
                data.lambda_c.segment(constraintRow, constraintDim) = constraint->lambda_;

                constraintRow += constraintDim;
            });

        /* Reorder the groups by size. It should converge faster as it deals
           the rows having less coupling with orders first. */
        std::vector<std::size_t> sortIndex(fIndices_.size());
        std::iota(sortIndex.begin(), sortIndex.end(), 0);
        std::sort(sortIndex.begin(), sortIndex.end(),
            [&](std::size_t const & i, std::size_t const & j)
            {
                return fIndices_[i].size() < fIndices_[j].size();
            });
        for (std::size_t i = 0; i < fIndices_.size() - 1; ++i)
        {
            if (sortIndex[i] == i)
            {
                continue;
            }
            std::size_t j;
            for (j = i + 1; j < fIndices_.size(); ++j)
            {
                if (sortIndex[j] == i)
                {
                    break;
                }
            }
            std::size_t const k = sortIndex[i];
            std::swap(lo_[i], lo_[k]);
            std::swap(hi_[i], hi_[k]);
            std::swap(fIndices_[i], fIndices_[k]);
            std::swap(sortIndex[i], sortIndex[j]);
        }

        // Compute JMinvJt, including cholesky decomposition of inertia matrix
        matrixN_t & A = pinocchio_overload::computeJMinvJt(model, data, J_, false);

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
        data.torque_residual = data.u - data.nle;
        pinocchio::cholesky::solve(model, data, data.torque_residual);

        // Compute b
        b_ = - gamma_;
        b_.noalias() -= J_ * data.torque_residual;

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
            data.lambda_c = pinocchio_overload::solveJMinvJtv(data, b_, true);

            // Return immediatly
            isSuccess = true;
        }
        else
        {
            // Full matrix is needed to enable vectorization
            A.triangularView<Eigen::Upper>() = A.transpose();

            // Run standard PGS algorithm
            isSuccess = ProjectedGaussSeidelSolver(A, b_, data.lambda_c);
        }

        // Update lagrangian multipliers associated with the constraint
        constraintRow = 0U;
        constraintsHolder.foreach(
            [&lambda_c = const_cast<vectorN_t &>(data.lambda_c), &constraintRow](
                std::shared_ptr<AbstractConstraintBase> const & constraint,
                constraintsHolderType_t const & /* holderType */)
            {
                if (!constraint->getIsEnabled())
                {
                    return;
                }
                Eigen::Index const constraintDim = static_cast<Eigen::Index>(constraint->getDim());
                constraint->lambda_ = lambda_c.segment(constraintRow, constraintDim);
                constraintRow += constraintDim;
            });

        // Compute resulting acceleration, no matter if computing forces was successful
        data.ddq.noalias() = J_.transpose() * data.lambda_c;
        pinocchio::cholesky::solve(model, data, data.ddq);
        data.ddq += data.torque_residual;

        return isSuccess;
    }
}
