#include "pinocchio/multibody/model.hpp"     // `pinocchio::Model`
#include "pinocchio/multibody/data.hpp"      // `pinocchio::Data`
#include "pinocchio/algorithm/cholesky.hpp"  // `pinocchio::cholesky::`

#include "jiminy/core/robot/PinocchioOverloadAlgorithms.h"
#include "jiminy/core/utilities/Random.h"
#include "jiminy/core/utilities/Helpers.h"
#include "jiminy/core/Constants.h"

#include "jiminy/core/solver/LCPSolvers.h"


namespace jiminy
{
    PGSSolver::PGSSolver(uint32_t const & maxIter,
                         uint32_t const & randomPermutationPeriod,
                         float64_t const & tolAbs,
                         float64_t const & tolRel) :
    maxIter_(maxIter),
    randomPermutationPeriod_(randomPermutationPeriod),
    tolAbs_(tolAbs),
    tolRel_(tolRel),
    indices_(),
    lastShuffle_(0U),
    b_(),
    y_(),
    yPrev_()
    {
        // Empty on purpose.
    }

    bool_t PGSSolver::ProjectedGaussSeidelIter(matrixN_t const & A,
                                               vectorN_t const & b,
                                               vectorN_t const & lo,
                                               vectorN_t const & hi,
                                               std::vector<std::vector<int32_t> > const & fIndices,
                                               vectorN_t & x)
    {
        // Shuffle coefficient update order to break any repeating cycles
        if (randomPermutationPeriod_ > 0 && lastShuffle_ > randomPermutationPeriod_)
        {
            shuffleIndices(indices_);
            lastShuffle_ = 0U;
        }
        ++lastShuffle_;

        // Backup previous solution
        yPrev_.noalias() = A * x;

        // Update every coefficients sequentially
        for (uint32_t const & i : indices_)
        {
            // Extract single coefficient
            float64_t & e = x[i];

            // Update a single coefficient
            e += (b[i] - A.col(i).dot(x)) / A(i, i);

            // Project the coefficient between lower and upper bounds
            std::vector<int32_t> const & fIdx = fIndices[i];
            std::size_t const fSize = fIdx.size();
            if (fSize == 0)
            {
                e = clamp(e, lo[i], hi[i]);
            }
            else
            {
                float64_t f = x[fIdx[0]];
                float64_t thr = hi[i] * f;
                if (thr > EPS)
                {
                    if (fSize == 1)
                    {
                        e = clamp(e, -thr, thr);
                    }
                    else
                    {
                        thr *= thr;
                        for (auto fIt = fIdx.begin() + 1; fIt != fIdx.end(); ++fIt)
                        {
                            f = x[*fIt];
                            thr -= f * f;
                        }
                        if (thr > EPS)
                        {
                            thr = std::sqrt(thr);
                            e = clamp(e, -thr, thr);
                        }
                        else
                        {
                            e = 0.0;
                        }
                    }
                }
                else
                {
                    e = 0.0;
                }
            }
        }

        // Check if terminate conditions are satisfied
        y_.noalias() = A * x;
        bool_t isSuccess = (
            (y_ - yPrev_).array().abs() < tolAbs_ ||
            ((y_ - yPrev_).array() / y_.array()).abs() < tolRel_
        ).all();

        return isSuccess;
    }

    bool_t PGSSolver::ProjectedGaussSeidelSolver(matrixN_t & A,
                                                 vectorN_t & b,
                                                 vectorN_t const & lo,
                                                 vectorN_t const & hi,
                                                 std::vector<std::vector<int32_t> > const & fIndices,
                                                 vectorN_t & x)
    {
        /* The implementation is partially adapted from Dart Simulator:
           https://github.com/dartsim/dart/blob/master/dart/constraint/PgsBoxedLcpSolver.cpp */
        assert(b.size() > 0 && "The number of inequality constraints must be larger than 0.");

        /* Adapt shuffling indices if the number of indices has changed.
           Note that it may converge faster to enforce constraints in reverse order,
           since usually constraints bounds dependending on others have lower indices
           by design, aka. the linear friction pyramid.
           TODO: take into account the actual value of 'fIndices' to order the indices. */
        size_t const nIndices = b.size();
        size_t const nIndicesOrig = indices_.size();
        if (nIndicesOrig < nIndices)
        {
            indices_.resize(nIndices);
            std::generate(indices_.begin() + nIndicesOrig, indices_.end(),
                          [n = static_cast<uint32_t>(nIndices - 1)]() mutable { return n--; });
        }
        else if (nIndicesOrig > nIndices)
        {
            size_t shiftIdx = nIndices;
            for (size_t i = 0; i < nIndices; ++i)
            {
                if (static_cast<size_t>(indices_[i]) >= nIndices)
                {
                    for (size_t j = shiftIdx; j < nIndicesOrig; ++j)
                    {
                        ++shiftIdx;
                        if (static_cast<size_t>(indices_[j]) < nIndices)
                        {
                            indices_[i] = indices_[j];
                            break;
                        }
                    }
                }
            }
            indices_.resize(nIndices);
        }

        // Normalizing
        // for (Eigen::Index i = 0; i < b.size(); ++i)
        // {
        //     b[i] /= A(i, i);
        //     A.col(i).array() /= A(i, i);
        // }

        // Perform multiple PGS loop until convergence or max iter reached
        for (uint32_t iter = 0; iter < maxIter_; ++iter)
        {
            bool_t isSuccess = ProjectedGaussSeidelIter(A, b, lo, hi, fIndices, x);
            if (isSuccess)
            {
                // Do NOT shuffle indices unless necessary to avoid discontinuities
                lastShuffle_ = 0U;
                return true;
            }
        }

        // Impossible to converge
        return false;
    }

    bool_t PGSSolver::BoxedForwardDynamics(pinocchio::Model const & model,
                                           pinocchio::Data & data,
                                           vectorN_t const & tau,
                                           Eigen::Ref<matrixN_t const> const & J,
                                           Eigen::Ref<vectorN_t const> const & gamma,
                                           float64_t const & inv_damping,
                                           vectorN_t const & lo,
                                           vectorN_t const & hi,
                                           std::vector<std::vector<int32_t> > const & fIndices)
    {
        // Define some proxies for convenience
        vectorN_t & f = data.lambda_c;

        // Compute JMinvJt, including cholesky decomposition of inertia matrix
        matrixN_t & A = pinocchio_overload::computeJMinvJt(model, data, J, true);

        // Compute the dynamic drift (control - nle)
        data.torque_residual = tau - data.nle;
        pinocchio::cholesky::solve(model, data, data.torque_residual);

        /* Add regularization term in case A is not inversible.
           Note that Mujoco defines an impedance function that depends on
           the distance instead of a constant value to model soft contacts.
           See: - http://mujoco.org/book/modeling.html#CSolver
                - http://mujoco.org/book/computation.html#soParameters  */
        A.diagonal() += clamp(
            A.diagonal() * inv_damping,
            PGS_MIN_REGULARIZER,
            INF);

        // Compute b
        b_.noalias() = - J * data.torque_residual;
        b_ -= gamma;

        // Compute resulting forces solving forward dynamics
        bool_t isSuccess = false;
        if (lo.array().isInf().all() && hi.array().isInf().all())
        {
            /* There is no inequality constraint, so the problem can be
               solved exactly and efficiently using cholesky decomposition.

               The implementation of this particular case is based on
               `pinocchio::forwardDynamics methods` without modification.
               See https://github.com/stack-of-tasks/pinocchio/blob/master/src/algorithm/contact-dynamics.hxx */

            // Compute the Lagrange Multipliers
            f = pinocchio_overload::solveJMinvJtv(data, b_, true);

            // Return immediatly
            isSuccess = true;
        }
        else
        {
            // Run standard PGS algorithm
            isSuccess = ProjectedGaussSeidelSolver(A, b_, lo, hi, fIndices, f);
        }

        // Compute resulting acceleration, no matter if computing forces was successful
        data.ddq.noalias() = J.transpose() * f;
        pinocchio::cholesky::solve(model, data, data.ddq);
        data.ddq += data.torque_residual;

        return isSuccess;
    }
}
