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
    b_()
    {
        // Empty on purpose.
    }

    bool_t PGSSolver::ProjectedGaussSeidelIter(matrixN_t const & A,
                                               vectorN_t const & b,
                                               vectorN_t const & lo,
                                               vectorN_t const & hi,
                                               std::vector<int32_t> const & fIdx,
                                               bool_t const & checkAbs,
                                               bool_t const & checkRel,
                                               vectorN_t & x)
    {
        bool_t isSuccess = true;

        // Shuffle coefficient update order to break any repeating cycles
        if (randomPermutationPeriod_ > 0 && lastShuffle_ > randomPermutationPeriod_)
        {
            shuffleIndices(indices_);
            lastShuffle_ = 0U;
        }
        ++lastShuffle_;

        // Update every coefficients sequentially
        for (int32_t const & i : indices_)
        {
            // Update a single coefficient
            float64_t const xPrev = x[i];
            x[i] += (b[i] - A.row(i).dot(x)) / A(i, i);

            // Project the coefficient between lower and upper bounds
            if (fIdx[i] < 0)
            {
                x[i] = clamp(x[i], lo[i], hi[i]);
            }
            else
            {
                float64_t const hiTmp = hi[i] * x[fIdx[i]];
                x[i] = clamp(x[i], - hiTmp, hiTmp);
            }

            // Check if still possible to terminate after complete update
            if (checkAbs)
            {
                isSuccess = isSuccess && (std::abs(x[i] - xPrev) < tolAbs_);
            }
            if (checkRel && std::abs(x[i]) > EPS_DIVISION)
            {
                isSuccess = isSuccess && (std::abs((x[i] - xPrev) / x[i]) < tolRel_);
            }
        }

        return isSuccess;
    }

    bool_t PGSSolver::ProjectedGaussSeidelSolver(matrixN_t & A,
                                                 vectorN_t & b,
                                                 vectorN_t const & lo,
                                                 vectorN_t const & hi,
                                                 std::vector<int32_t> const & fIdx,
                                                 vectorN_t & x)
    {
        /* The implementation is partially adapted from Dart Simulator:
           https://github.com/dartsim/dart/blob/master/dart/constraint/PgsBoxedLcpSolver.cpp */
        assert(b.size() > 0 && "The number of inequality constraints must be larger than 0.");

        /* Adapt shuffling indices if the number of indices has changed.
           Note that it may converge faster to enforce constraints in reverse order,
           since usually constraints bounds dependending on others have lower indices
           by design. For instance, for friction, x and y  */
        size_t const nIndicesOrig = indices_.size();
        size_t const nIndices = b.size();
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
        for (Eigen::Index i = 0; i < b.size(); ++i)
        {
            b[i] /= A(i, i);
            A.row(i).array() /= A(i, i);
        }

        // Perform multiple PGS loop until convergence or max iter reached
        for (uint32_t iter = 0; iter < maxIter_; ++iter)
        {
            bool_t isSuccess = ProjectedGaussSeidelIter(A, b, lo, hi, fIdx, false, true, x);
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
                                           std::vector<int32_t> const & fIdx)
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
            isSuccess = ProjectedGaussSeidelSolver(A, b_, lo, hi, fIdx, f);
        }

        // Compute resulting acceleration, no matter if computing forces was successful
        data.ddq.noalias() = J.transpose() * f;
        pinocchio::cholesky::solve(model, data, data.ddq);
        data.ddq += data.torque_residual;

        return isSuccess;
    }
}
