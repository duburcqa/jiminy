#include <math.h>
#include <climits>
#include <numeric>     /* iota */
#include <stdlib.h>     /* srand, rand */
#include <random>

#ifndef _WIN32
#include <pwd.h>
#include <unistd.h>
#include <getopt.h>
#else
#include <stdlib.h>
#include <stdio.h>
#endif

#include "pinocchio/multibody/model.hpp"
#include "pinocchio/algorithm/cholesky.hpp"

#include "jiminy/core/Utilities.h"
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

        // Shuffle coefficient update order to break repeating cycles
        if ((int32_t) indices_.size() != b.size())
        {
            indices_.resize(b.size());
            std::generate(indices_.begin(), indices_.end(),
                          [n = 0]() mutable { return n++; });
            lastShuffle_ = randomPermutationPeriod_;
        }
        if (std::isfinite(static_cast<float64_t>(randomPermutationPeriod_))  // MSVC does not support `isfinite`/`isnan` on integers...
         && lastShuffle_ % randomPermutationPeriod_ == 0)
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

        // Single loop of standard Projected Gauss Seidel algorithm
        bool_t isSuccess = ProjectedGaussSeidelIter(A, b, lo, hi, fIdx, true, false, x);

        // If termination condition has been reached, return right now
        if (isSuccess)
        {
            return true;
        }

        // Normalizing
        for (int32_t i = 0; i < b.size(); ++i)
        {
            b[i] /= A(i, i);
            A.row(i).array() /= A(i, i);
        }

        // Perform multiple PGS loop until convergence or max iter reached
        for (uint32_t iter = 1; iter < maxIter_; ++iter)
        {
            isSuccess = ProjectedGaussSeidelIter(A, b, lo, hi, fIdx, false, true, x);
            if (isSuccess)
            {
                return true;
            }
        }

        // Impossible to converge
        return false;
    }

    bool_t PGSSolver::BoxedForwardDynamics(pinocchio::Model const & model,
                                           pinocchio::Data & data,
                                           vectorN_t const & tau,
                                           matrixN_t const & J,
                                           vectorN_t const & gamma,
                                           float64_t const & inv_damping,
                                           vectorN_t const & lo,
                                           vectorN_t const & hi,
                                           std::vector<int32_t> const & fIdx)
    {
        // Define some proxies for convenience
        matrixN_t & A = data.JMinvJt;
        vectorN_t & f = data.lambda_c;

        // Keep previous value if size did not change, reset it otherwise
        if (f.size() != gamma.size())
        {
            f = vectorN_t::Zero(gamma.size());
        }

        // Compute the UDUt decomposition of data.M
        pinocchio::cholesky::decompose(model, data);

        // Compute the dynamic drift (control - nle)
        data.torque_residual = tau - data.nle;
        pinocchio::cholesky::solve(model, data, data.torque_residual);

        // Compute U^-1 * J.T
        data.sDUiJt = J.transpose();
        pinocchio::cholesky::Uiv(model, data, data.sDUiJt);
        for(int32_t k=0; k<model.nv; ++k)
        {
            data.sDUiJt.row(k) /= sqrt(data.D[k]);
        }

        // Compute A
        A.noalias() = data.sDUiJt.transpose() * data.sDUiJt;

        // Add regularization term in case A is not inversible
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
            /* There is no constraint, so the problem can be solved exactly
            and efficiently using cholesky decomposition.

            The implementation of this particular case is based on
            `pinocchio::forwardDynamics methods` without modification.
            See https://github.com/stack-of-tasks/pinocchio/blob/master/src/algorithm/contact-dynamics.hxx */

            // Compute Cholesky decomposition
            data.llt_JMinvJt.compute(A);

            // Compute the Lagrange Multipliers
            f = data.llt_JMinvJt.solve(b_);

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
