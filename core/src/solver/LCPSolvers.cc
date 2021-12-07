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
                         float64_t const & tolAbs,
                         float64_t const & tolRel) :
    maxIter_(maxIter),
    tolAbs_(tolAbs),
    tolRel_(tolRel),
    b_(),
    xPrev_(),
    y_(),
    yPrev_(),
    dy_()
    {
        // Empty on purpose.
    }

    void PGSSolver::ProjectedGaussSeidelIter(matrixN_t const & A,
                                             vectorN_t const & b,
                                             vectorN_t const & lo,
                                             vectorN_t const & hi,
                                             std::vector<std::vector<int32_t> > const & fIndices,
                                             vectorN_t & x)
    {
        // Backup previous solution
        xPrev_ = x;

        // Update every coefficients sequentially
        for (uint32_t i = 0; i < x.size(); ++i)
        {
            // Extract a single coefficient
            float64_t & e = x[i];

            // Update the coefficient if relevant
            std::vector<int32_t> const & fIdx = fIndices[i];
            std::size_t const fSize = fIdx.size();
            if ((fSize == 0 && (hi[i] - lo[i] > EPS)) || (hi[i] > EPS))
            {
                e += (b[i] - A.col(i).dot(x)) / A(i, i);
                if (fSize > 1)
                {
                    for (auto fIt = fIdx.begin() + 1; fIt != fIdx.end(); ++fIt)
                    {
                        e += A(i, *fIt) / A(i, i) * (x[*fIt] - xPrev_[*fIt]);
                    }
                }
            }

            // Project the coefficient between lower and upper bounds
            if (fSize == 0)
            {
                e = clamp(e, lo[i], hi[i]);
            }
            else
            {
                float64_t f = x[fIdx[0]];
                float64_t const thr = hi[i] * f;
                if (thr > EPS)
                {
                    if (fSize == 1)
                    {
                        // Specialization for speedup and numerical stability
                        e = clamp(e, -thr, thr);
                    }
                    else
                    {
                        // Generic case
                        float64_t squaredNorm = e * e;
                        for (auto fIt = fIdx.begin() + 1; fIt != fIdx.end(); ++fIt)
                        {
                            f = x[*fIt];
                            squaredNorm += f * f;
                        }
                        if (squaredNorm > thr * thr)
                        {
                            float64_t const scale = thr / std::sqrt(squaredNorm);
                            e *= scale;
                            for (auto fIt = fIdx.begin() + 1; fIt != fIdx.end(); ++fIt)
                            {
                                x[*fIt] *= scale;
                            }
                        }
                    }
                }
                else
                {
                    // Specialization for speedup
                    e = 0.0;
                    for (auto fIt = fIdx.begin() + 1; fIt != fIdx.end(); ++fIt)
                    {
                        x[*fIt] = 0.0;
                    }
                }
            }
        }
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

        // Initialize the residuals
        y_.noalias() = A * x - b;

        // Perform multiple PGS loop until convergence or max iter reached
        for (uint32_t iter = 0; iter < maxIter_; ++iter)
        {
            // Do a single iteration
            ProjectedGaussSeidelIter(A, b, lo, hi, fIndices, x);

            // Check if terminate conditions are satisfied
            yPrev_ = y_;
            y_.noalias() = A * x - b;
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
