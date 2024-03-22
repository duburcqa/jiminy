#include "jiminy/core/stepper/runge_kutta_dopri_stepper.h"

namespace jiminy
{
    RungeKuttaDOPRIStepper::RungeKuttaDOPRIStepper(const systemDynamics & f,
                                                   const std::vector<const Robot *> & robots,
                                                   double tolRel,
                                                   double tolAbs) noexcept :
    AbstractRungeKuttaStepper(f, robots, DOPRI::A, DOPRI::b, DOPRI::c, true),
    tolRel_{tolRel},
    tolAbs_{tolAbs},
    scale_(robots),
    otherSolution_(robots),
    error_(robots)
    {
    }

    bool RungeKuttaDOPRIStepper::adjustStep(
        const State & initialState, const State & solution, double & dt)
    {
        // Estimate the integration error
        const double error = computeError(initialState, solution, dt);

        // Make sure the error is well defined, otherwise throw an exception
        if (std::isnan(error))
        {
            THROW_ERROR(std::runtime_error, "The estimated integration error contains 'nan'.");
        }

        /* Adjustment algorithm from boost implementation.
           For technical reference, see original boost::odeint implementation:
           https://beta.boost.org/doc/libs/1_82_0/libs/numeric/odeint/doc/html/boost_numeric_odeint/odeint_in_detail/steppers.html#boost_numeric_odeint.odeint_in_detail.steppers.controlled_steppers
        */
        if (error < 1.0)
        {
            /* Increase step size only if the error is sufficiently small.
               The threshold must be chosen in a way to guarantee that it actually decreases. */
            if (error <
                std::min(DOPRI::ERROR_THRESHOLD, std::pow(DOPRI::SAFETY, DOPRI::STEPPER_ORDER)))
            {
                /* Prevent numeric rounding error when close to zero.
                   Multiply step size by 'DOPRI::SAFETY / (error ** (1 / DOPRI::STEPPER_ORDER))',
                   up to 'DOPRI::MAX_FACTOR'. */
                const double clippedError = std::max(
                    error, std::pow(DOPRI::MAX_FACTOR / DOPRI::SAFETY, -DOPRI::STEPPER_ORDER));
                dt *= DOPRI::SAFETY * std::pow(clippedError, -1.0 / DOPRI::STEPPER_ORDER);
            }
            return true;
        }
        else
        {
            dt *= std::max(DOPRI::SAFETY * std::pow(error, -1.0 / (DOPRI::STEPPER_ORDER - 2)),
                           DOPRI::MIN_FACTOR);
            return false;
        }
    }

    double RungeKuttaDOPRIStepper::computeError(
        const State & initialState, const State & solution, double dt)
    {
        // Compute error scale given absolute and relative tolerance
        otherSolution_.setZero();
        initialState.difference(otherSolution_, scale_);
        scale_.absInPlace();
        scale_ *= tolRel_;
        scale_ += tolAbs_;

        // Compute alternative solution
        stateIncrement_.setZero();
        for (std::size_t i = 0; i < ki_.size(); ++i)
        {
            stateIncrement_.sumInPlace(ki_[i], dt * DOPRI::e[i]);
        }
        initialState.sum(stateIncrement_, otherSolution_);

        // Evaluate error between both states to adjust step
        solution.difference(otherSolution_, error_);

        // Return element-wise maximum rescaled error
        error_ /= scale_;
        return error_.normInf();
    }
}
