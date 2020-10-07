#include "jiminy/core/stepper/RungeKuttaDOPRIStepper.h"

namespace jiminy
{
    RungeKuttaDOPRIStepper::RungeKuttaDOPRIStepper(systemDynamics f, /* Copy on purpose */
                                                   std::vector<Robot const *> const & robots,
                                                   float64_t const & tolRel,
                                                   float64_t const & tolAbs):
    AbstractRungeKuttaStepper(f, robots, DOPRI::A, DOPRI::b, DOPRI::c, true),
    tolRel_(tolRel),
    tolAbs_(tolAbs)
    {
        // Empty on purpose
    }

    bool_t RungeKuttaDOPRIStepper::adjustStep(state_t   const & initialState,
                                              state_t   const & solution,
                                              float64_t       & dt)
    {
        float64_t error = computeError(initialState, solution, dt);
        adjustStepImpl(error, dt);
        return error < 1.0;
    }

    float64_t RungeKuttaDOPRIStepper::computeError(state_t const & initialState,
                                                   state_t const & solution,
                                                   float64_t const & dt)
    {
        // Compute alternative solution.
        stateDerivative_t dvIntAlt = dt * DOPRI::e[0] * ki_[0];
        for (uint32_t i = 1; i < ki_.size(); ++i)
        {
            dvIntAlt += dt * DOPRI::e[i] * ki_[i];
        }
        state_t alternativeSolution = initialState + dvIntAlt;

        // Evaluate error between both states to adjust step
        float64_t errorNorm = solution.difference(alternativeSolution).norm();

        // Compute error scale
        float64_t scale = tolAbs_ + tolRel_ * initialState.normInf();
        return  errorNorm / scale;
    }

    void RungeKuttaDOPRIStepper::adjustStepImpl(float64_t const & error,
                                                float64_t       & dt)
    {
        // Adjustment algorithm from boost implementation.
        if (error < 1.0)
        {
            // Only increase if error is sufficiently small.
            if (error < 0.5)
            {
                // Prevent numeric rounding error when close to zero.
                float64_t newError = std::max(error,
                                              std::pow(DOPRI::MAX_FACTOR, -DOPRI::STEPPER_ORDER));
                dt *= DOPRI::SAFETY * std::pow(newError, -1.0 / DOPRI::STEPPER_ORDER);
            }
        }
        else
        {
            dt *= std::max(DOPRI::SAFETY * std::pow(error, -1.0 / (DOPRI::STEPPER_ORDER - 1)),
                           DOPRI::MIN_FACTOR);
        }
    }
}
