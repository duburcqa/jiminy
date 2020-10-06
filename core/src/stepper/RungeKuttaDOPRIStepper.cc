#include "jiminy/core/stepper/RungeKuttaDOPRIStepper.h"

namespace jiminy
{
    RungeKuttaDOPRIStepper::RungeKuttaDOPRIStepper(systemDynamics f, /* Copy on purpose */
                                                   std::vector<Robot const *> const & robots,
                                                   float64_t const & tolRel,
                                                   float64_t const & tolAbs):
    AbstractRungeKuttaStepper(f,
                    robots,
                    DOPRI::A,
                    DOPRI::b,
                    DOPRI::c,
                    true),
    tolRel_(tolRel),
    tolAbs_(tolAbs)
    {
        // Empty on purpose
    }


    bool RungeKuttaDOPRIStepper::adjustStep(state_t const & initialState,
                                            state_t const & solution,
                                            float64_t & dt)
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
        float64_t errorNorm = initialState.difference(alternativeSolution).norm();

        // Compute error scale
        float64_t scale = tolAbs_ + tolRel_ * initialState.normInf();
        return  errorNorm / scale;
    }


    void RungeKuttaDOPRIStepper::adjustStepImpl(float64_t const& error, float64_t & dt)
    {
        float64_t factor = 0.0;
        if (error < 1.0)
        {
            if (error < EPS)
            {
                factor = RK::MIN_FACTOR;
            }
            else
            {
                factor = std::min(RK::MAX_FACTOR,
                                  RK::SAFETY * std::pow(error, RK::ERROR_EXPONENT));
            }
        }
        else
        {
            factor = std::max(RK::MIN_FACTOR,
                              RK::SAFETY * std::pow(error, RK::ERROR_EXPONENT));
        }
        dt *= factor;
    }
}
