
#include "jiminy/core/stepper/RungeKuttaStepper.h"
#include "jiminy/core/stepper/LieGroup.h"

namespace jiminy
{
    RungeKuttaStepper::RungeKuttaStepper(systemDynamics f, /* Copy on purpose */
                                         std::vector<Robot const *> const & robots,
                                         float64_t const & tolRel,
                                         float64_t const & tolAbs,
                                         matrixN_t const & RungeKuttaMatrix,
                                         vectorN_t const & cNodes,
                                         vectorN_t const & bWeights,
                                         vectorN_t const & eWeights):
    AbstractStepper(f, robots),
    tolRel_(tolRel),
    tolAbs_(tolAbs),
    A_(RungeKuttaMatrix),
    c_(cNodes),
    b_(bWeights),
    e_(eWeights)
    {
        // TODO assert sizes
        // TODO: the number of points needed for the numerical scheme is known
        // in advance so maybe this could be preallocated better.
    }

    bool RungeKuttaStepper::tryStepImpl(state_t           & state,
                                        stateDerivative_t & stateDerivative,
                                        float64_t   const & t,
                                        float64_t         & dt)
    {
        // Compute all ki's
        std::vector<stateDerivative_t> ki;

        // First ki is simply the provided stateDerivative
        ki.push_back(stateDerivative);

        for (uint32_t i = 1; i < c_.size(); ++i)
        {
            state_t xi = state;
            for (uint32_t j = 0; j < i; ++j)
            {
                xi += dt * A_(i, j) * ki[j];
            }
            ki.push_back(f(t + c_[i] * dt, xi));
        }

        // Now we have all the ki's: compute the solution
        state_t alternativeSolution = state;

        for (uint32_t i = 0; i < ki.size(); ++i)
        {
            state += dt * b_[i] * ki[i];
            alternativeSolution += dt * e_[i] * ki[i];
        }

        // Copy last ki as state derivative
        stateDerivative = ki.back();

        // Evaluate error between both states to adjust step
        float64_t errorNorm = state.difference(alternativeSolution).norm();

        // Compute error scale
        float64_t scale = tolAbs_ + tolRel_ * state.normInf();
        float64_t normalizedError = errorNorm / scale;

        // Adjust step size: this is taken from scipy source code
        float64_t factor = 0.0;
        if (normalizedError < 1.0)
        {
            if (normalizedError < EPS)
            {
                factor = RK::MIN_FACTOR;
            }
            else
            {
                factor = std::min(RK::MAX_FACTOR,
                                  RK::SAFETY * std::pow(normalizedError, RK::ERROR_EXPONENT));
            }
        }
        else
        {
            factor = std::max(RK::MIN_FACTOR,
                              RK::SAFETY * std::pow(normalizedError, RK::ERROR_EXPONENT));
        }
        dt *= factor;

        // Step failed if normalizedError is larger than 1.0
        return normalizedError > 1.0;
    }
}
