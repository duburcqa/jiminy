
#include "jiminy/core/stepper/euler_explicit_stepper.h"

namespace jiminy
{
    EulerExplicitStepper::EulerExplicitStepper(const systemDynamics & f,
                                               const std::vector<const Robot *> & robots) :
    AbstractStepper(f, robots)
    {
    }

    bool EulerExplicitStepper::tryStepImpl(
        state_t & state, stateDerivative_t & stateDerivative, double t, double & dt)
    {
        // Simple explicit Euler: x(t + dt) = x(t) + dt dx(t)
        state.sumInPlace(stateDerivative, dt);

        // Compute the next state derivative
        stateDerivative = f(t, state);

        // By default INF is returned in case of fixed timestep. It must be managed externally.
        dt = INF;

        // Scheme never considers failure.
        return true;
    }
}
