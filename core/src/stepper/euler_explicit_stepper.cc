
#include "jiminy/core/stepper/euler_explicit_stepper.h"

namespace jiminy
{
    bool EulerExplicitStepper::tryStepImpl(
        State & state, StateDerivative & stateDerivative, double t, double & dt)
    {
        // Simple explicit Euler: x(t + dt) = x(t) + dt dx(t)
        state.sumInPlace(stateDerivative, dt);

        // Compute the next state derivative
        const double t_next = t + dt;
        stateDerivative = f(t_next, state);

        /* By default INF is returned no matter what for fixed-timestep integrators.
           The user is responsible for managing it externally. */
        dt = INF;

        // Scheme never considers failure
        return true;
    }
}
