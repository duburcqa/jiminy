
#include "jiminy/core/stepper/EulerExplicitStepper.h"

namespace jiminy
{
    EulerExplicitStepper::EulerExplicitStepper(systemDynamics f, /* Copy on purpose */
                                               std::vector<Robot const *> const & robots):
    AbstractStepper(f, robots)
    {
        // Empty
    }

    bool_t EulerExplicitStepper::tryStepImpl(state_t                 & state,
                                             stateDerivative_t       & stateDerivative,
                                             float64_t         const & t,
                                             float64_t               & dt)
    {
        // Simple explicit Euler: x(t + dt) = x(t) + dt dx(t)
        stateDerivative = f(t, state);
        state.sumInPlace(stateDerivative, dt);

        // Scheme never considers failure.
        return true;
    }
}
