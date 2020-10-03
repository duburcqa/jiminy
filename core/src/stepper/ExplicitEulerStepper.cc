
#include "jiminy/core/stepper/ExplicitEulerStepper.h"
#include "jiminy/core/stepper/LieGroup.h"

namespace jiminy
{
    ExplicitEulerStepper::ExplicitEulerStepper(systemDynamics f, /* Copy on purpose */
                                               std::vector<Robot const *> robots):
    AbstractStepper(f, robots)
    {
        // Empty
    }

    bool ExplicitEulerStepper::try_step_impl(state_t           & state,
                                             stateDerivative_t & stateDerivative,
                                             float64_t   const & t,
                                             float64_t         & dt)
    {
        // Simple explicit Euler: x(t + dt) = x(t) + dt dx(t)
        state += dt * fWrapper(t, state);

        // Scheme never considers failure.
        return true;
    }
}
