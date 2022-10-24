
#include "jiminy/core/stepper/EulerExplicitStepper.h"

namespace jiminy
{
    EulerExplicitStepper::EulerExplicitStepper(systemDynamics const & f,
                                               std::vector<Robot const *> const & robots):
    AbstractStepper(f, robots),
    stateBuffer_(robots)
    {
        // Empty on purpose
    }

    bool_t EulerExplicitStepper::tryStepImpl(state_t                 & state,
                                             stateDerivative_t       & stateDerivative,
                                             float64_t         const & t,
                                             float64_t               & dt)
    {
        /* TODO: Replace `sum` with `sumInPlace` once pinocchio is fixed:
           https://github.com/stack-of-tasks/pinocchio/pull/1775
           Simple explicit Euler: x(t + dt) = x(t) + dt dx(t) */
        stateDerivative = f(t, state);
        // state.sumInPlace(stateDerivative, dt);
        state.sum(dt * stateDerivative, stateBuffer_);
        state = stateBuffer_;

        /* By default INF is returned in case of fixed time step, so that the
           engine will always try to perform the latest timestep possible,
           or stop to the next breakpoint otherwise. */
        dt = INF;

        // Scheme never considers failure.
        return true;
    }
}
