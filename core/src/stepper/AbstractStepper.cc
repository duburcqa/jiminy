
#include "jiminy/core/stepper/AbstractStepper.h"
#include "jiminy/core/stepper/LieGroup.h"

namespace jiminy
{
    AbstractStepper::AbstractStepper(systemDynamics f,
                                     std::vector<Robot const *> robots):
        f_(std::move(f)),
        robots_(robots)
    {
        // Empty
    }

    bool AbstractStepper::try_step(std::vector<vectorN_t> & q,
                                   std::vector<vectorN_t> & v,
                                   std::vector<vectorN_t> & a,
                                   float64_t              & t,
                                   float64_t              & dt)
    {
        float64_t t_next = t + dt;
        state_t state(robots_, q, v);
        stateDerivative_t stateDerivative(v, a);
        bool result = try_step_impl(state, stateDerivative, t, dt);
        if (result)
        {
            t = t_next;
            q = state.q;
            v = state.v;
            a = stateDerivative.a;
        }
        return result;
    }

    stateDerivative_t AbstractStepper::fWrapper(float64_t const & t,
                                                state_t   const & state)
    {
        std::vector<vectorN_t> a;
        a.resize(state.v.size());
        f_(t, state.q, state.v, a);
        return {state.v, a};
    }
}
