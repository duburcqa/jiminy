
#include "jiminy/core/stepper/AbstractStepper.h"
#include "jiminy/core/stepper/LieGroup.h"

namespace jiminy
{
    AbstractStepper::AbstractStepper(systemDynamics f,
                                     std::vector<Robot const *> const & robots):
    f_(std::move(f)),
    robots_(robots),
    state_(robots),
    stateDerivative_(robots),
    fOutput_(robots)
    {
        // Empty on purpose.
    }

    bool_t AbstractStepper::tryStep(std::vector<vectorN_t> & q,
                                    std::vector<vectorN_t> & v,
                                    std::vector<vectorN_t> & a,
                                    float64_t              & t,
                                    float64_t              & dt)
    {
        float64_t t_next = t + dt;
        state_.q = q;
        state_.v = v;
        stateDerivative_.v = v;
        stateDerivative_.a = a;
        bool_t result = tryStepImpl(state_, stateDerivative_, t, dt);
        if (result)
        {
            t = t_next;
            q = state_.q;
            v = state_.v;
            a = stateDerivative_.a;
        }
        return result;
    }

    stateDerivative_t const & AbstractStepper::f(float64_t const & t,
                                                 state_t   const & state)
    {
        f_(t, state.q, state.v, fOutput_.a);
        fOutput_.v = state.v;
        return fOutput_;
    }
}
