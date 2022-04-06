
#include "jiminy/core/stepper/AbstractStepper.h"

namespace jiminy
{
    AbstractStepper::AbstractStepper(systemDynamics const & f,
                                     std::vector<Robot const *> const & robots):
    f_(f),
    robots_(robots),
    state_(robots),
    stateDerivative_(robots),
    fOutput_(robots)
    {
        // Empty on purpose.
    }

    bool_t AbstractStepper::tryStep(std::vector<vectorN_t> & qSplit,
                                    std::vector<vectorN_t> & vSplit,
                                    std::vector<vectorN_t> & aSplit,
                                    float64_t              & t,
                                    float64_t              & dt)
    {
        // Update buffers
        float64_t t_next = t + dt;
        state_.q = qSplit;
        state_.v = vSplit;
        stateDerivative_.v = vSplit;
        stateDerivative_.a = aSplit;

        // Try doing a single step
        bool_t result = tryStepImpl(state_, stateDerivative_, t, dt);

        // Make sure everything went fine
        if (result)
        {
            for (vectorN_t const & a : stateDerivative_.a)
            {
                if ((a.array() != a.array()).any())
                {
                    dt = qNAN;
                    result = false;
                }
            }
        }

        // Update output if successfull
        if (result)
        {
            t = t_next;
            qSplit = state_.q;
            vSplit = state_.v;
            aSplit = stateDerivative_.a;
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
