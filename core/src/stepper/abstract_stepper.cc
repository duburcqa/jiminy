
#include "jiminy/core/stepper/abstract_stepper.h"

namespace jiminy
{
    AbstractStepper::AbstractStepper(const systemDynamics & f,
                                     const std::vector<const Robot *> & robots) :
    f_(f),
    robots_(robots),
    state_(robots),
    stateDerivative_(robots),
    fOutput_(robots)
    {
    }

    bool_t AbstractStepper::tryStep(std::vector<vectorN_t> & qSplit,
                                    std::vector<vectorN_t> & vSplit,
                                    std::vector<vectorN_t> & aSplit,
                                    float64_t & t,
                                    float64_t & dt)
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
            for (const vectorN_t & a : stateDerivative_.a)
            {
                if ((a.array() != a.array()).any())
                {
                    dt = qNAN;
                    result = false;
                }
            }
        }

        // Update output if successful
        if (result)
        {
            t = t_next;
            qSplit = state_.q;
            vSplit = state_.v;
            aSplit = stateDerivative_.a;
        }
        return result;
    }

    const stateDerivative_t & AbstractStepper::f(const float64_t & t, const state_t & state)
    {
        f_(t, state.q, state.v, fOutput_.a);
        fOutput_.v = state.v;
        return fOutput_;
    }
}
