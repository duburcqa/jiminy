
#include "jiminy/core/stepper/abstract_stepper.h"

namespace jiminy
{
    AbstractStepper::AbstractStepper(const systemDynamics & f,
                                     const std::vector<const Robot *> & robots) noexcept :
    f_{f},
    robots_{robots},
    state_(robots),
    stateDerivative_(robots),
    fOutput_(robots)
    {
    }

    stepper::StatusInfo AbstractStepper::tryStep(std::vector<Eigen::VectorXd> & qSplit,
                                                 std::vector<Eigen::VectorXd> & vSplit,
                                                 std::vector<Eigen::VectorXd> & aSplit,
                                                 double & t,
                                                 double & dt)
    {
        // Initialize return status
        stepper::StatusInfo status{stepper::ReturnCode::IS_SUCCESS, {}};

        // Update buffers
        double t_next = t + dt;
        state_.q = qSplit;
        state_.v = vSplit;
        stateDerivative_.v = vSplit;
        stateDerivative_.a = aSplit;

        // Do a single step
        try
        {
            // Try doing step
            if (!tryStepImpl(state_, stateDerivative_, t, dt))
            {
                return {stepper::ReturnCode::IS_FAILURE, {}};
            }

            // Make sure everything went fine
            for (const Eigen::VectorXd & a : stateDerivative_.a)
            {
                if ((a.array() != a.array()).any())
                {
                    JIMINY_THROW(std::runtime_error,
                                 "The integrated acceleration contains 'nan'.");
                }
            }
        }
        catch (...)
        {
            return {stepper::ReturnCode::IS_ERROR, std::current_exception()};
        }

        // Update output if successful
        t = t_next;
        qSplit = state_.q;
        vSplit = state_.v;
        aSplit = stateDerivative_.a;

        return {stepper::ReturnCode::IS_SUCCESS, {}};
    }

    const StateDerivative & AbstractStepper::f(double t, const State & state)
    {
        f_(t, state.q, state.v, fOutput_.a);
        fOutput_.v = state.v;
        return fOutput_;
    }
}
