
#include "jiminy/core/stepper/AbstractRungeKuttaStepper.h"
#include "jiminy/core/stepper/LieGroup.h"


namespace jiminy
{
    AbstractRungeKuttaStepper::AbstractRungeKuttaStepper(systemDynamics f, /* Copy on purpose */
                                         std::vector<Robot const *> const & robots,
                                         matrixN_t const & RungeKuttaMatrix,
                                         vectorN_t const & bWeights,
                                         vectorN_t const & cNodes,
                                         bool_t    const & isFSAL):
    AbstractStepper(f, robots),
    A_(RungeKuttaMatrix),
    b_(bWeights),
    c_(cNodes),
    isFSAL_(isFSAL),
    ki_(cNodes.size(), stateDerivative_t(robots))
    {
        assert(A_.rows() == A_.cols());
        assert(c_.size() == A_.rows());
        assert(b_.size() == b_.rows());
    }

    bool_t AbstractRungeKuttaStepper::tryStepImpl(state_t                 & state,
                                                  stateDerivative_t       & stateDerivative,
                                                  float64_t         const & t,
                                                  float64_t               & dt)
    {
        // First ki is simply the provided stateDerivative
        ki_[0] = stateDerivative;

        for (uint32_t i = 1; i < c_.size(); ++i)
        {
            stateDerivative_t stateIncrement = dt * A_(i, 0) * ki_[0];
            for (uint32_t j = 1; j < i; ++j)
            {
                stateIncrement += dt * A_(i, j) * ki_[j];
            }
            ki_[i] = f(t + c_[i] * dt, state + stateIncrement);
        }

        /* Now we have all the ki's: compute the solution.
           Sum the velocities before summing into position the accuracy is greater
           for summing vectors than for summing velocities into lie groups. */
        stateDerivative_t dvInc = dt * b_[0] * ki_[0];
        for (uint32_t i = 1; i < ki_.size(); ++i)
        {
            dvInc += dt * b_[i] * ki_[i];
        }
        state_t const solution = state + dvInc;

        // Evaluate the solution's error for step adjustment
        bool_t const hasSucceeded = adjustStep(state, solution, dt);

        // Compute the  next state and state derivative if success
        if (hasSucceeded)
        {
            state = solution;
            if (isFSAL_)
            {
                stateDerivative = ki_.back();
            }
            else
            {
                stateDerivative = f(t, state);
            }
        }

        return hasSucceeded;
    }

    bool_t AbstractRungeKuttaStepper::adjustStep(state_t   const & initialState,
                                                 state_t   const & solution,
                                                 float64_t       & dt)
    {
        // Fixed-step by default, which never fails
        return true;
    }
}

