
#include "jiminy/core/stepper/AbstractRungeKuttaStepper.h"


namespace jiminy
{
    AbstractRungeKuttaStepper::AbstractRungeKuttaStepper(systemDynamics const & f,
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
    ki_(cNodes.size(), stateDerivative_t(robots)),
    stateIncrement_(robots),
    stateBuffer_(robots),
    candidateSolution_(robots)

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

        for (Eigen::Index i = 1; i < c_.size(); ++i)
        {
            stateIncrement_.setZero();
            for (Eigen::Index j = 0; j < i; ++j)
            {
                stateIncrement_.sumInPlace(ki_[j], dt * A_(i, j));  // Equivalent to `stateIncrement_ += dt * A_(i, j) * ki_[j]` but more efficient because it avoid temporaries
            }
            state.sum(stateIncrement_, stateBuffer_);
            ki_[i] = f(t + c_[i] * dt, stateBuffer_);
        }

        /* Now we have all the ki's: compute the solution.
           Sum the velocities before summing into position the accuracy is greater
           for summing vectors than for summing velocities into lie groups. */
        stateIncrement_.setZero();
        for (std::size_t i = 0; i < ki_.size(); ++i)
        {
            stateIncrement_.sumInPlace(ki_[i], dt * b_[i]);
        }
        state.sum(stateIncrement_, candidateSolution_);

        // Evaluate the solution's error for step adjustment
        bool_t const hasSucceeded = adjustStep(state, candidateSolution_, dt);

        // Compute the next state and state derivative if success
        if (hasSucceeded)
        {
            state = candidateSolution_;
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

    bool_t AbstractRungeKuttaStepper::adjustStep(state_t   const & /* initialState */,
                                                 state_t   const & /* solution */,
                                                 float64_t       & dt)
    {
        // Fixed-step by default, which never fails
        dt = INF;
        return true;
    }
}

