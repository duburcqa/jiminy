
#ifndef JIMINY_EXPLICIT_EULER_STEPPER_H
#define JIMINY_EXPLICIT_EULER_STEPPER_H

#include "jiminy/core/fwd.h"
#include "jiminy/core/stepper/abstract_stepper.h"

namespace jiminy
{
    /// \brief Fixed-step explicit Euler first-order scheme.
    class JIMINY_DLLAPI EulerExplicitStepper : public AbstractStepper
    {
    public:
        /// \param[in] f Dynamics function, with signature `a = f(t, q, v)`.
        /// \param[in] robots Robots whose dynamics the stepper will work on.
        EulerExplicitStepper(const systemDynamics & f, const std::vector<const Robot *> & robots);

    protected:
        /// \brief Internal tryStep method wrapping the arguments as state_t and stateDerivative_t.
        bool tryStepImpl(state_t & state,
                         stateDerivative_t & stateDerivative,
                         double t,
                         double & dt) final override;
    };
}

#endif  // end of JIMINY_EXPLICIT_EULER_STEPPER_H
