
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
        using AbstractStepper::AbstractStepper;

    protected:
        /// \brief Internal tryStep method wrapping the arguments as State and
        /// StateDerivative.
        bool tryStepImpl(State & state,
                         StateDerivative & stateDerivative,
                         double t,
                         double & dt) final override;
    };
}

#endif  // end of JIMINY_EXPLICIT_EULER_STEPPER_H
