///////////////////////////////////////////////////////////////////////////////////////////////
///
/// \brief      Implements a fixed-step explicit Euler first-order scheme.
/// \details    This simple scheme is mostly meant for debugging due to its simplicity,
///             but is not very efficient...
///
///////////////////////////////////////////////////////////////////////////////////////////////

#ifndef JIMINY_EXPLICIT_EULER_STEPPER_H
#define JIMINY_EXPLICIT_EULER_STEPPER_H

#include "jiminy/core/stepper/AbstractStepper.h"

namespace jiminy
{
    class EulerExplicitStepper: public AbstractStepper
    {
        public:
            /// \brief Constructor
            /// \param[in] f      Dynamics function, with signature a = f(t, q, v)
            /// \param[in] robots Robots whose dynamics the stepper will work on.
            EulerExplicitStepper(systemDynamics f, /* Copy on purpose */
                                 std::vector<Robot const *> const & robots);

        protected:
            /// \brief Internal tryStep method wrapping the arguments as state_t and stateDerivative_t.
            bool_t tryStepImpl(state_t                 & state,
                               stateDerivative_t       & stateDerivative,
                               float64_t         const & t,
                               float64_t               & dt) final override;
    };
}

#endif //end of JIMINY_EXPLICIT_EULER_STEPPER_H
