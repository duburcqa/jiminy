///////////////////////////////////////////////////////////////////////////////////////////////
///
/// \brief      Implements a 4th-order fixed-step Runge Kutta method
///
///////////////////////////////////////////////////////////////////////////////////////////////

#ifndef JIMINY_RUNGE_KUTTA_4_STEPPER_H
#define JIMINY_RUNGE_KUTTA_4_STEPPER_H

#include "jiminy/core/stepper/AbstractRungeKuttaStepper.h"

namespace jiminy
{
    namespace RK4
    {
        matrixN_t const A((matrixN_t(4, 4) << 0.0, 0.0,  0.0,  0.0,
                                              0.5, 0.0,  0.0,  0.0,
                                              0.0, 0.5,  0.0,  0.0,
                                              0.0, 0.0,  1.0,  0.0).finished());
        vectorN_t const c((vectorN_t(4) << 0.0, 0.5, 0.5, 1.0).finished());
        vectorN_t const b((vectorN_t(4) << 1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0).finished());
    }

    class RungeKutta4Stepper: public AbstractRungeKuttaStepper
    {
        public:
            /// \brief Constructor
            /// \param[in] f      Dynamics function, with signature a = f(t, q, v)
            /// \param[in] robots Robots whose dynamics the stepper will work on.
            RungeKutta4Stepper(systemDynamics f, /* Copy on purpose */
                               std::vector<Robot const *> const & robots);
    };
}

#endif //end of JIMINY_RUNGE_KUTTA_DOPRI_STEPPER_H
