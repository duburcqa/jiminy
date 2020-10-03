///////////////////////////////////////////////////////////////////////////////////////////////
///
/// \brief      Implements the Dormand-Prince Runge-Kutta algorithm.

///////////////////////////////////////////////////////////////////////////////////////////////

#ifndef JIMINY_RUNGE_KUTTA_DOPRI_STEPPER_H
#define JIMINY_RUNGE_KUTTA_DOPRI_STEPPER_H

#include "jiminy/core/stepper/RungeKuttaStepper.h"

namespace jiminy
{
    namespace DOPRI
    {
        matrixN_t const A((matrixN_t(7, 7) << 0.0, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
                                              1.0 / 5.0, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
                                              3.0 / 40.0, 9.0 / 40.0,  0.0,  0.0,  0.0,  0.0,  0.0,
                                              44.0 / 45.0, -56.0 / 15.0,  32.0 / 9.0,  0.0,  0.0,  0.0,  0.0,
                                              19372.0 /6561.0, -25360.0 / 21873.0, 64448.0 / 6561.0, -212.0 / 729.0,  0.0,  0.0,  0.0,
                                              9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0, -5103.0 / 18656.0, 0.0, 0.0,
                                              35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0, 0.0).finished());
        vectorN_t const c((vectorN_t(7) << 0.0,
                                           2.0 / 10.0,
                                           3.0 / 10.0,
                                           4.0 / 5.0,
                                           8.0 / 9.0,
                                           1.0,
                                           1.0).finished());
        vectorN_t const b((vectorN_t(7) << 35.0/ 384.0,
                                           0.0,
                                           500.0 / 1113.0,
                                           125.0 / 192.0,
                                           -2187.0 / 6784.0,
                                           11.0 / 84.0,
                                           0.0).finished());
        vectorN_t const e((vectorN_t(7) << 5179.0 / 57600.0,
                                           0.0,
                                           7571.0 / 16695.0,
                                           393.0 / 640.0,
                                           -92097.0 / 339200.0,
                                           187.0 / 2100.0,
                                           1.0 / 40.0).finished());
    }

    class RungeKuttaDOPRIStepper: public RungeKuttaStepper
    {
        public:
            /// \brief Constructor
            /// \param[in] f      Dynamics function, with signature a = f(t, q, v)
            /// \param[in] robots Robots whose dynamics the stepper will work on.
            /// \param[in] tolRel Relative tolerance, used to determine step success and timestep update.
            /// \param[in] tolAbs Relative tolerance, used to determine step success and timestep update.
            RungeKuttaDOPRIStepper(systemDynamics f, /* Copy on purpose */
                                   std::vector<Robot const *> robots,
                                   float64_t const & tolRel,
                                   float64_t const & tolAbs);

    };
}

#endif //end of JIMINY_RUNGE_KUTTA_DOPRI_STEPPER_H
