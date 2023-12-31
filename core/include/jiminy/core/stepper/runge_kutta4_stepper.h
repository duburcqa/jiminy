#ifndef JIMINY_RUNGE_KUTTA_4_STEPPER_H
#define JIMINY_RUNGE_KUTTA_4_STEPPER_H

#include "jiminy/core/fwd.h"
#include "jiminy/core/stepper/abstract_runge_kutta_stepper.h"

namespace jiminy
{
    namespace RK4
    {
        // clang-format off
        const Eigen::MatrixXd A((Eigen::MatrixXd(4, 4) <<
            0.0, 0.0,  0.0,  0.0,
            0.5, 0.0,  0.0,  0.0,
            0.0, 0.5,  0.0,  0.0,
            0.0, 0.0,  1.0,  0.0
        ).finished());
        const Eigen::VectorXd c((Eigen::VectorXd(4) <<
            0.0, 0.5, 0.5, 1.0
        ).finished());
        const Eigen::VectorXd b((Eigen::VectorXd(4) <<
            1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0
        ).finished());
        // clang-format on
    }

    /// \brief 4th-order fixed-step Runge Kutta method
    class JIMINY_DLLAPI RungeKutta4Stepper : public AbstractRungeKuttaStepper
    {
    public:
        /// \param[in] f Dynamics function, with signature `a = f(t, q, v)`.
        /// \param[in] robots Robots whose dynamics the stepper will work on.
        RungeKutta4Stepper(const systemDynamics & f,
                           const std::vector<const Robot *> & robots) noexcept;
    };
}

#endif  // end of JIMINY_RUNGE_KUTTA_DOPRI_STEPPER_H
