#include "jiminy/core/stepper/runge_kutta4_stepper.h"

namespace jiminy
{
    RungeKutta4Stepper::RungeKutta4Stepper(systemDynamics const & f,
                                           std::vector<Robot const *> const & robots):
    AbstractRungeKuttaStepper(f, robots, RK4::A, RK4::b, RK4::c, false)
    {
        // Empty on purpose
    }
}
