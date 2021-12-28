#include "jiminy/core/stepper/RungeKutta4Stepper.h"

namespace jiminy
{
    RungeKutta4Stepper::RungeKutta4Stepper(systemDynamics const & f,
                                           std::vector<Robot const *> const & robots):
    AbstractRungeKuttaStepper(f, robots, RK4::A, RK4::b, RK4::c, false)
    {
        // Empty on purpose
    }
}
