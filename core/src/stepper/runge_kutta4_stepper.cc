#include "jiminy/core/stepper/runge_kutta4_stepper.h"

namespace jiminy
{
    RungeKutta4Stepper::RungeKutta4Stepper(const systemDynamics & f,
                                           const std::vector<const Robot *> & robots) :
    AbstractRungeKuttaStepper(f, robots, RK4::A, RK4::b, RK4::c, false)
    {
    }
}
