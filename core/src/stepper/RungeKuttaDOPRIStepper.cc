#include "jiminy/core/stepper/RungeKuttaDOPRIStepper.h"

namespace jiminy
{
    RungeKuttaDOPRIStepper::RungeKuttaDOPRIStepper(systemDynamics f, /* Copy on purpose */
                                                   std::vector<Robot const *> robots,
                                                   float64_t const& tolRel,
                                                   float64_t const& tolAbs):
    RungeKuttaStepper(f,
                      robots,
                      tolRel,
                      tolAbs,
                      DOPRI::A,
                      DOPRI::c,
                      DOPRI::b,
                      DOPRI::e)
    {
        // Empty on purpose
    }
}
