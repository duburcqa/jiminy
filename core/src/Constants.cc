#include "jiminy/core/Constants.h"


namespace jiminy
{
    std::string const JOINT_PREFIX_BASE = "current";
    std::string const FREE_FLYER_PREFIX_BASE_NAME = JOINT_PREFIX_BASE + "Freeflyer";
    std::string const FLEXIBLE_JOINT_SUFFIX = "FlexibleJoint";

    float64_t const MIN_SIMULATION_TIMESTEP = 1e-6;
    float64_t const MAX_SIMULATION_TIMESTEP = 5e-3;

    uint8_t const MIN_DELAY_BUFFER_RESERVE = 20U;
    uint8_t const MAX_DELAY_BUFFER_EXCEED = 20U;
}
