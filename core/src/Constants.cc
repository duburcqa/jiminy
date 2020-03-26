#include "jiminy/core/Constants.h"


namespace jiminy
{
    std::string const JOINT_PREFIX_BASE = "current";
    std::string const FREE_FLYER_PREFIX_BASE_NAME = JOINT_PREFIX_BASE + "Freeflyer";
    std::string const FLEXIBLE_JOINT_SUFFIX = "FlexibleJoint";

    float64_t const TELEMETRY_TIME_DISCRETIZATION_FACTOR = 1e6; // Log the time rounded to the closest µs
    int64_t const MAX_TELEMETRY_BUFFER_SIZE = 256U * 1024U; // 256Ko

    uint8_t const MIN_DELAY_BUFFER_RESERVE = 20U;
    uint8_t const MAX_DELAY_BUFFER_EXCEED = 20U;

    float64_t const MIN_SIMULATION_TIMESTEP = 1.0 / TELEMETRY_TIME_DISCRETIZATION_FACTOR;
    float64_t const MAX_SIMULATION_TIMESTEP = 5e-3;
    float64_t const DEFAULT_SIMULATION_TIMESTEP = 1e-3;
    float64_t const MIN_STEPPER_TIMESTEP = 1e-10;
}
