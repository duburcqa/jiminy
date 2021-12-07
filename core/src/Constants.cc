#include "jiminy/core/Constants.h"


namespace jiminy
{
    std::string const JOINT_PREFIX_BASE = "current";
    std::string const FREE_FLYER_PREFIX_BASE_NAME = JOINT_PREFIX_BASE + "Freeflyer";
    std::string const FLEXIBLE_JOINT_SUFFIX = "Flexibility";

    std::string const TELEMETRY_FIELDNAME_DELIMITER = ".";
    std::string const TELEMETRY_CONSTANT_DELIMITER = "=";
    float64_t const TELEMETRY_DEFAULT_TIME_UNIT = 1e6;  // Log the time rounded to the closest µs
    int64_t const TELEMETRY_MIN_BUFFER_SIZE = 256U * 1024U;  // 256Ko

    uint8_t const DELAY_MIN_BUFFER_RESERVE = 20U;
    uint8_t const DELAY_MAX_BUFFER_EXCEED = 20U;

    float64_t const STEPPER_MIN_TIMESTEP = 1e-10;
    float64_t const SIMULATION_MIN_TIMESTEP = 1e-6;
    float64_t const SIMULATION_MAX_TIMESTEP = 0.02;

    uint32_t const PGS_MAX_ITERATIONS = 100U;
    float64_t const PGS_MIN_REGULARIZER = 1.0e-11;
}
