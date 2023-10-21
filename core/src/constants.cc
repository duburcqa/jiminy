#include "jiminy/core/constants.h"


namespace jiminy
{
    const std::string JOINT_PREFIX_BASE = "current";
    const std::string FREE_FLYER_PREFIX_BASE_NAME = JOINT_PREFIX_BASE + "Freeflyer";
    const std::string FLEXIBLE_JOINT_SUFFIX = "Flexibility";

    const std::string TELEMETRY_FIELDNAME_DELIMITER = ".";
    const std::string TELEMETRY_CONSTANT_DELIMITER = "=";
    const int64_t TELEMETRY_MIN_BUFFER_SIZE = 256U * 1024U;  // 256Ko

    const uint8_t DELAY_MIN_BUFFER_RESERVE = 20U;
    const uint8_t DELAY_MAX_BUFFER_EXCEED = 100U;

    const float64_t STEPPER_MIN_TIMESTEP = 1e-10;
    const float64_t SIMULATION_MIN_TIMESTEP = 1e-6;
    const float64_t SIMULATION_MAX_TIMESTEP = 0.02;

    const uint32_t INIT_ITERATIONS = 4U;
    const uint32_t PGS_MAX_ITERATIONS = 100U;
    const float64_t PGS_MIN_REGULARIZER = 1.0e-11;
}
