#ifndef JIMINY_CONSTANTS_H
#define JIMINY_CONSTANTS_H

#include <math.h>

#include "jiminy/core/types.h"


namespace jiminy
{
    extern const std::string JOINT_PREFIX_BASE;
    extern const std::string FREE_FLYER_PREFIX_BASE_NAME;
    extern const std::string FLEXIBLE_JOINT_SUFFIX;

    extern const std::string TELEMETRY_FIELDNAME_DELIMITER;
    extern const std::string TELEMETRY_CONSTANT_DELIMITER;
    extern const int64_t TELEMETRY_MIN_BUFFER_SIZE;
    extern const float64_t TELEMETRY_DEFAULT_TIME_UNIT;

    /// \brief Minimum memory allocation if buffer is full and oldest stored data must be kept.
    extern const uint8_t DELAY_MIN_BUFFER_RESERVE;

    /// \brief Maximum amount of outdated data allowed to kept for longer than strictly necessary.
    extern const uint8_t DELAY_MAX_BUFFER_EXCEED;

    extern const float64_t STEPPER_MIN_TIMESTEP;
    extern const float64_t SIMULATION_MIN_TIMESTEP;
    extern const float64_t SIMULATION_MAX_TIMESTEP;

    extern const uint32_t INIT_ITERATIONS;
    extern const uint32_t PGS_MAX_ITERATIONS;
    extern const float64_t PGS_MIN_REGULARIZER;
}

#endif  // JIMINY_CONSTANTS_H
