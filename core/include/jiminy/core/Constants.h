///////////////////////////////////////////////////////////////////////////////
///
/// \brief    Contains constants shared throughout the code.
///
/// \details  Constants are physical constants and conversion factors,
///           which should not change in the time, contrary to
///           configurations parameters which store constants that
///           depends on our system and may be tweaked.
///
///////////////////////////////////////////////////////////////////////////////

#ifndef JIMINY_CONSTANTS_H
#define JIMINY_CONSTANTS_H

#include <math.h>

#include "jiminy/core/Types.h"


namespace jiminy
{
    extern std::string const JOINT_PREFIX_BASE;
    extern std::string const FREE_FLYER_PREFIX_BASE_NAME;
    extern std::string const FLEXIBLE_JOINT_SUFFIX;

    extern std::string const TELEMETRY_FIELDNAME_DELIMITER;
    extern std::string const TELEMETRY_CONSTANT_DELIMITER;
    extern int64_t const TELEMETRY_MIN_BUFFER_SIZE;
    extern float64_t const TELEMETRY_DEFAULT_TIME_UNIT;

    extern uint8_t const DELAY_MIN_BUFFER_RESERVE;  ///< Minimum memory allocation is memory is full and the older data stored is dated less than the desired delay
    extern uint8_t const DELAY_MAX_BUFFER_EXCEED;   ///< Maximum number of data stored allowed to be dated more than the desired delay

    extern float64_t const STEPPER_MIN_TIMESTEP;
    extern float64_t const SIMULATION_MIN_TIMESTEP;
    extern float64_t const SIMULATION_MAX_TIMESTEP;

    extern uint32_t const PGS_MAX_ITERATIONS;
    extern uint32_t const PGS_RANDOM_PERMUTATION_PERIOD;
    extern float64_t const PGS_MIN_REGULARIZER;
}

#endif  // JIMINY_CONSTANTS_H
