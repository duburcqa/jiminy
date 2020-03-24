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

    extern int64_t const MAX_TELEMETRY_BUFFER_SIZE;
    extern float64_t const TELEMETRY_TIME_DISCRETIZATION_FACTOR;

    extern uint8_t const MIN_DELAY_BUFFER_RESERVE; ///< Minimum memory allocation is memory is full and the older data stored is dated less than the desired delay
    extern uint8_t const MAX_DELAY_BUFFER_EXCEED;  ///< Maximum number of data stored allowed to be dated more than the desired delay

    extern float64_t const MIN_SIMULATION_TIMESTEP;
    extern float64_t const MAX_SIMULATION_TIMESTEP;
}

#endif  // JIMINY_CONSTANTS_H
