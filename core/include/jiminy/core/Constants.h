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
    std::string const JOINT_PREFIX_BASE{"current"};
    std::string const FREE_FLYER_PREFIX_BASE_NAME{JOINT_PREFIX_BASE + "Freeflyer"};
    std::string const FLEXIBLE_JOINT_SUFFIX{"FlexibleJoint"};

    constexpr float64_t MIN_SIMULATION_TIMESTEP = 1e-6;
    constexpr float64_t MAX_SIMULATION_TIMESTEP = 5e-3;

    constexpr uint8_t MIN_DELAY_BUFFER_RESERVE = 20U; ///< Minimum memory allocation is memory is full and the older data stored is dated less than the desired delay
    constexpr uint8_t MAX_DELAY_BUFFER_EXCEED = 20U;  ///< Maximum number of data stored allowed to be dated more than the desired delay
}

#endif  // JIMINY_CONSTANTS_H
