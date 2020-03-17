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
    float64_t const MIN_SIMULATION_TIMESTEP = 1e-6;
    float64_t const MAX_SIMULATION_TIMESTEP = 5e-3;
}

#endif  // JIMINY_CONSTANTS_H
