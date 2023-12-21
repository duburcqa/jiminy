#ifndef JIMINY_CONSTANTS_H
#define JIMINY_CONSTANTS_H

#include <string_view>
#include <math.h>

#include "jiminy/core/fwd.h"
#include "jiminy/core/traits.h"


namespace jiminy
{
    inline constexpr std::string_view TELEMETRY_FIELDNAME_DELIMITER{"."};
    inline constexpr std::string_view TELEMETRY_CONSTANT_DELIMITER{"="};
    /// \brief Special constant storing time unit
    inline constexpr std::string_view TIME_UNIT{"Global.TIME_UNIT"};
    /// \brief Special variable storing timesteps
    inline constexpr std::string_view GLOBAL_TIME{"Global.Time"};

    inline constexpr float64_t STEPPER_MIN_TIMESTEP{1e-10};
    inline constexpr float64_t SIMULATION_MIN_TIMESTEP{1e-6};
    inline constexpr float64_t SIMULATION_MAX_TIMESTEP{0.02};
}

#endif  // JIMINY_CONSTANTS_H
