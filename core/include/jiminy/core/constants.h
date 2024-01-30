#ifndef JIMINY_CONSTANTS_H
#define JIMINY_CONSTANTS_H

#include <string_view>  // `std::string_view`
#include <limits>       // `std::numeric_limits`

namespace jiminy
{
    // ******************************* Jiminy-specific constants ******************************* //

    inline constexpr std::string_view TELEMETRY_FIELDNAME_DELIMITER{"."};
    inline constexpr std::string_view TELEMETRY_CONSTANT_DELIMITER{"="};
    /// \brief Special constant storing time unit
    inline constexpr std::string_view TIME_UNIT{"Global.TIME_UNIT"};
    /// \brief Special variable storing timesteps
    inline constexpr std::string_view GLOBAL_TIME{"Global.Time"};

    inline constexpr double STEPPER_MIN_TIMESTEP{1e-10};
    inline constexpr double SIMULATION_MIN_TIMESTEP{1e-6};
    inline constexpr double SIMULATION_MAX_TIMESTEP{0.02};

    // ******************************* Constant of the universe ******************************** //

    inline constexpr double INF = std::numeric_limits<double>::infinity();
    inline constexpr double EPS = std::numeric_limits<double>::epsilon();
    inline constexpr double qNAN = std::numeric_limits<double>::quiet_NaN();
}

#endif  // JIMINY_CONSTANTS_H
