#ifndef JIMINY_FORWARD_TELEMETRY_H
#define JIMINY_FORWARD_TELEMETRY_H

#include "jiminy/core/fwd.h"


namespace jiminy
{
    /// \brief Version of the telemetry format.
    inline constexpr int32_t TELEMETRY_VERSION{1};
    /// \brief Number of integers in the data section.
    inline constexpr std::string_view NUM_INTS{"NumIntEntries"};
    /// \brief Number of floats in the data section.
    inline constexpr std::string_view NUM_FLOATS{"NumFloatEntries"};
    /// \brief Marker of the beginning the constants section.
    inline constexpr std::string_view START_CONSTANTS{"StartConstants"};
    /// \brief Marker of the beginning the columns section.
    inline constexpr std::string_view START_COLUMNS{"StartColumns"};
    /// \brief Marker of the beginning of a line of data.
    inline constexpr std::string_view START_LINE_TOKEN{"StartLine"};
    /// \brief Marker of the beginning of the data section.
    inline constexpr std::string_view START_DATA{"StartData"};

    struct JIMINY_DLLAPI LogData
    {
        int32_t version;
        static_map_t<std::string, std::string> constants;
        double timeUnit;
        VectorX<int64_t> times;
        std::vector<std::string> variableNames;
        MatrixX<int64_t> integerValues;
        MatrixX<double> floatValues;
    };
}

#endif  // JIMINY_FORWARD_TELEMETRY_H
