
#include "jiminy/core/telemetry/fwd.h"
#include "jiminy/core/telemetry/telemetry_data.h"


namespace jiminy
{
    void TelemetryData::reset() noexcept
    {
        constantRegistry_.clear();
        integerRegistry_.clear();
        floatRegistry_.clear();
        isRegisteringAvailable_ = true;
    }

    void TelemetryData::registerConstant(const std::string & name, const std::string & value)
    {
        // Check if registration is possible
        if (!isRegisteringAvailable_)
        {
            JIMINY_THROW(bad_control_flow, "Registration already locked.");
        }

        // Check if already in memory
        auto variableIt =
            std::find_if(constantRegistry_.begin(),
                         constantRegistry_.end(),
                         [&name](const std::pair<std::string, std::string> & element) -> bool
                         { return element.first == name; });
        if (variableIt != constantRegistry_.end())
        {
            JIMINY_THROW(bad_control_flow, "Entry '", name, "' already exists.");
        }

        // Register new constant
        constantRegistry_.emplace_back(name, value);
    }

    std::vector<char> TelemetryData::formatHeader()
    {
        std::vector<char> header{};

        // Define helper to easily insert new lines in header
        auto insertLineInHeader = [&header](auto &&... args) -> void
        {
            std::ostringstream sstr;
            auto format = [&header](const auto & var)
            {
                using T = std::decay_t<decltype(var)>;

                if constexpr (std::is_same_v<T, std::string_view> ||
                              std::is_same_v<T, std::string>)
                {
                    header.insert(header.end(), var.cbegin(), var.cend());
                }
                else
                {
                    const std::string str = toString(var);
                    std::move(str.cbegin(), str.cend(), std::back_inserter(header));
                }
            };
            (format(std::forward<decltype(args)>(args)), ...);
            header.push_back('\0');
        };

        // Lock registering
        isRegisteringAvailable_ = false;

        // Make sure provided header is empty
        header.clear();

        // Record format version
        header.resize(sizeof(int32_t));
        header[0] = ((TELEMETRY_VERSION & 0x000000ff) >> 0);
        header[1] = ((TELEMETRY_VERSION & 0x0000ff00) >> 8);
        header[2] = ((TELEMETRY_VERSION & 0x00ff0000) >> 16);
        header[3] = ((TELEMETRY_VERSION & 0xff000000) >> 24);

        // Record constants
        insertLineInHeader(START_CONSTANTS);
        for (const auto & [name, value] : constantRegistry_)
        {
            insertLineInHeader(START_LINE_TOKEN, name, TELEMETRY_CONSTANT_DELIMITER, value);
        }

        // Record number of integer variables (+1 because we add Global.Time)
        insertLineInHeader(
            START_LINE_TOKEN, NUM_INTS, TELEMETRY_CONSTANT_DELIMITER, integerRegistry_.size() + 1);

        // Record number of floating-point variables
        insertLineInHeader(
            START_LINE_TOKEN, NUM_FLOATS, TELEMETRY_CONSTANT_DELIMITER, floatRegistry_.size());

        // Insert column token
        insertLineInHeader(START_COLUMNS);

        // Record Global.Time - integers, floats
        insertLineInHeader(GLOBAL_TIME);

        // Record integers
        for (const std::pair<std::string, int64_t> & keyValue : integerRegistry_)
        {
            insertLineInHeader(keyValue.first);
        }

        // Record floats
        for (const std::pair<std::string, double> & keyValue : floatRegistry_)
        {
            insertLineInHeader(keyValue.first);
        }

        // Start data section
        insertLineInHeader(START_DATA);

        return header;
    }

    template<>
    static_map_t<std::string, int64_t, false> * TelemetryData::getRegistry<int64_t>()
    {
        return &integerRegistry_;
    }

    template<>
    static_map_t<std::string, double, false> * TelemetryData::getRegistry<double>()
    {
        return &floatRegistry_;
    }
}
