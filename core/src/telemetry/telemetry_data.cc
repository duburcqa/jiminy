#include "jiminy/core/constants.h"
#include "jiminy/core/exceptions.h"

#include "jiminy/core/telemetry/telemetry_data.h"


namespace jiminy
{
    TelemetryData::TelemetryData() :
    constantsRegistry_(),
    integersRegistry_(),
    floatsRegistry_(),
    isRegisteringAvailable_(false)
    {
        reset();
    }

    void TelemetryData::reset()
    {
        constantsRegistry_.clear();
        integersRegistry_.clear();
        floatsRegistry_.clear();
        isRegisteringAvailable_ = true;
    }

    hresult_t TelemetryData::registerConstant(const std::string & name, const std::string & value)
    {
        // Check if registration is possible
        if (!isRegisteringAvailable_)
        {
            PRINT_ERROR("Registration is locked.");
            return hresult_t::ERROR_GENERIC;
        }

        // Check if already in memory
        auto variableIt =
            std::find_if(constantsRegistry_.begin(),
                         constantsRegistry_.end(),
                         [&name](const std::pair<std::string, std::string> & element) -> bool_t
                         { return element.first == name; });
        if (variableIt != constantsRegistry_.end())
        {
            PRINT_ERROR("Entry already exists.");
            return hresult_t::ERROR_GENERIC;
        }

        // Register new constant
        constantsRegistry_.emplace_back(name, value);
        return hresult_t::SUCCESS;
    }

    void TelemetryData::formatHeader(std::vector<char_t> & header)
    {
        // Define helper to easily insert new lines in header
        auto insertLineInHeader = [&header](auto &&... args) -> void
        {
            std::ostringstream sstr;
            auto format = [&header](const auto & var)
            {
                if constexpr (std::is_same_v<decltype(var), std::string_view> ||
                              std::is_same_v<decltype(var), std::string>)
                {
                    header.insert(header.end(), var.cbegin(), var.cend());
                }
                else
                {
                    const std::string str = toString(var);
                    std::move(str.cbegin(), str.cend(), std::back_inserter(header));
                }
            };
            (format(args), ...);
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
        for (const auto & [name, value] : constantsRegistry_)
        {
            insertLineInHeader(START_LINE_TOKEN, name, TELEMETRY_CONSTANT_DELIMITER, value);
        }

        // Record number of integer variables (+1 because we add Global.Time)
        insertLineInHeader(START_LINE_TOKEN,
                           NUM_INTS,
                           TELEMETRY_CONSTANT_DELIMITER,
                           integersRegistry_.size() + 1);

        // Record number of floating-point variables
        insertLineInHeader(
            START_LINE_TOKEN, NUM_FLOATS, TELEMETRY_CONSTANT_DELIMITER, floatsRegistry_.size());

        // Insert column token
        insertLineInHeader(START_COLUMNS);

        // Record Global.Time - integers, floats
        insertLineInHeader(GLOBAL_TIME);

        // Record integers
        for (const std::pair<std::string, int64_t> & keyValue : integersRegistry_)
        {
            insertLineInHeader(keyValue.first);
        }

        // Record floats
        for (const std::pair<std::string, float64_t> & keyValue : floatsRegistry_)
        {
            insertLineInHeader(keyValue.first);
        }

        // Start data section
        insertLineInHeader(START_DATA);
    }

    template<>
    std::deque<std::pair<std::string, int64_t>> * TelemetryData::getRegistry<int64_t>()
    {
        return &integersRegistry_;
    }

    template<>
    std::deque<std::pair<std::string, float64_t>> * TelemetryData::getRegistry<float64_t>()
    {
        return &floatsRegistry_;
    }
}
