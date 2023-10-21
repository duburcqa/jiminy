#include "jiminy/core/constants.h"

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

    hresult_t TelemetryData::registerConstant(const std::string & variableNameIn,
                                              const std::string & constantValueIn)
    {
        // Check if registration is possible
        if (!isRegisteringAvailable_)
        {
            PRINT_ERROR("Registration is locked.");
            return hresult_t::ERROR_GENERIC;
        }

        // Check if already in memory
        auto variableIt = std::find_if(
            constantsRegistry_.begin(),
            constantsRegistry_.end(),
            [&variableNameIn](const std::pair<std::string, std::string> & element) -> bool_t
            { return element.first == variableNameIn; });
        if (variableIt != constantsRegistry_.end())
        {
            PRINT_ERROR("Entry already exists.");
            return hresult_t::ERROR_GENERIC;
        }

        // Register new constant
        constantsRegistry_.emplace_back(variableNameIn, constantValueIn);
        return hresult_t::SUCCESS;
    }

    void TelemetryData::formatHeader(std::vector<char_t> & header)
    {
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
        header.insert(
            header.end(), START_CONSTANTS.data(), START_CONSTANTS.data() + START_CONSTANTS.size());
        header.push_back('\0');
        for (const std::pair<std::string, std::string> & keyValue : constantsRegistry_)
        {
            for (auto strPtr : std::array<const std::string *, 4>{
                     {&START_LINE_TOKEN,
                      &keyValue.first,
                      &TELEMETRY_CONSTANT_DELIMITER,
                      &keyValue.second}
            })
            {
                header.insert(header.end(), strPtr->begin(), strPtr->end());
            }
            header.push_back('\0');
        }

        // Record entries numbers
        std::string entriesNumbers;
        entriesNumbers += START_LINE_TOKEN + NUM_INTS;
        entriesNumbers += std::to_string(integersRegistry_.size() + 1);  // +1 because we add
                                                                         // Global.Time
        entriesNumbers += '\0';
        entriesNumbers += START_LINE_TOKEN + NUM_FLOATS;
        entriesNumbers += std::to_string(floatsRegistry_.size());
        entriesNumbers += '\0';
        header.insert(
            header.end(), entriesNumbers.data(), entriesNumbers.data() + entriesNumbers.size());

        // Insert column token
        header.insert(
            header.end(), START_COLUMNS.data(), START_COLUMNS.data() + START_COLUMNS.size());
        header.push_back('\0');

        // Record Global.Time - integers, floats
        header.insert(header.end(), GLOBAL_TIME.data(), GLOBAL_TIME.data() + GLOBAL_TIME.size());
        header.push_back('\0');

        // Record integers
        for (const std::pair<std::string, int64_t> & keyValue : integersRegistry_)
        {
            header.insert(header.end(), keyValue.first.begin(), keyValue.first.end());
            header.push_back('\0');
        }

        // Record floats
        for (const std::pair<std::string, float64_t> & keyValue : floatsRegistry_)
        {
            header.insert(header.end(), keyValue.first.begin(), keyValue.first.end());
            header.push_back('\0');
        }

        // Start data section
        header.insert(header.end(), START_DATA.data(), START_DATA.data() + START_DATA.size());
        header.push_back('\0');
    }

    template<>
    std::deque<std::pair<std::string, int64_t>> * TelemetryData::getRegistry<int64_t>(void)
    {
        return &integersRegistry_;
    }

    template<>
    std::deque<std::pair<std::string, float64_t>> * TelemetryData::getRegistry<float64_t>(void)
    {
        return &floatsRegistry_;
    }
}
