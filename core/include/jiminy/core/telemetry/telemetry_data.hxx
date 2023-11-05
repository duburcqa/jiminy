#ifndef JIMINY_TELEMETRY_DATA_HXX
#define JIMINY_TELEMETRY_DATA_HXX

#include <string>


namespace jiminy
{
    template<typename T>
    hresult_t TelemetryData::registerVariable(const std::string & variableName,
                                              T *& positionInBufferOut)
    {
        // Get the right registry
        std::deque<std::pair<std::string, T>> * registry = getRegistry<T>();

        // Check if already in memory
        auto variableIt =
            std::find_if(registry->begin(),
                         registry->end(),
                         [&variableName](const std::pair<std::string, T> & element) -> bool_t
                         { return element.first == variableName; });
        if (variableIt != registry->end())
        {
            positionInBufferOut = &(variableIt->second);
            return hresult_t::SUCCESS;
        }

        // Check if registration is possible
        if (!isRegisteringAvailable_)
        {
            PRINT_ERROR("Entry not found and registration is not available.");
            return hresult_t::ERROR_GENERIC;
        }

        // Create new variable in registry
        registry->push_back({variableName, {}});
        positionInBufferOut = &(registry->back().second);

        return hresult_t::SUCCESS;
    }
}

#endif  // JIMINY_TELEMETRY_DATA_HXX
