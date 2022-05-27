///////////////////////////////////////////////////////////////////////////////
///
/// \brief   Manage the data structures of the telemetry.
///
///////////////////////////////////////////////////////////////////////////////

#ifndef JIMINY_TELEMETRY_DATA_TPP
#define JIMINY_TELEMETRY_DATA_TPP

#include <iostream>
#include <string>


namespace jiminy
{
    template<typename T>
    hresult_t TelemetryData::registerVariable(std::string const & variableName,
                                              T * & positionInBufferOut)
    {
        // Get the right registry
        std::deque<std::pair<std::string, T> > * registry = getRegistry<T>();

        // Check if already in memory
        auto variableIt = std::find_if(
            registry->begin(),
            registry->end(),
            [&variableName](std::pair<std::string, T> const & element) -> bool_t
            {
                return element.first == variableName;
            });
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
} // namespace jiminy

#endif // JIMINY_TELEMETRY_DATA_TPP