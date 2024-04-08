#ifndef JIMINY_TELEMETRY_DATA_HXX
#define JIMINY_TELEMETRY_DATA_HXX

#include <string>


namespace jiminy
{
    template<typename T>
    T * TelemetryData::registerVariable(const std::string & name)
    {
        // Get the right registry
        static_map_t<std::string, T, false> * registry = getRegistry<T>();

        // Check if already in memory
        auto variableIt = std::find_if(registry->begin(),
                                       registry->end(),
                                       [&name](const std::pair<std::string, T> & element) -> bool
                                       { return element.first == name; });
        if (variableIt != registry->end())
        {
            return &(variableIt->second);
        }

        // Check if registration is possible
        if (!isRegisteringAvailable_)
        {
            JIMINY_THROW(std::invalid_argument,
                         "Entry '",
                         name,
                         "' not found and registration is not available.");
        }

        // Create new variable in registry
        registry->emplace_back(name, T{});
        return &(registry->back().second);
    }
}

#endif  // JIMINY_TELEMETRY_DATA_HXX
