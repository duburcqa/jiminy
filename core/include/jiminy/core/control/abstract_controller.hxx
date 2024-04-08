#ifndef JIMINY_ABSTRACT_CONTROLLER_HXX
#define JIMINY_ABSTRACT_CONTROLLER_HXX

#include <string>
#include <vector>


namespace jiminy
{
    template<typename T>
    void AbstractController::registerVariable(const std::string_view & name, const T & value)
    {
        if (isTelemetryConfigured_)
        {
            JIMINY_THROW(bad_control_flow,
                         "Telemetry already initialized. Impossible to register new variables.");
        }

        // Check in local cache before.
        auto variableIt = std::find_if(variableRegistry_.begin(),
                                       variableRegistry_.end(),
                                       [&name](const auto & element)
                                       { return element.first == name; });
        if (variableIt != variableRegistry_.end())
        {
            JIMINY_THROW(bad_control_flow, "Variable '", name, "' already registered.");
        }
        variableRegistry_.emplace_back(name, &value);
    }
}

#endif  // JIMINY_ABSTRACT_CONTROLLER_HXX
