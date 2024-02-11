#ifndef JIMINY_ABSTRACT_CONTROLLER_HXX
#define JIMINY_ABSTRACT_CONTROLLER_HXX

#include <string>
#include <vector>


namespace jiminy
{
    /// \brief Register a constant to the telemetry.
    ///
    /// \param[in] fieldnames Name of the variable.
    /// \param[in] values Variable to add to the telemetry.
    template<typename T>
    void AbstractController::registerConstant(const std::string_view & fieldname, const T & value)
    {
        // Delayed variable registration (Taken into account by 'configureTelemetry')

        if (isTelemetryConfigured_)
        {
            THROW_ERROR(bad_control_flow,
                        "Telemetry already initialized. Impossible to register new variables.");
        }

        // Check in local cache before.
        auto constantIt = std::find_if(constantRegistry_.begin(),
                                       constantRegistry_.end(),
                                       [&fieldname](const auto & element)
                                       { return element.first == fieldname; });
        if (constantIt != constantRegistry_.end())
        {
            THROW_ERROR(bad_control_flow, "Constant already registered.");
        }
        constantRegistry_.emplace_back(fieldname, toString(value));
    }

    template<typename T>
    void AbstractController::registerVariable(const std::string_view & name, const T & value)
    {
        if (isTelemetryConfigured_)
        {
            THROW_ERROR(bad_control_flow,
                        "Telemetry already initialized. Impossible to register new variables.");
        }

        // Check in local cache before.
        auto variableIt = std::find_if(variableRegistry_.begin(),
                                       variableRegistry_.end(),
                                       [&name](const auto & element)
                                       { return element.first == name; });
        if (variableIt != variableRegistry_.end())
        {
            THROW_ERROR(bad_control_flow, "Variable already registered.");
        }
        variableRegistry_.emplace_back(name, &value);
    }
}

#endif  // JIMINY_ABSTRACT_CONTROLLER_HXX
