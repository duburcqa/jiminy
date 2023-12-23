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
    ///
    /// \return Return code to determine whether the execution of the method was successful.
    template<typename T>
    hresult_t AbstractController::registerConstant(const std::string_view & fieldname,
                                                   const T & value)
    {
        // Delayed variable registration (Taken into account by 'configureTelemetry')

        if (isTelemetryConfigured_)
        {
            PRINT_ERROR("Telemetry already initialized. Impossible to register new variables.");
            return hresult_t::ERROR_INIT_FAILED;
        }

        // Check in local cache before.
        auto constantIt = std::find_if(registeredConstants_.begin(),
                                       registeredConstants_.end(),
                                       [&fieldname](const auto & element)
                                       { return element.first == fieldname; });
        if (constantIt != registeredConstants_.end())
        {
            PRINT_ERROR("Constant already registered.");
            return hresult_t::ERROR_BAD_INPUT;
        }
        registeredConstants_.emplace_back(fieldname, toString(value));

        return hresult_t::SUCCESS;
    }

    template<typename T>
    hresult_t AbstractController::registerVariable(const std::string_view & name, const T & value)
    {
        if (isTelemetryConfigured_)
        {
            PRINT_ERROR("Telemetry already initialized. Impossible to register new variables.");
            return hresult_t::ERROR_INIT_FAILED;
        }

        // Check in local cache before.
        auto variableIt = std::find_if(registeredVariables_.begin(),
                                       registeredVariables_.end(),
                                       [&name](const auto & element)
                                       { return element.first == name; });
        if (variableIt != registeredVariables_.end())
        {
            PRINT_ERROR("Variable already registered.");
            return hresult_t::ERROR_BAD_INPUT;
        }
        registeredVariables_.emplace_back(name, &value);

        return hresult_t::SUCCESS;
    }
}

#endif  // JIMINY_ABSTRACT_CONTROLLER_HXX
