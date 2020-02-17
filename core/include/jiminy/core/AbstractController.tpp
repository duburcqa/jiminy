
///////////////////////////////////////////////////////////////////////////////
///
/// \brief Contains templated function implementation of the AbstractController class.
///
///////////////////////////////////////////////////////////////////////////////

#ifndef SIMU_ABSTRACT_CONTROLLER_TPP
#define SIMU_ABSTRACT_CONTROLLER_TPP

#include <string>
#include <vector>


namespace jiminy
{
    ///////////////////////////////////////////////////////////////////////////////////////////////
    ///
    /// \brief      Register a constant float64 to the telemetry.
    ///
    /// \param[in]  fieldNames      Name of the variable.
    /// \param[in]  values          Variable to add to the telemetry
    ///
    /// \return     Return code to determine whether the execution of the method was successful.
    ///
    ///////////////////////////////////////////////////////////////////////////////////////////////
    template<typename T>
    result_t AbstractController::registerConstant(std::string const & fieldName,
                                                  T           const & value)
    {
        // Delayed variable registration (Taken into account by 'configureTelemetry')

        result_t returnCode = result_t::SUCCESS;

        if (isTelemetryConfigured_)
        {
            std::cout << "Error - AbstractController::registerConstant - Telemetry already initialized. Impossible to register new variables." << std::endl;
            returnCode = result_t::ERROR_INIT_FAILED;
        }

        if (returnCode == result_t::SUCCESS)
        {
            registeredConstants_.emplace_back(fieldName, std::to_string(value));
        }

        return returnCode;
    }
}

#endif // SIMU_ABSTRACT_CONTROLLER_TPP


