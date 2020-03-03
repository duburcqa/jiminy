
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
    /// \brief      Register a constant to the telemetry.
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
            // Check in local cache before.
            auto constantIt = std::find_if(registeredConstants_.begin(),
                                           registeredConstants_.end(),
                                           [&fieldName](auto const & element)
                                           {
                                               return element.first == fieldName;
                                           });
            if (constantIt != registeredConstants_.end())
            {
                std::cout << "Error - AbstractController::registerConstant - Constant already registered." << std::endl;
                return result_t::ERROR_BAD_INPUT;
            }
            registeredConstants_.emplace_back(fieldName, std::to_string(value));
        }

        return returnCode;
    }

    template<>
    inline result_t AbstractController::registerConstant<char const *>(std::string   const & fieldName,
                                                                       char  const * const & value)
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
            // Check in local cache before.
            auto constantIt = std::find_if(registeredConstants_.begin(),
                                           registeredConstants_.end(),
                                           [&fieldName](auto const & element)
                                           {
                                               return element.first == fieldName;
                                           });
            if (constantIt != registeredConstants_.end())
            {
                std::cout << "Error - AbstractController::registerConstant - Constant already registered." << std::endl;
                return result_t::ERROR_BAD_INPUT;
            }
            registeredConstants_.emplace_back(fieldName, std::string(value));
        }

        return returnCode;
    }

    template<>
    inline result_t AbstractController::registerConstant<char *>(std::string   const & fieldName,
                                                                 char        * const & value)
    {
        return registerConstant<char const *>(fieldName, value);
    }

    // TODO: Improve to support any type of Eigen object (and const or not)
    // Note that fully specificied template should go in cc file instead of h, unless inline is used.
    template<>
    inline result_t AbstractController::registerConstant<Eigen::Ref<vectorN_t const> >(std::string                 const & fieldName,
                                                                                       Eigen::Ref<vectorN_t const> const & value)
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
            // Check in local cache before.
            auto constantIt = std::find_if(registeredConstants_.begin(),
                                           registeredConstants_.end(),
                                           [&fieldName](auto const & element)
                                           {
                                               return element.first == fieldName;
                                           });
            if (constantIt != registeredConstants_.end())
            {
                std::cout << "Error - AbstractController::registerConstant - Constant already registered." << std::endl;
                return result_t::ERROR_BAD_INPUT;
            }
            Eigen::IOFormat HeavyFmt(Eigen::FullPrecision, 0, ", ", ";\n", "[", "]", "[", "]");
            std::stringstream matrixStream;
            matrixStream << value.format(HeavyFmt);
            std::string matrixPrint = matrixStream.str();
            registeredConstants_.emplace_back(fieldName, matrixPrint);
        }

        return returnCode;
    }

    template<>
    inline result_t AbstractController::registerConstant<Eigen::Map<vectorN_t> >(std::string           const & fieldName,
                                                                                 Eigen::Map<vectorN_t> const & value)
    {
        return registerConstant<Eigen::Ref<vectorN_t const> >(fieldName, value);
    }
}

#endif // SIMU_ABSTRACT_CONTROLLER_TPP


