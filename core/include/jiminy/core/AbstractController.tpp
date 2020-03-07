
///////////////////////////////////////////////////////////////////////////////
///
/// \brief Contains templated function implementation of the AbstractController class.
///
///////////////////////////////////////////////////////////////////////////////

#ifndef JIMINY_ABSTRACT_CONTROLLER_TPP
#define JIMINY_ABSTRACT_CONTROLLER_TPP

#include <string>
#include <vector>


namespace jiminy
{
    using std::to_string;

    inline std::string to_string(char const * var) {
        return {var};
    }

    inline std::string to_string(Eigen::Ref<matrixN_t const> var) {
        Eigen::IOFormat HeavyFmt(Eigen::FullPrecision, 0, ", ", ";\n", "[", "]", "[", "]");
        std::stringstream matrixStream;
        matrixStream << var.format(HeavyFmt);
        return matrixStream.str();
    }

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

        if (isTelemetryConfigured_)
        {
            std::cout << "Error - AbstractController::registerConstant - Telemetry already initialized. Impossible to register new variables." << std::endl;
            return result_t::ERROR_INIT_FAILED;
        }

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
        registeredConstants_.emplace_back(fieldName, to_string(value));

        return result_t::SUCCESS;
    }
}

#endif // JIMINY_ABSTRACT_CONTROLLER_TPP


