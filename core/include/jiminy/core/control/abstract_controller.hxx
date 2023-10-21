#ifndef JIMINY_ABSTRACT_CONTROLLER_HXX
#define JIMINY_ABSTRACT_CONTROLLER_HXX

#include <string>
#include <vector>


namespace jiminy
{
    using std::to_string;

    template<typename DerivedType>
    std::string to_string(const Eigen::MatrixBase<DerivedType> & var)
    {
        Eigen::IOFormat HeavyFmt(Eigen::FullPrecision, 0, ", ", ";\n", "[", "]", "[", "]");
        std::stringstream matrixStream;
        matrixStream << var.format(HeavyFmt);
        return matrixStream.str();
    }

    inline std::string to_string(const char_t * var)
    {
        return {var};
    }

    inline std::string to_string(const std::string & var)
    {
        return var;
    }

    /// \brief Register a constant to the telemetry.
    ///
    /// \param[in] fieldnames Name of the variable.
    /// \param[in] values Variable to add to the telemetry.
    ///
    /// \return Return code to determine whether the execution of the method was successful.
    template<typename T>
    hresult_t AbstractController::registerConstant(const std::string & fieldname, const T & value)
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
        registeredConstants_.emplace_back(fieldname, to_string(value));

        return hresult_t::SUCCESS;
    }

    template<typename T>
    hresult_t AbstractController::registerVariable(const std::string & fieldname, const T & value)
    {
        if (isTelemetryConfigured_)
        {
            PRINT_ERROR("Telemetry already initialized. Impossible to register new variables.");
            return hresult_t::ERROR_INIT_FAILED;
        }

        // Check in local cache before.
        auto variableIt = std::find_if(registeredVariables_.begin(),
                                       registeredVariables_.end(),
                                       [&fieldname](const auto & element)
                                       { return element.first == fieldname; });
        if (variableIt != registeredVariables_.end())
        {
            PRINT_ERROR("Variable already registered.");
            return hresult_t::ERROR_BAD_INPUT;
        }
        registeredVariables_.emplace_back(fieldname, &value);

        return hresult_t::SUCCESS;
    }
}

#endif  // JIMINY_ABSTRACT_CONTROLLER_HXX
