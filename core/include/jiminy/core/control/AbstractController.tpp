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

    template<typename DerivedType>
    std::string to_string(Eigen::MatrixBase<DerivedType> const & var)
    {
        Eigen::IOFormat HeavyFmt(Eigen::FullPrecision, 0, ", ", ";\n", "[", "]", "[", "]");
        std::stringstream matrixStream;
        matrixStream << var.format(HeavyFmt);
        return matrixStream.str();
    }

    inline std::string to_string(char_t const * var)
    {
        return {var};
    }

    inline std::string to_string(std::string const & var)
    {
        return var;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    ///
    /// \brief      Register a constant to the telemetry.
    ///
    /// \param[in]  fieldnames      Name of the variable.
    /// \param[in]  values          Variable to add to the telemetry
    ///
    /// \return     Return code to determine whether the execution of the method was successful.
    ///
    ///////////////////////////////////////////////////////////////////////////////////////////////
    template<typename T>
    hresult_t AbstractController::registerConstant(std::string const & fieldName,
                                                   T           const & value)
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
                                       [&fieldName](auto const & element)
                                       {
                                           return element.first == fieldName;
                                       });
        if (constantIt != registeredConstants_.end())
        {
            PRINT_ERROR("Constant already registered.");
            return hresult_t::ERROR_BAD_INPUT;
        }
        registeredConstants_.emplace_back(fieldName, to_string(value));

        return hresult_t::SUCCESS;
    }
}

#endif // JIMINY_ABSTRACT_CONTROLLER_TPP


