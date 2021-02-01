#ifndef JIMINY_TELEMETRY_SENDER_TPP
#define JIMINY_TELEMETRY_SENDER_TPP

#include <iostream>
#include <string>


namespace jiminy
{
    template <typename Derived>
    hresult_t TelemetrySender::registerVariable(std::vector<std::string>   const & fieldnames,
                                                Eigen::MatrixBase<Derived> const & initialValues)
    {
        hresult_t returnCode = hresult_t::SUCCESS;
        for (uint32_t i=0; i < initialValues.size(); ++i)
        {
            if (returnCode == hresult_t::SUCCESS)
            {
                returnCode = registerVariable(fieldnames[i], initialValues[i]);
            }
        }
        return returnCode;
    }

    template <typename Derived>
    void TelemetrySender::updateValue(std::vector<std::string>   const & fieldnames,
                                      Eigen::MatrixBase<Derived> const & values)
    {
        for (uint32_t i=0; i < values.size(); ++i)
        {
            updateValue(fieldnames[i], values[i]);
        }
    }
} // namespace jiminy

#endif // JIMINY_TELEMETRY_SENDER_TPP