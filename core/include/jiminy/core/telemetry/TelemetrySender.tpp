#ifndef JIMINY_TELEMETRY_SENDER_TPP
#define JIMINY_TELEMETRY_SENDER_TPP

#include "jiminy/core/telemetry/TelemetryData.h"
#include "jiminy/core/Constants.h"


namespace jiminy
{
    template<typename T>
    hresult_t TelemetrySender::registerVariable(std::string const & fieldname,
                                                T           const * value)
    {
        T * positionInBuffer = nullptr;
        std::string const fullFieldName = objectName_ + TELEMETRY_FIELDNAME_DELIMITER + fieldname;

        hresult_t returnCode = telemetryData_->registerVariable(fullFieldName, positionInBuffer);
        if (returnCode == hresult_t::SUCCESS)
        {
            bufferPosition_.emplace_back(telemetry_data_pair_t<T>{value, positionInBuffer});
            *positionInBuffer = *value;
        }

        return returnCode;
    }

    template<typename Derived>
    hresult_t TelemetrySender::registerVariable(std::vector<std::string>   const & fieldnames,
                                                Eigen::MatrixBase<Derived> const & values)
    {
        hresult_t returnCode = hresult_t::SUCCESS;
        for (Eigen::Index i=0; i < values.size(); ++i)
        {
            if (returnCode == hresult_t::SUCCESS)
            {
                returnCode = registerVariable(fieldnames[i], &values[i]);
            }
        }
        return returnCode;
    }
} // namespace jiminy

#endif // JIMINY_TELEMETRY_SENDER_TPP