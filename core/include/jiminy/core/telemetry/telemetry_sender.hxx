#ifndef JIMINY_TELEMETRY_SENDER_HXX
#define JIMINY_TELEMETRY_SENDER_HXX

#include "jiminy/core/constants.h"
#include "jiminy/core/telemetry/telemetry_data.h"


namespace jiminy
{
    template<typename T>
    hresult_t TelemetrySender::registerVariable(const std::string & fieldname, const T * value)
    {
        T * positionInBuffer = nullptr;
        const std::string fullFieldName = objectName_ + TELEMETRY_FIELDNAME_DELIMITER + fieldname;

        hresult_t returnCode = telemetryData_->registerVariable(fullFieldName, positionInBuffer);
        if (returnCode == hresult_t::SUCCESS)
        {
            bufferPosition_.emplace_back(telemetry_data_pair_t<T>{value, positionInBuffer});
            *positionInBuffer = *value;
        }

        return returnCode;
    }

    template<typename Derived>
    hresult_t TelemetrySender::registerVariable(const std::vector<std::string> & fieldnames,
                                                const Eigen::MatrixBase<Derived> & values)
    {
        hresult_t returnCode = hresult_t::SUCCESS;
        for (Eigen::Index i = 0; i < values.size(); ++i)
        {
            if (returnCode == hresult_t::SUCCESS)
            {
                returnCode = registerVariable(fieldnames[i], &values[i]);
            }
        }
        return returnCode;
    }
}

#endif  // JIMINY_TELEMETRY_SENDER_HXX
