#ifndef JIMINY_TELEMETRY_SENDER_HXX
#define JIMINY_TELEMETRY_SENDER_HXX

#include "jiminy/core/constants.h"
#include "jiminy/core/exceptions.h"
#include "jiminy/core/utilities/helpers.h"
#include "jiminy/core/telemetry/telemetry_data.h"


namespace jiminy
{
    template<typename Scalar>
    std::enable_if_t<std::is_arithmetic_v<Scalar>, hresult_t>
    TelemetrySender::registerVariable(const std::string & name, const Scalar * value)
    {
        Scalar * positionInBuffer = nullptr;
        const std::string fullName =
            addCircumfix(name, objectName_, {}, TELEMETRY_FIELDNAME_DELIMITER);

        hresult_t returnCode = telemetryData_->registerVariable(fullName, positionInBuffer);
        if (returnCode == hresult_t::SUCCESS)
        {
            bufferPosition_.emplace_back(telemetry_data_pair_t<Scalar>{value, positionInBuffer});
            *positionInBuffer = *value;
        }

        return returnCode;
    }

    template<typename KeyType, typename Derived>
    hresult_t TelemetrySender::registerVariable(const std::vector<KeyType> & fieldnames,
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
