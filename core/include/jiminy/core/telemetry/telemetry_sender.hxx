#ifndef JIMINY_TELEMETRY_SENDER_HXX
#define JIMINY_TELEMETRY_SENDER_HXX

#include "jiminy/core/utilities/helpers.h"
#include "jiminy/core/telemetry/telemetry_data.h"


namespace jiminy
{
    template<typename Scalar>
    std::enable_if_t<std::is_arithmetic_v<Scalar>, void>
    TelemetrySender::registerVariable(const std::string & name, const Scalar * valuePtr)
    {
        const std::string fullName =
            addCircumfix(name, objectName_, {}, TELEMETRY_FIELDNAME_DELIMITER);
        Scalar * positionInBuffer = telemetryData_->registerVariable<Scalar>(fullName);

        *positionInBuffer = *valuePtr;
        bufferPosition_.emplace_back(telemetry_data_pair_t<Scalar>{valuePtr, positionInBuffer});
    }

    template<typename KeyType, typename Derived>
    void TelemetrySender::registerVariable(const std::vector<KeyType> & fieldnames,
                                           const Eigen::MatrixBase<Derived> & values)
    {
        for (Eigen::Index i = 0; i < values.size(); ++i)
        {
            registerVariable(fieldnames[i], &values[i]);
        }
    }
}

#endif  // JIMINY_TELEMETRY_SENDER_HXX
