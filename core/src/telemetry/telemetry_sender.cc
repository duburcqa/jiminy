#include "jiminy/core/telemetry/telemetry_sender.h"


namespace jiminy
{
    void TelemetrySender::configure(std::shared_ptr<TelemetryData> telemetryData,
                                    const std::string_view & name)
    {
        objectName_ = name;
        telemetryData_ = telemetryData;
        bufferPosition_.clear();
    }

    void TelemetrySender::registerConstant(const std::string & name, const std::string & value)
    {
        const std::string fullFieldName =
            addCircumfix(name, objectName_, {}, TELEMETRY_FIELDNAME_DELIMITER);
        return telemetryData_->registerConstant(fullFieldName, value);
    }

    void TelemetrySender::updateValues()
    {
        // Write the value directly in the buffer holder using the pointer stored in the map.
        for (const auto & pair : bufferPosition_)
        {
            std::visit([](auto && arg) { *arg.second = *arg.first; }, pair);
        }
    }

    uint32_t TelemetrySender::getNumEntries() const
    {
        return static_cast<uint32_t>(bufferPosition_.size());
    }

    const std::string & TelemetrySender::getName() const
    {
        return objectName_;
    }
}
