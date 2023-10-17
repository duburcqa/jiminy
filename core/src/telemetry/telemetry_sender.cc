#include "jiminy/core/telemetry/telemetry_sender.h"


namespace jiminy
{
    TelemetrySender::TelemetrySender(void) :
    objectName_(DEFAULT_TELEMETRY_NAMESPACE),
    telemetryData_(nullptr),
    bufferPosition_()
    {
        // Empty on purpose
    }

    void TelemetrySender::configureObject(std::shared_ptr<TelemetryData> telemetryDataInstance,
                                          std::string const & objectName)
    {
        objectName_ = objectName;
        telemetryData_ = telemetryDataInstance;
        bufferPosition_.clear();
    }

    hresult_t TelemetrySender::registerConstant(std::string const & variableName,
                                                std::string const & value)
    {
        std::string const fullFieldName = objectName_ + TELEMETRY_FIELDNAME_DELIMITER + variableName;
        return telemetryData_->registerConstant(fullFieldName, value);
    }

    void TelemetrySender::updateValues(void)
    {
        // Write the value directly in the buffer holder using the pointer stored in the map.
        for (auto const & pair : bufferPosition_)
        {
            std::visit([](auto && arg) { *arg.second = *arg.first; }, pair);
        }
    }

    uint32_t TelemetrySender::getLocalNumEntries(void) const
    {
        return static_cast<uint32_t>(bufferPosition_.size());
    }

    std::string const & TelemetrySender::getObjectName(void) const
    {
        return objectName_;
    }
} // End of namespace jiminy.
