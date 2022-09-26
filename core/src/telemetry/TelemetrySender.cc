///////////////////////////////////////////////////////////////////////////////
///
/// \brief Implementation of the TelemetrySender class and localLogs class.
///
///////////////////////////////////////////////////////////////////////////////

#include "jiminy/core/telemetry/TelemetryData.h"
#include "jiminy/core/Constants.h"

#include "jiminy/core/telemetry/TelemetrySender.h"


namespace jiminy
{
    TelemetrySender::TelemetrySender(void) :
    objectName_(DEFAULT_TELEMETRY_NAMESPACE),
    telemetryData_(nullptr),
    intBufferPosition_(),
    floatBufferPosition_()
    {
        // Empty on purpose
    }

    template<>
    void TelemetrySender::updateValue<int64_t>(std::string const & fieldnameIn,
                                               int64_t     const & value)
    {
        auto it = intBufferPosition_.find(fieldnameIn);
        if (it == intBufferPosition_.end())
        {
            PRINT_ERROR("Cannot log the variable: it was never registered as an int64_t before! |", fieldnameIn.c_str(), "|");
            return;
        }

        // Write the value directly in the buffer holder using the pointer stored in the map.
        *(it->second) = value;
    }

    template<>
    void TelemetrySender::updateValue<float64_t>(std::string const & fieldnameIn,
                                                 float64_t   const & value)
    {
        auto it = floatBufferPosition_.find(fieldnameIn);
        if (it == floatBufferPosition_.end())
        {
            PRINT_ERROR("Cannot log the variable: it was never registered as a float64_t before! |", fieldnameIn.c_str(), "|");
            return;
        }

        // Write the value directly in the buffer holder using the pointer stored in the map.
        *(it->second) = value;
    }

    template<>
    hresult_t TelemetrySender::registerVariable<int64_t>(std::string const & fieldnameIn,
                                                         int64_t     const & initialValue)
    {
        int64_t * positionInBuffer = nullptr;
        std::string const fullFieldName = objectName_ + TELEMETRY_FIELDNAME_DELIMITER + fieldnameIn;

        hresult_t returnCode = telemetryData_->registerVariable(fullFieldName, positionInBuffer);
        if (returnCode == hresult_t::SUCCESS)
        {
            intBufferPosition_[fieldnameIn] = positionInBuffer;
            updateValue(fieldnameIn, initialValue);
        }

        return returnCode;
    }

    template<>
    hresult_t TelemetrySender::registerVariable<float64_t>(std::string const & fieldnameIn,
                                                           float64_t   const & initialValue)
    {
        float64_t * positionInBuffer = nullptr;
        std::string const fullFieldName = objectName_ + TELEMETRY_FIELDNAME_DELIMITER + fieldnameIn;

        hresult_t returnCode = telemetryData_->registerVariable(fullFieldName, positionInBuffer);
        if (returnCode == hresult_t::SUCCESS)
        {
            floatBufferPosition_[fieldnameIn] = positionInBuffer;
            updateValue(fieldnameIn, initialValue);
        }

        return returnCode;
    }

    hresult_t TelemetrySender::registerConstant(std::string const & variableNameIn,
                                                std::string const & valueIn)
    {
        std::string const fullFieldName = objectName_ + TELEMETRY_FIELDNAME_DELIMITER + variableNameIn;
        return telemetryData_->registerConstant(fullFieldName, valueIn);
    }

    void TelemetrySender::configureObject(std::shared_ptr<TelemetryData> telemetryDataInstance,
                                          std::string const & objectNameIn)
    {
        objectName_ = objectNameIn;
        telemetryData_ = telemetryDataInstance;
        intBufferPosition_.clear();
        floatBufferPosition_.clear();
    }

    uint32_t TelemetrySender::getLocalNumEntries(void) const
    {
        return static_cast<uint32_t>(floatBufferPosition_.size() + intBufferPosition_.size());
    }

    std::string const & TelemetrySender::getObjectName(void) const
    {
        return objectName_;
    }
} // End of namespace jiminy.
