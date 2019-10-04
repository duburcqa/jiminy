///////////////////////////////////////////////////////////////////////////////
///
/// \brief Implementation of the TelemetrySender class and localLogs class.
///
///////////////////////////////////////////////////////////////////////////////

#include <iostream>

#include "jiminy/core/TelemetryData.h"
#include "jiminy/core/TelemetrySender.h"


namespace jiminy
{
    TelemetrySender::TelemetrySender(void) : 
    objectName_(DEFAULT_OBJECT_NAME), 
    telemetryData_(nullptr),
    intBufferPositon_(), 
    floatBufferPositon_()
    {
        // Empty.
    }

    TelemetrySender::~TelemetrySender(void)
    {
        // Empty.
    }

    template <>
    void TelemetrySender::updateValue<int32_t>(std::string const & fieldNameIn, 
                                               int32_t     const & value)
    {
        auto it = intBufferPositon_.find(fieldNameIn);
        if (intBufferPositon_.end() == it)
        {
            std::cout << "Error - TelemetrySender::updateValue - Cannot log the variable: it was never registered as an int32_t before! |" << fieldNameIn.c_str() << "|" << std::endl;
            return;
        }

        // Write the value directly in the buffer holder using the pointer stored in the map.
        *(it->second) = value;
    }

    template <>
    void TelemetrySender::updateValue<float64_t>(std::string const & fieldNameIn, 
                                                 float64_t   const & value)
    {
        auto it = floatBufferPositon_.find(fieldNameIn);
        if (floatBufferPositon_.end() == it)
        {
            std::cout << "Error - TelemetrySender::updateValue - Cannot log the variable: it was never registered as an float64_t before! |" << fieldNameIn.c_str() << "|" << std::endl;
            return;
        }

        // Write the value directly in the buffer holder using the pointer stored in the map.
        *(it->second) = static_cast<float32_t>(value);
    }

    template<>
    result_t TelemetrySender::registerNewEntry<int32_t>(std::string const & fieldNameIn, 
                                                        int32_t     const & initialValue)
    {
        int32_t * positionInBuffer = nullptr;
        std::string const fullFieldName = objectName_ + "." + fieldNameIn;

        result_t returnCode = telemetryData_->registerVariable<int32_t>(fullFieldName, positionInBuffer);
        if (returnCode == result_t::SUCCESS)
        {
            intBufferPositon_[fieldNameIn] = positionInBuffer;

            updateValue(fieldNameIn, initialValue);
        }

        return returnCode;
    }

    template<>
    result_t TelemetrySender::registerNewEntry<float64_t>(std::string const & fieldNameIn, 
                                                          float64_t   const & initialValue)
    {
        float32_t * positionInBuffer = nullptr;
        std::string const fullFieldName = objectName_ + "." + fieldNameIn;

        result_t returnCode = telemetryData_->registerVariable<float32_t>(fullFieldName, positionInBuffer);
        if (returnCode == result_t::SUCCESS)
        {
            floatBufferPositon_[fieldNameIn] = positionInBuffer;

            updateValue(fieldNameIn, initialValue);
        }

        return returnCode;
    }

    result_t TelemetrySender::addConstantEntry(std::string const & variableNameIn, 
                                               std::string const & valueIn)
    {
        std::string const fullFieldName = objectName_ + "." + variableNameIn;
        return telemetryData_->registerConstant(fullFieldName, valueIn);
    }

    void TelemetrySender::configureObject(std::shared_ptr<TelemetryData> const & telemetryDataInstance,
                                          std::string                    const & objectNameIn)
    {
        objectName_ = objectNameIn;
        telemetryData_ = std::shared_ptr<TelemetryData>(telemetryDataInstance);
        intBufferPositon_.clear();
        floatBufferPositon_.clear();
    }

    uint32_t TelemetrySender::getLocalNumEntries(void) const
    {
        return static_cast<uint32_t>(floatBufferPositon_.size() + intBufferPositon_.size());
    }

    std::string const & TelemetrySender::getObjectName(void) const
    {
        return objectName_;
    }
} // End of namespace jiminy.