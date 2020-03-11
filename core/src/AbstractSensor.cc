#include "jiminy/core/AbstractSensor.h"
#include "jiminy/core/Model.h"


namespace jiminy
{
    AbstractSensorBase::AbstractSensorBase(std::string const & name) :
    baseSensorOptions_(nullptr),
    sensorOptionsHolder_(),
    isInitialized_(false),
    isAttached_(false),
    isTelemetryConfigured_(false),
    model_(nullptr),
    name_(name),
    data_(),
    telemetrySender_()
    {
        // Initialize the options
        setOptions(getDefaultOptions());
    }

    result_t AbstractSensorBase::configureTelemetry(std::shared_ptr<TelemetryData> const & telemetryData)
    {
        result_t returnCode = result_t::SUCCESS;

        if (!isInitialized_)
        {
            std::cout << "Error - AbstractSensorBase::configureTelemetry - The sensor is not initialized." << std::endl;
            returnCode = result_t::ERROR_INIT_FAILED;
        }

        if (returnCode == result_t::SUCCESS)
        {
            if (!isTelemetryConfigured_)
            {
                if (telemetryData)
                {
                    telemetrySender_.configureObject(telemetryData, getTelemetryName());
                    returnCode = telemetrySender_.registerVariable(getFieldNames(), data_);
                    if (returnCode == result_t::SUCCESS)
                    {
                        isTelemetryConfigured_ = true;
                    }
                }
                else
                {
                    std::cout << "Error - AbstractSensorBase::configureTelemetry - Telemetry not initialized. Impossible to log sensor data." << std::endl;
                    returnCode = result_t::ERROR_INIT_FAILED;
                }
            }
        }

        return returnCode;
    }

    void AbstractSensorBase::updateTelemetry(void)
    {
        if(isTelemetryConfigured_)
        {
            updateDataBuffer(); // Force update the internal measurement buffer if necessary
            telemetrySender_.updateValue(getFieldNames(), data_);
        }
    }

    result_t AbstractSensorBase::setOptions(configHolder_t const & sensorOptions)
    {
        sensorOptionsHolder_ = sensorOptions;
        baseSensorOptions_ = std::make_unique<abstractSensorOptions_t const>(sensorOptionsHolder_);
        return result_t::SUCCESS;
    }

    configHolder_t AbstractSensorBase::getOptions(void) const
    {
        return sensorOptionsHolder_;
    }

    bool_t const & AbstractSensorBase::getIsInitialized(void) const
    {
        return isInitialized_;
    }

    bool_t const & AbstractSensorBase::getIsTelemetryConfigured(void) const
    {
        return isTelemetryConfigured_;
    }

    std::string const & AbstractSensorBase::getName(void) const
    {
        return name_;
    }
}