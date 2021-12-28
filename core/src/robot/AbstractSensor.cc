#include "jiminy/core/robot/Robot.h"

#include "jiminy/core/utilities/Random.h"
#include "jiminy/core/robot/AbstractSensor.h"


namespace jiminy
{
    AbstractSensorBase::AbstractSensorBase(std::string const & name) :
    baseSensorOptions_(nullptr),
    sensorOptionsHolder_(),
    isInitialized_(false),
    isAttached_(false),
    isTelemetryConfigured_(false),
    robot_(),
    name_(name),
    telemetrySender_()
    {
        // Initialize the options
        setOptions(getDefaultSensorOptions());
    }

    hresult_t AbstractSensorBase::configureTelemetry(std::shared_ptr<TelemetryData> telemetryData,
                                                     std::string const & objectPrefixName)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (!isInitialized_)
        {
            PRINT_ERROR("Sensor '", name_, "' of type '", getType(), "' is not initialized.");
            returnCode = hresult_t::ERROR_INIT_FAILED;
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            if (!isTelemetryConfigured_)
            {
                if (telemetryData)
                {
                    std::string objectName = getTelemetryName();
                    if (!objectPrefixName.empty())
                    {
                        objectName = objectPrefixName + TELEMETRY_FIELDNAME_DELIMITER + objectName;
                    }
                    telemetrySender_.configureObject(telemetryData, objectName);
                    returnCode = telemetrySender_.registerVariable(getFieldnames(), get());
                    if (returnCode == hresult_t::SUCCESS)
                    {
                        isTelemetryConfigured_ = true;
                    }
                }
                else
                {
                    PRINT_ERROR("Telemetry not initialized. Impossible to log sensor data.");
                    returnCode = hresult_t::ERROR_INIT_FAILED;
                }
            }
        }

        return returnCode;
    }

    void AbstractSensorBase::updateTelemetry(void)
    {
        if (isTelemetryConfigured_)
        {
            telemetrySender_.updateValue(getFieldnames(), get());
        }
    }

    void AbstractSensorBase::skewMeasurement(void)
    {
        // Add white noise
        if (baseSensorOptions_->noiseStd.size())
        {
            get() += randVectorNormal(baseSensorOptions_->noiseStd);
        }

        // Add bias
        if (baseSensorOptions_->bias.size())
        {
            get() += baseSensorOptions_->bias;
        }
    }

    hresult_t AbstractSensorBase::setOptions(configHolder_t const & sensorOptions)
    {
        sensorOptionsHolder_ = sensorOptions;
        baseSensorOptions_ = std::make_unique<abstractSensorOptions_t const>(sensorOptionsHolder_);
        return hresult_t::SUCCESS;
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