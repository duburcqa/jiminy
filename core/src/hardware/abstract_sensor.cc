#include "jiminy/core/exceptions.h"
#include "jiminy/core/robot/robot.h"
#include "jiminy/core/utilities/random.h"

#include "jiminy/core/hardware/abstract_sensor.h"


namespace jiminy
{
    AbstractSensorBase::AbstractSensorBase(const std::string & name) :
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
                                                     const std::string & objectPrefixName)
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
                        objectName = addCircumfix(
                            objectName, objectPrefixName, {}, TELEMETRY_FIELDNAME_DELIMITER);
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

    void AbstractSensorBase::updateTelemetry()
    {
        if (isTelemetryConfigured_)
        {
            telemetrySender_.updateValues();
        }
    }

    void AbstractSensorBase::measureData()
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

    hresult_t AbstractSensorBase::setOptions(const GenericConfig & sensorOptions)
    {
        sensorOptionsHolder_ = sensorOptions;
        baseSensorOptions_ = std::make_unique<const abstractSensorOptions_t>(sensorOptionsHolder_);
        return hresult_t::SUCCESS;
    }

    GenericConfig AbstractSensorBase::getOptions() const
    {
        return sensorOptionsHolder_;
    }

    bool_t AbstractSensorBase::getIsInitialized() const
    {
        return isInitialized_;
    }

    bool_t AbstractSensorBase::getIsTelemetryConfigured() const
    {
        return isTelemetryConfigured_;
    }

    const std::string & AbstractSensorBase::getName() const
    {
        return name_;
    }
}