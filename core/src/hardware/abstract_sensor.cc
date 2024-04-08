#include "jiminy/core/telemetry/telemetry_sender.h"
#include "jiminy/core/robot/robot.h"

#include "jiminy/core/hardware/abstract_sensor.h"


namespace jiminy
{
    AbstractSensorBase::AbstractSensorBase(const std::string & name) noexcept :
    generator_{std::seed_seq{std::random_device{}()}},
    name_{name},
    telemetrySender_{std::make_unique<TelemetrySender>()}
    {
        // Initialize the options
        setOptions(getDefaultSensorOptions());
    }

    AbstractSensorBase::~AbstractSensorBase() = default;

    void AbstractSensorBase::configureTelemetry(std::shared_ptr<TelemetryData> telemetryData,
                                                const std::string & prefix)
    {
        if (isTelemetryConfigured_)
        {
            return;
        }

        if (!isInitialized_)
        {
            JIMINY_THROW(bad_control_flow,
                        "Sensor '",
                        name_,
                        "' of type '",
                        getType(),
                        "' is not initialized.");
        }

        if (!telemetryData)
        {
            JIMINY_THROW(bad_control_flow,
                        "Telemetry not initialized. Impossible to log sensor data.");
        }

        std::string name = getTelemetryName();
        if (!prefix.empty())
        {
            name = addCircumfix(name, prefix, {}, TELEMETRY_FIELDNAME_DELIMITER);
        }
        telemetrySender_->configure(telemetryData, name);
        telemetrySender_->registerVariable(getFieldnames(), get());
        isTelemetryConfigured_ = true;
    }

    void AbstractSensorBase::updateTelemetry()
    {
        if (isTelemetryConfigured_)
        {
            telemetrySender_->updateValues();
        }
    }

    void AbstractSensorBase::measureData()
    {
        // Add white noise
        if (baseSensorOptions_->noiseStd.size())
        {
            get() += normal(generator_, 0.0F, baseSensorOptions_->noiseStd.cast<float>())
                         .cast<double>();
        }

        // Add bias
        if (baseSensorOptions_->bias.size())
        {
            get() += baseSensorOptions_->bias;
        }
    }

    void AbstractSensorBase::setOptions(const GenericConfig & sensorOptions)
    {
        // Make sure that no simulation is already running
        auto robot = robot_.lock();
        if (robot && robot->getIsLocked())
        {
            JIMINY_THROW(bad_control_flow,
                        "Robot already locked, probably because a simulation is running. "
                        "Please stop it before setting sensor options.");
        }

        // Set sensor options
        sensorOptionsGeneric_ = sensorOptions;
        baseSensorOptions_ = std::make_unique<const AbstractSensorOptions>(sensorOptionsGeneric_);
    }

    GenericConfig AbstractSensorBase::getOptions() const noexcept
    {
        return sensorOptionsGeneric_;
    }

    bool AbstractSensorBase::getIsInitialized() const
    {
        return isInitialized_;
    }

    bool AbstractSensorBase::getIsTelemetryConfigured() const
    {
        return isTelemetryConfigured_;
    }

    const std::string & AbstractSensorBase::getName() const
    {
        return name_;
    }
}
