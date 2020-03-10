///////////////////////////////////////////////////////////////////////////////
///
/// \brief Contains templated function implementation of the Model class.
///
///////////////////////////////////////////////////////////////////////////////

#ifndef JIMINY_MODEL_TPP
#define JIMINY_MODEL_TPP

namespace jiminy
{

    template<typename TMotor>
    result_t Model::addMotor(std::string const & motorName,
                             std::shared_ptr<AbstractMotorBase> & motor)
    {
        if (getIsLocked())
        {
            std::cout << "Error - Model::addMotors - Model is locked, probably because a simulation is running.";
            std::cout << " Please stop it before adding motors." << std::endl;
            return result_t::ERROR_GENERIC;
        }

        if (!isInitialized_)
        {
            std::cout << "Error - Model::addMotors - Model not initialized." << std::endl;
            return result_t::ERROR_INIT_FAILED;
        }

        if (motor)
        {
            std::cout << "Error - Model::addMotor - Shared pointer 'motor' already associated with an existing motor." << std::endl;
            return result_t::ERROR_BAD_INPUT;
        }

        auto motorIt = motorsHolder_.find(motorName);
        if (motorIt != motorsHolder_.end())
        {
            std::cout << "Error - Model::addMotor - A motor with the same name already exists." << std::endl;
            return result_t::ERROR_BAD_INPUT;
        }

        // Create the motor and add it
        motorsHolder_.emplace(
            std::piecewise_construct,
            std::forward_as_tuple(motorName),
            std::forward_as_tuple(
                new TMotor(*this, motorsSharedHolder_, motorName))
        );

        // Get a shared pointer to the motor
        getMotor(motorName, motor);

        // Refresh the attributes of the model
        refreshMotorProxies();

        return result_t::SUCCESS;
    }

    template<typename TSensor>
    result_t Model::addSensor(std::string const & sensorName,
                              std::shared_ptr<AbstractSensorBase> & sensor)
    {
        // The sensors' names must be unique, even if their type is different.

        if (getIsLocked())
        {
            std::cout << "Error - Model::addSensor - Model is locked, probably because a simulation is running.";
            std::cout << " Please stop it before adding sensors." << std::endl;
            return result_t::ERROR_GENERIC;
        }

        if (!isInitialized_)
        {
            std::cout << "Error - Model::addSensor - Model not initialized." << std::endl;
            return result_t::ERROR_INIT_FAILED;
        }

        if (sensor)
        {
            std::cout << "Error - Model::addSensor - Shared pointer 'sensor' already associated with an existing sensor." << std::endl;
            return result_t::ERROR_BAD_INPUT;
        }

        std::string sensorType = TSensor::type_;
        auto sensorGroupIt = sensorsGroupHolder_.find(sensorType);
        if (sensorGroupIt != sensorsGroupHolder_.end())
        {
            auto sensorIt = sensorGroupIt->second.find(sensorName);
            if (sensorIt != sensorGroupIt->second.end())
            {
                std::cout << "Error - Model::addSensor - A sensor with the same type and name already exists." << std::endl;
                return result_t::ERROR_BAD_INPUT;
            }
        }

        // Create a new sensor data holder if necessary
        if (sensorGroupIt == sensorsGroupHolder_.end())
        {
            sensorsSharedHolder_[sensorType] = std::make_shared<SensorSharedDataHolder_t>();
            sensorTelemetryOptions_[sensorType] = false;
        }

        // Create the sensor and add it to its group
        sensorsGroupHolder_[sensorType].emplace(
            std::piecewise_construct,
            std::forward_as_tuple(sensorName),
            std::forward_as_tuple(
                new TSensor(*this, sensorsSharedHolder_.at(sensorType), sensorName))
        );

        // Get a shared pointer to the sensor
        getSensor(sensorType, sensorName, sensor);

        return result_t::SUCCESS;
    }
}

#endif // JIMINY_MODEL_TPP