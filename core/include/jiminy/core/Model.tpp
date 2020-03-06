namespace jiminy
{
    template<typename TSensor>
    result_t Model::addSensor(std::string              const & sensorName,
                              std::shared_ptr<TSensor>       & sensor)
    {
        // The sensor name must be unique, even if their type is different.

        if (!isInitialized_)
        {
            std::cout << "Error - Model::addSensor - Model not initialized." << std::endl;
            return result_t::ERROR_INIT_FAILED;
        }

        std::string sensorType;
        sensorsGroupHolder_t::iterator sensorGroupIt;
        sensorType = TSensor::type_; // cannot use sensor->getType() is case of nullptr
        sensorGroupIt = sensorsGroupHolder_.find(sensorType);
        if (sensorGroupIt != sensorsGroupHolder_.end())
        {
            sensorsHolder_t::const_iterator it = sensorGroupIt->second.find(sensorName);
            if (it != sensorGroupIt->second.end())
            {
                std::cout << "Error - Model::addSensor - A sensor with the same type and name already exists." << std::endl;
                return result_t::ERROR_BAD_INPUT;
            }
        }

        // Create a new sensor data holder if necessary
        if (sensorGroupIt == sensorsGroupHolder_.end())
        {
            sensorsDataHolder_[sensorType] = std::make_shared<SensorDataHolder_t>();
            sensorTelemetryOptions_[sensorType] = false;
        }

        // Create the sensor and add it to its group
        sensorsGroupHolder_[sensorType][sensorName] =
            std::shared_ptr<AbstractSensorBase>(new TSensor(*this,
                                                            sensorsDataHolder_.at(sensorType),
                                                            sensorName));

        // Get a pointer to the sensor
        getSensor<TSensor>(sensorType, sensorName, sensor);

        return result_t::SUCCESS;
    }

    template<typename TSensor>
    result_t Model::getSensor(std::string              const & sensorType,
                              std::string              const & sensorName,
                              std::shared_ptr<TSensor>       & sensor)
    {
        if (!isInitialized_)
        {
            std::cout << "Error - Model::getSensorOptions - Model not initialized." << std::endl;
            return result_t::ERROR_INIT_FAILED;
        }

        sensorsGroupHolder_t::iterator sensorGroupIt;
        sensorGroupIt = sensorsGroupHolder_.find(sensorType);
        if (sensorGroupIt == sensorsGroupHolder_.end())
        {
            std::cout << "Error - Model::getSensorOptions - This type of sensor does not exist." << std::endl;
            return result_t::ERROR_BAD_INPUT;
        }

        sensorsHolder_t::iterator sensorIt;
        sensorIt = sensorGroupIt->second.find(sensorName);
        if (sensorIt == sensorGroupIt->second.end())
        {
            std::cout << "Error - Model::getSensorOptions - No sensor with this type and name exists." << std::endl;
            return result_t::ERROR_BAD_INPUT;
        }

        sensor = std::static_pointer_cast<TSensor>(sensorIt->second);

        return result_t::SUCCESS;
    }
}