#include <fstream>
#include <exception>

#include "jiminy/core/utilities/helpers.h"
#include "jiminy/core/utilities/pinocchio.h"
#include "jiminy/core/utilities/json.h"
#include "jiminy/core/io/file_device.h"
#include "jiminy/core/hardware/abstract_motor.h"
#include "jiminy/core/hardware/abstract_sensor.h"

#include "jiminy/core/robot/robot.h"


namespace jiminy
{

    Robot::Robot() noexcept :
    motorSharedStorage_{std::make_shared<MotorSharedStorage>()}
    {
    }

    Robot::~Robot()
    {
        // Detach all the motors and sensors
        detachSensors();
        detachMotors();
    }

    hresult_t Robot::initialize(const std::string & urdfPath,
                                bool hasFreeflyer,
                                const std::vector<std::string> & meshPackageDirs,
                                bool loadVisualMeshes)
    {
        // Detach all the motors and sensors
        detachSensors();
        detachMotors();

        /* Delete the current model and generate a new one.
           Note that is also refresh all proxies automatically. */
        return Model::initialize(urdfPath, hasFreeflyer, meshPackageDirs, loadVisualMeshes);
    }

    hresult_t Robot::initialize(const pinocchio::Model & pinocchioModel,
                                const pinocchio::GeometryModel & collisionModel,
                                const pinocchio::GeometryModel & visualModel)
    {
        // Detach all the motors and sensors
        detachSensors();
        detachMotors();

        /* Delete the current model and generate a new one.
           Note that is also refresh all proxies automatically. */
        return Model::initialize(pinocchioModel, collisionModel, visualModel);
    }

    void Robot::reset(const uniform_random_bit_generator_ref<uint32_t> & g)
    {
        // Reset the model
        Model::reset(g);

        // Reset the motors
        if (!motors_.empty())
        {
            (*motors_.begin())->resetAll();
        }

        // Reset the sensors
        for (auto & sensorGroupItem : sensors_)
        {
            if (!sensorGroupItem.second.empty())
            {
                (*sensorGroupItem.second.begin())->resetAll(g());
            }
        }

        // Reset the telemetry flag
        isTelemetryConfigured_ = false;
    }

    hresult_t Robot::configureTelemetry(std::shared_ptr<TelemetryData> telemetryData,
                                        const std::string & prefix)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (!isInitialized_)
        {
            PRINT_ERROR("The robot is not initialized.");
            returnCode = hresult_t::ERROR_INIT_FAILED;
        }

        isTelemetryConfigured_ = false;
        if (returnCode == hresult_t::SUCCESS)
        {
            telemetryData_ = telemetryData;
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            if (!isTelemetryConfigured_)
            {
                for (const auto & [sensorType, sensorGroup] : sensors_)
                {
                    for (const auto & sensor : sensorGroup)
                    {
                        if (returnCode == hresult_t::SUCCESS)
                        {
                            if (sensorTelemetryOptions_[sensorType])
                            {
                                returnCode = sensor->configureTelemetry(telemetryData_, prefix);
                            }
                        }
                    }
                }
            }
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            isTelemetryConfigured_ = true;
        }

        return returnCode;
    }

    hresult_t Robot::attachMotor(std::shared_ptr<AbstractMotorBase> motor)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (!isInitialized_)
        {
            PRINT_ERROR("The robot is not initialized.");
            returnCode = hresult_t::ERROR_INIT_FAILED;
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            if (getIsLocked())
            {
                PRINT_ERROR("Robot is locked, probably because a simulation is running. Please "
                            "stop it before adding motors.");
                returnCode = hresult_t::ERROR_GENERIC;
            }
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            const std::string & motorName = motor->getName();
            auto motorIt = std::find_if(motors_.begin(),
                                        motors_.end(),
                                        [&motorName](const auto & elem)
                                        { return (elem->getName() == motorName); });
            if (motorIt != motors_.end())
            {
                PRINT_ERROR("A motor with the same name already exists.");
                returnCode = hresult_t::ERROR_BAD_INPUT;
            }
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            // Define robot notification method, responsible for updating the robot if
            // necessary after changing the motor parameters, for example the armature.
            auto notifyRobot =
                [robot_ = std::weak_ptr<Robot>(shared_from_this())](AbstractMotorBase & motorIn)
            {
                // Make sure the robot still exists
                auto robot = robot_.lock();
                if (!robot)
                {
                    PRINT_ERROR("Robot has been deleted. Impossible to notify motor update.");
                    return hresult_t::ERROR_GENERIC;
                }

                // Update rotor inertia and effort limit of pinocchio model
                Eigen::Index jointVelocityOrigIndex;
                getJointVelocityFirstIndex(
                    robot->pinocchioModelOrig_, motorIn.getJointName(), jointVelocityOrigIndex);
                robot->pinocchioModel_.rotorInertia[motorIn.getJointVelocityIndex()] =
                    robot->pinocchioModelOrig_.rotorInertia[jointVelocityOrigIndex] +
                    motorIn.getArmature();
                robot->pinocchioModel_.effortLimit[motorIn.getJointVelocityIndex()] =
                    motorIn.getCommandLimit();

                return hresult_t::SUCCESS;
            };

            // Attach the motor
            returnCode = motor->attach(shared_from_this(), notifyRobot, motorSharedStorage_.get());
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            // Add the motor to the holder
            motors_.push_back(motor);

            // Refresh the motors proxies
            refreshMotorProxies();
        }

        return returnCode;
    }

    hresult_t Robot::detachMotor(const std::string & motorName)
    {
        if (!isInitialized_)
        {
            PRINT_ERROR("Robot not initialized.");
            return hresult_t::ERROR_INIT_FAILED;
        }

        if (getIsLocked())
        {
            PRINT_ERROR("Robot is locked, probably because a simulation is running. Please stop "
                        "it before removing motors.");
            return hresult_t::ERROR_GENERIC;
        }

        auto motorIt = std::find_if(motors_.cbegin(),
                                    motors_.cend(),
                                    [&motorName](const auto & elem)
                                    { return (elem->getName() == motorName); });
        if (motorIt == motors_.cend())
        {
            PRINT_ERROR("No motor with this name exists.");
            return hresult_t::ERROR_BAD_INPUT;
        }

        // Reset effortLimit and rotorInertia
        const std::shared_ptr<AbstractMotorBase> & motor = *motorIt;
        Eigen::Index jointVelocityOrigIndex;
        ::jiminy::getJointVelocityFirstIndex(
            pinocchioModelOrig_, motor->getJointName(), jointVelocityOrigIndex);
        pinocchioModel_.rotorInertia[motor->getJointVelocityIndex()] =
            pinocchioModelOrig_.rotorInertia[jointVelocityOrigIndex];
        pinocchioModel_.effortLimit[motor->getJointVelocityIndex()] = 0.0;

        // Detach the motor
        motor->detach();  // Cannot fail at this point

        // Remove the motor from the holder
        motors_.erase(motorIt);

        // Refresh the motors proxies
        refreshMotorProxies();

        return hresult_t::SUCCESS;
    }

    hresult_t Robot::detachMotors(std::vector<std::string> motorsNames)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (motorsNames.empty())
        {
            // Remove all sensors if none is specified
            if (returnCode == hresult_t::SUCCESS)
            {
                if (!motorNames_.empty())
                {
                    returnCode = detachMotors(motorNames_);
                }
            }
        }
        else
        {
            // Make sure that no motor names are duplicates
            if (checkDuplicates(motorsNames))
            {
                PRINT_ERROR("Duplicated motor names.");
                returnCode = hresult_t::ERROR_BAD_INPUT;
            }

            if (returnCode == hresult_t::SUCCESS)
            {
                // Make sure that every motor name exist
                if (!checkInclusion(motorNames_, motorsNames))
                {
                    PRINT_ERROR("At least one of the motor names does not exist.");
                    returnCode = hresult_t::ERROR_BAD_INPUT;
                }
            }

            // Detach motors one-by-one
            for (const std::string & name : motorsNames)
            {
                if (returnCode == hresult_t::SUCCESS)
                {
                    returnCode = detachMotor(name);
                }
            }
        }

        return returnCode;
    }

    hresult_t Robot::attachSensor(std::shared_ptr<AbstractSensorBase> sensor)
    {
        // The sensors' names must be unique, even if their type is different.

        hresult_t returnCode = hresult_t::SUCCESS;

        if (!isInitialized_)
        {
            PRINT_ERROR("The robot is not initialized.");
            returnCode = hresult_t::ERROR_INIT_FAILED;
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            if (getIsLocked())
            {
                PRINT_ERROR("Robot is locked, probably because a simulation is running. Please "
                            "stop it before adding sensors.");
                returnCode = hresult_t::ERROR_GENERIC;
            }
        }

        const std::string & sensorName = sensor->getName();
        const std::string & sensorType = sensor->getType();
        SensorTree::const_iterator sensorGroupIt;
        if (returnCode == hresult_t::SUCCESS)
        {
            sensorGroupIt = sensors_.find(sensorType);
            if (sensorGroupIt != sensors_.end())
            {
                auto sensorIt = std::find_if(sensorGroupIt->second.begin(),
                                             sensorGroupIt->second.end(),
                                             [&sensorName](const auto & elem)
                                             { return (elem->getName() == sensorName); });
                if (sensorIt != sensorGroupIt->second.end())
                {
                    PRINT_ERROR("A sensor with the same type and name already exists.");
                    returnCode = hresult_t::ERROR_BAD_INPUT;
                }
            }
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            // Create a new sensor data holder if necessary
            if (sensorGroupIt == sensors_.end())
            {
                sensorSharedStorageMap_.emplace(sensorType,
                                                std::make_shared<SensorSharedStorage>());
                sensorTelemetryOptions_.emplace(sensorType,
                                                true);  // Enable the telemetry by default
            }

            // Attach the sensor
            returnCode =
                sensor->attach(shared_from_this(), sensorSharedStorageMap_[sensorType].get());
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            // Create the sensor and add it to its group
            sensors_[sensorType].push_back(sensor);

            // Refresh the sensors proxies
            refreshSensorProxies();
        }

        return returnCode;
    }

    hresult_t Robot::detachSensor(const std::string & sensorType, const std::string & sensorName)
    {
        if (getIsLocked())
        {
            PRINT_ERROR("Robot is locked, probably because a simulation is running. Please stop "
                        "it before removing sensors.");
            return hresult_t::ERROR_GENERIC;
        }

        if (!isInitialized_)
        {
            PRINT_ERROR("Robot not initialized.");
            return hresult_t::ERROR_INIT_FAILED;
        }

        // FIXME: remove explicit conversion to `std::string` when moving to C++20
        auto sensorGroupIt = sensors_.find(std::string{sensorType});
        if (sensorGroupIt == sensors_.end())
        {
            PRINT_ERROR("This type of sensor does not exist.");
            return hresult_t::ERROR_BAD_INPUT;
        }

        SensorVector::iterator sensorIt;
        sensorIt = std::find_if(sensorGroupIt->second.begin(),
                                sensorGroupIt->second.end(),
                                [&sensorName](const auto & elem)
                                { return (elem->getName() == sensorName); });
        if (sensorIt == sensorGroupIt->second.end())
        {
            PRINT_ERROR("No sensor with this type and name exists.");
            return hresult_t::ERROR_BAD_INPUT;
        }

        // Detach the sensor
        (*sensorIt)->detach();  // Cannot fail at this point

        // Remove the sensor from its group
        sensorGroupIt->second.erase(sensorIt);

        // Remove the sensor group if there is no more sensors left
        if (sensorGroupIt->second.empty())
        {
            sensors_.erase(sensorType);
            sensorSharedStorageMap_.erase(sensorType);
            sensorTelemetryOptions_.erase(sensorType);
        }

        // Refresh the sensors proxies
        refreshSensorProxies();

        return hresult_t::SUCCESS;
    }

    hresult_t Robot::detachSensors(const std::string & sensorType)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (!sensorType.empty())
        {
            auto sensorGroupIt = sensors_.find(sensorType);
            if (sensorGroupIt == sensors_.end())
            {
                PRINT_ERROR("No sensor with this type exists.");
                returnCode = hresult_t::ERROR_BAD_INPUT;
            }

            std::vector<std::string> sensorGroupNames =
                sensorNames_[sensorType];  // Make a copy since calling detachSensors update it !
            for (const std::string & sensorName : sensorGroupNames)
            {
                if (returnCode == hresult_t::SUCCESS)
                {
                    returnCode = detachSensor(sensorType, sensorName);
                }
            }
        }
        else
        {
            std::vector<std::string> sensorsTypesNames;
            sensorsTypesNames.reserve(sensors_.size());
            std::transform(sensors_.begin(),
                           sensors_.end(),
                           std::back_inserter(sensorsTypesNames),
                           [](const auto & pair) -> std::string { return pair.first; });
            for (const std::string & sensorTypeName : sensorsTypesNames)
            {
                if (returnCode == hresult_t::SUCCESS)
                {
                    returnCode = detachSensors(sensorTypeName);
                }
            }
        }

        return returnCode;
    }

    hresult_t Robot::refreshProxies()
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (!isInitialized_)
        {
            PRINT_ERROR("Robot not initialized.");
            returnCode = hresult_t::ERROR_INIT_FAILED;
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            returnCode = Model::refreshProxies();
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            returnCode = refreshMotorProxies();
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            returnCode = refreshSensorProxies();
        }

        return returnCode;
    }

    hresult_t Robot::refreshMotorProxies()
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (!isInitialized_)
        {
            PRINT_ERROR("Robot not initialized.");
            returnCode = hresult_t::ERROR_INIT_FAILED;
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            // Determine the number of motors
            nmotors_ = motors_.size();

            // Extract the motor names
            motorNames_.clear();
            motorNames_.reserve(nmotors_);
            std::transform(motors_.begin(),
                           motors_.end(),
                           std::back_inserter(motorNames_),
                           [](const auto & elem) -> std::string { return elem->getName(); });

            // Generate the fieldnames associated with command
            logCommandFieldnames_.clear();
            logCommandFieldnames_.reserve(nmotors_);
            std::transform(
                motors_.begin(),
                motors_.end(),
                std::back_inserter(logCommandFieldnames_),
                [](const auto & elem) -> std::string
                { return addCircumfix(elem->getName(), toString(JOINT_PREFIX_BASE, "Command")); });

            // Generate the fieldnames associated with motor efforts
            logMotorEffortFieldnames_.clear();
            logMotorEffortFieldnames_.reserve(nmotors_);
            std::transform(
                motors_.begin(),
                motors_.end(),
                std::back_inserter(logMotorEffortFieldnames_),
                [](const auto & elem) -> std::string
                { return addCircumfix(elem->getName(), toString(JOINT_PREFIX_BASE, "Effort")); });
        }

        return returnCode;
    }

    hresult_t Robot::refreshSensorProxies()
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (!isInitialized_)
        {
            PRINT_ERROR("Robot not initialized.");
            returnCode = hresult_t::ERROR_INIT_FAILED;
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            // Extract the motor names
            sensorNames_.clear();
            sensorNames_.reserve(sensors_.size());
            for (const auto & [sensorType, sensorGroup] : sensors_)
            {
                std::vector<std::string> sensorGroupNames;
                sensorGroupNames.reserve(sensorGroup.size());
                std::transform(sensorGroup.begin(),
                               sensorGroup.end(),
                               std::back_inserter(sensorGroupNames),
                               [](const auto & elem) -> std::string { return elem->getName(); });
                sensorNames_.emplace(sensorType, std::move(sensorGroupNames));
            }
        }

        return returnCode;
    }

    hresult_t Robot::getMotor(const std::string & motorName,
                              std::shared_ptr<AbstractMotorBase> & motor)
    {
        if (!isInitialized_)
        {
            PRINT_ERROR("Robot not initialized.");
            return hresult_t::ERROR_INIT_FAILED;
        }

        auto motorIt = std::find_if(motors_.begin(),
                                    motors_.end(),
                                    [&motorName](const auto & elem)
                                    { return (elem->getName() == motorName); });
        if (motorIt == motors_.end())
        {
            PRINT_ERROR("No motor with this name exists.");
            return hresult_t::ERROR_BAD_INPUT;
        }

        motor = *motorIt;

        return hresult_t::SUCCESS;
    }

    hresult_t Robot::getMotor(const std::string & motorName,
                              std::weak_ptr<const AbstractMotorBase> & motor) const
    {
        if (!isInitialized_)
        {
            PRINT_ERROR("Robot not initialized.");
            return hresult_t::ERROR_INIT_FAILED;
        }

        auto motorIt = std::find_if(motors_.begin(),
                                    motors_.end(),
                                    [&motorName](const auto & elem)
                                    { return (elem->getName() == motorName); });
        if (motorIt == motors_.end())
        {
            PRINT_ERROR("No motor with this name exists.");
            return hresult_t::ERROR_BAD_INPUT;
        }

        motor = std::const_pointer_cast<const AbstractMotorBase>(*motorIt);

        return hresult_t::SUCCESS;
    }

    const Robot::MotorVector & Robot::getMotors() const
    {
        return motors_;
    }

    hresult_t Robot::getSensor(const std::string & sensorType,
                               const std::string & sensorName,
                               std::weak_ptr<const AbstractSensorBase> & sensor) const
    {
        if (!isInitialized_)
        {
            PRINT_ERROR("Robot not initialized.");
            return hresult_t::ERROR_INIT_FAILED;
        }

        auto sensorGroupIt = sensors_.find(sensorType);
        if (sensorGroupIt == sensors_.end())
        {
            PRINT_ERROR("This type of sensor does not exist.");
            return hresult_t::ERROR_BAD_INPUT;
        }

        auto sensorIt = std::find_if(sensorGroupIt->second.begin(),
                                     sensorGroupIt->second.end(),
                                     [&sensorName](const auto & elem)
                                     { return (elem->getName() == sensorName); });
        if (sensorIt == sensorGroupIt->second.end())
        {
            PRINT_ERROR("No sensor with this type and name exists.");
            return hresult_t::ERROR_BAD_INPUT;
        }

        sensor = std::const_pointer_cast<const AbstractSensorBase>(*sensorIt);

        return hresult_t::SUCCESS;
    }

    hresult_t Robot::getSensor(const std::string & sensorType,
                               const std::string & sensorName,
                               std::shared_ptr<AbstractSensorBase> & sensor)
    {
        if (!isInitialized_)
        {
            PRINT_ERROR("Robot not initialized.");
            return hresult_t::ERROR_INIT_FAILED;
        }

        auto sensorGroupIt = sensors_.find(sensorType);
        if (sensorGroupIt == sensors_.end())
        {
            PRINT_ERROR("This type of sensor does not exist.");
            return hresult_t::ERROR_BAD_INPUT;
        }

        auto sensorIt = std::find_if(sensorGroupIt->second.begin(),
                                     sensorGroupIt->second.end(),
                                     [&sensorName](const auto & elem)
                                     { return (elem->getName() == sensorName); });
        if (sensorIt == sensorGroupIt->second.end())
        {
            PRINT_ERROR("No sensor with this type and name exists.");
            return hresult_t::ERROR_BAD_INPUT;
        }

        sensor = *sensorIt;

        return hresult_t::SUCCESS;
    }

    const Robot::SensorTree & Robot::getSensors() const
    {
        return sensors_;
    }

    hresult_t Robot::setOptions(const GenericConfig & robotOptions)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        GenericConfig::const_iterator modelOptionsIt;
        modelOptionsIt = robotOptions.find("model");
        if (modelOptionsIt == robotOptions.end())
        {
            PRINT_ERROR("'model' options are missing.");
            returnCode = hresult_t::ERROR_INIT_FAILED;
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            const GenericConfig & modelOptions = boost::get<GenericConfig>(modelOptionsIt->second);
            returnCode = setModelOptions(modelOptions);
        }

        GenericConfig::const_iterator motorsOptionsIt;
        if (returnCode == hresult_t::SUCCESS)
        {
            motorsOptionsIt = robotOptions.find("motors");
            if (motorsOptionsIt == robotOptions.end())
            {
                PRINT_ERROR("'motors' options are missing.");
                returnCode = hresult_t::ERROR_INIT_FAILED;
            }
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            const GenericConfig & motorsOptions =
                boost::get<GenericConfig>(motorsOptionsIt->second);
            returnCode = setMotorsOptions(motorsOptions);
        }

        GenericConfig::const_iterator sensorOptionsIt;
        if (returnCode == hresult_t::SUCCESS)
        {
            sensorOptionsIt = robotOptions.find("sensors");
            if (sensorOptionsIt == robotOptions.end())
            {
                PRINT_ERROR("'sensors' options are missing.");
                returnCode = hresult_t::ERROR_INIT_FAILED;
            }
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            const GenericConfig & sensorOptions =
                boost::get<GenericConfig>(sensorOptionsIt->second);
            returnCode = setSensorsOptions(sensorOptions);
        }

        GenericConfig::const_iterator telemetryOptionsIt;
        if (returnCode == hresult_t::SUCCESS)
        {
            telemetryOptionsIt = robotOptions.find("telemetry");
            if (telemetryOptionsIt == robotOptions.end())
            {
                PRINT_ERROR("'telemetry' options are missing.");
                returnCode = hresult_t::ERROR_INIT_FAILED;
            }
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            const GenericConfig & telemetryOptions =
                boost::get<GenericConfig>(telemetryOptionsIt->second);
            returnCode = setTelemetryOptions(telemetryOptions);
        }

        return returnCode;
    }

    GenericConfig Robot::getOptions() const noexcept
    {
        GenericConfig robotOptions;
        robotOptions["model"] = getModelOptions();
        GenericConfig motorsOptions;
        robotOptions["motors"] = getMotorsOptions();
        GenericConfig sensorOptions;
        robotOptions["sensors"] = getSensorsOptions();
        GenericConfig telemetryOptions;
        robotOptions["telemetry"] = getTelemetryOptions();
        return robotOptions;
    }

    hresult_t Robot::setMotorOptions(const std::string & motorName,
                                     const GenericConfig & motorOptions)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (getIsLocked())
        {
            PRINT_ERROR("Robot is locked, probably because a simulation is running. Please stop "
                        "it before updating the motor options.");
            returnCode = hresult_t::ERROR_GENERIC;
        }

        MotorVector::iterator motorIt;
        if (returnCode == hresult_t::SUCCESS)
        {
            motorIt = std::find_if(motors_.begin(),
                                   motors_.end(),
                                   [&motorName](const auto & elem)
                                   { return (elem->getName() == motorName); });
            if (motorIt == motors_.end())
            {
                PRINT_ERROR("No motor with this name exists.");
                returnCode = hresult_t::ERROR_BAD_INPUT;
            }
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            returnCode = (*motorIt)->setOptions(motorOptions);
        }

        return returnCode;
    }

    hresult_t Robot::setMotorsOptions(const GenericConfig & motorsOptions)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (getIsLocked())
        {
            PRINT_ERROR("Robot is locked, probably because a simulation is running. Please stop "
                        "it before updating the motor options.");
            returnCode = hresult_t::ERROR_GENERIC;
        }

        for (const auto & motor : motors_)
        {
            if (returnCode == hresult_t::SUCCESS)
            {
                auto motorOptionIt = motorsOptions.find(motor->getName());
                if (motorOptionIt != motorsOptions.end())
                {
                    returnCode =
                        motor->setOptions(boost::get<GenericConfig>(motorOptionIt->second));
                }
                else
                {
                    returnCode = motor->setOptionsAll(motorsOptions);
                    break;
                }
            }
        }

        return returnCode;
    }

    hresult_t Robot::getMotorOptions(const std::string & motorName,
                                     GenericConfig & motorOptions) const
    {
        auto motorIt = std::find_if(motors_.begin(),
                                    motors_.end(),
                                    [&motorName](const auto & elem)
                                    { return (elem->getName() == motorName); });
        if (motorIt == motors_.end())
        {
            PRINT_ERROR("No motor with this name exists.");
            return hresult_t::ERROR_BAD_INPUT;
        }

        motorOptions = (*motorIt)->getOptions();

        return hresult_t::SUCCESS;
    }

    GenericConfig Robot::getMotorsOptions() const
    {
        GenericConfig motorsOptions;
        for (const auto & motor : motors_)
        {
            motorsOptions[motor->getName()] = motor->getOptions();
        }
        return motorsOptions;
    }

    hresult_t Robot::setSensorOptions(const std::string & sensorType,
                                      const std::string & sensorName,
                                      const GenericConfig & sensorOptions)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (getIsLocked())
        {
            PRINT_ERROR("Robot is locked, probably because a simulation is running. Please stop "
                        "it before updating the sensor options.");
            returnCode = hresult_t::ERROR_GENERIC;
        }

        auto sensorGroupIt = sensors_.find(sensorType);
        if (returnCode == hresult_t::SUCCESS)
        {
            if (sensorGroupIt == sensors_.end())
            {
                PRINT_ERROR("This type of sensor does not exist.");
                returnCode = hresult_t::ERROR_BAD_INPUT;
            }
        }

        auto sensorIt = std::find_if(sensorGroupIt->second.begin(),
                                     sensorGroupIt->second.end(),
                                     [&sensorName](const auto & elem)
                                     { return (elem->getName() == sensorName); });
        if (returnCode == hresult_t::SUCCESS)
        {
            if (sensorIt == sensorGroupIt->second.end())
            {
                PRINT_ERROR("No sensor with this type and name exists.");
                returnCode = hresult_t::ERROR_BAD_INPUT;
            }
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            returnCode = (*sensorIt)->setOptions(sensorOptions);
        }

        return returnCode;
    }

    hresult_t Robot::setSensorsOptions(const std::string & sensorType,
                                       const GenericConfig & sensorsOptions)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (getIsLocked())
        {
            PRINT_ERROR("Robot is locked, probably because a simulation is running. Please stop "
                        "it before updating the sensor options.");
            returnCode = hresult_t::ERROR_GENERIC;
        }

        SensorTree::iterator sensorGroupIt;
        if (returnCode == hresult_t::SUCCESS)
        {
            sensorGroupIt = sensors_.find(sensorType);
            if (sensorGroupIt == sensors_.end())
            {
                PRINT_ERROR("This type of sensor does not exist.");
                returnCode = hresult_t::ERROR_BAD_INPUT;
            }
        }

        for (const auto & sensor : sensorGroupIt->second)
        {
            if (returnCode == hresult_t::SUCCESS)
            {
                auto sensorOptionIt = sensorsOptions.find(sensor->getName());
                if (sensorOptionIt != sensorsOptions.end())
                {
                    returnCode =
                        sensor->setOptions(boost::get<GenericConfig>(sensorOptionIt->second));
                }
                else
                {
                    returnCode = sensor->setOptionsAll(sensorsOptions);
                    break;
                }
            }
        }

        return returnCode;
    }

    hresult_t Robot::setSensorsOptions(const GenericConfig & sensorsOptions)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (getIsLocked())
        {
            PRINT_ERROR("Robot is locked, probably because a simulation is running. Please stop "
                        "it before updating the sensor options.");
            returnCode = hresult_t::ERROR_GENERIC;
        }

        for (const auto & [sensorType, sensorGroup] : sensors_)
        {
            if (returnCode == hresult_t::SUCCESS)
            {
                // FIXME: remove explicit conversion to `std::string` when moving to C++20
                auto sensorGroupOptionsIt = sensorsOptions.find(std::string{sensorType});
                if (sensorGroupOptionsIt != sensorsOptions.end())
                {
                    const GenericConfig & sensorGroupOptions =
                        boost::get<GenericConfig>(sensorGroupOptionsIt->second);

                    for (const auto & sensor : sensorGroup)
                    {
                        if (returnCode == hresult_t::SUCCESS)
                        {
                            const std::string & sensorName = sensor->getName();

                            auto sensorOptionsIt = sensorGroupOptions.find(sensorName);
                            if (sensorOptionsIt != sensorGroupOptions.end())
                            {
                                returnCode = sensor->setOptions(
                                    boost::get<GenericConfig>(sensorOptionsIt->second));
                            }
                            else
                            {
                                PRINT_ERROR("No sensor with this name exists.");
                                returnCode = hresult_t::ERROR_BAD_INPUT;
                            }
                        }
                    }
                }
                else
                {
                    PRINT_ERROR("This type of sensor does not exist.");
                    returnCode = hresult_t::ERROR_BAD_INPUT;
                }
            }
        }

        return returnCode;
    }

    hresult_t Robot::getSensorOptions(const std::string & sensorType,
                                      const std::string & sensorName,
                                      GenericConfig & sensorOptions) const
    {
        auto sensorGroupIt = sensors_.find(sensorType);
        if (sensorGroupIt == sensors_.end())
        {
            PRINT_ERROR("This type of sensor does not exist.");
            return hresult_t::ERROR_BAD_INPUT;
        }

        auto sensorIt = std::find_if(sensorGroupIt->second.begin(),
                                     sensorGroupIt->second.end(),
                                     [&sensorName](const auto & elem)
                                     { return (elem->getName() == sensorName); });
        if (sensorIt == sensorGroupIt->second.end())
        {
            PRINT_ERROR("No sensor with this type and name exists.");
            return hresult_t::ERROR_BAD_INPUT;
        }

        sensorOptions = (*sensorIt)->getOptions();

        return hresult_t::SUCCESS;
    }

    hresult_t Robot::getSensorsOptions(const std::string & sensorType,
                                       GenericConfig & sensorsOptions) const
    {
        auto sensorGroupIt = sensors_.find(sensorType);
        if (sensorGroupIt == sensors_.end())
        {
            PRINT_ERROR("This type of sensor does not exist.");
            return hresult_t::ERROR_BAD_INPUT;
        }
        sensorsOptions.clear();
        for (const auto & sensor : sensorGroupIt->second)
        {
            sensorsOptions[sensor->getName()] = sensor->getOptions();
        }

        return hresult_t::SUCCESS;
    }

    GenericConfig Robot::getSensorsOptions() const
    {
        GenericConfig sensorsOptions;
        for (const auto & [sensorType, sensorGroup] : sensors_)
        {
            GenericConfig sensorGroupOptions;
            for (const auto & sensor : sensorGroup)
            {
                sensorGroupOptions[sensor->getName()] = sensor->getOptions();
            }
            sensorsOptions[sensorType] = sensorGroupOptions;
        }
        return sensorsOptions;
    }

    hresult_t Robot::setModelOptions(const GenericConfig & modelOptions)
    {
        return Model::setOptions(modelOptions);
    }

    GenericConfig Robot::getModelOptions() const
    {
        return Model::getOptions();
    }

    hresult_t Robot::setTelemetryOptions(const GenericConfig & telemetryOptions)
    {
        if (getIsLocked())
        {
            PRINT_ERROR("Robot is locked, probably because a simulation is running. Please stop "
                        "it before updating the telemetry options.");
            return hresult_t::ERROR_GENERIC;
        }

        for (auto & [sensorType, sensorGroupTelemetryOption] : sensorTelemetryOptions_)
        {
            const std::string optionTelemetryName = toString("enable", sensorType, "s");
            auto sensorTelemetryOptionIt = telemetryOptions.find(optionTelemetryName);
            if (sensorTelemetryOptionIt == telemetryOptions.end())
            {
                PRINT_ERROR("Missing field.");
                return hresult_t::ERROR_GENERIC;
            }
            sensorGroupTelemetryOption = boost::get<bool>(sensorTelemetryOptionIt->second);
        }

        return hresult_t::SUCCESS;
    }

    GenericConfig Robot::getTelemetryOptions() const
    {
        GenericConfig telemetryOptions;
        for (const auto & [sensorType, sensorGroupTelemetryOption] : sensorTelemetryOptions_)
        {
            const std::string optionTelemetryName = toString("enable", sensorType, "s");
            telemetryOptions[optionTelemetryName] = sensorGroupTelemetryOption;
        }
        return telemetryOptions;
    }

    hresult_t Robot::dumpOptions(const std::string & filepath) const
    {
        std::shared_ptr<AbstractIODevice> device = std::make_shared<FileDevice>(filepath);
        return jsonDump(getOptions(), device);
    }

    hresult_t Robot::loadOptions(const std::string & filepath)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        std::shared_ptr<AbstractIODevice> device = std::make_shared<FileDevice>(filepath);
        GenericConfig robotOptions;
        returnCode = jsonLoad(robotOptions, device);

        if (returnCode == hresult_t::SUCCESS)
        {
            returnCode = setOptions(robotOptions);
        }

        return returnCode;
    }

    bool Robot::getIsTelemetryConfigured() const
    {
        return isTelemetryConfigured_;
    }

    void Robot::computeMotorEfforts(double t,
                                    const Eigen::VectorXd & q,
                                    const Eigen::VectorXd & v,
                                    const Eigen::VectorXd & a,
                                    const Eigen::VectorXd & command)
    {
        if (!motors_.empty())
        {
            (*motors_.begin())->computeEffortAll(t, q, v, a, command);
        }
    }

    const Eigen::VectorXd & Robot::getMotorEfforts() const
    {
        static const Eigen::VectorXd motorsEffortsEmpty;

        if (!motors_.empty())
        {
            return (*motors_.begin())->getAll();
        }

        return motorsEffortsEmpty;
    }

    double Robot::getMotorEffort(const std::string & motorName) const
    {
        static const double motorEffortEmpty = -1;

        auto motorIt = std::find_if(motors_.begin(),
                                    motors_.end(),
                                    [&motorName](const auto & elem)
                                    { return (elem->getName() == motorName); });
        if (motorIt != motors_.end())
        {
            return (*motorIt)->get();
        }

        return motorEffortEmpty;
    }

    void Robot::computeSensorMeasurements(double t,
                                          const Eigen::VectorXd & q,
                                          const Eigen::VectorXd & v,
                                          const Eigen::VectorXd & a,
                                          const Eigen::VectorXd & uMotor,
                                          const ForceVector & fExternal)
    {
        /* Note that it is assumed that the kinematic quantities have been
           updated previously to be consistent with (q, v, a, u). If not,
           one is supposed to call  `pinocchio::forwardKinematics` and
           `pinocchio::updateFramePlacements` before calling this method. */

        for (const auto & sensorGroupItem : sensors_)
        {
            if (!sensorGroupItem.second.empty())
            {
                (*sensorGroupItem.second.begin())->setAll(t, q, v, a, uMotor, fExternal);
            }
        }
    }

    SensorMeasurementTree Robot::getSensorMeasurements() const
    {
        SensorMeasurementTree data;
        auto sensorGroupIt = sensors_.cbegin();
        auto sensorsSharedIt = sensorSharedStorageMap_.cbegin();
        for (; sensorGroupIt != sensors_.cend(); ++sensorGroupIt, ++sensorsSharedIt)
        {
            auto & [sensorType, sensorGroup] = *sensorGroupIt;
            SensorMeasurementTree::mapped_type sensorsMeasurementsStack(
                &sensorsSharedIt->second->measurements_);
            for (const auto & sensor : sensorGroup)
            {
                // FIXME: manually casting to const is really necessary ?
                auto sensorConst = std::const_pointer_cast<const AbstractSensorBase>(sensor);
                sensorsMeasurementsStack.insert(
                    {sensorConst->getName(), sensorConst->getIndex(), sensorConst->get()});
            }
            data.emplace(sensorType, std::move(sensorsMeasurementsStack));
        }
        return data;
    }

    Eigen::Ref<const Eigen::VectorXd> Robot::getSensorMeasurement(
        const std::string & sensorType, const std::string & sensorName) const
    {
        static const Eigen::VectorXd sensorDataEmpty;
        static const Eigen::Ref<const Eigen::VectorXd> sensorDataRefEmpty(sensorDataEmpty);

        auto sensorGroupIt = sensors_.find(sensorType);
        if (sensorGroupIt != sensors_.end())
        {
            auto sensorIt = std::find_if(sensorGroupIt->second.begin(),
                                         sensorGroupIt->second.end(),
                                         [&sensorName](const auto & elem)
                                         { return (elem->getName() == sensorName); });
            if (sensorIt != sensorGroupIt->second.end())
            {
                return (*sensorIt)->get();
            }
        }

        return sensorDataRefEmpty;
    }

    void Robot::updateTelemetry()
    {
        for (const auto & sensorGroupItem : sensors_)
        {
            if (!sensorGroupItem.second.empty())
            {
                (*sensorGroupItem.second.begin())->updateTelemetryAll();
            }
        }
    }

    hresult_t Robot::getLock(std::unique_ptr<LockGuardLocal> & lock)
    {
        if (mutexLocal_->isLocked())
        {
            PRINT_ERROR("Robot already locked. Please release the current lock first.");
            return hresult_t::ERROR_GENERIC;
        }

        lock = std::make_unique<LockGuardLocal>(*mutexLocal_);

        return hresult_t::SUCCESS;
    }

    bool Robot::getIsLocked() const
    {
        return mutexLocal_->isLocked();
    }

    const std::vector<std::string> & Robot::getMotorNames() const
    {
        return motorNames_;
    }

    std::vector<pinocchio::JointIndex> Robot::getMotorJointIndices() const
    {
        std::vector<pinocchio::JointIndex> motorJointIndices;
        motorJointIndices.reserve(nmotors_);
        std::transform(motors_.begin(),
                       motors_.end(),
                       std::back_inserter(motorJointIndices),
                       [](const auto & motor) -> pinocchio::JointIndex
                       { return motor->getJointIndex(); });
        return motorJointIndices;
    }

    std::vector<std::vector<Eigen::Index>> Robot::getMotorsPositionIndices() const
    {
        std::vector<std::vector<Eigen::Index>> motorPositionIndices;
        motorPositionIndices.reserve(nmotors_);
        std::transform(motors_.begin(),
                       motors_.end(),
                       std::back_inserter(motorPositionIndices),
                       [](const auto & elem) -> std::vector<Eigen::Index>
                       {
                           const Eigen::Index & jointPositionIndex = elem->getJointPositionIndex();
                           if (elem->getJointType() == JointModelType::ROTARY_UNBOUNDED)
                           {
                               return {jointPositionIndex, jointPositionIndex + 1};
                           }
                           else
                           {
                               return {jointPositionIndex};
                           }
                       });
        return motorPositionIndices;
    }

    std::vector<Eigen::Index> Robot::getMotorVelocityIndices() const
    {
        std::vector<Eigen::Index> motorVelocityIndices;
        motorVelocityIndices.reserve(nmotors_);
        std::transform(motors_.begin(),
                       motors_.end(),
                       std::back_inserter(motorVelocityIndices),
                       [](const auto & elem) -> Eigen::Index
                       { return elem->getJointVelocityIndex(); });
        return motorVelocityIndices;
    }

    const Eigen::VectorXd & Robot::getCommandLimit() const
    {
        return pinocchioModel_.effortLimit;
    }

    const std::unordered_map<std::string, std::vector<std::string>> & Robot::getSensorNames() const
    {
        return sensorNames_;
    }

    const std::vector<std::string> & Robot::getSensorNames(const std::string & sensorType) const
    {
        static const std::vector<std::string> sensorNamesEmpty{};

        auto sensorsNamesIt = sensorNames_.find(sensorType);
        if (sensorsNamesIt != sensorNames_.end())
        {
            return sensorsNamesIt->second;
        }
        else
        {
            return sensorNamesEmpty;
        }
    }

    const std::vector<std::string> & Robot::getLogCommandFieldnames() const
    {
        return logCommandFieldnames_;
    }

    const std::vector<std::string> & Robot::getLogMotorEffortFieldnames() const
    {
        return logMotorEffortFieldnames_;
    }

    uint64_t Robot::nmotors() const
    {
        return nmotors_;
    }
}
