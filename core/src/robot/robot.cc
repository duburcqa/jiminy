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

    void Robot::initialize(const std::string & urdfPath,
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

    void Robot::initialize(const pinocchio::Model & pinocchioModel,
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

    void Robot::configureTelemetry(std::shared_ptr<TelemetryData> telemetryData,
                                   const std::string & prefix)
    {
        if (!isInitialized_)
        {
            THROW_ERROR(bad_control_flow, "Robot is initialized.");
        }

        telemetryData_ = telemetryData;

        isTelemetryConfigured_ = false;
        for (const auto & [sensorType, sensorGroup] : sensors_)
        {
            for (const auto & sensor : sensorGroup)
            {
                if (sensorTelemetryOptions_[sensorType])
                {
                    sensor->configureTelemetry(telemetryData_, prefix);
                }
            }
        }
        isTelemetryConfigured_ = true;
    }

    void Robot::attachMotor(std::shared_ptr<AbstractMotorBase> motor)
    {
        if (!isInitialized_)
        {
            THROW_ERROR(bad_control_flow, "Robot not initialized.");
        }

        if (getIsLocked())
        {
            THROW_ERROR(std::logic_error,
                        "Robot already locked, probably because a simulation is running. "
                        "Please stop it before adding motors.");
        }

        const std::string & motorName = motor->getName();
        auto motorIt = std::find_if(motors_.begin(),
                                    motors_.end(),
                                    [&motorName](const auto & elem)
                                    { return (elem->getName() == motorName); });
        if (motorIt != motors_.end())
        {
            THROW_ERROR(std::logic_error,
                        "Another motor with name '",
                        motorName,
                        "' is already attached.");
        }

        // Define robot notification method, responsible for updating the robot if
        // necessary after changing the motor parameters, for example the armature.
        auto notifyRobot =
            [robot_ = std::weak_ptr<Robot>(shared_from_this())](AbstractMotorBase & motorIn)
        {
            // Make sure the robot still exists
            auto robot = robot_.lock();
            if (!robot)
            {
                THROW_ERROR(std::runtime_error,
                            "Robot has been deleted. Impossible to notify motor update.");
            }

            // Update rotor inertia and effort limit of pinocchio model
            const Eigen::Index jointVelocityOrigIndex =
                getJointVelocityFirstIndex(robot->pinocchioModelOrig_, motorIn.getJointName());
            robot->pinocchioModel_.rotorInertia[motorIn.getJointVelocityIndex()] =
                robot->pinocchioModelOrig_.rotorInertia[jointVelocityOrigIndex] +
                motorIn.getArmature();
            robot->pinocchioModel_.effortLimit[motorIn.getJointVelocityIndex()] =
                motorIn.getCommandLimit();
        };

        // Attach the motor
        motor->attach(shared_from_this(), notifyRobot, motorSharedStorage_.get());

        // Add the motor to the holder
        motors_.push_back(motor);

        // Refresh the motors proxies
        refreshMotorProxies();
    }

    void Robot::detachMotor(const std::string & motorName)
    {
        if (!isInitialized_)
        {
            THROW_ERROR(bad_control_flow, "Robot not initialized.");
        }

        if (getIsLocked())
        {
            THROW_ERROR(std::logic_error,
                        "Robot already locked, probably because a simulation is running. "
                        "Please stop it before removing motors.");
        }

        auto motorIt = std::find_if(motors_.cbegin(),
                                    motors_.cend(),
                                    [&motorName](const auto & elem)
                                    { return (elem->getName() == motorName); });
        if (motorIt == motors_.cend())
        {
            THROW_ERROR(std::logic_error, "No motor with name '", motorName, "' is attached.");
        }

        // Reset effortLimit and rotorInertia
        const std::shared_ptr<AbstractMotorBase> & motor = *motorIt;
        const Eigen::Index jointVelocityOrigIndex =
            ::jiminy::getJointVelocityFirstIndex(pinocchioModelOrig_, motor->getJointName());
        pinocchioModel_.rotorInertia[motor->getJointVelocityIndex()] =
            pinocchioModelOrig_.rotorInertia[jointVelocityOrigIndex];
        pinocchioModel_.effortLimit[motor->getJointVelocityIndex()] = 0.0;

        // Detach the motor
        motor->detach();

        // Remove the motor from the holder
        motors_.erase(motorIt);

        // Refresh the motors proxies
        refreshMotorProxies();
    }

    void Robot::detachMotors(std::vector<std::string> motorsNames)
    {
        if (motorsNames.empty())
        {
            // Remove all sensors if none is specified
            if (!motorNames_.empty())
            {
                detachMotors(motorNames_);
            }
        }
        else
        {
            // Make sure that no motor names are duplicates
            if (checkDuplicates(motorsNames))
            {
                THROW_ERROR(std::invalid_argument, "Duplicated motor names found.");
            }

            // Make sure that every motor name exist
            if (!checkInclusion(motorNames_, motorsNames))
            {
                THROW_ERROR(std::invalid_argument,
                            "At least one of the motor names does not exist.");
            }

            // Detach motors one-by-one
            for (const std::string & name : motorsNames)
            {
                detachMotor(name);
            }
        }
    }

    void Robot::attachSensor(std::shared_ptr<AbstractSensorBase> sensor)
    {
        // The sensors' names must be unique, even if their type is different.

        if (!isInitialized_)
        {
            THROW_ERROR(bad_control_flow, "The robot is not initialized.");
        }

        if (getIsLocked())
        {
            THROW_ERROR(std::logic_error,
                        "Robot already locked, probably because a simulation is running. "
                        "Please stop it before removing motors.");
        }

        const std::string & sensorName = sensor->getName();
        const std::string & sensorType = sensor->getType();
        SensorTree::const_iterator sensorGroupIt;
        sensorGroupIt = sensors_.find(sensorType);
        if (sensorGroupIt != sensors_.end())
        {
            auto sensorIt = std::find_if(sensorGroupIt->second.begin(),
                                         sensorGroupIt->second.end(),
                                         [&sensorName](const auto & elem)
                                         { return (elem->getName() == sensorName); });
            if (sensorIt != sensorGroupIt->second.end())
            {
                THROW_ERROR(std::invalid_argument,
                            "A sensor with the same type and name already exists.");
            }
        }

        // Create a new sensor data holder if necessary
        if (sensorGroupIt == sensors_.end())
        {
            sensorSharedStorageMap_.emplace(sensorType, std::make_shared<SensorSharedStorage>());
            sensorTelemetryOptions_.emplace(sensorType,
                                            true);  // Enable the telemetry by default
        }

        // Attach the sensor
        sensor->attach(shared_from_this(), sensorSharedStorageMap_[sensorType].get());

        // Create the sensor and add it to its group
        sensors_[sensorType].push_back(sensor);

        // Refresh the sensors proxies
        refreshSensorProxies();
    }

    void Robot::detachSensor(const std::string & sensorType, const std::string & sensorName)
    {
        if (getIsLocked())
        {
            THROW_ERROR(std::logic_error,
                        "Robot already locked, probably because a simulation is running. "
                        "Please stop it before removing motors.");
        }

        if (!isInitialized_)
        {
            THROW_ERROR(bad_control_flow, "Robot not initialized.");
        }

        // FIXME: remove explicit conversion to `std::string` when moving to C++20
        auto sensorGroupIt = sensors_.find(std::string{sensorType});
        if (sensorGroupIt == sensors_.end())
        {
            THROW_ERROR(std::invalid_argument,
                        "None of the attached sensors has type '",
                        sensorType,
                        "'.");
        }

        SensorVector::iterator sensorIt;
        sensorIt = std::find_if(sensorGroupIt->second.begin(),
                                sensorGroupIt->second.end(),
                                [&sensorName](const auto & elem)
                                { return (elem->getName() == sensorName); });
        if (sensorIt == sensorGroupIt->second.end())
        {
            THROW_ERROR(std::invalid_argument,
                        "None of the attached sensors of type '",
                        sensorType,
                        "' has name '",
                        sensorName,
                        "'.");
        }

        // Detach the sensor
        (*sensorIt)->detach();

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
    }

    void Robot::detachSensors(const std::string & sensorType)
    {
        if (!sensorType.empty())
        {
            auto sensorGroupIt = sensors_.find(sensorType);
            if (sensorGroupIt == sensors_.end())
            {
                THROW_ERROR(std::invalid_argument,
                            "None of the attached sensors has type '",
                            sensorType,
                            "'.");
            }

            std::vector<std::string> sensorGroupNames =
                sensorNames_[sensorType];  // Make a copy since calling detachSensors update it !
            for (const std::string & sensorName : sensorGroupNames)
            {
                detachSensor(sensorType, sensorName);
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
                detachSensors(sensorTypeName);
            }
        }
    }

    void Robot::refreshProxies()
    {
        if (!isInitialized_)
        {
            THROW_ERROR(bad_control_flow, "Robot not initialized.");
        }

        Model::refreshProxies();
        refreshMotorProxies();
        refreshSensorProxies();
    }

    void Robot::refreshMotorProxies()
    {
        if (!isInitialized_)
        {
            THROW_ERROR(bad_control_flow, "Robot not initialized.");
        }

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

    void Robot::refreshSensorProxies()
    {
        if (!isInitialized_)
        {
            THROW_ERROR(bad_control_flow, "Robot not initialized.");
        }

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

    std::shared_ptr<AbstractMotorBase> Robot::getMotor(const std::string & motorName)
    {
        if (!isInitialized_)
        {
            THROW_ERROR(bad_control_flow, "Robot not initialized.");
        }

        auto motorIt = std::find_if(motors_.begin(),
                                    motors_.end(),
                                    [&motorName](const auto & elem)
                                    { return (elem->getName() == motorName); });
        if (motorIt == motors_.end())
        {
            THROW_ERROR(
                std::invalid_argument, "None of the attached motors has name '", motorName, "'.");
        }
        return *motorIt;
    }

    std::weak_ptr<const AbstractMotorBase> Robot::getMotor(const std::string & motorName) const
    {
        if (!isInitialized_)
        {
            THROW_ERROR(bad_control_flow, "Robot not initialized.");
        }

        auto motorIt = std::find_if(motors_.begin(),
                                    motors_.end(),
                                    [&motorName](const auto & elem)
                                    { return (elem->getName() == motorName); });
        if (motorIt == motors_.end())
        {
            THROW_ERROR(
                std::invalid_argument, "None of the attached motors has name '", motorName, "'.");
        }
        return std::const_pointer_cast<const AbstractMotorBase>(*motorIt);
    }

    const Robot::MotorVector & Robot::getMotors() const
    {
        return motors_;
    }

    std::shared_ptr<AbstractSensorBase> Robot::getSensor(const std::string & sensorType,
                                                         const std::string & sensorName)
    {
        if (!isInitialized_)
        {
            THROW_ERROR(bad_control_flow, "Robot not initialized.");
        }

        auto sensorGroupIt = sensors_.find(sensorType);
        if (sensorGroupIt == sensors_.end())
        {
            THROW_ERROR(std::invalid_argument,
                        "None of the attached sensors has type '",
                        sensorType,
                        "'.");
        }

        auto sensorIt = std::find_if(sensorGroupIt->second.begin(),
                                     sensorGroupIt->second.end(),
                                     [&sensorName](const auto & elem)
                                     { return (elem->getName() == sensorName); });
        if (sensorIt == sensorGroupIt->second.end())
        {
            THROW_ERROR(std::invalid_argument,
                        "None of the attached sensors of type '",
                        sensorType,
                        "' has name '",
                        sensorName,
                        "'.");
        }

        return *sensorIt;
    }

    std::weak_ptr<const AbstractSensorBase> Robot::getSensor(const std::string & sensorType,
                                                             const std::string & sensorName) const
    {
        if (!isInitialized_)
        {
            THROW_ERROR(bad_control_flow, "Robot not initialized.");
        }

        auto sensorGroupIt = sensors_.find(sensorType);
        if (sensorGroupIt == sensors_.end())
        {
            THROW_ERROR(std::invalid_argument,
                        "None of the attached sensors has type '",
                        sensorType,
                        "'.");
        }

        auto sensorIt = std::find_if(sensorGroupIt->second.begin(),
                                     sensorGroupIt->second.end(),
                                     [&sensorName](const auto & elem)
                                     { return (elem->getName() == sensorName); });
        if (sensorIt == sensorGroupIt->second.end())
        {
            THROW_ERROR(std::invalid_argument,
                        "None of the attached sensors of type '",
                        sensorType,
                        "' has name '",
                        sensorName,
                        "'.");
        }

        return std::const_pointer_cast<const AbstractSensorBase>(*sensorIt);
    }

    const Robot::SensorTree & Robot::getSensors() const
    {
        return sensors_;
    }

    void Robot::setOptions(const GenericConfig & robotOptions)
    {
        GenericConfig::const_iterator modelOptionsIt;
        modelOptionsIt = robotOptions.find("model");
        if (modelOptionsIt == robotOptions.end())
        {
            THROW_ERROR(std::invalid_argument, "'model' options are missing.");
        }

        const GenericConfig & modelOptions = boost::get<GenericConfig>(modelOptionsIt->second);
        setModelOptions(modelOptions);

        GenericConfig::const_iterator motorsOptionsIt;
        motorsOptionsIt = robotOptions.find("motors");
        if (motorsOptionsIt == robotOptions.end())
        {
            THROW_ERROR(std::invalid_argument, "'motors' options are missing.");
        }

        const GenericConfig & motorsOptions = boost::get<GenericConfig>(motorsOptionsIt->second);
        setMotorsOptions(motorsOptions);

        GenericConfig::const_iterator sensorOptionsIt = robotOptions.find("sensors");
        if (sensorOptionsIt == robotOptions.end())
        {
            THROW_ERROR(std::invalid_argument, "'sensors' options are missing.");
        }

        const GenericConfig & sensorOptions = boost::get<GenericConfig>(sensorOptionsIt->second);
        setSensorsOptions(sensorOptions);

        GenericConfig::const_iterator telemetryOptionsIt = robotOptions.find("telemetry");
        if (telemetryOptionsIt == robotOptions.end())
        {
            THROW_ERROR(std::invalid_argument, "'telemetry' options are missing.");
        }

        const GenericConfig & telemetryOptions =
            boost::get<GenericConfig>(telemetryOptionsIt->second);
        setTelemetryOptions(telemetryOptions);
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

    void Robot::setMotorOptions(const std::string & motorName, const GenericConfig & motorOptions)
    {
        if (getIsLocked())
        {
            THROW_ERROR(std::logic_error,
                        "Robot already locked, probably because a simulation is running. "
                        "Please stop it before removing motors.");
        }

        MotorVector::iterator motorIt;
        motorIt = std::find_if(motors_.begin(),
                               motors_.end(),
                               [&motorName](const auto & elem)
                               { return (elem->getName() == motorName); });
        if (motorIt == motors_.end())
        {
            THROW_ERROR(
                std::invalid_argument, "None of the attached motors has name '", motorName, "'.");
        }

        (*motorIt)->setOptions(motorOptions);
    }

    void Robot::setMotorsOptions(const GenericConfig & motorsOptions)
    {
        if (getIsLocked())
        {
            THROW_ERROR(std::logic_error,
                        "Robot already locked, probably because a simulation is running. "
                        "Please stop it before removing motors.");
        }

        for (const auto & motor : motors_)
        {
            auto motorOptionIt = motorsOptions.find(motor->getName());
            if (motorOptionIt != motorsOptions.end())
            {
                motor->setOptions(boost::get<GenericConfig>(motorOptionIt->second));
            }
            else
            {
                motor->setOptionsAll(motorsOptions);
                break;
            }
        }
    }

    GenericConfig Robot::getMotorOptions(const std::string & motorName) const
    {
        auto motorIt = std::find_if(motors_.begin(),
                                    motors_.end(),
                                    [&motorName](const auto & elem)
                                    { return (elem->getName() == motorName); });
        if (motorIt == motors_.end())
        {
            THROW_ERROR(
                std::invalid_argument, "None of the attached motors has name '", motorName, "'.");
        }
        return (*motorIt)->getOptions();
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

    void Robot::setSensorOptions(const std::string & sensorType,
                                 const std::string & sensorName,
                                 const GenericConfig & sensorOptions)
    {
        if (getIsLocked())
        {
            THROW_ERROR(std::logic_error,
                        "Robot already locked, probably because a simulation is running. "
                        "Please stop it before removing motors.");
        }

        auto sensorGroupIt = sensors_.find(sensorType);
        if (sensorGroupIt == sensors_.end())
        {
            THROW_ERROR(std::invalid_argument,
                        "None of the attached sensors has type '",
                        sensorType,
                        "'.");
        }

        auto sensorIt = std::find_if(sensorGroupIt->second.begin(),
                                     sensorGroupIt->second.end(),
                                     [&sensorName](const auto & elem)
                                     { return (elem->getName() == sensorName); });
        if (sensorIt == sensorGroupIt->second.end())
        {
            THROW_ERROR(std::invalid_argument,
                        "None of the attached sensors of type '",
                        sensorType,
                        "' has name '",
                        sensorName,
                        "'.");
        }

        (*sensorIt)->setOptions(sensorOptions);
    }

    void Robot::setSensorsOptions(const std::string & sensorType,
                                  const GenericConfig & sensorsOptions)
    {
        if (getIsLocked())
        {
            THROW_ERROR(std::logic_error,
                        "Robot already locked, probably because a simulation is running. "
                        "Please stop it before removing motors.");
        }

        SensorTree::iterator sensorGroupIt;
        sensorGroupIt = sensors_.find(sensorType);
        if (sensorGroupIt == sensors_.end())
        {
            THROW_ERROR(std::invalid_argument,
                        "None of the attached sensors has type '",
                        sensorType,
                        "'.");
        }

        for (const auto & sensor : sensorGroupIt->second)
        {
            auto sensorOptionIt = sensorsOptions.find(sensor->getName());
            if (sensorOptionIt != sensorsOptions.end())
            {
                sensor->setOptions(boost::get<GenericConfig>(sensorOptionIt->second));
            }
            else
            {
                sensor->setOptionsAll(sensorsOptions);
                break;
            }
        }
    }

    void Robot::setSensorsOptions(const GenericConfig & sensorsOptions)
    {
        if (getIsLocked())
        {
            THROW_ERROR(std::logic_error,
                        "Robot already locked, probably because a simulation is running. "
                        "Please stop it before removing motors.");
        }

        for (const auto & [sensorType, sensorGroup] : sensors_)
        {
            // FIXME: remove explicit conversion to `std::string` when moving to C++20
            auto sensorGroupOptionsIt = sensorsOptions.find(std::string{sensorType});
            if (sensorGroupOptionsIt != sensorsOptions.end())
            {
                const GenericConfig & sensorGroupOptions =
                    boost::get<GenericConfig>(sensorGroupOptionsIt->second);

                for (const auto & sensor : sensorGroup)
                {
                    const std::string & sensorName = sensor->getName();

                    auto sensorOptionsIt = sensorGroupOptions.find(sensorName);
                    if (sensorOptionsIt != sensorGroupOptions.end())
                    {
                        sensor->setOptions(boost::get<GenericConfig>(sensorOptionsIt->second));
                    }
                    else
                    {
                        THROW_ERROR(std::invalid_argument,
                                    "None of the attached sensors of type '",
                                    sensorType,
                                    "' has name '",
                                    sensorName,
                                    "'.");
                    }
                }
            }
            else
            {
                THROW_ERROR(std::invalid_argument,
                            "None of the attached sensors has type '",
                            sensorType,
                            "'.");
            }
        }
    }

    GenericConfig Robot::getSensorOptions(const std::string & sensorType,
                                          const std::string & sensorName) const
    {
        auto sensorGroupIt = sensors_.find(sensorType);
        if (sensorGroupIt == sensors_.end())
        {
            THROW_ERROR(std::invalid_argument,
                        "None of the attached sensors has type '",
                        sensorType,
                        "'.");
        }

        auto sensorIt = std::find_if(sensorGroupIt->second.begin(),
                                     sensorGroupIt->second.end(),
                                     [&sensorName](const auto & elem)
                                     { return (elem->getName() == sensorName); });
        if (sensorIt == sensorGroupIt->second.end())
        {
            THROW_ERROR(std::invalid_argument,
                        "None of the attached sensors of type '",
                        sensorType,
                        "' has name '",
                        sensorName,
                        "'.");
        }

        return (*sensorIt)->getOptions();
    }

    GenericConfig Robot::getSensorsOptions(const std::string & sensorType) const
    {
        auto sensorGroupIt = sensors_.find(sensorType);
        if (sensorGroupIt == sensors_.end())
        {
            THROW_ERROR(std::invalid_argument,
                        "None of the attached sensors has type '",
                        sensorType,
                        "'.");
        }

        GenericConfig sensorsOptions{};
        for (const auto & sensor : sensorGroupIt->second)
        {
            sensorsOptions[sensor->getName()] = sensor->getOptions();
        }
        return sensorsOptions;
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

    void Robot::setModelOptions(const GenericConfig & modelOptions)
    {
        return Model::setOptions(modelOptions);
    }

    GenericConfig Robot::getModelOptions() const
    {
        return Model::getOptions();
    }

    void Robot::setTelemetryOptions(const GenericConfig & telemetryOptions)
    {
        if (getIsLocked())
        {
            THROW_ERROR(std::logic_error,
                        "Robot already locked, probably because a simulation is running. "
                        "Please stop it before removing motors.");
        }

        for (auto & [sensorType, sensorGroupTelemetryOption] : sensorTelemetryOptions_)
        {
            const std::string optionTelemetryName = toString("enable", sensorType, "s");
            auto sensorTelemetryOptionIt = telemetryOptions.find(optionTelemetryName);
            if (sensorTelemetryOptionIt == telemetryOptions.end())
            {
                THROW_ERROR(std::invalid_argument, "Missing field '", optionTelemetryName, "'.");
            }
            sensorGroupTelemetryOption = boost::get<bool>(sensorTelemetryOptionIt->second);
        }
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

    void Robot::dumpOptions(const std::string & filepath) const
    {
        std::shared_ptr<AbstractIODevice> device = std::make_shared<FileDevice>(filepath);
        return jsonDump(getOptions(), device);
    }

    void Robot::loadOptions(const std::string & filepath)
    {
        std::shared_ptr<AbstractIODevice> device = std::make_shared<FileDevice>(filepath);
        GenericConfig robotOptions;
        jsonLoad(robotOptions, device);

        setOptions(robotOptions);
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

    std::unique_ptr<LockGuardLocal> Robot::getLock()
    {
        if (mutexLocal_->isLocked())
        {
            THROW_ERROR(std::logic_error,
                        "Robot already locked. Please release it first prior requesting lock.");
        }

        return std::make_unique<LockGuardLocal>(*mutexLocal_);
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

    Eigen::Index Robot::nmotors() const
    {
        return nmotors_;
    }
}
