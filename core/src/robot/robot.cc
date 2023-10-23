#include <fstream>
#include <exception>

#include "jiminy/core/hardware/abstract_motor.h"
#include "jiminy/core/hardware/abstract_sensor.h"
#include "jiminy/core/telemetry/telemetry_data.h"
#include "jiminy/core/io/file_device.h"
#include "jiminy/core/utilities/helpers.h"
#include "jiminy/core/utilities/pinocchio.h"
#include "jiminy/core/utilities/json.h"

#include "jiminy/core/robot/robot.h"


namespace jiminy
{
    Robot::Robot() :
    Model(),
    isTelemetryConfigured_(false),
    telemetryData_(nullptr),
    motorsHolder_(),
    sensorsGroupHolder_(),
    sensorTelemetryOptions_(),
    motorsNames_(),
    sensorsNames_(),
    logFieldnamesCommand_(),
    logFieldnamesMotorEffort_(),
    nmotors_(0U),
    mutexLocal_(std::make_unique<MutexLocal>()),
    motorsSharedHolder_(std::make_shared<MotorSharedDataHolder_t>()),
    sensorsSharedHolder_()
    {
        // Empty on purpose
    }

    Robot::~Robot()
    {
        // Detach all the motors and sensors
        detachSensors({});
        detachMotors({});
    }

    hresult_t Robot::initialize(const std::string & urdfPath,
                                const bool_t & hasFreeflyer,
                                const std::vector<std::string> & meshPackageDirs,
                                const bool_t & loadVisualMeshes)
    {
        // Detach all the motors and sensors
        detachSensors({});
        detachMotors({});

        /* Delete the current model and generate a new one.
           Note that is also refresh all proxies automatically. */
        return Model::initialize(urdfPath, hasFreeflyer, meshPackageDirs, loadVisualMeshes);
    }

    hresult_t Robot::initialize(const pinocchio::Model & pncModel,
                                const pinocchio::GeometryModel & collisionModel,
                                const pinocchio::GeometryModel & visualModel)
    {
        // Detach all the motors and sensors
        detachSensors({});
        detachMotors({});

        /* Delete the current model and generate a new one.
           Note that is also refresh all proxies automatically. */
        return Model::initialize(pncModel, collisionModel, visualModel);
    }

    void Robot::reset()
    {
        // Reset the model
        Model::reset();

        // Reset the motors
        if (!motorsHolder_.empty())
        {
            (*motorsHolder_.begin())->resetAll();
        }

        // Reset the sensors
        for (auto & sensorGroup : sensorsGroupHolder_)
        {
            if (!sensorGroup.second.empty())
            {
                (*sensorGroup.second.begin())->resetAll();
            }
        }

        // Reset the telemetry flag
        isTelemetryConfigured_ = false;
    }

    hresult_t Robot::configureTelemetry(std::shared_ptr<TelemetryData> telemetryData,
                                        const std::string & objectPrefixName)
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
                for (const auto & sensorGroup : sensorsGroupHolder_)
                {
                    for (const auto & sensor : sensorGroup.second)
                    {
                        if (returnCode == hresult_t::SUCCESS)
                        {
                            if (sensorTelemetryOptions_[sensorGroup.first])
                            {
                                returnCode =
                                    sensor->configureTelemetry(telemetryData_, objectPrefixName);
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
            auto motorIt = std::find_if(motorsHolder_.begin(),
                                        motorsHolder_.end(),
                                        [&motorName](const auto & elem)
                                        { return (elem->getName() == motorName); });
            if (motorIt != motorsHolder_.end())
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
                int32_t jointVelocityOrigIdx;
                ::jiminy::getJointVelocityIdx(
                    robot->pncModelOrig_, motorIn.getJointName(), jointVelocityOrigIdx);
                robot->pncModel_.rotorInertia[motorIn.getJointVelocityIdx()] =
                    robot->pncModelOrig_.rotorInertia[jointVelocityOrigIdx] +
                    motorIn.getArmature();
                robot->pncModel_.effortLimit[motorIn.getJointVelocityIdx()] =
                    motorIn.getCommandLimit();

                return hresult_t::SUCCESS;
            };

            // Attach the motor
            returnCode = motor->attach(shared_from_this(), notifyRobot, motorsSharedHolder_.get());
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            // Add the motor to the holder
            motorsHolder_.push_back(motor);

            // Refresh the motors proxies
            refreshMotorsProxies();
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

        auto motorIt = std::find_if(motorsHolder_.begin(),
                                    motorsHolder_.end(),
                                    [&motorName](const auto & elem)
                                    { return (elem->getName() == motorName); });
        if (motorIt == motorsHolder_.end())
        {
            PRINT_ERROR("No motor with this name exists.");
            return hresult_t::ERROR_BAD_INPUT;
        }

        // Reset effortLimit and rotorInertia
        const std::shared_ptr<AbstractMotorBase> & motor = *motorIt;
        int32_t jointVelocityOrigIdx;
        ::jiminy::getJointVelocityIdx(pncModelOrig_, motor->getJointName(), jointVelocityOrigIdx);
        pncModel_.rotorInertia[motor->getJointVelocityIdx()] =
            pncModelOrig_.rotorInertia[jointVelocityOrigIdx];
        pncModel_.effortLimit[motor->getJointVelocityIdx()] = 0.0;

        // Detach the motor
        motor->detach();  // Cannot fail at this point

        // Remove the motor from the holder
        motorsHolder_.erase(motorIt);

        // Refresh the motors proxies
        refreshMotorsProxies();

        return hresult_t::SUCCESS;
    }

    hresult_t Robot::detachMotors(const std::vector<std::string> & motorsNames)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (!motorsNames.empty())
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
                if (!checkInclusion(motorsNames_, motorsNames))
                {
                    PRINT_ERROR("At least one of the motor names does not exist.");
                    returnCode = hresult_t::ERROR_BAD_INPUT;
                }
            }

            for (const std::string & name : motorsNames)
            {
                if (returnCode == hresult_t::SUCCESS)
                {
                    returnCode = detachMotor(name);
                }
            }
        }
        else
        {
            if (returnCode == hresult_t::SUCCESS)
            {
                if (!motorsNames_.empty())
                {
                    returnCode = detachMotors(std::vector<std::string>(motorsNames_));
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
        sensorsGroupHolder_t::const_iterator sensorGroupIt;
        if (returnCode == hresult_t::SUCCESS)
        {
            sensorGroupIt = sensorsGroupHolder_.find(sensorType);
            if (sensorGroupIt != sensorsGroupHolder_.end())
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
            if (sensorGroupIt == sensorsGroupHolder_.end())
            {
                sensorsSharedHolder_.emplace(
                    std::make_pair(sensorType, std::make_shared<SensorSharedDataHolder_t>()));
                sensorTelemetryOptions_.emplace(std::make_pair(sensorType, true));  // Enable the
                                                                                    // telemetry by
                                                                                    // default
            }

            // Attach the sensor
            returnCode =
                sensor->attach(shared_from_this(), sensorsSharedHolder_[sensorType].get());
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            // Create the sensor and add it to its group
            sensorsGroupHolder_[sensorType].push_back(sensor);

            // Refresh the sensors proxies
            refreshSensorsProxies();
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

        auto sensorGroupIt = sensorsGroupHolder_.find(sensorType);
        if (sensorGroupIt == sensorsGroupHolder_.end())
        {
            PRINT_ERROR("This type of sensor does not exist.");
            return hresult_t::ERROR_BAD_INPUT;
        }

        sensorsHolder_t::iterator sensorIt;
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
            sensorsGroupHolder_.erase(sensorType);
            sensorsSharedHolder_.erase(sensorType);
            sensorTelemetryOptions_.erase(sensorType);
        }

        // Refresh the sensors proxies
        refreshSensorsProxies();

        return hresult_t::SUCCESS;
    }

    hresult_t Robot::detachSensors(const std::string & sensorType)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (!sensorType.empty())
        {
            auto sensorGroupIt = sensorsGroupHolder_.find(sensorType);
            if (sensorGroupIt == sensorsGroupHolder_.end())
            {
                PRINT_ERROR("No sensor with this type exists.");
                returnCode = hresult_t::ERROR_BAD_INPUT;
            }

            std::vector<std::string> sensorGroupNames =
                sensorsNames_[sensorType];  // Make a copy since calling detachSensors update it !
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
            sensorsTypesNames.reserve(sensorsGroupHolder_.size());
            std::transform(sensorsGroupHolder_.begin(),
                           sensorsGroupHolder_.end(),
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
            returnCode = refreshMotorsProxies();
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            returnCode = refreshSensorsProxies();
        }

        return returnCode;
    }

    hresult_t Robot::refreshMotorsProxies()
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
            nmotors_ = motorsHolder_.size();

            // Extract the motor names
            motorsNames_.clear();
            motorsNames_.reserve(nmotors_);
            std::transform(motorsHolder_.begin(),
                           motorsHolder_.end(),
                           std::back_inserter(motorsNames_),
                           [](const auto & elem) -> std::string { return elem->getName(); });

            // Generate the fieldnames associated with command
            logFieldnamesCommand_.clear();
            logFieldnamesCommand_.reserve(nmotors_);
            std::transform(motorsHolder_.begin(),
                           motorsHolder_.end(),
                           std::back_inserter(logFieldnamesCommand_),
                           [](const auto & elem) -> std::string {
                               return addCircumfix(elem->getName(), JOINT_PREFIX_BASE + "Command");
                           });

            // Generate the fieldnames associated with motor efforts
            logFieldnamesMotorEffort_.clear();
            logFieldnamesMotorEffort_.reserve(nmotors_);
            std::transform(motorsHolder_.begin(),
                           motorsHolder_.end(),
                           std::back_inserter(logFieldnamesMotorEffort_),
                           [](const auto & elem) -> std::string {
                               return addCircumfix(elem->getName(), JOINT_PREFIX_BASE + "Effort");
                           });
        }

        return returnCode;
    }

    hresult_t Robot::refreshSensorsProxies()
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
            sensorsNames_.clear();
            sensorsNames_.reserve(sensorsGroupHolder_.size());
            for (const auto & sensorGroup : sensorsGroupHolder_)
            {
                std::vector<std::string> sensorGroupNames;
                sensorGroupNames.reserve(sensorGroup.second.size());
                std::transform(sensorGroup.second.begin(),
                               sensorGroup.second.end(),
                               std::back_inserter(sensorGroupNames),
                               [](const auto & elem) -> std::string { return elem->getName(); });
                sensorsNames_.insert({sensorGroup.first, std::move(sensorGroupNames)});
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

        auto motorIt = std::find_if(motorsHolder_.begin(),
                                    motorsHolder_.end(),
                                    [&motorName](const auto & elem)
                                    { return (elem->getName() == motorName); });
        if (motorIt == motorsHolder_.end())
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

        auto motorIt = std::find_if(motorsHolder_.begin(),
                                    motorsHolder_.end(),
                                    [&motorName](const auto & elem)
                                    { return (elem->getName() == motorName); });
        if (motorIt == motorsHolder_.end())
        {
            PRINT_ERROR("No motor with this name exists.");
            return hresult_t::ERROR_BAD_INPUT;
        }

        motor = std::const_pointer_cast<const AbstractMotorBase>(*motorIt);

        return hresult_t::SUCCESS;
    }

    const Robot::motorsHolder_t & Robot::getMotors() const
    {
        return motorsHolder_;
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

        auto sensorGroupIt = sensorsGroupHolder_.find(sensorType);
        if (sensorGroupIt == sensorsGroupHolder_.end())
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

        auto sensorGroupIt = sensorsGroupHolder_.find(sensorType);
        if (sensorGroupIt == sensorsGroupHolder_.end())
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

    const Robot::sensorsGroupHolder_t & Robot::getSensors() const
    {
        return sensorsGroupHolder_;
    }

    hresult_t Robot::setOptions(const configHolder_t & robotOptions)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        configHolder_t::const_iterator modelOptionsIt;
        modelOptionsIt = robotOptions.find("model");
        if (modelOptionsIt == robotOptions.end())
        {
            PRINT_ERROR("'model' options are missing.");
            returnCode = hresult_t::ERROR_INIT_FAILED;
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            const configHolder_t & modelOptions =
                boost::get<configHolder_t>(modelOptionsIt->second);
            returnCode = setModelOptions(modelOptions);
        }

        configHolder_t::const_iterator motorsOptionsIt;
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
            const configHolder_t & motorsOptions =
                boost::get<configHolder_t>(motorsOptionsIt->second);
            returnCode = setMotorsOptions(motorsOptions);
        }

        configHolder_t::const_iterator sensorsOptionsIt;
        if (returnCode == hresult_t::SUCCESS)
        {
            sensorsOptionsIt = robotOptions.find("sensors");
            if (sensorsOptionsIt == robotOptions.end())
            {
                PRINT_ERROR("'sensors' options are missing.");
                returnCode = hresult_t::ERROR_INIT_FAILED;
            }
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            const configHolder_t & sensorsOptions =
                boost::get<configHolder_t>(sensorsOptionsIt->second);
            returnCode = setSensorsOptions(sensorsOptions);
        }

        configHolder_t::const_iterator telemetryOptionsIt;
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
            const configHolder_t & telemetryOptions =
                boost::get<configHolder_t>(telemetryOptionsIt->second);
            returnCode = setTelemetryOptions(telemetryOptions);
        }

        return returnCode;
    }

    configHolder_t Robot::getOptions() const
    {
        configHolder_t robotOptions;
        robotOptions["model"] = getModelOptions();
        configHolder_t motorsOptions;
        robotOptions["motors"] = getMotorsOptions();
        configHolder_t sensorsOptions;
        robotOptions["sensors"] = getSensorsOptions();
        configHolder_t telemetryOptions;
        robotOptions["telemetry"] = getTelemetryOptions();
        return robotOptions;
    }

    hresult_t Robot::setMotorOptions(const std::string & motorName,
                                     const configHolder_t & motorOptions)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (getIsLocked())
        {
            PRINT_ERROR("Robot is locked, probably because a simulation is running. Please stop "
                        "it before updating the motor options.");
            returnCode = hresult_t::ERROR_GENERIC;
        }

        motorsHolder_t::iterator motorIt;
        if (returnCode == hresult_t::SUCCESS)
        {
            motorIt = std::find_if(motorsHolder_.begin(),
                                   motorsHolder_.end(),
                                   [&motorName](const auto & elem)
                                   { return (elem->getName() == motorName); });
            if (motorIt == motorsHolder_.end())
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

    hresult_t Robot::setMotorsOptions(const configHolder_t & motorsOptions)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (getIsLocked())
        {
            PRINT_ERROR("Robot is locked, probably because a simulation is running. Please stop "
                        "it before updating the motor options.");
            returnCode = hresult_t::ERROR_GENERIC;
        }

        for (const auto & motor : motorsHolder_)
        {
            if (returnCode == hresult_t::SUCCESS)
            {
                auto motorOptionIt = motorsOptions.find(motor->getName());
                if (motorOptionIt != motorsOptions.end())
                {
                    returnCode =
                        motor->setOptions(boost::get<configHolder_t>(motorOptionIt->second));
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
                                     configHolder_t & motorOptions) const
    {
        auto motorIt = std::find_if(motorsHolder_.begin(),
                                    motorsHolder_.end(),
                                    [&motorName](const auto & elem)
                                    { return (elem->getName() == motorName); });
        if (motorIt == motorsHolder_.end())
        {
            PRINT_ERROR("No motor with this name exists.");
            return hresult_t::ERROR_BAD_INPUT;
        }

        motorOptions = (*motorIt)->getOptions();

        return hresult_t::SUCCESS;
    }

    configHolder_t Robot::getMotorsOptions() const
    {
        configHolder_t motorsOptions;
        for (const motorsHolder_t::value_type & motor : motorsHolder_)
        {
            motorsOptions[motor->getName()] = motor->getOptions();
        }
        return motorsOptions;
    }

    hresult_t Robot::setSensorOptions(const std::string & sensorType,
                                      const std::string & sensorName,
                                      const configHolder_t & sensorOptions)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (getIsLocked())
        {
            PRINT_ERROR("Robot is locked, probably because a simulation is running. Please stop "
                        "it before updating the sensor options.");
            returnCode = hresult_t::ERROR_GENERIC;
        }

        auto sensorGroupIt = sensorsGroupHolder_.find(sensorType);
        if (returnCode == hresult_t::SUCCESS)
        {
            if (sensorGroupIt == sensorsGroupHolder_.end())
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
                                       const configHolder_t & sensorsOptions)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (getIsLocked())
        {
            PRINT_ERROR("Robot is locked, probably because a simulation is running. Please stop "
                        "it before updating the sensor options.");
            returnCode = hresult_t::ERROR_GENERIC;
        }

        sensorsGroupHolder_t::iterator sensorGroupIt;
        if (returnCode == hresult_t::SUCCESS)
        {
            sensorGroupIt = sensorsGroupHolder_.find(sensorType);
            if (sensorGroupIt == sensorsGroupHolder_.end())
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
                        sensor->setOptions(boost::get<configHolder_t>(sensorOptionIt->second));
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

    hresult_t Robot::setSensorsOptions(const configHolder_t & sensorsOptions)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (getIsLocked())
        {
            PRINT_ERROR("Robot is locked, probably because a simulation is running. Please stop "
                        "it before updating the sensor options.");
            returnCode = hresult_t::ERROR_GENERIC;
        }

        for (const auto & sensorGroup : sensorsGroupHolder_)
        {
            if (returnCode == hresult_t::SUCCESS)
            {
                const std::string & sensorType = sensorGroup.first;

                auto sensorGroupOptionsIt = sensorsOptions.find(sensorType);
                if (sensorGroupOptionsIt != sensorsOptions.end())
                {
                    const configHolder_t & sensorGroupOptions =
                        boost::get<configHolder_t>(sensorGroupOptionsIt->second);

                    for (const auto & sensor : sensorGroup.second)
                    {
                        if (returnCode == hresult_t::SUCCESS)
                        {
                            const std::string & sensorName = sensor->getName();

                            auto sensorOptionsIt = sensorGroupOptions.find(sensorName);
                            if (sensorOptionsIt != sensorGroupOptions.end())
                            {
                                const configHolder_t & sensorOptions =
                                    boost::get<configHolder_t>(sensorOptionsIt->second);

                                returnCode = sensor->setOptions(sensorOptions);
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
                                      configHolder_t & sensorOptions) const
    {
        auto sensorGroupIt = sensorsGroupHolder_.find(sensorType);
        if (sensorGroupIt == sensorsGroupHolder_.end())
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
                                       configHolder_t & sensorsOptions) const
    {
        auto sensorGroupIt = sensorsGroupHolder_.find(sensorType);
        if (sensorGroupIt == sensorsGroupHolder_.end())
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

    configHolder_t Robot::getSensorsOptions() const
    {
        configHolder_t sensorsOptions;
        for (const auto & sensorGroup : sensorsGroupHolder_)
        {
            configHolder_t sensorsGroupOptions;
            for (const auto & sensor : sensorGroup.second)
            {
                sensorsGroupOptions[sensor->getName()] = sensor->getOptions();
            }
            sensorsOptions[sensorGroup.first] = sensorsGroupOptions;
        }
        return sensorsOptions;
    }

    hresult_t Robot::setModelOptions(const configHolder_t & modelOptions)
    {
        return Model::setOptions(modelOptions);
    }

    configHolder_t Robot::getModelOptions() const
    {
        return Model::getOptions();
    }

    hresult_t Robot::setTelemetryOptions(const configHolder_t & telemetryOptions)
    {
        if (getIsLocked())
        {
            PRINT_ERROR("Robot is locked, probably because a simulation is running. Please stop "
                        "it before updating the telemetry options.");
            return hresult_t::ERROR_GENERIC;
        }

        for (auto & sensorGroupTelemetryOption : sensorTelemetryOptions_)
        {
            std::string optionTelemetryName = "enable" + sensorGroupTelemetryOption.first + "s";
            auto sensorTelemetryOptionIt = telemetryOptions.find(optionTelemetryName);
            if (sensorTelemetryOptionIt == telemetryOptions.end())
            {
                PRINT_ERROR("Missing field.");
                return hresult_t::ERROR_GENERIC;
            }
            sensorGroupTelemetryOption.second =
                boost::get<bool_t>(sensorTelemetryOptionIt->second);
        }

        return hresult_t::SUCCESS;
    }

    configHolder_t Robot::getTelemetryOptions() const
    {
        configHolder_t telemetryOptions;
        for (const auto & sensorGroupTelemetryOption : sensorTelemetryOptions_)
        {
            std::string optionTelemetryName = "enable" + sensorGroupTelemetryOption.first + "s";
            telemetryOptions[optionTelemetryName] = sensorGroupTelemetryOption.second;
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
        configHolder_t robotOptions;
        returnCode = jsonLoad(robotOptions, device);

        if (returnCode == hresult_t::SUCCESS)
        {
            returnCode = setOptions(robotOptions);
        }

        return returnCode;
    }

    const bool_t & Robot::getIsTelemetryConfigured() const
    {
        return isTelemetryConfigured_;
    }

    void Robot::computeMotorsEfforts(const float64_t & t,
                                     const Eigen::VectorXd & q,
                                     const Eigen::VectorXd & v,
                                     const Eigen::VectorXd & a,
                                     const Eigen::VectorXd & command)
    {
        if (!motorsHolder_.empty())
        {
            (*motorsHolder_.begin())->computeEffortAll(t, q, v, a, command);
        }
    }

    const Eigen::VectorXd & Robot::getMotorsEfforts() const
    {
        static const Eigen::VectorXd motorsEffortsEmpty;

        if (!motorsHolder_.empty())
        {
            return (*motorsHolder_.begin())->getAll();
        }

        return motorsEffortsEmpty;
    }

    const float64_t & Robot::getMotorEffort(const std::string & motorName) const
    {
        static const float64_t motorEffortEmpty = -1;

        auto motorIt = std::find_if(motorsHolder_.begin(),
                                    motorsHolder_.end(),
                                    [&motorName](const auto & elem)
                                    { return (elem->getName() == motorName); });
        if (motorIt != motorsHolder_.end())
        {
            return (*motorIt)->get();
        }

        return motorEffortEmpty;
    }

    void Robot::setSensorsData(const float64_t & t,
                               const Eigen::VectorXd & q,
                               const Eigen::VectorXd & v,
                               const Eigen::VectorXd & a,
                               const Eigen::VectorXd & uMotor,
                               const forceVector_t & fExternal)
    {
        /* Note that it is assumed that the kinematic quantities have been
           updated previously to be consistent with (q, v, a, u). If not,
           one is supposed to call  `pinocchio::forwardKinematics` and
           `pinocchio::updateFramePlacements` before calling this method. */

        for (const auto & sensorGroup : sensorsGroupHolder_)
        {
            if (!sensorGroup.second.empty())
            {
                (*sensorGroup.second.begin())->setAll(t, q, v, a, uMotor, fExternal);
            }
        }
    }

    sensorsDataMap_t Robot::getSensorsData() const
    {
        sensorsDataMap_t data;
        sensorsGroupHolder_t::const_iterator sensorsGroupIt = sensorsGroupHolder_.begin();
        sensorsSharedHolder_t::const_iterator sensorsSharedIt = sensorsSharedHolder_.begin();
        for (; sensorsGroupIt != sensorsGroupHolder_.end(); ++sensorsGroupIt, ++sensorsSharedIt)
        {
            sensorDataTypeMap_t dataType(sensorsSharedIt->second->dataMeasured_);
            for (auto & sensor : sensorsGroupIt->second)
            {
                auto & sensorConst = const_cast<const AbstractSensorBase &>(*sensor);
                dataType.emplace(sensorConst.getName(), sensorConst.getIdx(), sensorConst.get());
            }
            data.emplace(sensorsGroupIt->first, std::move(dataType));
        }
        return data;
    }

    Eigen::Ref<const Eigen::VectorXd> Robot::getSensorData(const std::string & sensorType,
                                                           const std::string & sensorName) const
    {
        static const Eigen::VectorXd sensorDataEmpty;
        static const Eigen::Ref<const Eigen::VectorXd> sensorDataRefEmpty(sensorDataEmpty);

        auto sensorGroupIt = sensorsGroupHolder_.find(sensorType);
        if (sensorGroupIt != sensorsGroupHolder_.end())
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
        for (const auto & sensorGroup : sensorsGroupHolder_)
        {
            if (!sensorGroup.second.empty())
            {
                (*sensorGroup.second.begin())->updateTelemetryAll();
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

    const bool_t & Robot::getIsLocked() const
    {
        return mutexLocal_->isLocked();
    }

    const std::vector<std::string> & Robot::getMotorsNames() const
    {
        return motorsNames_;
    }

    std::vector<jointIndex_t> Robot::getMotorsModelIdx() const
    {
        std::vector<jointIndex_t> motorsModelIdx;
        motorsModelIdx.reserve(nmotors_);
        std::transform(motorsHolder_.begin(),
                       motorsHolder_.end(),
                       std::back_inserter(motorsModelIdx),
                       [](const auto & motor) -> jointIndex_t
                       { return motor->getJointModelIdx(); });
        return motorsModelIdx;
    }

    std::vector<std::vector<int32_t>> Robot::getMotorsPositionIdx() const
    {
        std::vector<std::vector<int32_t>> motorsPositionIdx;
        motorsPositionIdx.reserve(nmotors_);
        std::transform(motorsHolder_.begin(),
                       motorsHolder_.end(),
                       std::back_inserter(motorsPositionIdx),
                       [](const auto & elem) -> std::vector<int32_t>
                       {
                           int32_t const & jointPositionIdx = elem->getJointPositionIdx();
                           if (elem->getJointType() == joint_t::ROTARY_UNBOUNDED)
                           {
                               return {jointPositionIdx, jointPositionIdx + 1};
                           }
                           else
                           {
                               return {jointPositionIdx};
                           }
                       });
        return motorsPositionIdx;
    }

    std::vector<int32_t> Robot::getMotorsVelocityIdx() const
    {
        std::vector<int32_t> motorsVelocityIdx;
        motorsVelocityIdx.reserve(nmotors_);
        std::transform(motorsHolder_.begin(),
                       motorsHolder_.end(),
                       std::back_inserter(motorsVelocityIdx),
                       [](const auto & elem) -> int32_t { return elem->getJointVelocityIdx(); });
        return motorsVelocityIdx;
    }

    const Eigen::VectorXd & Robot::getCommandLimit() const
    {
        return pncModel_.effortLimit;
    }

    const std::unordered_map<std::string, std::vector<std::string>> &
    Robot::getSensorsNames() const
    {
        return sensorsNames_;
    }

    const std::vector<std::string> & Robot::getSensorsNames(const std::string & sensorType) const
    {
        static const std::vector<std::string> sensorsNamesEmpty{};

        auto sensorsNamesIt = sensorsNames_.find(sensorType);
        if (sensorsNamesIt != sensorsNames_.end())
        {
            return sensorsNamesIt->second;
        }
        else
        {
            return sensorsNamesEmpty;
        }
    }

    const std::vector<std::string> & Robot::getCommandFieldnames() const
    {
        return logFieldnamesCommand_;
    }

    const std::vector<std::string> & Robot::getMotorEffortFieldnames() const
    {
        return logFieldnamesMotorEffort_;
    }

    const uint64_t & Robot::nmotors() const
    {
        return nmotors_;
    }
}
