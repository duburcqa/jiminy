
#include <iostream>
#include <fstream>
#include <exception>

#include "jiminy/core/robot/AbstractMotor.h"
#include "jiminy/core/robot/AbstractSensor.h"
#include "jiminy/core/telemetry/TelemetryData.h"
#include "jiminy/core/io/FileDevice.h"
#include "jiminy/core/utilities/Helpers.h"
#include "jiminy/core/utilities/Pinocchio.h"
#include "jiminy/core/utilities/Json.h"

#include "jiminy/core/robot/Robot.h"


namespace jiminy
{
    Robot::Robot(void) :
    Model(),
    isTelemetryConfigured_(false),
    telemetryData_(nullptr),
    motorsHolder_(),
    sensorsGroupHolder_(),
    sensorTelemetryOptions_(),
    motorsNames_(),
    sensorsNames_(),
    commandFieldnames_(),
    motorEffortFieldnames_(),
    nmotors_(0U),
    mutexLocal_(std::make_unique<MutexLocal>()),
    motorsSharedHolder_(std::make_shared<MotorSharedDataHolder_t>()),
    sensorsSharedHolder_()
    {
        // Empty on purpose
    }

    Robot::~Robot(void)
    {
        // Detach all the motors and sensors
        detachSensors({});
        detachMotors({});
    }

    hresult_t Robot::initialize(std::string              const & urdfPath,
                                bool_t                   const & hasFreeflyer,
                                std::vector<std::string> const & meshPackageDirs)
    {
        // Detach all the motors and sensors
        detachSensors({});
        detachMotors({});

        /* Delete the current model and generate a new one.
           Note that is also refresh all proxies automatically. */
        return Model::initialize(urdfPath, hasFreeflyer, meshPackageDirs);
    }

    hresult_t Robot::initialize(pinocchio::Model         const & pncModel,
                                pinocchio::GeometryModel const & collisionModel,
                                pinocchio::GeometryModel const & visualModel)
    {
        // Detach all the motors and sensors
        detachSensors({});
        detachMotors({});

        /* Delete the current model and generate a new one.
           Note that is also refresh all proxies automatically. */
        return Model::initialize(pncModel, collisionModel, visualModel);
    }

    void Robot::reset(void)
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
                                        std::string const & objectPrefixName)
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
                for (auto const & sensorGroup : sensorsGroupHolder_)
                {
                    for (auto const & sensor : sensorGroup.second)
                    {
                        if (returnCode == hresult_t::SUCCESS)
                        {
                            if (sensorTelemetryOptions_[sensorGroup.first])
                            {
                                returnCode = sensor->configureTelemetry(telemetryData_, objectPrefixName);
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
                PRINT_ERROR("Robot is locked, probably because a simulation is running. "
                            "Please stop it before adding motors.");
                returnCode = hresult_t::ERROR_GENERIC;
            }
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            std::string const & motorName = motor->getName();
            auto motorIt = std::find_if(motorsHolder_.begin(), motorsHolder_.end(),
                                        [&motorName](auto const & elem)
                                        {
                                            return (elem->getName() == motorName);
                                        });
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
            auto notifyRobot = [robot_=std::weak_ptr<Robot>(shared_from_this())](AbstractMotorBase & motorIn)
                {
                    // Make sure the robot still exists
                    auto robot = robot_.lock();
                    if (!robot)
                    {
                        PRINT_ERROR("Robot has been deleted. Impossible to notify motor update.");
                        return hresult_t::ERROR_GENERIC;
                    }

                    // Update rotor inertia of pinocchio model
                    float64_t const & armature = motorIn.getArmature();
                    std::string const & jointName = motorIn.getJointName();
                    int32_t jointVelocityIdx;
                    ::jiminy::getJointVelocityIdx(robot->pncModel_, jointName, jointVelocityIdx);
                    robot->pncModel_.rotorInertia[jointVelocityIdx] = armature;
                    ::jiminy::getJointVelocityIdx(robot->pncModelOrig_, jointName, jointVelocityIdx);
                    robot->pncModelOrig_.rotorInertia[jointVelocityIdx] = armature;
                    return hresult_t::SUCCESS;
                };

            // Attach the motor
            returnCode = motor->attach(shared_from_this(),
                                       notifyRobot,
                                       motorsSharedHolder_.get());
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

    hresult_t Robot::detachMotor(std::string const & motorName)
    {
        if (!isInitialized_)
        {
            PRINT_ERROR("Robot not initialized.");
            return hresult_t::ERROR_INIT_FAILED;
        }

        if (getIsLocked())
        {
            PRINT_ERROR("Robot is locked, probably because a simulation is running. "
                        "Please stop it before removing motors.");
            return hresult_t::ERROR_GENERIC;
        }

        auto motorIt = std::find_if(motorsHolder_.begin(), motorsHolder_.end(),
                                    [&motorName](auto const & elem)
                                    {
                                        return (elem->getName() == motorName);
                                    });
        if (motorIt == motorsHolder_.end())
        {
            PRINT_ERROR("No motor with this name exists.");
            return hresult_t::ERROR_BAD_INPUT;
        }

        // Detach the motor
        (*motorIt)->detach();  // It cannot fail at this point

        // Remove the motor from the holder
        motorsHolder_.erase(motorIt);

        // Refresh the motors proxies
        refreshMotorsProxies();

        return hresult_t::SUCCESS;
    }

    hresult_t Robot::detachMotors(std::vector<std::string> const & motorsNames)
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

            for (std::string const & name : motorsNames)
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
                PRINT_ERROR("Robot is locked, probably because a simulation is running."
                            " Please stop it before adding sensors.");
                returnCode = hresult_t::ERROR_GENERIC;
            }
        }

        std::string const & sensorName = sensor->getName();
        std::string const & sensorType = sensor->getType();
        sensorsGroupHolder_t::const_iterator sensorGroupIt;
        if (returnCode == hresult_t::SUCCESS)
        {
            sensorGroupIt = sensorsGroupHolder_.find(sensorType);
            if (sensorGroupIt != sensorsGroupHolder_.end())
            {
                auto sensorIt = std::find_if(sensorGroupIt->second.begin(),
                                             sensorGroupIt->second.end(),
                                             [&sensorName](auto const & elem)
                                             {
                                                 return (elem->getName() == sensorName);
                                             });
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
                sensorsSharedHolder_.emplace(std::make_pair(
                    sensorType, std::make_shared<SensorSharedDataHolder_t>()));
                sensorTelemetryOptions_.emplace(std::make_pair(sensorType, true));  // Enable the telemetry by default
            }

            // Attach the sensor
            returnCode = sensor->attach(shared_from_this(),
                                        sensorsSharedHolder_[sensorType].get());
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

    hresult_t Robot::detachSensor(std::string const & sensorType,
                                  std::string const & sensorName)
    {
        if (getIsLocked())
        {
            PRINT_ERROR("Robot is locked, probably because a simulation is running. "
                        "Please stop it before removing sensors.");
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
                                [&sensorName](auto const & elem)
                                {
                                    return (elem->getName() == sensorName);
                                });
        if (sensorIt == sensorGroupIt->second.end())
        {
            PRINT_ERROR("No sensor with this type and name exists.");
            return hresult_t::ERROR_BAD_INPUT;
        }

        // Detach the sensor
        (*sensorIt)->detach();  // It cannot fail at this point

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

    hresult_t Robot::detachSensors(std::string const & sensorType)
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

            std::vector<std::string> sensorGroupNames = sensorsNames_[sensorType];  // Make a copy since calling detachSensors update it !
            for (std::string const & sensorName : sensorGroupNames)
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
            std::transform(sensorsGroupHolder_.begin(), sensorsGroupHolder_.end(),
                           std::back_inserter(sensorsTypesNames),
                           [](auto const & pair) -> std::string
                           {
                               return pair.first;
                           });
            for (std::string const & sensorTypeName : sensorsTypesNames)
            {
                if (returnCode == hresult_t::SUCCESS)
                {
                    returnCode = detachSensors(sensorTypeName);
                }
            }
        }

        return returnCode;
    }

    hresult_t Robot::refreshProxies(void)
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

    hresult_t Robot::refreshMotorsProxies(void)
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
            std::transform(motorsHolder_.begin(), motorsHolder_.end(),
                           std::back_inserter(motorsNames_),
                           [](auto const & elem) -> std::string
                           {
                               return elem->getName();
                           });

            // Generate the fieldnames associated with command
            commandFieldnames_.clear();
            commandFieldnames_.reserve(nmotors_);
            std::transform(motorsHolder_.begin(), motorsHolder_.end(),
                           std::back_inserter(commandFieldnames_),
                           [](auto const & elem) -> std::string
                           {
                                return addCircumfix(elem->getName(), JOINT_PREFIX_BASE + "Command");
                           });

            // Generate the fieldnames associated with motor efforts
            motorEffortFieldnames_.clear();
            motorEffortFieldnames_.reserve(nmotors_);
            std::transform(motorsHolder_.begin(), motorsHolder_.end(),
                           std::back_inserter(motorEffortFieldnames_),
                           [](auto const & elem) -> std::string
                           {
                                return addCircumfix(elem->getName(), JOINT_PREFIX_BASE + "Effort");
                           });
        }

        return returnCode;
    }

    hresult_t Robot::refreshSensorsProxies(void)
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
            for (auto const & sensorGroup : sensorsGroupHolder_)
            {
                std::vector<std::string> sensorGroupNames;
                sensorGroupNames.reserve(sensorGroup.second.size());
                std::transform(sensorGroup.second.begin(), sensorGroup.second.end(),
                               std::back_inserter(sensorGroupNames),
                               [](auto const & elem) -> std::string
                               {
                                   return elem->getName();
                               });
                sensorsNames_.insert({sensorGroup.first, std::move(sensorGroupNames)});
            }
        }

        return returnCode;
    }

    hresult_t Robot::getMotor(std::string const & motorName,
                              std::shared_ptr<AbstractMotorBase> & motor)
    {
        if (!isInitialized_)
        {
            PRINT_ERROR("Robot not initialized.");
            return hresult_t::ERROR_INIT_FAILED;
        }

        auto motorIt = std::find_if(motorsHolder_.begin(), motorsHolder_.end(),
                                    [&motorName](auto const & elem)
                                    {
                                        return (elem->getName() == motorName);
                                    });
        if (motorIt == motorsHolder_.end())
        {
            PRINT_ERROR("No motor with this name exists.");
            return hresult_t::ERROR_BAD_INPUT;
        }

        motor = *motorIt;

        return hresult_t::SUCCESS;
    }

    hresult_t Robot::getMotor(std::string const & motorName,
                              std::weak_ptr<AbstractMotorBase const> & motor) const
    {
        if (!isInitialized_)
        {
            PRINT_ERROR("Robot not initialized.");
            return hresult_t::ERROR_INIT_FAILED;
        }

        auto motorIt = std::find_if(motorsHolder_.begin(), motorsHolder_.end(),
                                    [&motorName](auto const & elem)
                                    {
                                        return (elem->getName() == motorName);
                                    });
        if (motorIt == motorsHolder_.end())
        {
            PRINT_ERROR("No motor with this name exists.");
            return hresult_t::ERROR_BAD_INPUT;
        }

        motor = std::const_pointer_cast<AbstractMotorBase const>(*motorIt);

        return hresult_t::SUCCESS;
    }

    Robot::motorsHolder_t const & Robot::getMotors(void) const
    {
        return motorsHolder_;
    }

    hresult_t Robot::getSensor(std::string const & sensorType,
                               std::string const & sensorName,
                               std::weak_ptr<AbstractSensorBase const> & sensor) const
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
                                     [&sensorName](auto const & elem)
                                     {
                                         return (elem->getName() == sensorName);
                                     });
        if (sensorIt == sensorGroupIt->second.end())
        {
            PRINT_ERROR("No sensor with this type and name exists.");
            return hresult_t::ERROR_BAD_INPUT;
        }

        sensor = std::const_pointer_cast<AbstractSensorBase const>(*sensorIt);

        return hresult_t::SUCCESS;
    }

    hresult_t Robot::getSensor(std::string const & sensorType,
                               std::string const & sensorName,
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
                                     [&sensorName](auto const & elem)
                                     {
                                         return (elem->getName() == sensorName);
                                     });
        if (sensorIt == sensorGroupIt->second.end())
        {
            PRINT_ERROR("No sensor with this type and name exists.");
            return hresult_t::ERROR_BAD_INPUT;
        }

        sensor = *sensorIt;

        return hresult_t::SUCCESS;
    }

    Robot::sensorsGroupHolder_t const & Robot::getSensors(void) const
    {
        return sensorsGroupHolder_;
    }

    hresult_t Robot::setOptions(configHolder_t const & robotOptions)
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
            configHolder_t const & modelOptions =
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
            configHolder_t const & motorsOptions =
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
            configHolder_t const & sensorsOptions =
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
            configHolder_t const & telemetryOptions =
                boost::get<configHolder_t>(telemetryOptionsIt->second);
            returnCode = setTelemetryOptions(telemetryOptions);
        }

        return returnCode;
    }

    configHolder_t Robot::getOptions(void) const
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

    hresult_t Robot::setMotorOptions(std::string    const & motorName,
                                     configHolder_t const & motorOptions)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (getIsLocked())
        {
            PRINT_ERROR("Robot is locked, probably because a simulation is running. "
                        "Please stop it before updating the motor options.");
            returnCode = hresult_t::ERROR_GENERIC;
        }

        motorsHolder_t::iterator motorIt;
        if (returnCode == hresult_t::SUCCESS)
        {
            motorIt = std::find_if(motorsHolder_.begin(), motorsHolder_.end(),
                                   [&motorName](auto const & elem)
                                   {
                                       return (elem->getName() == motorName);
                                   });
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

    hresult_t Robot::setMotorsOptions(configHolder_t const & motorsOptions)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (getIsLocked())
        {
            PRINT_ERROR("Robot is locked, probably because a simulation is running. "
                        "Please stop it before updating the motor options.");
            returnCode = hresult_t::ERROR_GENERIC;
        }

        for (auto const & motor : motorsHolder_)
        {
            if (returnCode == hresult_t::SUCCESS)
            {
                auto motorOptionIt = motorsOptions.find(motor->getName());
                if (motorOptionIt != motorsOptions.end())
                {
                    returnCode = motor->setOptions(
                        boost::get<configHolder_t>(motorOptionIt->second));
                }
                else
                {
                    returnCode = motor->setOptionsAll(motorsOptions);
                    break;
                }
            }
        }

        // Propagate the user-defined motor inertia at Pinocchio model level
        pncModelOrig_.rotorInertia = getArmatures();
        pncModel_.rotorInertia = pncModelOrig_.rotorInertia;

        return returnCode;
    }

    hresult_t Robot::getMotorOptions(std::string    const & motorName,
                                     configHolder_t       & motorOptions) const
    {
        auto motorIt = std::find_if(motorsHolder_.begin(), motorsHolder_.end(),
                                    [&motorName](auto const & elem)
                                    {
                                        return (elem->getName() == motorName);
                                    });
        if (motorIt == motorsHolder_.end())
        {
            PRINT_ERROR("No motor with this name exists.");
            return hresult_t::ERROR_BAD_INPUT;
        }

        motorOptions = (*motorIt)->getOptions();

        return hresult_t::SUCCESS;
    }

    configHolder_t Robot::getMotorsOptions(void) const
    {
        configHolder_t motorsOptions;
        for (motorsHolder_t::value_type const & motor : motorsHolder_)
        {
            motorsOptions[motor->getName()] = motor->getOptions();
        }
        return motorsOptions;
    }

    hresult_t Robot::setSensorOptions(std::string    const & sensorType,
                                      std::string    const & sensorName,
                                      configHolder_t const & sensorOptions)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (getIsLocked())
        {
            PRINT_ERROR("Robot is locked, probably because a simulation is running. "
                        "Please stop it before updating the sensor options.");
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
                                     [&sensorName](auto const & elem)
                                     {
                                         return (elem->getName() == sensorName);
                                     });
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

    hresult_t Robot::setSensorsOptions(std::string    const & sensorType,
                                       configHolder_t const & sensorsOptions)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (getIsLocked())
        {
            PRINT_ERROR("Robot is locked, probably because a simulation is running. "
                        "Please stop it before updating the sensor options.");
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

        for (auto const & sensor : sensorGroupIt->second)
        {
            if (returnCode == hresult_t::SUCCESS)
            {
                auto sensorOptionIt = sensorsOptions.find(sensor->getName());
                if (sensorOptionIt != sensorsOptions.end())
                {
                    returnCode = sensor->setOptions(
                        boost::get<configHolder_t>(sensorOptionIt->second));
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

    hresult_t Robot::setSensorsOptions(configHolder_t const & sensorsOptions)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (getIsLocked())
        {
            PRINT_ERROR("Robot is locked, probably because a simulation is running. "
                        "Please stop it before updating the sensor options.");
            returnCode = hresult_t::ERROR_GENERIC;
        }

        for (auto const & sensorGroup : sensorsGroupHolder_)
        {
            if (returnCode == hresult_t::SUCCESS)
            {
                std::string const & sensorType = sensorGroup.first;

                auto sensorGroupOptionsIt = sensorsOptions.find(sensorType);
                if (sensorGroupOptionsIt != sensorsOptions.end())
                {
                    configHolder_t const & sensorGroupOptions =
                        boost::get<configHolder_t>(sensorGroupOptionsIt->second);

                    for (auto const & sensor : sensorGroup.second)
                    {
                        if (returnCode == hresult_t::SUCCESS)
                        {
                            std::string const & sensorName = sensor->getName();

                            auto sensorOptionsIt = sensorGroupOptions.find(sensorName);
                            if (sensorOptionsIt != sensorGroupOptions.end())
                            {
                                configHolder_t const & sensorOptions =
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

    hresult_t Robot::getSensorOptions(std::string    const & sensorType,
                                      std::string    const & sensorName,
                                      configHolder_t       & sensorOptions) const
    {
        auto sensorGroupIt = sensorsGroupHolder_.find(sensorType);
        if (sensorGroupIt == sensorsGroupHolder_.end())
        {
            PRINT_ERROR("This type of sensor does not exist.");
            return hresult_t::ERROR_BAD_INPUT;
        }

        auto sensorIt = std::find_if(sensorGroupIt->second.begin(),
                                     sensorGroupIt->second.end(),
                                     [&sensorName](auto const & elem)
                                     {
                                         return (elem->getName() == sensorName);
                                     });
        if (sensorIt == sensorGroupIt->second.end())
        {
            PRINT_ERROR("No sensor with this type and name exists.");
            return hresult_t::ERROR_BAD_INPUT;
        }

        sensorOptions = (*sensorIt)->getOptions();

        return hresult_t::SUCCESS;
    }

    hresult_t Robot::getSensorsOptions(std::string    const & sensorType,
                                       configHolder_t       & sensorsOptions) const
    {
        auto sensorGroupIt = sensorsGroupHolder_.find(sensorType);
        if (sensorGroupIt == sensorsGroupHolder_.end())
        {
            PRINT_ERROR("This type of sensor does not exist.");
            return hresult_t::ERROR_BAD_INPUT;
        }
        sensorsOptions.clear();
        for (auto const & sensor : sensorGroupIt->second)
        {
            sensorsOptions[sensor->getName()] = sensor->getOptions();
        }

        return hresult_t::SUCCESS;
    }

    configHolder_t Robot::getSensorsOptions(void) const
    {
        configHolder_t sensorsOptions;
        for (auto const & sensorGroup : sensorsGroupHolder_)
        {
            configHolder_t sensorsGroupOptions;
            for (auto const & sensor : sensorGroup.second)
            {
                sensorsGroupOptions[sensor->getName()] = sensor->getOptions();
            }
            sensorsOptions[sensorGroup.first] = sensorsGroupOptions;
        }
        return sensorsOptions;
    }

    hresult_t Robot::setModelOptions(configHolder_t const & modelOptions)
    {
        return Model::setOptions(modelOptions);
    }

    configHolder_t Robot::getModelOptions(void) const
    {
        return Model::getOptions();
    }

    hresult_t Robot::setTelemetryOptions(configHolder_t const & telemetryOptions)
    {
        if (getIsLocked())
        {
            PRINT_ERROR("Robot is locked, probably because a simulation is running. "
                        "Please stop it before updating the telemetry options.");
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
            sensorGroupTelemetryOption.second = boost::get<bool_t>(sensorTelemetryOptionIt->second);
        }

        return hresult_t::SUCCESS;
    }

    configHolder_t Robot::getTelemetryOptions(void) const
    {
        configHolder_t telemetryOptions;
        for (auto const & sensorGroupTelemetryOption : sensorTelemetryOptions_)
        {
            std::string optionTelemetryName = "enable" + sensorGroupTelemetryOption.first + "s";
            telemetryOptions[optionTelemetryName] = sensorGroupTelemetryOption.second;
        }
        return telemetryOptions;
    }

    hresult_t Robot::dumpOptions(std::string const & filepath) const
    {
        std::shared_ptr<AbstractIODevice> device =
            std::make_shared<FileDevice>(filepath);
        return jsonDump(getOptions(), device);
    }

    hresult_t Robot::loadOptions(std::string const & filepath)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        std::shared_ptr<AbstractIODevice> device =
            std::make_shared<FileDevice>(filepath);
        configHolder_t robotOptions;
        returnCode = jsonLoad(robotOptions, device);

        if (returnCode == hresult_t::SUCCESS)
        {
            returnCode = setOptions(robotOptions);
        }

        return returnCode;
    }

    bool_t const & Robot::getIsTelemetryConfigured(void) const
    {
        return isTelemetryConfigured_;
    }

    void Robot::computeMotorsEfforts(float64_t const & t,
                                     vectorN_t const & q,
                                     vectorN_t const & v,
                                     vectorN_t const & a,
                                     vectorN_t const & command)
    {
        if (!motorsHolder_.empty())
        {
            (*motorsHolder_.begin())->computeEffortAll(t, q, v, a, command);
        }
    }

    vectorN_t const & Robot::getMotorsEfforts(void) const
    {
        static vectorN_t const motorsEffortsEmpty;

        if (!motorsHolder_.empty())
        {
            return (*motorsHolder_.begin())->getAll();
        }

        return motorsEffortsEmpty;
    }

    float64_t const & Robot::getMotorEffort(std::string const & motorName) const
    {
        static float64_t const motorEffortEmpty = -1;

        auto motorIt = std::find_if(motorsHolder_.begin(), motorsHolder_.end(),
                                    [&motorName](auto const & elem)
                                    {
                                        return (elem->getName() == motorName);
                                    });
        if (motorIt != motorsHolder_.end())
        {
            return (*motorIt)->get();
        }

        return motorEffortEmpty;
    }

    void Robot::setSensorsData(float64_t     const & t,
                               vectorN_t     const & q,
                               vectorN_t     const & v,
                               vectorN_t     const & a,
                               vectorN_t     const & uMotor,
                               forceVector_t const & fExternal)
    {
        /* Note that it is assumed that the kinematic quantities have been
           updated previously to be consistent with (q, v, a, u). If not,
           one is supposed to call  `pinocchio::forwardKinematics` and
           `pinocchio::updateFramePlacements` before calling this method. */

        for (auto const & sensorGroup : sensorsGroupHolder_)
        {
            if (!sensorGroup.second.empty())
            {
                (*sensorGroup.second.begin())->setAll(t, q, v, a, uMotor, fExternal);
            }
        }
    }

    sensorsDataMap_t Robot::getSensorsData(void) const
    {
        sensorsDataMap_t data;
        sensorsGroupHolder_t::const_iterator sensorsGroupIt = sensorsGroupHolder_.begin();
        sensorsSharedHolder_t::const_iterator sensorsSharedIt = sensorsSharedHolder_.begin();
        for (; sensorsGroupIt != sensorsGroupHolder_.end() ; ++sensorsGroupIt, ++sensorsSharedIt)
        {
            sensorDataTypeMap_t dataType(std::cref(sensorsSharedIt->second->dataMeasured_));  // Need explicit call to `std::reference_wrapper` for gcc<7.3
            for (auto & sensor : sensorsGroupIt->second)
            {
                auto & sensorConst = const_cast<AbstractSensorBase const &>(*sensor);
                dataType.emplace(sensorConst.getName(),
                                 sensorConst.getIdx(),
                                 sensorConst.get());
            }
            data.emplace(sensorsGroupIt->first, std::move(dataType));
        }
        return data;
    }

    Eigen::Ref<vectorN_t const> Robot::getSensorData(std::string const & sensorType,
                                                     std::string const & sensorName) const
    {
        static vectorN_t const sensorDataEmpty;
        static Eigen::Ref<vectorN_t const> const sensorDataRefEmpty(sensorDataEmpty);

        auto sensorGroupIt = sensorsGroupHolder_.find(sensorType);
        if (sensorGroupIt != sensorsGroupHolder_.end())
        {
            auto sensorIt = std::find_if(sensorGroupIt->second.begin(),
                                         sensorGroupIt->second.end(),
                                         [&sensorName](auto const & elem)
                                         {
                                             return (elem->getName() == sensorName);
                                         });
            if (sensorIt != sensorGroupIt->second.end())
            {
                return (*sensorIt)->get();
            }
        }

        return sensorDataRefEmpty;
    }

    void Robot::updateTelemetry(void)
    {
        for (auto const & sensorGroup : sensorsGroupHolder_)
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

    bool_t const & Robot::getIsLocked(void) const
    {
        return mutexLocal_->isLocked();
    }

    std::vector<std::string> const & Robot::getMotorsNames(void) const
    {
        return motorsNames_;
    }

    std::vector<jointIndex_t> Robot::getMotorsModelIdx(void) const
    {
        std::vector<jointIndex_t> motorsModelIdx;
        motorsModelIdx.reserve(nmotors_);
        std::transform(motorsHolder_.begin(), motorsHolder_.end(),
                       std::back_inserter(motorsModelIdx),
                       [](auto const & motor) -> jointIndex_t
                       {
                           return motor->getJointModelIdx();
                       });
        return motorsModelIdx;
    }

    std::vector<std::vector<int32_t> > Robot::getMotorsPositionIdx(void) const
    {
        std::vector<std::vector<int32_t> > motorsPositionIdx;
        motorsPositionIdx.reserve(nmotors_);
        std::transform(motorsHolder_.begin(), motorsHolder_.end(),
                       std::back_inserter(motorsPositionIdx),
                       [](auto const & elem) -> std::vector<int32_t>
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

    std::vector<int32_t> Robot::getMotorsVelocityIdx(void) const
    {
        std::vector<int32_t> motorsVelocityIdx;
        motorsVelocityIdx.reserve(nmotors_);
        std::transform(motorsHolder_.begin(), motorsHolder_.end(),
                       std::back_inserter(motorsVelocityIdx),
                       [](auto const & elem) -> int32_t
                       {
                           return elem->getJointVelocityIdx();
                       });
        return motorsVelocityIdx;
    }

    std::unordered_map<std::string, std::vector<std::string> > const & Robot::getSensorsNames(void) const
    {
        return sensorsNames_;
    }

    std::vector<std::string> const & Robot::getSensorsNames(std::string const & sensorType) const
    {
        static std::vector<std::string> const sensorsNamesEmpty {};

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

    vectorN_t const & Robot::getCommandLimit(void) const
    {
        static vectorN_t commandLimit;
        commandLimit.resize(pncModel_.nv);

        commandLimit.setConstant(qNAN);
        for (auto const & motor : motorsHolder_)
        {
            auto const & motorOptions = motor->baseMotorOptions_;
            int32_t const & motorsVelocityIdx = motor->getJointVelocityIdx();
            if (motorOptions->enableCommandLimit)
            {
                commandLimit[motorsVelocityIdx] = motor->getCommandLimit();
            }
            else
            {
                commandLimit[motorsVelocityIdx] = INF;
            }

        }

        return commandLimit;
    }

    vectorN_t const & Robot::getArmatures(void) const
    {
        static vectorN_t armatures;
        armatures.resize(pncModel_.nv);

        armatures.setZero();
        for (auto const & motor : motorsHolder_)
        {
            int32_t const & motorsVelocityIdx = motor->getJointVelocityIdx();
            armatures[motorsVelocityIdx] = motor->getArmature();
        }

        return armatures;
    }

    std::vector<std::string> const & Robot::getCommandFieldnames(void) const
    {
        return commandFieldnames_;
    }

    std::vector<std::string> const & Robot::getMotorEffortFieldnames(void) const
    {
        return motorEffortFieldnames_;
    }

    uint64_t const & Robot::nmotors(void) const
    {
        return nmotors_;
    }
}
