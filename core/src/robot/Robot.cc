
#include <iostream>
#include <fstream>
#include <exception>

#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/jacobian.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"

#include "jiminy/core/robot/AbstractConstraint.h"
#include "jiminy/core/robot/AbstractMotor.h"
#include "jiminy/core/robot/AbstractSensor.h"
#include "jiminy/core/telemetry/TelemetryData.h"
#include "jiminy/core/io/FileDevice.h"

#include "jiminy/core/robot/Robot.h"


namespace jiminy
{
    Robot::Robot(void) :
    isTelemetryConfigured_(false),
    telemetryData_(nullptr),
    motorsHolder_(),
    sensorsGroupHolder_(),
    sensorTelemetryOptions_(),
    motorsNames_(),
    sensorsNames_(),
    motorEffortFieldnames_(),
    nmotors_(-1),
    constraintsHolder_(),
    constraintsJacobian_(),
    constraintsDrift_(),
    mutexLocal_(),
    motorsSharedHolder_(nullptr),
    sensorsSharedHolder_(),
    zeroAccelerationVector_(vectorN_t::Zero(0))
    {
        // Empty on purpose
    }

    Robot::~Robot(void)
    {
        // Detach all the motors and sensors
        detachMotors();
        detachSensors();
    }

    hresult_t Robot::initialize(std::string const & urdfPath,
                                bool_t      const & hasFreeflyer)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        // Remove all the motors and sensors
        motorsHolder_.clear();
        motorsSharedHolder_ = std::make_shared<MotorSharedDataHolder_t>();
        sensorsGroupHolder_.clear();
        sensorsSharedHolder_.clear();
        sensorTelemetryOptions_.clear();

        /* Delete the current model and generate a new one.
           Note that is also refresh all proxies automatically. */
        returnCode = Model::initialize(urdfPath, hasFreeflyer);

        return returnCode;
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
            std::cout << "Error - Robot::configureTelemetry - The robot is not initialized." << std::endl;
            returnCode = hresult_t::ERROR_INIT_FAILED;
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            telemetryData_ = std::move(telemetryData);
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
                            if (sensorTelemetryOptions_.at(sensorGroup.first))
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

        if (getIsLocked())
        {
            std::cout << "Error - Robot::addMotors - Robot is locked, probably because a simulation is running."\
                         " Please stop it before adding motors." << std::endl;
            returnCode = hresult_t::ERROR_GENERIC;
        }

        std::string const & motorName = motor->getName();
        auto motorIt = std::find_if(motorsHolder_.begin(), motorsHolder_.end(),
                                    [&motorName](auto const & elem)
                                    {
                                        return (elem->getName() == motorName);
                                    });
        if (returnCode == hresult_t::SUCCESS)
        {
            if (motorIt != motorsHolder_.end())
            {
                std::cout << "Error - Robot::attachMotor - A motor with the same name already exists." << std::endl;
                returnCode = hresult_t::ERROR_BAD_INPUT;
            }
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            // Attach the motor
            returnCode = motor->attach(this, motorsSharedHolder_.get());
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            // Add the motor to the holder
            motorsHolder_.emplace_back(std::move(motor));

            // Refresh the motors proxies
            refreshMotorsProxies();
        }

        return returnCode;
    }

    hresult_t Robot::detachMotor(std::string const & motorName)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (getIsLocked())
        {
            std::cout << "Error - Robot::detachMotor - Robot is locked, probably because a simulation is running."\
                         " Please stop it before removing motors." << std::endl;
            returnCode = hresult_t::ERROR_GENERIC;
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            if (!isInitialized_)
            {
                std::cout << "Error - Robot::detachMotor - Robot not initialized." << std::endl;
                returnCode = hresult_t::ERROR_INIT_FAILED;
            }
        }

        auto motorIt = std::find_if(motorsHolder_.begin(), motorsHolder_.end(),
                                    [&motorName](auto const & elem)
                                    {
                                        return (elem->getName() == motorName);
                                    });
        if (returnCode == hresult_t::SUCCESS)
        {
            if (motorIt == motorsHolder_.end())
            {
                std::cout << "Error - Robot::detachMotor - No motor with this name exists." << std::endl;
                returnCode = hresult_t::ERROR_BAD_INPUT;
            }
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            // Detach the motor
            returnCode = (*motorIt)->detach();
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            // Remove the motor from the holder
            motorsHolder_.erase(motorIt);
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            // Refresh the motors proxies
            refreshMotorsProxies();
        }

        return returnCode;
    }

    hresult_t Robot::detachMotors(std::vector<std::string> const & motorsNames)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (!motorsNames.empty())
        {
            // Make sure that no motor names are duplicates
            if (checkDuplicates(motorsNames))
            {
                std::cout << "Error - Robot::detachMotors - Duplicated motor names." << std::endl;
                returnCode = hresult_t::ERROR_BAD_INPUT;
            }

            if (returnCode == hresult_t::SUCCESS)
            {
                // Make sure that every motor name exist
                if (!checkInclusion(motorsNames_, motorsNames))
                {
                    std::cout << "Error - Robot::detachMotors - At least one of the motor names does not exist." << std::endl;
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

        if (getIsLocked())
        {
            std::cout << "Error - Robot::attachSensor - Robot is locked, probably because a simulation is running."\
                         " Please stop it before adding sensors." << std::endl;
            returnCode = hresult_t::ERROR_GENERIC;
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
                    std::cout << "Error - Robot::attachSensor - A sensor with the same type and name already exists." << std::endl;
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
                sensorTelemetryOptions_.emplace(std::make_pair(sensorType, true)); // Enable the telemetry by default
            }

            // Attach the sensor
            returnCode = sensor->attach(this, sensorsSharedHolder_.at(sensorType).get());
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            // Create the sensor and add it to its group
            sensorsGroupHolder_[sensorType].emplace_back(std::move(sensor));

            // Refresh the sensors proxies
            refreshSensorsProxies();
        }

        return returnCode;
    }

    hresult_t Robot::detachSensor(std::string const & sensorType,
                                  std::string const & sensorName)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (getIsLocked())
        {
            std::cout << "Error - Robot::detachSensor - Robot is locked, probably because a simulation is running."\
                         " Please stop it before removing sensors." << std::endl;
            returnCode = hresult_t::ERROR_GENERIC;
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            if (!isInitialized_)
            {
                std::cout << "Error - Robot::detachSensor - Robot not initialized." << std::endl;
                returnCode = hresult_t::ERROR_INIT_FAILED;
            }
        }

        auto sensorGroupIt = sensorsGroupHolder_.find(sensorType);
        if (returnCode == hresult_t::SUCCESS)
        {
            if (sensorGroupIt == sensorsGroupHolder_.end())
            {
                std::cout << "Error - Robot::detachSensor - This type of sensor does not exist." << std::endl;
                returnCode = hresult_t::ERROR_BAD_INPUT;
            }
        }

        sensorsHolder_t::iterator sensorIt;
        if (returnCode == hresult_t::SUCCESS)
        {
            sensorIt = std::find_if(sensorGroupIt->second.begin(),
                                    sensorGroupIt->second.end(),
                                    [&sensorName](auto const & elem)
                                    {
                                        return (elem->getName() == sensorName);
                                    });
            if (sensorIt == sensorGroupIt->second.end())
            {
                std::cout << "Error - Robot::detachSensors - No sensor with this type and name exists." << std::endl;
                returnCode = hresult_t::ERROR_BAD_INPUT;
            }
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            // Detach the motor
            returnCode = (*sensorIt)->detach();
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            // Remove the sensor from its group
            sensorGroupIt->second.erase(sensorIt);

            // Remove the sensor group if there is no more sensors left.
            if (sensorGroupIt->second.empty())
            {
                sensorsGroupHolder_.erase(sensorType);
                sensorsSharedHolder_.erase(sensorType);
                sensorTelemetryOptions_.erase(sensorType);
            }
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            // Refresh the sensors proxies
            refreshSensorsProxies();
        }

        return returnCode;
    }

    hresult_t Robot::detachSensors(std::string const & sensorType)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (!sensorType.empty())
        {
            auto sensorGroupIt = sensorsGroupHolder_.find(sensorType);
            if (sensorGroupIt == sensorsGroupHolder_.end())
            {
                std::cout << "Error - Robot::detachSensors - No sensor with this type exists." << std::endl;
                returnCode = hresult_t::ERROR_BAD_INPUT;
            }

            std::vector<std::string> sensorGroupNames = sensorsNames_.at(sensorType); // Make a copy since calling detachSensors update it !
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


    hresult_t Robot::addConstraint(std::string const & constraintName,
                                   std::shared_ptr<AbstractConstraint> constraint)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        auto constraintIt = std::find_if(constraintsHolder_.begin(),
                                         constraintsHolder_.end(),
                                         [&constraintName](auto const & element)
                                         {
                                             return element.name_ == constraintName;
                                         });
        if (constraintIt != constraintsHolder_.end())
        {
            std::cout << "Error - Robot::addConstraint - A constraint with name " << constraintName <<  " already exists." << std::endl;
            returnCode = hresult_t::ERROR_BAD_INPUT;
        }
        else
        {
            returnCode = constraint->attach(this);
            if (returnCode == hresult_t::SUCCESS)
            {
                constraintsHolder_.push_back(robotConstraint_t(constraintName, constraint));
            }
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            // Required to resize constraintsJacobian_ to the right size.
            refreshConstraintsProxies();
        }

        return returnCode;
    }

    hresult_t Robot::removeConstraint(std::string const & constraintName)
    {
        // Lookup constraint.
        auto constraintIt = std::find_if(constraintsHolder_.begin(),
                                         constraintsHolder_.end(),
                                         [&constraintName](auto const & element)
                                         {
                                             return element.name_ == constraintName;
                                         });
        if (constraintIt == constraintsHolder_.end())
        {
            std::cout << "Error - Robot::removeConstraint - No constraint with this name exists." << std::endl;
            return hresult_t::ERROR_BAD_INPUT;
        }

        constraintIt->constraint_->detach();
        constraintsHolder_.erase(constraintIt);

        // Required to resize constraintsJacobian_ to the right size.
        refreshConstraintsProxies();

        return hresult_t::SUCCESS;
    }

    hresult_t Robot::getConstraint(std::string const & constraintName,
                                   std::shared_ptr<AbstractConstraint> & constraint) const
    {
        // Lookup constraint.
        auto constraintIt = std::find_if(constraintsHolder_.begin(),
                                         constraintsHolder_.end(),
                                         [&constraintName](auto const & element)
                                         {
                                             return element.name_ == constraintName;
                                         });
        if (constraintIt == constraintsHolder_.end())
        {
            std::cout << "Error - Robot::getConstraint - No constraint with this name exists." << std::endl;
            return hresult_t::ERROR_BAD_INPUT;
        }
        else
        {
            constraint = constraintIt->constraint_;
        }
        return hresult_t::SUCCESS;
    }


    hresult_t Robot::refreshProxies(void)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (!isInitialized_)
        {
            std::cout << "Error - Robot::refreshProxies - Robot not initialized." << std::endl;
            returnCode = hresult_t::ERROR_INIT_FAILED;
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            returnCode = Model::refreshProxies();
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            returnCode = refreshConstraintsProxies();
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

    hresult_t Robot::refreshConstraintsProxies(void)
    {
        hresult_t returnCode = hresult_t::SUCCESS;
        vectorN_t q = pinocchio::neutral(pncModel_);
        vectorN_t v = vectorN_t::Zero(pncModel_.nv);

        // Resize zeroAccelerationVector_ to the right size
        zeroAccelerationVector_ = vectorN_t::Zero(pncModel_.nv);

        int constraintSize = 0;
        for (auto & constraint : constraintsHolder_)
        {
            if (returnCode == hresult_t::SUCCESS)
            {
                returnCode = constraint.constraint_->refreshProxies();
            }
            if (returnCode == hresult_t::SUCCESS)
            {
                // Call constraint on neutral position and zero velocity.
                matrixN_t J = constraint.constraint_->getJacobian(q);
                vectorN_t drift = constraint.constraint_->getDrift(q, v);

                // Verify dimensions.
                if (J.cols() != pncModel_.nv)
                {
                    std::cout << "Error - Robot::refreshConstraintsProxies: constraint "\
                                 "has an invalid jacobian (wrong number of columns)." << std::endl;
                    returnCode = hresult_t::ERROR_GENERIC;
                }
                if (drift.size() != J.rows())
                {
                    std::cout << "Error - Robot::refreshConstraintsProxies: constraint "\
                                 "has inconsistent jacobian and drift (size mismatch)." << std::endl;
                    returnCode = hresult_t::ERROR_GENERIC;
                }
                if (returnCode == hresult_t::SUCCESS)
                {
                    // Store constraint size.
                    constraint.dim_ = J.rows();
                    constraintSize += J.rows();
                }
            }
        }

        // Reset jacobian and drift to 0.
        if (returnCode == hresult_t::SUCCESS)
        {
            constraintsJacobian_ = matrixN_t::Zero(constraintSize, pncModel_.nv);
            constraintsDrift_ = vectorN_t::Zero(constraintSize);
        }
        return returnCode;
    }

    hresult_t Robot::refreshMotorsProxies(void)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (!isInitialized_)
        {
            std::cout << "Error - Robot::refreshMotorsProxies - Robot not initialized." << std::endl;
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

            // Generate the fieldnames associated with the motor efforts
            motorEffortFieldnames_.clear();
            motorEffortFieldnames_.reserve(nmotors_);
            std::transform(motorsHolder_.begin(), motorsHolder_.end(),
                           std::back_inserter(motorEffortFieldnames_),
                           [](auto const & elem) -> std::string
                           {
                               if (elem->getJointType() == joint_t::LINEAR)
                               {
                                   return addCircumfix(elem->getName(), JOINT_PREFIX_BASE + "Force");
                               }
                               else
                               {
                                   return addCircumfix(elem->getName(), JOINT_PREFIX_BASE + "Torque");
                               }
                           });
        }

        return returnCode;
    }

    hresult_t Robot::refreshSensorsProxies(void)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (!isInitialized_)
        {
            std::cout << "Error - Robot::refreshSensorsProxies - Robot not initialized." << std::endl;
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

    hresult_t Robot::getMotor(std::string       const   & motorName,
                              AbstractMotorBase const * & motor) const
    {
        if (!isInitialized_)
        {
            std::cout << "Error - Robot::getMotor - Robot not initialized." << std::endl;
            return hresult_t::ERROR_INIT_FAILED;
        }

        auto motorIt = std::find_if(motorsHolder_.begin(), motorsHolder_.end(),
                                    [&motorName](auto const & elem)
                                    {
                                        return (elem->getName() == motorName);
                                    });
        if (motorIt == motorsHolder_.end())
        {
            std::cout << "Error - Robot::getMotor - No motor with this name exists." << std::endl;
            return hresult_t::ERROR_BAD_INPUT;
        }

        motor = motorIt->get();

        return hresult_t::SUCCESS;
    }

    hresult_t Robot::getMotor(std::string const & motorName,
                              std::shared_ptr<AbstractMotorBase> & motor)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        AbstractMotorBase const * motorConst;
        returnCode = const_cast<Robot const *>(this)->getMotor(motorName, motorConst);

        if (returnCode == hresult_t::SUCCESS)
        {
            motor = std::move(const_cast<AbstractMotorBase *>(motorConst)->shared_from_this());
        }

        return returnCode;
    }

    Robot::motorsHolder_t const & Robot::getMotors(void) const
    {
        return motorsHolder_;
    }

    hresult_t Robot::getSensor(std::string        const   & sensorType,
                               std::string        const   & sensorName,
                               AbstractSensorBase const * & sensor) const
    {
        if (!isInitialized_)
        {
            std::cout << "Error - Robot::getSensor - Robot not initialized." << std::endl;
            return hresult_t::ERROR_INIT_FAILED;
        }

        auto sensorGroupIt = sensorsGroupHolder_.find(sensorType);
        if (sensorGroupIt == sensorsGroupHolder_.end())
        {
            std::cout << "Error - Robot::getSensor - This type of sensor does not exist." << std::endl;
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
            std::cout << "Error - Robot::getSensor - No sensor with this type and name exists." << std::endl;
            return hresult_t::ERROR_BAD_INPUT;
        }

        sensor = sensorIt->get();

        return hresult_t::SUCCESS;
    }

    hresult_t Robot::getSensor(std::string const & sensorType,
                               std::string const & sensorName,
                               std::shared_ptr<AbstractSensorBase> & sensor)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        AbstractSensorBase const * sensorPtr;
        returnCode = const_cast<Robot const *>(this)->getSensor(sensorType, sensorName, sensorPtr);

        if (returnCode == hresult_t::SUCCESS)
        {
            sensor = std::move(const_cast<AbstractSensorBase *>(sensorPtr)->shared_from_this());
        }

        return returnCode;
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
            std::cout << "Error - Robot::setOptions - 'model' options are missing." << std::endl;
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
                std::cout << "Error - Robot::setOptions - 'motors' options are missing." << std::endl;
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
                std::cout << "Error - Robot::setOptions - 'sensors' options are missing." << std::endl;
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
                std::cout << "Error - Robot::setOptions - 'telemetry' options are missing." << std::endl;
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
            std::cout << "Error - Robot::setMotorOptions - Robot is locked, probably because a simulation is running."\
                         " Please stop it before updating the motor options." << std::endl;
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
                std::cout << "Error - Robot::setMotorOptions - No motor with this name exists." << std::endl;
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
            std::cout << "Error - Robot::setMotorsOptions - Robot is locked, probably because a simulation is running."\
                         " Please stop it before updating the motor options." << std::endl;
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
            std::cout << "Error - Robot::getMotorOptions - No motor with this name exists." << std::endl;
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
            std::cout << "Error - Robot::setSensorOptions - Robot is locked, probably because a simulation is running."\
                         " Please stop it before updating the sensor options." << std::endl;
            returnCode = hresult_t::ERROR_GENERIC;
        }

        auto sensorGroupIt = sensorsGroupHolder_.find(sensorType);
        if (returnCode == hresult_t::SUCCESS)
        {
            if (sensorGroupIt == sensorsGroupHolder_.end())
            {
                std::cout << "Error - Robot::setSensorOptions - This type of sensor does not exist." << std::endl;
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
                std::cout << "Error - Robot::setSensorOptions - No sensor with this type and name exists." << std::endl;
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
            std::cout << "Error - Robot::setSensorsOptions - Robot is locked, probably because a simulation is running."\
                         " Please stop it before updating the sensor options." << std::endl;
            returnCode = hresult_t::ERROR_GENERIC;
        }

        sensorsGroupHolder_t::iterator sensorGroupIt;
        if (returnCode == hresult_t::SUCCESS)
        {
            sensorGroupIt = sensorsGroupHolder_.find(sensorType);
            if (sensorGroupIt == sensorsGroupHolder_.end())
            {
                std::cout << "Error - Robot::setSensorsOptions - This type of sensor does not exist." << std::endl;
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
            std::cout << "Error - Robot::setSensorsOptions - Robot is locked, probably because a simulation is running."\
                         " Please stop it before updating the sensor options." << std::endl;
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
                                std::cout << "Error - Robot::setSensorsOptions - No sensor with this name exists." << std::endl;
                                returnCode = hresult_t::ERROR_BAD_INPUT;
                            }
                        }
                    }
                }
                else
                {
                    std::cout << "Error - Robot::setSensorsOptions - This type of sensor does not exist." << std::endl;
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
            std::cout << "Error - Robot::getSensorOptions - This type of sensor does not exist." << std::endl;
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
            std::cout << "Error - Robot::getSensorOptions - No sensor with this type and name exists." << std::endl;
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
            std::cout << "Error - Robot::getSensorsOptions - This type of sensor does not exist." << std::endl;
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
            std::cout << "Error - Robot::setTelemetryOptions - Robot is locked, probably because a simulation is running."\
                         " Please stop it before updating the telemetry options." << std::endl;
            return hresult_t::ERROR_GENERIC;
        }

        for (auto & sensorGroupTelemetryOption : sensorTelemetryOptions_)
        {
            std::string optionTelemetryName = "enable" + sensorGroupTelemetryOption.first + "s";
            auto sensorTelemetryOptionIt = telemetryOptions.find(optionTelemetryName);
            if (sensorTelemetryOptionIt == telemetryOptions.end())
            {
                std::cout << "Error - Robot::setTelemetryOptions - Missing field." << std::endl;
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

    void Robot::computeMotorsEfforts(float64_t            const  & t,
                                     Eigen::Ref<vectorN_t const> const & q,
                                     Eigen::Ref<vectorN_t const> const & v,
                                     vectorN_t                   const & a,
                                     vectorN_t                   const & u)
    {
        if (!motorsHolder_.empty())
        {
            (*motorsHolder_.begin())->computeEffortAll(t, q, v, a, u);
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

    void Robot::setSensorsData(float64_t                   const & t,
                               Eigen::Ref<vectorN_t const> const & q,
                               Eigen::Ref<vectorN_t const> const & v,
                               Eigen::Ref<vectorN_t const> const & a,
                               vectorN_t                   const & u)
    {
        // Update kinematic quantities before updating sensors.
        // There is no need to update frame placement as this has already been done
        // before.
        pinocchio::forwardKinematics(pncModel_, pncData_, q, v, a);
        for (auto const & sensorGroup : sensorsGroupHolder_)
        {
            if (!sensorGroup.second.empty())
            {
                (*sensorGroup.second.begin())->setAll(t, q, v, a, u);
            }
        }
    }

    void Robot::computeConstraints(Eigen::Ref<vectorN_t const> const & q,
                                   Eigen::Ref<vectorN_t const> const & v)
    {
        // Compute joint jacobian.
        pinocchio::computeJointJacobians(pncModel_, pncData_, q);
        pinocchio::forwardKinematics(pncModel_, pncData_, q, v, zeroAccelerationVector_);

        uint32_t currentRow = 0;
        for (auto & constraint : constraintsHolder_)
        {
            matrixN_t J = constraint.constraint_->getJacobian(q);
            vectorN_t drift = constraint.constraint_->getDrift(q, v);

            uint32_t constraintDim = J.rows();
            // Resize matrix if needed.
            if (constraintDim != constraint.dim_)
            {
                constraintsJacobian_.conservativeResize(
                    constraintsJacobian_.rows() + constraintDim - constraint.dim_,
                    Eigen::NoChange);
                constraintsDrift_.conservativeResize(constraintsDrift_.size() + constraintDim - constraint.dim_);
                constraint.dim_ = constraintDim;
            }
            constraintsJacobian_.block(currentRow, 0, constraintDim, pncModel_.nv) = J;
            constraintsDrift_.segment(currentRow, constraintDim) = drift;
            currentRow += constraintDim;
        }
     }

    sensorsDataMap_t Robot::getSensorsData(void) const
    {
        sensorsDataMap_t data;
        for (auto & sensorGroup : sensorsGroupHolder_)
        {
            sensorDataTypeMap_t dataType;
            for (auto & sensor : sensorGroup.second)
            {
                auto & sensorConst = const_cast<AbstractSensorBase const &>(*sensor);
                dataType.emplace(sensorConst.getName(),
                                 sensorConst.getIdx(),
                                 sensorConst.get());
            }
            data.emplace(sensorGroup.first, std::move(dataType));
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

    hresult_t Robot::getLock(std::unique_ptr<MutexLocal::LockGuardLocal> & lock)
    {
        if (mutexLocal_.isLocked())
        {
            std::cout << "Error - Robot::getLock - Robot already locked. Please release the current lock first." << std::endl;
            return hresult_t::ERROR_GENERIC;
        }

        lock = std::move(std::make_unique<MutexLocal::LockGuardLocal>(mutexLocal_));

        return hresult_t::SUCCESS;
    }

    bool_t const & Robot::getIsLocked(void) const
    {
        return mutexLocal_.isLocked();
    }

    std::vector<std::string> const & Robot::getMotorsNames(void) const
    {
        return motorsNames_;
    }

    std::vector<int32_t> Robot::getMotorsModelIdx(void) const
    {
        std::vector<int32_t> motorsModelIdx;
        motorsModelIdx.reserve(nmotors_);
        std::transform(motorsHolder_.begin(), motorsHolder_.end(),
                       std::back_inserter(motorsModelIdx),
                       [](auto const & elem) -> int32_t
                       {
                           return elem->getJointModelIdx();
                       });
        return motorsModelIdx;
    }

    std::vector<int32_t> Robot::getMotorsPositionIdx(void) const
    {
        std::vector<int32_t> motorsPositionIdx;
        motorsPositionIdx.reserve(nmotors_);
        std::transform(motorsHolder_.begin(), motorsHolder_.end(),
                       std::back_inserter(motorsPositionIdx),
                       [](auto const & elem) -> int32_t
                       {
                           return elem->getJointPositionIdx();
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

    vectorN_t Robot::getEffortLimit(void) const
    {
        vectorN_t effortLimit = vectorN_t::Constant(pncModel_.nv, qNAN); // Do NOT use robot_->pncModel_.effortLimit, since we don't care about effort limits for non-physical joints
        for (auto const & motor : motorsHolder_)
        {
            auto const & motorOptions = motor->baseMotorOptions_;
            int32_t const & motorsVelocityIdx = motor->getJointVelocityIdx();
            if (motorOptions->enableEffortLimit)
            {
                effortLimit[motorsVelocityIdx] = motor->getEffortLimit();
            }
            else
            {
                effortLimit[motorsVelocityIdx] = INF;
            }

        }
        return effortLimit;
    }

    vectorN_t Robot::getMotorInertia(void) const
    {
        vectorN_t motorInertia = vectorN_t::Zero(pncModel_.nv);
        for (auto const & motor : motorsHolder_)
        {
            int32_t const & motorsVelocityIdx = motor->getJointVelocityIdx();
            motorInertia[motorsVelocityIdx] = motor->getRotorInertia();
        }
        return motorInertia;
    }

    std::vector<std::string> const & Robot::getMotorEffortFieldnames(void) const
    {
        return motorEffortFieldnames_;
    }

    int32_t const & Robot::nmotors(void) const
    {
        return nmotors_;
    }

    /// \brief Get jacobian of the constraints.
    matrixN_t const & Robot::getConstraintsJacobian(void) const
    {
        return constraintsJacobian_;
    }

    /// \brief Get drift of the constraints.
    vectorN_t const & Robot::getConstraintsDrift(void) const
    {
        return constraintsDrift_;
    }

    /// \brief Returns true if at least one constraint is active on the robot.
    bool_t Robot::hasConstraint(void) const
    {
        return !constraintsHolder_.empty();
    }
}
