
#include <fstream>
#include <exception>

#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/frames.hpp"

#include "jiminy/core/robot/AbstractMotor.h"
#include "jiminy/core/robot/AbstractSensor.h"
#include "jiminy/core/telemetry/TelemetryData.h"
#include "jiminy/core/io/FileDevice.h"

#include "jiminy/core/robot/Robot.h"


namespace jiminy
{
    Robot::Robot(void) :
    pncModel_(),
    pncData_(pncModel_),
    pncModelRigidOrig_(),
    pncDataRigidOrig_(pncModelRigidOrig_),
    mdlOptions_(nullptr),
    contactForces_(),
    isInitialized_(false),
    isTelemetryConfigured_(false),
    urdfPath_(),
    hasFreeflyer_(false),
    mdlOptionsHolder_(),
    telemetryData_(nullptr),
    motorsHolder_(),
    sensorsGroupHolder_(),
    sensorTelemetryOptions_(),
    contactFramesNames_(),
    contactFramesIdx_(),
    motorsNames_(),
    rigidJointsNames_(),
    rigidJointsModelIdx_(),
    rigidJointsPositionIdx_(),
    rigidJointsVelocityIdx_(),
    flexibleJointsNames_(),
    flexibleJointsModelIdx_(),
    positionLimitMin_(),
    positionLimitMax_(),
    velocityLimit_(),
    positionFieldNames_(),
    velocityFieldNames_(),
    accelerationFieldNames_(),
    motorTorqueFieldNames_(),
    mutexLocal_(),
    pncModelFlexibleOrig_(),
    motorsSharedHolder_(nullptr),
    sensorsSharedHolder_(),
    nq_(0),
    nv_(0),
    nx_(0)
    {
        setOptions(getDefaultOptions());
    }

    Robot::~Robot(void)
    {
        // Detach the motors
        detachMotors();

        // Detach the sensors
        detachSensors();
    }

    hresult_t Robot::initialize(std::string const & urdfPath,
                                bool_t      const & hasFreeflyer)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        // Remove all sensors, if any
        motorsHolder_.clear();
        motorsSharedHolder_ = std::make_shared<MotorSharedDataHolder_t>();
        sensorsGroupHolder_.clear();
        sensorsSharedHolder_.clear();
        sensorTelemetryOptions_.clear();

        // Initialize the URDF model
        returnCode = loadUrdfModel(urdfPath, hasFreeflyer);
        isInitialized_ = true;

        if (returnCode == hresult_t::SUCCESS)
        {
            // Backup the original model and data
            pncModelRigidOrig_ = pncModel_;
            pncDataRigidOrig_ = pinocchio::Data(pncModelRigidOrig_);

            /* Get the list of joint names of the rigid model and
               remove the 'universe' and 'root' if any, since they
               are not actual joints. */
            rigidJointsNames_ = pncModelRigidOrig_.names;
            rigidJointsNames_.erase(rigidJointsNames_.begin()); // remove the 'universe'
            if (hasFreeflyer)
            {
                rigidJointsNames_.erase(rigidJointsNames_.begin()); // remove the 'root'
            }
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            // Create the flexible model
            returnCode = generateModelFlexible();
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            // Add biases to the dynamics properties of the model
            returnCode = generateModelBiased();
        }

        if (returnCode != hresult_t::SUCCESS)
        {
            // Set the initialization flag
            isInitialized_ = false;
        }

        return returnCode;
    }

    void Robot::reset(void)
    {
        if (isInitialized_)
        {
            // Update the biases added to the dynamics properties of the model.
            generateModelBiased();
        }

        // Reset the sensors
        for (auto & sensorGroup : sensorsGroupHolder_)
        {
            for (auto & sensor : sensorGroup.second)
            {
                sensor->reset();
            }
        }

        // Reset the motors
        for (auto & motor : motorsHolder_)
        {
            motor->reset();
        }

        // Reset the telemetry state
        isTelemetryConfigured_ = false;
    }

    hresult_t Robot::configureTelemetry(std::shared_ptr<TelemetryData> const & telemetryData)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (!isInitialized_)
        {
            std::cout << "Error - Robot::configureTelemetry - The robot is not initialized." << std::endl;
            returnCode = hresult_t::ERROR_INIT_FAILED;
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            telemetryData_ = std::shared_ptr<TelemetryData>(telemetryData);
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
                                returnCode = sensor->configureTelemetry(telemetryData_);
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

    hresult_t Robot::addContactPoints(std::vector<std::string> const & frameNames)
    {
        if (getIsLocked())
        {
            std::cout << "Error - Robot::addContactPoints - Robot is locked, probably because a simulation is running.";
            std::cout << " Please stop it before adding contact points." << std::endl;
            return hresult_t::ERROR_GENERIC;
        }

        if (!isInitialized_)
        {
            std::cout << "Error - Robot::addContactPoints - Robot not initialized." << std::endl;
            return hresult_t::ERROR_INIT_FAILED;
        }

        // Make sure that the frame list is not empty
        if (frameNames.empty())
        {
            std::cout << "Error - Robot::addContactPoints - The list of frames must not be empty." << std::endl;
            return hresult_t::ERROR_BAD_INPUT;
        }

        // Make sure that no frame are duplicates
        if (checkDuplicates(frameNames))
        {
            std::cout << "Error - Robot::addContactPoints - Some frames are duplicates." << std::endl;
            return hresult_t::ERROR_BAD_INPUT;
        }

        // Make sure that no motor is associated with any of the joint in the list
        if (checkIntersection(contactFramesNames_, frameNames))
        {
            std::cout << "Error - Robot::addContactPoints - At least one of the frame is already been associated with a contact point." << std::endl;
            return hresult_t::ERROR_BAD_INPUT;
        }

        // Make sure that all the frames exist
        for (std::string const & frame : frameNames)
        {
            if (!pncModel_.existFrame(frame))
            {
                std::cout << "Error - Robot::addContactPoints - At least one of the frame does not exist." << std::endl;
                return hresult_t::ERROR_BAD_INPUT;
            }
        }

        // Add the list of frames to the set of contact points
        contactFramesNames_.insert(contactFramesNames_.end(), frameNames.begin(), frameNames.end());

        // Reset the contact force internal buffer
        contactForces_ = forceVector_t(contactFramesNames_.size(), pinocchio::Force::Zero());

        // Refresh proxies associated with the contact points only
        refreshContactProxies();

        return hresult_t::SUCCESS;
    }

    hresult_t Robot::removeContactPoints(std::vector<std::string> const & frameNames)
    {
        if (getIsLocked())
        {
            std::cout << "Error - Robot::removeContactPoints - Robot is locked, probably because a simulation is running.";
            std::cout << " Please stop it before removing contact points." << std::endl;
            return hresult_t::ERROR_GENERIC;
        }

        if (!isInitialized_)
        {
            std::cout << "Error - Robot::removeContactPoints - Robot not initialized." << std::endl;
            return hresult_t::ERROR_INIT_FAILED;
        }

        // Make sure that no frame are duplicates
        if (checkDuplicates(frameNames))
        {
            std::cout << "Error - Robot::removeContactPoints - Some frames are duplicates." << std::endl;
            return hresult_t::ERROR_BAD_INPUT;
        }

        // Make sure that no motor is associated with any of the joint in jointNames
        if (!checkInclusion(contactFramesNames_, frameNames))
        {
            std::cout << "Error - Robot::removeContactPoints - At least one of the frame is not associated with any contact point." << std::endl;
            return hresult_t::ERROR_BAD_INPUT;
        }

        // Remove the list of frames from the set of contact points
        if (!frameNames.empty())
        {
            eraseVector(contactFramesNames_, frameNames);
        }
        else
        {
            contactFramesNames_.clear();
        }

        // Reset the contact force internal buffer
        contactForces_ = forceVector_t(contactFramesNames_.size(), pinocchio::Force::Zero());

        // Refresh proxies associated with the contact points only
        refreshContactProxies();

        return hresult_t::SUCCESS;
    }

    hresult_t Robot::attachMotor(std::shared_ptr<AbstractMotorBase> const & motor)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (getIsLocked())
        {
            std::cout << "Error - Robot::addMotors - Robot is locked, probably because a simulation is running.";
            std::cout << " Please stop it before adding motors." << std::endl;
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
            returnCode = motor->attach(this, motorsSharedHolder_);
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            // Add the motor to the holder
            motorsHolder_.emplace_back(motor);

            // Refresh the attributes of the robot
            refreshMotorProxies();
        }

        return returnCode;
    }

    hresult_t Robot::detachMotor(std::string const & motorName)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (getIsLocked())
        {
            std::cout << "Error - Robot::detachMotor - Robot is locked, probably because a simulation is running.";
            std::cout << " Please stop it before removing motors." << std::endl;
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
            // Refresh proxies associated with the motors only
            refreshMotorProxies();
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

    hresult_t Robot::attachSensor(std::shared_ptr<AbstractSensorBase> const & sensor)
    {
        // The sensors' names must be unique, even if their type is different.

        hresult_t returnCode = hresult_t::SUCCESS;

        if (getIsLocked())
        {
            std::cout << "Error - Robot::attachSensor - Robot is locked, probably because a simulation is running.";
            std::cout << " Please stop it before adding sensors." << std::endl;
            returnCode = hresult_t::ERROR_GENERIC;
        }

        std::string const & sensorName = sensor->getName();
        std::string const & sensorType = sensor->getType();
        auto sensorGroupIt = sensorsGroupHolder_.find(sensorType);
        if (returnCode == hresult_t::SUCCESS)
        {
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
                sensorsSharedHolder_[sensorType] = std::make_shared<SensorSharedDataHolder_t>();
                sensorTelemetryOptions_[sensorType] = false;
            }

            // Create the sensor and add it to its group
            sensorsGroupHolder_[sensorType].emplace_back(sensor);

            // Attach the sensor
            returnCode = sensor->attach(this, sensorsSharedHolder_.at(sensorType));
        }

        return returnCode;
    }

    hresult_t Robot::detachSensor(std::string const & sensorType,
                                 std::string const & sensorName)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (getIsLocked())
        {
            std::cout << "Error - Robot::detachSensor - Robot is locked, probably because a simulation is running.";
            std::cout << " Please stop it before removing sensors." << std::endl;
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

            for (std::string const & sensorName : getSensorsNames(sensorType))
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

    hresult_t Robot::generateModelFlexible(void)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (!isInitialized_)
        {
            std::cout << "Error - Robot::generateModelFlexible - Robot not initialized." << std::endl;
            returnCode = hresult_t::ERROR_INIT_FAILED;
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            flexibleJointsNames_.clear();
            flexibleJointsModelIdx_.clear();
            pncModelFlexibleOrig_ = pncModelRigidOrig_;
            for(flexibleJointData_t const & flexibleJoint : mdlOptions_->dynamics.flexibilityConfig)
            {
                std::string const & jointName = flexibleJoint.jointName;

                // Look if given joint exists in the joint list.
                if (returnCode == hresult_t::SUCCESS)
                {
                    int32_t jointIdx;
                    returnCode = getJointPositionIdx(pncModel_, jointName, jointIdx);
                }

                // Add joints to model.
                if (returnCode == hresult_t::SUCCESS)
                {
                    std::string newName =
                        removeFieldnameSuffix(jointName, "Joint") + FLEXIBLE_JOINT_SUFFIX;
                    flexibleJointsNames_.emplace_back(newName);
                    insertFlexibilityInModel(pncModelFlexibleOrig_, jointName, newName); // Ignore return code, as check has already been done.
                }
            }
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            getJointsModelIdx(pncModelFlexibleOrig_,
                              flexibleJointsNames_,
                              flexibleJointsModelIdx_);
        }

        return returnCode;
    }

    hresult_t Robot::generateModelBiased(void)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (!isInitialized_)
        {
            std::cout << "Error - Robot::generateModelBiased - Robot not initialized." << std::endl;
            returnCode = hresult_t::ERROR_INIT_FAILED;
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            // Reset the robot either with the original rigid or flexible model
            if (mdlOptions_->dynamics.enableFlexibleModel)
            {
                pncModel_ = pncModelFlexibleOrig_;
            }
            else
            {
                pncModel_ = pncModelRigidOrig_;
            }

            for (std::string const & jointName : rigidJointsNames_)
            {
                int32_t const jointIdx = pncModel_.getJointId(jointName);

                vector3_t & comRelativePositionBody =
                    const_cast<vector3_t &>(pncModel_.inertias[jointIdx].lever());
                comRelativePositionBody +=
                    randVectorNormal(3U, mdlOptions_->dynamics.centerOfMassPositionBodiesBiasStd);

                // Cannot be less than 1g for numerical stability
                float64_t & massBody =
                    const_cast<float64_t &>(pncModel_.inertias[jointIdx].mass());
                massBody =
                    std::max(massBody +
                        randNormal(0.0, mdlOptions_->dynamics.massBodiesBiasStd), 1.0e-3);

                // Cannot be less 1g applied at 1mm of distance from the rotation center
                vector6_t & inertiaBody =
                    const_cast<vector6_t &>(pncModel_.inertias[jointIdx].inertia().data());
                inertiaBody =
                    clamp(inertiaBody +
                        randVectorNormal(6U, mdlOptions_->dynamics.inertiaBodiesBiasStd), 1.0e-9);

                vector3_t & relativePositionBody =
                    pncModel_.jointPlacements[jointIdx].translation();
                relativePositionBody +=
                    randVectorNormal(3U, mdlOptions_->dynamics.relativePositionBodiesBiasStd);
            }

            // Initialize Pinocchio Data internal state
            pncData_ = pinocchio::Data(pncModel_);
            pinocchio::forwardKinematics(pncModel_, pncData_,
                                         vectorN_t::Zero(pncModel_.nq),
                                         vectorN_t::Zero(pncModel_.nv));
            pinocchio::updateFramePlacements(pncModel_, pncData_);
        }

        /* Initialize the internal proxies.
           Be careful, the internal proxies of the sensors and motors are
           not up-to-date at the point, so one cannot use them. */
        if (returnCode == hresult_t::SUCCESS)
        {
            returnCode = refreshProxies();
        }

        // Refresh the motors
        for (auto & motor : motorsHolder_)
        {
            if (returnCode == hresult_t::SUCCESS)
            {
                returnCode = motor->refreshProxies();
            }
        }

        // Refresh the sensors
        for (auto & sensorGroup : sensorsGroupHolder_)
        {
            for (auto & sensor : sensorGroup.second)
            {
                if (returnCode == hresult_t::SUCCESS)
                {
                    returnCode = sensor->refreshProxies();
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
            std::cout << "Error - Robot::refreshProxies - Robot not initialized." << std::endl;
            returnCode = hresult_t::ERROR_INIT_FAILED;
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            // Extract the dimensions of the configuration and velocity vectors
            nq_ = pncModel_.nq;
            nv_ = pncModel_.nv;
            nx_ = nq_ + nv_;

            // Extract some rigid joints indices in the model
            getJointsModelIdx(pncModel_, rigidJointsNames_, rigidJointsModelIdx_);
            getJointsPositionIdx(pncModel_, rigidJointsNames_, rigidJointsPositionIdx_, false);
            getJointsVelocityIdx(pncModel_, rigidJointsNames_, rigidJointsVelocityIdx_, false);

            /* Generate the fieldnames associated with the configuration,
               velocity, and acceleration vectors. */
            positionFieldNames_.clear();
            positionFieldNames_.resize(nq_);
            velocityFieldNames_.clear();
            velocityFieldNames_.resize(nv_);
            accelerationFieldNames_.clear();
            accelerationFieldNames_.resize(nv_);
            std::vector<std::string> const & jointNames = pncModel_.names;
            std::vector<std::string> jointShortNames =
                removeFieldnamesSuffix(jointNames, "Joint");
            for (uint32_t i=0; i<jointNames.size(); ++i)
            {
                std::string const & jointName = jointNames[i];
                int32_t const jointIdx = pncModel_.getJointId(jointName);

                int32_t idx_q = pncModel_.joints[jointIdx].idx_q();

                if (idx_q >= 0) // Otherwise the joint is not part of the vectorial representation
                {
                    int32_t idx_v = pncModel_.joints[jointIdx].idx_v();

                    joint_t jointType;
                    std::string jointPrefix;
                    if (returnCode == hresult_t::SUCCESS)
                    {
                        returnCode = getJointTypeFromId(pncModel_, jointIdx, jointType);
                    }
                    if (returnCode == hresult_t::SUCCESS)
                    {
                        if (jointType == joint_t::FREE)
                        {
                            // Discard the joint name for FREE joint type since it is unique if any
                            jointPrefix = FREE_FLYER_PREFIX_BASE_NAME;
                            jointShortNames[i] = "";
                        }
                        else
                        {
                            jointPrefix = JOINT_PREFIX_BASE;
                        }
                    }

                    std::vector<std::string> jointTypePositionSuffixes;
                    std::vector<std::string> jointPositionFieldnames;
                    if (returnCode == hresult_t::SUCCESS)
                    {
                        returnCode = getJointTypePositionSuffixes(jointType,
                                                                  jointTypePositionSuffixes);
                    }
                    if (returnCode == hresult_t::SUCCESS)
                    {
                        for (std::string const & suffix : jointTypePositionSuffixes)
                        {
                            jointPositionFieldnames.emplace_back(
                                jointPrefix + "Position" + jointShortNames[i] + suffix);
                        }
                    }
                    if (returnCode == hresult_t::SUCCESS)
                    {
                        std::copy(jointPositionFieldnames.begin(),
                                  jointPositionFieldnames.end(),
                                  positionFieldNames_.begin() + idx_q);
                    }

                    std::vector<std::string> jointTypeVelocitySuffixes;
                    std::vector<std::string> jointVelocityFieldnames;
                    std::vector<std::string> jointAccelerationFieldnames;
                    if (returnCode == hresult_t::SUCCESS)
                    {
                        returnCode = getJointTypeVelocitySuffixes(jointType,
                                                                  jointTypeVelocitySuffixes);
                    }
                    if (returnCode == hresult_t::SUCCESS)
                    {
                        for (std::string const & suffix : jointTypeVelocitySuffixes)
                        {
                            jointVelocityFieldnames.emplace_back(
                                jointPrefix + "Velocity" + jointShortNames[i] + suffix);
                            jointAccelerationFieldnames.emplace_back(
                                jointPrefix + "Acceleration" + jointShortNames[i] + suffix);
                        }
                    }
                    if (returnCode == hresult_t::SUCCESS)
                    {
                        std::copy(jointVelocityFieldnames.begin(),
                                  jointVelocityFieldnames.end(),
                                  velocityFieldNames_.begin() + idx_v);
                        std::copy(jointAccelerationFieldnames.begin(),
                                  jointAccelerationFieldnames.end(),
                                  accelerationFieldNames_.begin() + idx_v);
                    }
                }
            }
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            // Get the joint position limits from the URDF or the user options
            if (mdlOptions_->joints.positionLimitFromUrdf)
            {
                positionLimitMin_.resize(rigidJointsPositionIdx_.size());
                positionLimitMax_.resize(rigidJointsPositionIdx_.size());
                for (uint32_t i=0; i < rigidJointsPositionIdx_.size(); ++i)
                {
                    positionLimitMin_[i] = pncModel_.lowerPositionLimit[rigidJointsPositionIdx_[i]];
                    positionLimitMax_[i] = pncModel_.upperPositionLimit[rigidJointsPositionIdx_[i]];
                }
            }
            else
            {
                positionLimitMin_ = mdlOptions_->joints.positionLimitMin;
                positionLimitMax_ = mdlOptions_->joints.positionLimitMax;
            }

            // Get the joint velocity limits from the URDF or the user options
            if (mdlOptions_->joints.velocityLimitFromUrdf)
            {
                velocityLimit_.resize(rigidJointsVelocityIdx_.size());
                for (uint32_t i=0; i < rigidJointsVelocityIdx_.size(); ++i)
                {
                    velocityLimit_[i] = pncModel_.velocityLimit[rigidJointsVelocityIdx_[i]];
                }
            }
            else
            {
                velocityLimit_ = mdlOptions_->joints.velocityLimit;
            }
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            returnCode = refreshContactProxies();
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            returnCode = refreshMotorProxies();
        }

        return returnCode;
    }

    hresult_t Robot::refreshContactProxies(void)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (!isInitialized_)
        {
            std::cout << "Error - Robot::refreshContactProxies - Robot not initialized." << std::endl;
            returnCode = hresult_t::ERROR_INIT_FAILED;
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            // Extract the contact frames indices in the model
            getFramesIdx(pncModel_, contactFramesNames_, contactFramesIdx_);
        }

        return returnCode;
    }

    hresult_t Robot::refreshMotorProxies(void)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (!isInitialized_)
        {
            std::cout << "Error - Robot::refreshMotorProxies - Robot not initialized." << std::endl;
            returnCode = hresult_t::ERROR_INIT_FAILED;
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            // Extract the motor names
            motorsNames_.clear();
            motorsNames_.reserve(motorsHolder_.size());
            std::transform(motorsHolder_.begin(), motorsHolder_.end(),
                           std::back_inserter(motorsNames_),
                           [](auto const & elem) -> std::string
                           {
                               return elem->getName();
                           });

            // Generate the fieldnames associated with the motor torques
            motorTorqueFieldNames_.clear();
            for (std::string const & jointName : removeFieldnamesSuffix(motorsNames_, "Joint"))
            {
                motorTorqueFieldNames_.emplace_back(JOINT_PREFIX_BASE + "Torque" + jointName);
            }
        }

        return returnCode;
    }

    hresult_t Robot::setMotorOptions(std::string    const & motorName,
                                     configHolder_t const & motorOptions)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (getIsLocked())
        {
            std::cout << "Error - Robot::setMotorOptions - Robot is locked, probably because a simulation is running.";
            std::cout << " Please stop it before updating the motor options." << std::endl;
            returnCode = hresult_t::ERROR_INIT_FAILED;
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            if (!isInitialized_)
            {
                std::cout << "Error - Robot::setMotorOptions - Robot not initialized." << std::endl;
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
            std::cout << "Error - Robot::setMotorsOptions - Robot is locked, probably because a simulation is running.";
            std::cout << " Please stop it before updating the motor options." << std::endl;
            returnCode = hresult_t::ERROR_INIT_FAILED;
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            if (!isInitialized_)
            {
                std::cout << "Error - Robot::setMotorsOptions - Robot not initialized." << std::endl;
                returnCode = hresult_t::ERROR_INIT_FAILED;
            }
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

    hresult_t Robot::getMotorsOptions(configHolder_t & motorsOptions) const
    {
        if (!isInitialized_)
        {
            std::cout << "Error - Robot::getMotorsOptions - Robot not initialized." << std::endl;
            return hresult_t::ERROR_INIT_FAILED;
        }

        motorsOptions.clear();
        for (motorsHolder_t::value_type const & motor : motorsHolder_)
        {
            motorsOptions[motor->getName()] = motor->getOptions();
        }

        return hresult_t::SUCCESS;
    }

    hresult_t Robot::getMotorOptions(std::string    const & motorName,
                                     configHolder_t       & motorOptions) const
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (!isInitialized_)
        {
            std::cout << "Error - Robot::getMotorOptions - Robot not initialized." << std::endl;
            return hresult_t::ERROR_INIT_FAILED;
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
                std::cout << "Error - Robot::getMotorOptions - No motor with this name exists." << std::endl;
                returnCode = hresult_t::ERROR_BAD_INPUT;
            }
        }

        motorOptions = (*motorIt)->getOptions();

        return hresult_t::SUCCESS;
    }

    hresult_t Robot::getMotor(std::string const & motorName,
                              std::shared_ptr<AbstractMotorBase const> & motor) const
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

        motor = (*motorIt);

        return hresult_t::SUCCESS;
    }

    Robot::motorsHolder_t const & Robot::getMotors(void) const
    {
        return motorsHolder_;
    }

    hresult_t Robot::getSensor(std::string const & sensorType,
                               std::string const & sensorName,
                               std::shared_ptr<AbstractSensorBase const> & sensor) const
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

        sensor = (*sensorIt);

        return hresult_t::SUCCESS;
    }

    Robot::sensorsGroupHolder_t const & Robot::getSensors(void) const
    {
        return sensorsGroupHolder_;
    }

    hresult_t Robot::setSensorOptions(std::string    const & sensorType,
                                      std::string    const & sensorName,
                                      configHolder_t const & sensorOptions)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (getIsLocked())
        {
            std::cout << "Error - Robot::setSensorOptions - Robot is locked, probably because a simulation is running.";
            std::cout << " Please stop it before updating the sensor options." << std::endl;
            returnCode = hresult_t::ERROR_INIT_FAILED;
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            if (!isInitialized_)
            {
                std::cout << "Error - Robot::setSensorOptions - Robot not initialized." << std::endl;
                returnCode = hresult_t::ERROR_INIT_FAILED;
            }
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
            std::cout << "Error - Robot::setSensorsOptions - Robot is locked, probably because a simulation is running.";
            std::cout << " Please stop it before updating the sensor options." << std::endl;
            returnCode = hresult_t::ERROR_INIT_FAILED;
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            if (!isInitialized_)
            {
                std::cout << "Error - Robot::setSensorsOptions - Robot not initialized." << std::endl;
                returnCode = hresult_t::ERROR_INIT_FAILED;
            }
        }

        auto sensorGroupIt = sensorsGroupHolder_.find(sensorType);
        if (returnCode == hresult_t::SUCCESS)
        {
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
            std::cout << "Error - Robot::setSensorsOptions - Robot is locked, probably because a simulation is running.";
            std::cout << " Please stop it before updating the sensor options." << std::endl;
            returnCode = hresult_t::ERROR_INIT_FAILED;
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            if (!isInitialized_)
            {
                std::cout << "Error - Robot::setSensorsOptions - Robot not initialized." << std::endl;
                returnCode = hresult_t::ERROR_INIT_FAILED;
            }
        }

        for (auto const & sensorGroup : sensorsGroupHolder_)
        {
            for (auto const & sensor : sensorGroup.second)
            {
                if (returnCode == hresult_t::SUCCESS)
                {
                    // TODO: missing check for sensor type and name availability
                    returnCode = sensor->setOptions(boost::get<configHolder_t>(
                        boost::get<configHolder_t>(
                            sensorsOptions.at(sensorGroup.first)).at(sensor->getName())));
                }
            }
        }

        return returnCode;
    }


    hresult_t Robot::getSensorsOptions(std::string    const & sensorType,
                                       configHolder_t       & sensorsOptions) const
    {
        if (!isInitialized_)
        {
            std::cout << "Error - Robot::getSensorsOptions - Robot not initialized." << std::endl;
            return hresult_t::ERROR_INIT_FAILED;
        }

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

    hresult_t Robot::getSensorsOptions(configHolder_t & sensorsOptions) const
    {
        if (!isInitialized_)
        {
            std::cout << "Error - Robot::getSensorsOptions - Robot not initialized." << std::endl;
            return hresult_t::ERROR_INIT_FAILED;
        }

        sensorsOptions.clear();
        for (auto const & sensorGroup : sensorsGroupHolder_)
        {
            configHolder_t sensorsGroupOptions;
            for (auto const & sensor : sensorGroup.second)
            {
                sensorsGroupOptions[sensor->getName()] = sensor->getOptions();
            }
            sensorsOptions[sensorGroup.first] = sensorsGroupOptions;
        }

        return hresult_t::SUCCESS;
    }

    hresult_t Robot::getSensorOptions(std::string    const & sensorType,
                                      std::string    const & sensorName,
                                      configHolder_t       & sensorOptions) const
    {
        if (!isInitialized_)
        {
            std::cout << "Error - Robot::getSensorOptions - Robot not initialized." << std::endl;
            return hresult_t::ERROR_INIT_FAILED;
        }

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

    hresult_t Robot::setTelemetryOptions(configHolder_t const & telemetryOptions)
    {
        if (getIsLocked())
        {
            std::cout << "Error - Robot::setTelemetryOptions - Robot is locked, probably because a simulation is running.";
            std::cout << " Please stop it before updating the telemetry options." << std::endl;
            return hresult_t::ERROR_INIT_FAILED;
        }

        if (!isInitialized_)
        {
            std::cout << "Error - Robot::setTelemetryOptions - Robot not initialized." << std::endl;
            return hresult_t::ERROR_INIT_FAILED;
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

    hresult_t Robot::getTelemetryOptions(configHolder_t & telemetryOptions) const
    {
        if (!isInitialized_)
        {
            std::cout << "Error - Robot::getTelemetryOptions - Robot not initialized." << std::endl;
            return hresult_t::ERROR_INIT_FAILED;
        }

        telemetryOptions.clear();
        for (auto const & sensorGroupTelemetryOption : sensorTelemetryOptions_)
        {
            std::string optionTelemetryName = "enable" + sensorGroupTelemetryOption.first + "s";
            telemetryOptions[optionTelemetryName] = sensorGroupTelemetryOption.second;
        }

        return hresult_t::SUCCESS;
    }

    hresult_t Robot::setOptions(configHolder_t mdlOptions)
    {
        if (getIsLocked())
        {
            std::cout << "Error - Robot::setOptions - Robot is locked, probably because a simulation is running.";
            std::cout << " Please stop it before updating the robot options." << std::endl;
            return hresult_t::ERROR_INIT_FAILED;
        }

        bool_t internalBuffersMustBeUpdated = false;
        bool_t isFlexibleModelInvalid = false;
        bool_t isCurrentModelInvalid = false;
        if (isInitialized_)
        {
            /* Check that the following user parameters has the right dimension,
               then update the required internal buffers to reflect changes, if any. */
            configHolder_t & jointOptionsHolder =
                boost::get<configHolder_t>(mdlOptions.at("joints"));
            if (!boost::get<bool_t>(jointOptionsHolder.at("positionLimitFromUrdf")))
            {
                vectorN_t & positionLimitMin = boost::get<vectorN_t>(jointOptionsHolder.at("positionLimitMin"));
                if ((int32_t) rigidJointsPositionIdx_.size() != positionLimitMin.size())
                {
                    std::cout << "Error - Robot::setOptions - Wrong vector size for 'positionLimitMin'." << std::endl;
                    return hresult_t::ERROR_BAD_INPUT;
                }
                vectorN_t positionLimitMinDiff = positionLimitMin - mdlOptions_->joints.positionLimitMin;
                internalBuffersMustBeUpdated |= (positionLimitMinDiff.array().abs() >= EPS).all();
                vectorN_t & positionLimitMax = boost::get<vectorN_t>(jointOptionsHolder.at("positionLimitMax"));
                if ((uint32_t) rigidJointsPositionIdx_.size() != positionLimitMax.size())
                {
                    std::cout << "Error - Robot::setOptions - Wrong vector size for 'positionLimitMax'." << std::endl;
                    return hresult_t::ERROR_BAD_INPUT;
                }
                vectorN_t positionLimitMaxDiff = positionLimitMax - mdlOptions_->joints.positionLimitMax;
                internalBuffersMustBeUpdated |= (positionLimitMaxDiff.array().abs() >= EPS).all();
            }
            if (!boost::get<bool_t>(jointOptionsHolder.at("velocityLimitFromUrdf")))
            {
                vectorN_t & velocityLimit = boost::get<vectorN_t>(jointOptionsHolder.at("velocityLimit"));
                if ((int32_t) rigidJointsVelocityIdx_.size() != velocityLimit.size())
                {
                    std::cout << "Error - Robot::setOptions - Wrong vector size for 'velocityLimit'." << std::endl;
                    return hresult_t::ERROR_BAD_INPUT;
                }
                vectorN_t velocityLimitDiff = velocityLimit - mdlOptions_->joints.velocityLimit;
                internalBuffersMustBeUpdated |= (velocityLimitDiff.array().abs() >= EPS).all();
            }

            // Check if the flexible model and its associated proxies must be regenerated
            configHolder_t & dynOptionsHolder =
                boost::get<configHolder_t>(mdlOptions.at("dynamics"));
            bool_t const & enableFlexibleModel = boost::get<bool_t>(dynOptionsHolder.at("enableFlexibleModel"));
            flexibilityConfig_t const & flexibilityConfig =
                boost::get<flexibilityConfig_t>(dynOptionsHolder.at("flexibilityConfig"));

            if (mdlOptions_
            && (flexibilityConfig.size() != mdlOptions_->dynamics.flexibilityConfig.size()
                || !std::equal(flexibilityConfig.begin(),
                                flexibilityConfig.end(),
                                mdlOptions_->dynamics.flexibilityConfig.begin())))
            {
                isFlexibleModelInvalid = true;
            }
            else if (mdlOptions_ && enableFlexibleModel != mdlOptions_->dynamics.enableFlexibleModel)
            {
                isCurrentModelInvalid = true;
            }
        }

        // Update the internal options
        mdlOptionsHolder_ = mdlOptions;

        // Create a fast struct accessor
        mdlOptions_ = std::make_unique<modelOptions_t const>(mdlOptionsHolder_);

        if (isFlexibleModelInvalid)
        {
            // Force flexible model regeneration
            generateModelFlexible();
        }

        if (isFlexibleModelInvalid || isCurrentModelInvalid)
        {
            // Trigger biased model regeneration
            reset();
        }
        else if (internalBuffersMustBeUpdated)
        {
            // Update the info extracted from the model
            refreshProxies();
        }

        return hresult_t::SUCCESS;
    }

    configHolder_t Robot::getOptions(void) const
    {
        return mdlOptionsHolder_;
    }

    bool_t const & Robot::getIsInitialized(void) const
    {
        return isInitialized_;
    }

    bool_t const & Robot::getIsTelemetryConfigured(void) const
    {
        return isTelemetryConfigured_;
    }

    std::string const & Robot::getUrdfPath(void) const
    {
        return urdfPath_;
    }

    bool_t const & Robot::getHasFreeflyer(void) const
    {
        return hasFreeflyer_;
    }

    hresult_t Robot::loadUrdfModel(std::string const & urdfPath,
                                   bool_t      const & hasFreeflyer)
    {
        if (!std::ifstream(urdfPath.c_str()).good())
        {
            std::cout << "Error - Robot::loadUrdfModel - The URDF file does not exist. Impossible to load it." << std::endl;
            return hresult_t::ERROR_BAD_INPUT;
        }

        urdfPath_ = urdfPath;
        hasFreeflyer_ = hasFreeflyer;

        try
        {
            pinocchio::Model pncModel;
            if (hasFreeflyer)
            {
                pinocchio::urdf::buildModel(urdfPath,
                                            pinocchio::JointModelFreeFlyer(),
                                            pncModel);
            }
            else
            {
                pinocchio::urdf::buildModel(urdfPath, pncModel);
            }
            pncModel_ = pncModel;
        }
        catch (std::exception& e)
        {
            std::cout << "Error - Robot::loadUrdfModel - Something is wrong with the URDF. Impossible to build a model from it." << std::endl;
            return hresult_t::ERROR_BAD_INPUT;
        }

        return hresult_t::SUCCESS;
    }

    void Robot::computeMotorsTorques(float64_t const & t,
                                     vectorN_t const & q,
                                     vectorN_t const & v,
                                     vectorN_t const & a,
                                     vectorN_t const & u)
    {
        (*motorsHolder_.begin())->computeAllEffort(t, q, v, a, u);
    }

    vectorN_t const & Robot::getMotorsTorques(void) const
    {
        return (*motorsHolder_.begin())->getAll();
    }

    float64_t const & Robot::getMotorTorque(std::string const & motorName) const
    {
        // TODO : it should handle the case where motorName is not found
        auto motorIt = std::find_if(motorsHolder_.begin(), motorsHolder_.end(),
                                    [&motorName](auto const & elem)
                                    {
                                        return (elem->getName() == motorName);
                                    });
        return (*motorIt)->get();
    }

    void Robot::setSensorsData(float64_t const & t,
                               vectorN_t const & q,
                               vectorN_t const & v,
                               vectorN_t const & a,
                               vectorN_t const & u)
    {
        for (auto const & sensorGroup : sensorsGroupHolder_)
        {
            if (!sensorGroup.second.empty())
            {
                (*sensorGroup.second.begin())->setAll(t, q, v, a, u);
            }
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
                dataType.emplace(sensor->getName(),
                                 sensor->getIdx(),
                                 sensor->get());
            }

            data.emplace(std::piecewise_construct,
                         std::forward_as_tuple(sensorGroup.first),
                         std::forward_as_tuple(std::move(dataType)));
        }
        return data;
    }

    matrixN_t Robot::getSensorsData(std::string const & sensorType) const
    {
        return (*sensorsGroupHolder_.at(sensorType).begin())->getAll();
    }

    vectorN_t Robot::getSensorData(std::string const & sensorType,
                                   std::string const & sensorName) const
    {
        // TODO : it should handle the case where sensorType/sensorName is not found
        auto & sensorGroup = sensorsGroupHolder_.at(sensorType);
        auto sensorIt = std::find_if(sensorGroup.begin(),
                                     sensorGroup.end(),
                                     [&sensorName](auto const & elem)
                                     {
                                         return (elem->getName() == sensorName);
                                     });
        return *(*sensorIt)->get();
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

    std::vector<std::string> const & Robot::getContactFramesNames(void) const
    {
        return contactFramesNames_;
    }

    std::vector<int32_t> const & Robot::getContactFramesIdx(void) const
    {
        return contactFramesIdx_;
    }

    std::vector<std::string> const & Robot::getMotorsNames(void) const
    {
        return motorsNames_;
    }

    std::vector<int32_t> Robot::getMotorsModelIdx(void) const
    {
        std::vector<int32_t> motorsModelIdx;
        motorsModelIdx.reserve(motorsHolder_.size());
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
        motorsPositionIdx.reserve(motorsHolder_.size());
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
        motorsVelocityIdx.reserve(motorsHolder_.size());
        std::transform(motorsHolder_.begin(), motorsHolder_.end(),
                       std::back_inserter(motorsVelocityIdx),
                       [](auto const & elem) -> int32_t
                       {
                           return elem->getJointVelocityIdx();
                       });
        return motorsVelocityIdx;
    }

    std::unordered_map<std::string, std::vector<std::string> > Robot::getSensorsNames(void) const
    {
        std::unordered_map<std::string, std::vector<std::string> > sensorNames;
        for (auto const & sensorGroup : sensorsGroupHolder_)
        {
            sensorNames.insert({sensorGroup.first, getSensorsNames(sensorGroup.first)});
        }
        return sensorNames;
    }

    std::vector<std::string> Robot::getSensorsNames(std::string const & sensorType) const
    {
        std::vector<std::string> sensorsNames;
        auto sensorGroupIt = sensorsGroupHolder_.find(sensorType);
        if (sensorGroupIt != sensorsGroupHolder_.end())
        {
            sensorsNames.reserve(sensorGroupIt->second.size());
            std::transform(sensorGroupIt->second.begin(), sensorGroupIt->second.end(),
                           std::back_inserter(sensorsNames),
                           [](auto const & elem) -> std::string
                           {
                               return elem->getName();
                           });
        }
        return sensorsNames;
    }

    std::vector<std::string> const & Robot::getPositionFieldNames(void) const
    {
        return positionFieldNames_;
    }

    vectorN_t const & Robot::getPositionLimitMin(void) const
    {
        return positionLimitMin_;
    }

    vectorN_t const & Robot::getPositionLimitMax(void) const
    {
        return positionLimitMax_;
    }

    std::vector<std::string> const & Robot::getVelocityFieldNames(void) const
    {
        return velocityFieldNames_;
    }

    vectorN_t const & Robot::getVelocityLimit(void) const
    {
        return velocityLimit_;
    }

    vectorN_t Robot::getTorqueLimit(void) const
    {
        vectorN_t torqueLimit = vectorN_t::Zero(pncModel_.nv);
        for (auto const & motor : motorsHolder_)
        {
            auto const & motorOptions = motor->baseMotorOptions_;
            int32_t const & motorsVelocityIdx = motor->getJointVelocityIdx();
            if (motorOptions->enableTorqueLimit)
            {
                torqueLimit[motorsVelocityIdx] = motor->getTorqueLimit();
            }
            else
            {
                torqueLimit[motorsVelocityIdx] = -1;
            }
        }
        return torqueLimit;
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

    std::vector<std::string> const & Robot::getAccelerationFieldNames(void) const
    {
        return accelerationFieldNames_;
    }

    std::vector<std::string> const & Robot::getMotorTorqueFieldNames(void) const
    {
        return motorTorqueFieldNames_;
    }

    std::vector<std::string> const & Robot::getRigidJointsNames(void) const
    {
        return rigidJointsNames_;
    }

    std::vector<int32_t> const & Robot::getRigidJointsModelIdx(void) const
    {
        return rigidJointsModelIdx_;
    }

    std::vector<int32_t> const & Robot::getRigidJointsPositionIdx(void) const
    {
        return rigidJointsPositionIdx_;
    }

    std::vector<int32_t> const & Robot::getRigidJointsVelocityIdx(void) const
    {
        return rigidJointsVelocityIdx_;
    }

    std::vector<std::string> const & Robot::getFlexibleJointsNames(void) const
    {
        static std::vector<std::string> const flexibleJointsNamesEmpty {};
        if (mdlOptions_->dynamics.enableFlexibleModel)
        {
            return flexibleJointsNames_;
        }
        else
        {
            return flexibleJointsNamesEmpty;
        }
    }

    std::vector<int32_t> const & Robot::getFlexibleJointsModelIdx(void) const
    {
        static std::vector<int32_t> const flexibleJointsModelIdxEmpty {};
        if (mdlOptions_->dynamics.enableFlexibleModel)
        {
            return flexibleJointsModelIdx_;
        }
        else
        {
            return flexibleJointsModelIdxEmpty;
        }
    }

    uint32_t const & Robot::nq(void) const
    {
        return nq_;
    }

    uint32_t const & Robot::nv(void) const
    {
        return nv_;
    }

    uint32_t const & Robot::nx(void) const
    {
        return nx_;
    }
}
