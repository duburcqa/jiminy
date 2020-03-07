
#include <fstream>
#include <exception>

#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/frames.hpp"

#include "jiminy/core/TelemetryData.h"
#include "jiminy/core/AbstractSensor.h"
#include "jiminy/core/Model.h"


namespace jiminy
{
    Model::Model(void) :
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
    motorsModelIdx_(),
    motorsPositionIdx_(),
    motorsVelocityIdx_(),
    rigidJointsNames_(),
    rigidJointsModelIdx_(),
    rigidJointsPositionIdx_(),
    rigidJointsVelocityIdx_(),
    flexibleJointsNames_(),
    flexibleJointsModelIdx_(),
    positionLimitMin_(),
    positionLimitMax_(),
    velocityLimit_(),
    torqueLimit_(),
    positionFieldNames_(),
    velocityFieldNames_(),
    accelerationFieldNames_(),
    motorTorqueFieldNames_(),
    mutexLocal_(),
    pncModelFlexibleOrig_(),
    motorsSharedHolder_(),
    sensorsSharedHolder_(),
    nq_(0),
    nv_(0),
    nx_(0)
    {
        setOptions(getDefaultOptions());
    }

    result_t Model::initialize(std::string const & urdfPath,
                               bool_t      const & hasFreeflyer)
    {
        result_t returnCode = result_t::SUCCESS;

        // Remove all sensors, if any
        sensorsGroupHolder_.clear();
        sensorsSharedHolder_.clear();
        sensorTelemetryOptions_.clear();

        // Initialize the URDF model
        returnCode = loadUrdfModel(urdfPath, hasFreeflyer);
        isInitialized_ = true;

        if (returnCode == result_t::SUCCESS)
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

        if (returnCode == result_t::SUCCESS)
        {
            // Create the flexible model
            returnCode = generateModelFlexible();
        }

        if (returnCode == result_t::SUCCESS)
        {
            // Add biases to the dynamics properties of the model
            returnCode = generateModelBiased();
        }

        if (returnCode != result_t::SUCCESS)
        {
            // Set the initialization flag
            isInitialized_ = false;
        }

        return returnCode;
    }

    void Model::reset(void)
    {
        if (isInitialized_)
        {
            // Update the biases added to the dynamics properties of the model.
            generateModelBiased();
        }

        // Reset the sensors
        for (sensorsGroupHolder_t::value_type & sensorGroup : sensorsGroupHolder_)
        {
            for (sensorsHolder_t::value_type & sensor : sensorGroup.second)
            {
                sensor.second->reset();
            }
        }

        // Reset the telemetry state
        isTelemetryConfigured_ = false;
    }

    result_t Model::configureTelemetry(std::shared_ptr<TelemetryData> const & telemetryData)
    {
        result_t returnCode = result_t::SUCCESS;

        if (!isInitialized_)
        {
            std::cout << "Error - Model::configureTelemetry - The model is not initialized." << std::endl;
            returnCode = result_t::ERROR_INIT_FAILED;
        }

        if (returnCode == result_t::SUCCESS)
        {
            telemetryData_ = std::shared_ptr<TelemetryData>(telemetryData);
        }

        if (returnCode == result_t::SUCCESS)
        {
            if (!isTelemetryConfigured_)
            {
                for (sensorsGroupHolder_t::value_type const & sensorGroup : sensorsGroupHolder_)
                {
                    for (sensorsHolder_t::value_type const & sensor : sensorGroup.second)
                    {
                        if (returnCode == result_t::SUCCESS)
                        {
                            if (sensorTelemetryOptions_.at(sensorGroup.first))
                            {
                                returnCode = sensor.second->configureTelemetry(telemetryData_);
                            }
                        }
                    }
                }
            }
        }

        if (returnCode == result_t::SUCCESS)
        {
            isTelemetryConfigured_ = true;
        }

        return returnCode;
    }

    result_t Model::addContactPoints(std::vector<std::string> const & frameNames)
    {
        if (getIsLocked())
        {
            std::cout << "Error - Model::addContactPoints - Model is locked, probably because a simulation is running.";
            std::cout << " Please stop it before adding contact points." << std::endl;
            return result_t::ERROR_GENERIC;
        }

        if (!isInitialized_)
        {
            std::cout << "Error - Model::addContactPoints - Model not initialized." << std::endl;
            return result_t::ERROR_INIT_FAILED;
        }

        // Make sure that no frame in the list are duplicates
        if (checkDuplicates(frameNames))
        {
            std::cout << "Error - Model::addContactPoints - Some frames in the list are duplicates." << std::endl;
            return result_t::ERROR_BAD_INPUT;
        }

        // Make sure that no motor is associated with any of the joint in the list
        if (checkIntersection(contactFramesNames_, frameNames))
        {
            std::cout << "Error - Model::addContactPoints - At least one of the frame in the list is already been associated with a contact point." << std::endl;
            return result_t::ERROR_BAD_INPUT;
        }

        // Make sure that all the frames in the list exist
        for (std::string const & frame : frameNames)
        {
            if (!pncModel_.existFrame(frame))
            {
                std::cout << "Error - Model::addContactPoints - At least one of the frame in the list does not exist." << std::endl;
                return result_t::ERROR_BAD_INPUT;
            }
        }

        if (!frameNames.empty())
        {
            // Add the list of frames to the set of contact points
            contactFramesNames_.insert(contactFramesNames_.end(), frameNames.begin(), frameNames.end());

            // Reset the contact force internal buffer
            contactForces_ = forceVector_t(contactFramesNames_.size(), pinocchio::Force::Zero());

            // Refresh the attributes of the model
            refreshModelProxies();
        }

        return result_t::SUCCESS;
    }

    result_t Model::removeContactPoints(std::vector<std::string> const & frameNames)
    {
        if (getIsLocked())
        {
            std::cout << "Error - Model::removeContactPoints - Model is locked, probably because a simulation is running.";
            std::cout << " Please stop it before removing contact points." << std::endl;
            return result_t::ERROR_GENERIC;
        }

        if (!isInitialized_)
        {
            std::cout << "Error - Model::removeContactPoints - Model not initialized." << std::endl;
            return result_t::ERROR_INIT_FAILED;
        }

        // Make sure that no frame in the list are duplicates
        if (checkDuplicates(frameNames))
        {
            std::cout << "Error - Model::removeContactPoints - Some frames in the list are duplicates." << std::endl;
            return result_t::ERROR_BAD_INPUT;
        }

        // Make sure that no motor is associated with any of the joint in jointNames
        if (!checkInclusion(contactFramesNames_, frameNames))
        {
            std::cout << "Error - Model::removeContactPoints - At least one of the frame in the list is not associated with any contact point." << std::endl;
            return result_t::ERROR_BAD_INPUT;
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

        // Refresh the attributes of the model
        refreshModelProxies();

        return result_t::SUCCESS;
    }

    result_t Model::addMotors(std::vector<std::string> const & jointNames)
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

        // Make sure that no joint in the list are duplicates
        if (checkDuplicates(jointNames))
        {
            std::cout << "Error - Model::addMotors - Some joints in the list are duplicates." << std::endl;
            return result_t::ERROR_BAD_INPUT;
        }

        // Make sure that no motor is associated with any of the joint in jointNames
        if (checkIntersection(motorsNames_, jointNames))
        {
            std::cout << "Error - Model::addMotors - At least one of the joint in the list is already associated with a motor." << std::endl;
            return result_t::ERROR_BAD_INPUT;
        }

        // Make sure that all the joints in the list exist
        for (std::string const & joint : jointNames)
        {
            if (!pncModel_.existJointName(joint))
            {
                std::cout << "Error - Model::addContactPoints - At least one of the joint in the list does not exist." << std::endl;
                return result_t::ERROR_BAD_INPUT;
            }
        }

        if (!jointNames.empty())
        {
            // Add the list of joints to the set of motors
            motorsNames_.insert(motorsNames_.end(), jointNames.begin(), jointNames.end());

            /* Make sure that no user-specified torque limits or motor inertias are used,
                since it does not make sense to add motors otherwise */
            if (!mdlOptions_->joints.torqueLimitFromUrdf || mdlOptions_->joints.enableMotorInertia)
            {
                std::cout << "Warning - Model::addMotors - Be careful, adding motors disable user-specified torque limits and motor inertia to prevent dimensional mismatch." << std::endl;
                configHolder_t mdlOptions = getOptions();
                configHolder_t & jointOptionsHolder = boost::get<configHolder_t>(mdlOptions.at("joints"));
                boost::get<bool_t>(jointOptionsHolder.at("torqueLimitFromUrdf")) = true;
                boost::get<bool_t>(jointOptionsHolder.at("enableMotorInertia")) = false;
                setOptions(mdlOptions);
            }

            // Refresh the attributes of the model
            refreshModelProxies();
        }

        return result_t::SUCCESS;
    }

    result_t Model::removeMotors(std::vector<std::string> const & jointNames)
    {
        if (getIsLocked())
        {
            std::cout << "Error - Model::removeMotors - Model is locked, probably because a simulation is running.";
            std::cout << " Please stop it before removing motors." << std::endl;
            return result_t::ERROR_GENERIC;
        }

        if (!isInitialized_)
        {
            std::cout << "Error - Model::removeMotors - Model not initialized." << std::endl;
            return result_t::ERROR_INIT_FAILED;
        }

        // Make sure that no joint in the list are duplicates
        if (checkDuplicates(jointNames))
        {
            std::cout << "Error - Model::removeMotors - Some joints in the list are duplicates." << std::endl;
            return result_t::ERROR_BAD_INPUT;
        }

        // Make sure that no motor is associated with any of the joint in jointNames
        if (!checkInclusion(motorsNames_, jointNames))
        {
            std::cout << "Error - Model::removeMotors - At least one of the joint in the list is not associated with any motor." << std::endl;
            return result_t::ERROR_BAD_INPUT;
        }

        // Remove the list of joints from the set of motors
        if (!jointNames.empty())
        {
            eraseVector(motorsNames_, jointNames);
        }
        else
        {
            motorsNames_.clear();
        }

        /* Make sure that no user-specified torque limits or motor inertias are used,
            since it does not make sense to add motors otherwise */
        if (!mdlOptions_->joints.torqueLimitFromUrdf || mdlOptions_->joints.enableMotorInertia)
        {
            std::cout << "Warning - Model::removeMotors - Be careful, adding motors disable user-specified torque limits and motor inertia to prevent dimensional mismatch." << std::endl;
            configHolder_t mdlOptions = getOptions();
            configHolder_t & jointOptionsHolder = boost::get<configHolder_t>(mdlOptions.at("joints"));
            boost::get<bool_t>(jointOptionsHolder.at("torqueLimitFromUrdf")) = true;
            boost::get<bool_t>(jointOptionsHolder.at("enableMotorInertia")) = false;
            setOptions(mdlOptions);
        }

        // Refresh the attributes of the model
        refreshModelProxies();

        return result_t::SUCCESS;
    }

    result_t Model::removeSensor(std::string const & sensorType,
                                 std::string const & sensorName)
    {
        if (getIsLocked())
        {
            std::cout << "Error - Model::removeSensor - Model is locked, probably because a simulation is running.";
            std::cout << " Please stop it before removing sensors." << std::endl;
            return result_t::ERROR_GENERIC;
        }

        if (!isInitialized_)
        {
            std::cout << "Error - Model::removeSensor - Model not initialized." << std::endl;
            return result_t::ERROR_INIT_FAILED;
        }

        sensorsGroupHolder_t::iterator sensorGroupIt;
        sensorGroupIt = sensorsGroupHolder_.find(sensorType);
        if (sensorGroupIt == sensorsGroupHolder_.end())
        {
            std::cout << "Error - Model::removeSensor - This type of sensor does not exist." << std::endl;
            return result_t::ERROR_BAD_INPUT;
        }

        sensorsHolder_t::iterator sensorIt;
        sensorIt = sensorGroupIt->second.find(sensorName);
        if (sensorIt == sensorGroupIt->second.end())
        {
            std::cout << "Error - Model::removeSensors - No sensor with this type and name exists." << std::endl;
            return result_t::ERROR_BAD_INPUT;
        }

        // Remove the sensor from its group
        sensorGroupIt->second.erase(sensorIt);

        // Remove the sensor group if there is no more sensors left.
        if (sensorGroupIt->second.empty())
        {
            sensorsGroupHolder_.erase(sensorType);
            sensorsSharedHolder_.erase(sensorType);
            sensorTelemetryOptions_.erase(sensorType);
        }

        return result_t::SUCCESS;
    }

    result_t Model::removeSensors(std::string const & sensorType)
    {
        if (getIsLocked())
        {
            std::cout << "Error - Model::removeSensors - Model is locked, probably because a simulation is running.";
            std::cout << " Please stop it before removing sensors." << std::endl;
            return result_t::ERROR_GENERIC;
        }

        if (!isInitialized_)
        {
            std::cout << "Error - Model::removeSensors - Model not initialized." << std::endl;
            return result_t::ERROR_INIT_FAILED;
        }

        if (!sensorType.empty())
        {
            sensorsGroupHolder_t::iterator sensorGroupIt;
            sensorGroupIt = sensorsGroupHolder_.find(sensorType);
            if (sensorGroupIt == sensorsGroupHolder_.end())
            {
                std::cout << "Error - Model::removeSensors - No sensor with this type exists." << std::endl;
                return result_t::ERROR_BAD_INPUT;
            }

            sensorsGroupHolder_.erase(sensorGroupIt);
            sensorsSharedHolder_.erase(sensorType);
            sensorTelemetryOptions_.erase(sensorType);
        }
        else
        {
            sensorsGroupHolder_.clear();
            sensorsSharedHolder_.clear();
            sensorTelemetryOptions_.clear();
        }

        return result_t::SUCCESS;
    }

    result_t Model::generateModelFlexible(void)
    {
        result_t returnCode = result_t::SUCCESS;

        if (!isInitialized_)
        {
            std::cout << "Error - Model::generateModelFlexible - Model not initialized." << std::endl;
            returnCode = result_t::ERROR_INIT_FAILED;
        }

        if (returnCode == result_t::SUCCESS)
        {
            flexibleJointsNames_.clear();
            flexibleJointsModelIdx_.clear();
            pncModelFlexibleOrig_ = pncModelRigidOrig_;
            for(flexibleJointData_t const & flexibleJoint : mdlOptions_->dynamics.flexibilityConfig)
            {
                std::string const & jointName = flexibleJoint.jointName;

                // Look if given joint exists in the joint list.
                if(returnCode == result_t::SUCCESS)
                {
                    int32_t jointIdx;
                    returnCode = getJointPositionIdx(pncModel_, jointName, jointIdx);
                }

                // Add joints to model.
                if(returnCode == result_t::SUCCESS)
                {
                    std::string newName =
                        removeFieldnameSuffix(jointName, "Joint") + FLEXIBLE_JOINT_SUFFIX;
                    flexibleJointsNames_.emplace_back(newName);
                    insertFlexibilityInModel(pncModelFlexibleOrig_, jointName, newName); // Ignore return code, as check has already been done.
                }
            }
        }

        if (returnCode == result_t::SUCCESS)
        {
            getJointsModelIdx(pncModelFlexibleOrig_,
                              flexibleJointsNames_,
                              flexibleJointsModelIdx_);
        }

        return returnCode;
    }

    result_t Model::generateModelBiased(void)
    {
        result_t returnCode = result_t::SUCCESS;

        if (!isInitialized_)
        {
            std::cout << "Error - Model::generateModelBiased - Model not initialized." << std::endl;
            returnCode = result_t::ERROR_INIT_FAILED;
        }

        if (returnCode == result_t::SUCCESS)
        {
            // Reset the model either with the original rigid or flexible model
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

            // Initialize the internal proxies
            returnCode = refreshModelProxies();
        }

        return returnCode;
    }

    result_t Model::refreshModelProxies(void)
    {
        result_t returnCode = result_t::SUCCESS;

        if (!isInitialized_)
        {
            std::cout << "Error - Model::updateFieldnames - Model not initialized." << std::endl;
            returnCode = result_t::ERROR_INIT_FAILED;
        }

        if (returnCode == result_t::SUCCESS)
        {
            // Extract the dimensions of the configuration and velocity vectors
            nq_ = pncModel_.nq;
            nv_ = pncModel_.nv;
            nx_ = nq_ + nv_;

            // Extract some rigid joints indices in the model
            getJointsModelIdx(pncModel_, rigidJointsNames_, rigidJointsModelIdx_);
            getJointsPositionIdx(pncModel_, rigidJointsNames_, rigidJointsPositionIdx_, false);
            getJointsVelocityIdx(pncModel_, rigidJointsNames_, rigidJointsVelocityIdx_, false);

            // Extract the contact frames indices in the model
            getFramesIdx(pncModel_, contactFramesNames_, contactFramesIdx_);

            // Extract some motor indices in the model
            getJointsModelIdx(pncModel_, motorsNames_, motorsModelIdx_);
            getJointsPositionIdx(pncModel_, motorsNames_, motorsPositionIdx_, true);
            getJointsVelocityIdx(pncModel_, motorsNames_, motorsVelocityIdx_, true);

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
                    if (returnCode == result_t::SUCCESS)
                    {
                        returnCode = getJointTypeFromId(pncModel_, jointIdx, jointType);
                    }
                    if (returnCode == result_t::SUCCESS)
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
                    if (returnCode == result_t::SUCCESS)
                    {
                        returnCode = getJointTypePositionSuffixes(jointType,
                                                                  jointTypePositionSuffixes);
                    }
                    if (returnCode == result_t::SUCCESS)
                    {
                        for (std::string const & suffix : jointTypePositionSuffixes)
                        {
                            jointPositionFieldnames.emplace_back(
                                jointPrefix + "Position" + jointShortNames[i] + suffix);
                        }
                    }
                    if (returnCode == result_t::SUCCESS)
                    {
                        std::copy(jointPositionFieldnames.begin(),
                                  jointPositionFieldnames.end(),
                                  positionFieldNames_.begin() + idx_q);
                    }

                    std::vector<std::string> jointTypeVelocitySuffixes;
                    std::vector<std::string> jointVelocityFieldnames;
                    std::vector<std::string> jointAccelerationFieldnames;
                    if (returnCode == result_t::SUCCESS)
                    {
                        returnCode = getJointTypeVelocitySuffixes(jointType,
                                                                  jointTypeVelocitySuffixes);
                    }
                    if (returnCode == result_t::SUCCESS)
                    {
                        for (std::string const & suffix : jointTypeVelocitySuffixes)
                        {
                            jointVelocityFieldnames.emplace_back(
                                jointPrefix + "Velocity" + jointShortNames[i] + suffix);
                            jointAccelerationFieldnames.emplace_back(
                                jointPrefix + "Acceleration" + jointShortNames[i] + suffix);
                        }
                    }
                    if (returnCode == result_t::SUCCESS)
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

        if (returnCode == result_t::SUCCESS)
        {
            // Generate the fieldnames associated with the motor torques
            motorTorqueFieldNames_.clear();
            for (std::string const & jointName : removeFieldnamesSuffix(motorsNames_, "Joint"))
            {
                motorTorqueFieldNames_.emplace_back(JOINT_PREFIX_BASE + "Torque" + jointName);
            }

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

            // Get the motor torque limits from the URDF or the user options
            if (mdlOptions_->joints.torqueLimitFromUrdf)
            {
                // effortLimit is given in the velocity vector space
                torqueLimit_.resize(motorsVelocityIdx_.size());
                for (uint32_t i=0; i < motorsVelocityIdx_.size(); ++i)
                {
                    torqueLimit_[i] = pncModel_.effortLimit[motorsVelocityIdx_[i]];
                }
            }
            else
            {
                torqueLimit_ = mdlOptions_->joints.torqueLimit;
            }

            // Get the motor inertia from the URDF or the user options
            pncModel_.rotorInertia.setConstant(0.0);
            if (mdlOptions_->joints.enableMotorInertia)
            {
                for (uint32_t i=0; i < motorsVelocityIdx_.size(); ++i)
                {
                    pncModel_.rotorInertia[motorsVelocityIdx_[i]] = mdlOptions_->joints.motorInertia[i];
                }
            }
        }

        return returnCode;
    }

    result_t Model::getSensorsOptions(std::string    const & sensorType,
                                      configHolder_t       & sensorsOptions) const
    {
        if (!isInitialized_)
        {
            std::cout << "Error - Model::getSensorsOptions - Model not initialized." << std::endl;
            return result_t::ERROR_INIT_FAILED;
        }

        sensorsGroupHolder_t::const_iterator sensorGroupIt;
        sensorGroupIt = sensorsGroupHolder_.find(sensorType);
        if (sensorGroupIt == sensorsGroupHolder_.end())
        {
            std::cout << "Error - Model::getSensorsOptions - This type of sensor does not exist." << std::endl;
            return result_t::ERROR_BAD_INPUT;
        }
        sensorsOptions = configHolder_t();
        for (sensorsHolder_t::value_type const & sensor : sensorGroupIt->second)
        {
            sensorsOptions[sensor.first] = sensor.second->getOptions();
        }

        return result_t::SUCCESS;
    }

    result_t Model::getSensorsOptions(configHolder_t & sensorsOptions) const
    {
        if (!isInitialized_)
        {
            std::cout << "Error - Model::getSensorsOptions - Model not initialized." << std::endl;
            return result_t::ERROR_INIT_FAILED;
        }

        sensorsOptions = configHolder_t();
        for (sensorsGroupHolder_t::value_type const & sensorGroup : sensorsGroupHolder_)
        {
            configHolder_t sensorsGroupOptions;
            for (sensorsHolder_t::value_type const & sensor : sensorGroup.second)
            {
                sensorsGroupOptions[sensor.first] = sensor.second->getOptions();
            }
            sensorsOptions[sensorGroup.first] = sensorsGroupOptions;
        }

        return result_t::SUCCESS;
    }

    result_t Model::getSensorOptions(std::string    const & sensorType,
                                     std::string    const & sensorName,
                                     configHolder_t       & sensorOptions) const
    {
        if (!isInitialized_)
        {
            std::cout << "Error - Model::getSensorOptions - Model not initialized." << std::endl;
            return result_t::ERROR_INIT_FAILED;
        }

        sensorsGroupHolder_t::const_iterator sensorGroupIt;
        sensorGroupIt = sensorsGroupHolder_.find(sensorType);
        if (sensorGroupIt == sensorsGroupHolder_.end())
        {
            std::cout << "Error - Model::getSensorOptions - This type of sensor does not exist." << std::endl;
            return result_t::ERROR_BAD_INPUT;
        }

        sensorsHolder_t::const_iterator sensorIt;
        sensorIt = sensorGroupIt->second.find(sensorName);
        if (sensorIt == sensorGroupIt->second.end())
        {
            std::cout << "Error - Model::getSensorOptions - No sensor with this type and name exists." << std::endl;
            return result_t::ERROR_BAD_INPUT;
        }

        sensorOptions = sensorIt->second->getOptions();

        return result_t::SUCCESS;
    }

    result_t Model::setSensorOptions(std::string    const & sensorType,
                                     std::string    const & sensorName,
                                     configHolder_t const & sensorOptions)
    {
        result_t returnCode = result_t::SUCCESS;

        if (getIsLocked())
        {
            std::cout << "Error - Model::setSensorOptions - Model is locked, probably because a simulation is running.";
            std::cout << " Please stop it before updating the sensor options." << std::endl;
            returnCode = result_t::ERROR_INIT_FAILED;
        }

        if (returnCode == result_t::SUCCESS)
        {
            if (!isInitialized_)
            {
                std::cout << "Error - Model::setSensorOptions - Model not initialized." << std::endl;
                returnCode = result_t::ERROR_INIT_FAILED;
            }
        }

        sensorsGroupHolder_t::iterator sensorGroupIt;
        sensorGroupIt = sensorsGroupHolder_.find(sensorType);
        if (returnCode == result_t::SUCCESS)
        {
            if (sensorGroupIt == sensorsGroupHolder_.end())
            {
                std::cout << "Error - Model::setSensorOptions - This type of sensor does not exist." << std::endl;
                returnCode = result_t::ERROR_BAD_INPUT;
            }
        }

        sensorsHolder_t::iterator sensorIt;
        sensorIt = sensorGroupIt->second.find(sensorName);
        if (returnCode == result_t::SUCCESS)
        {
            if (sensorIt == sensorGroupIt->second.end())
            {
                std::cout << "Error - Model::setSensorOptions - No sensor with this type and name exists." << std::endl;
                returnCode = result_t::ERROR_BAD_INPUT;
            }
        }

        if (returnCode == result_t::SUCCESS)
        {
            returnCode = sensorIt->second->setOptions(sensorOptions);
        }

        return returnCode;
    }

    result_t Model::setSensorsOptions(std::string    const & sensorType,
                                      configHolder_t const & sensorsOptions)
    {
        result_t returnCode = result_t::SUCCESS;

        if (getIsLocked())
        {
            std::cout << "Error - Model::setSensorsOptions - Model is locked, probably because a simulation is running.";
            std::cout << " Please stop it before updating the sensor options." << std::endl;
            returnCode = result_t::ERROR_INIT_FAILED;
        }

        if (returnCode == result_t::SUCCESS)
        {
            if (!isInitialized_)
            {
                std::cout << "Error - Model::setSensorsOptions - Model not initialized." << std::endl;
                returnCode = result_t::ERROR_INIT_FAILED;
            }
        }

        sensorsGroupHolder_t::iterator sensorGroupIt;
        sensorGroupIt = sensorsGroupHolder_.find(sensorType);
        if (returnCode == result_t::SUCCESS)
        {
            if (sensorGroupIt == sensorsGroupHolder_.end())
            {
                std::cout << "Error - Model::setSensorsOptions - This type of sensor does not exist." << std::endl;
                returnCode = result_t::ERROR_BAD_INPUT;
            }
        }

        for (sensorsHolder_t::value_type const & sensor : sensorGroupIt->second)
        {
            if (returnCode == result_t::SUCCESS)
            {
                configHolder_t::const_iterator it = sensorsOptions.find(sensor.first);
                if (it != sensorsOptions.end())
                {
                    returnCode = sensor.second->setOptions(boost::get<configHolder_t>(it->second));
                }
                else
                {
                    returnCode = sensor.second->setOptionsAll(sensorsOptions);
                    break;
                }
            }
        }

        return returnCode;
    }

    result_t Model::setSensorsOptions(configHolder_t const & sensorsOptions)
    {
        result_t returnCode = result_t::SUCCESS;

        if (getIsLocked())
        {
            std::cout << "Error - Model::setSensorsOptions - Model is locked, probably because a simulation is running.";
            std::cout << " Please stop it before updating the sensor options." << std::endl;
            returnCode = result_t::ERROR_INIT_FAILED;
        }

        if (returnCode == result_t::SUCCESS)
        {
            if (!isInitialized_)
            {
                std::cout << "Error - Model::setSensorsOptions - Model not initialized." << std::endl;
                returnCode = result_t::ERROR_INIT_FAILED;
            }
        }

        for (sensorsGroupHolder_t::value_type const & sensorGroup : sensorsGroupHolder_)
        {
            for (sensorsHolder_t::value_type const & sensor : sensorGroup.second)
            {
                if (returnCode == result_t::SUCCESS)
                {
                    // TODO: missing check for sensor type and name availability
                    returnCode = sensor.second->setOptions(boost::get<configHolder_t>(
                        boost::get<configHolder_t>(
                            sensorsOptions.at(sensorGroup.first)).at(sensor.first)));
                }
            }
        }

        return returnCode;
    }

    result_t Model::getTelemetryOptions(configHolder_t & telemetryOptions) const
    {
        if (!isInitialized_)
        {
            std::cout << "Error - Model::setSensorsOptions - Model not initialized." << std::endl;
            return result_t::ERROR_INIT_FAILED;
        }

        telemetryOptions.clear();
        for (auto const & sensorGroupTelemetryOption : sensorTelemetryOptions_)
        {
            std::string optionTelemetryName = "enable" + sensorGroupTelemetryOption.first + "s";
            telemetryOptions[optionTelemetryName] = sensorGroupTelemetryOption.second;
        }

        return result_t::SUCCESS;
    }

    result_t Model::setTelemetryOptions(configHolder_t const & telemetryOptions)
    {
        if (getIsLocked())
        {
            std::cout << "Error - Model::setTelemetryOptions - Model is locked, probably because a simulation is running.";
            std::cout << " Please stop it before updating the telemetry options." << std::endl;
            return result_t::ERROR_INIT_FAILED;
        }

        if (!isInitialized_)
        {
            std::cout << "Error - Model::setTelemetryOptions - Model not initialized." << std::endl;
            return result_t::ERROR_INIT_FAILED;
        }

        for (auto & sensorGroupTelemetryOption : sensorTelemetryOptions_)
        {
            std::string optionTelemetryName = "enable" + sensorGroupTelemetryOption.first + "s";
            configHolder_t::const_iterator sensorTelemetryOptionIt =
                telemetryOptions.find(optionTelemetryName);
            if (sensorTelemetryOptionIt == telemetryOptions.end())
            {
                std::cout << "Error - Model::setTelemetryOptions - Missing field." << std::endl;
                return result_t::ERROR_GENERIC;
            }
            sensorGroupTelemetryOption.second = boost::get<bool_t>(sensorTelemetryOptionIt->second);
        }

        return result_t::SUCCESS;
    }

    configHolder_t Model::getOptions(void) const
    {
        return mdlOptionsHolder_;
    }

    result_t Model::setOptions(configHolder_t mdlOptions)
    {
        if (getIsLocked())
        {
            std::cout << "Error - Model::setOptions - Model is locked, probably because a simulation is running.";
            std::cout << " Please stop it before updating the model options." << std::endl;
            return result_t::ERROR_INIT_FAILED;
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
                if((int32_t) rigidJointsPositionIdx_.size() != positionLimitMin.size())
                {
                    std::cout << "Error - Model::setOptions - Wrong vector size for 'positionLimitMin'." << std::endl;
                    return result_t::ERROR_BAD_INPUT;
                }
                internalBuffersMustBeUpdated |= positionLimitMin != mdlOptions_->joints.positionLimitMin;
                vectorN_t & positionLimitMax = boost::get<vectorN_t>(jointOptionsHolder.at("positionLimitMax"));
                if((uint32_t) rigidJointsPositionIdx_.size() != positionLimitMax.size())
                {
                    std::cout << "Error - Model::setOptions - Wrong vector size for 'positionLimitMax'." << std::endl;
                    return result_t::ERROR_BAD_INPUT;
                }
                internalBuffersMustBeUpdated |= (positionLimitMax != mdlOptions_->joints.positionLimitMax);
            }
            if (!boost::get<bool_t>(jointOptionsHolder.at("velocityLimitFromUrdf")))
            {
                vectorN_t & velocityLimit = boost::get<vectorN_t>(jointOptionsHolder.at("velocityLimit"));
                if((int32_t) rigidJointsVelocityIdx_.size() != velocityLimit.size())
                {
                    std::cout << "Error - Model::setOptions - Wrong vector size for 'velocityLimit'." << std::endl;
                    return result_t::ERROR_BAD_INPUT;
                }
                internalBuffersMustBeUpdated |= velocityLimit != mdlOptions_->joints.velocityLimit;
            }
            if (!boost::get<bool_t>(jointOptionsHolder.at("torqueLimitFromUrdf")))
            {
                vectorN_t & torqueLimit = boost::get<vectorN_t>(jointOptionsHolder.at("torqueLimit"));
                if((int32_t) motorsVelocityIdx_.size() != torqueLimit.size())
                {
                    std::cout << "Error - Model::setOptions - Wrong vector size for 'torqueLimit'." << std::endl;
                    return result_t::ERROR_BAD_INPUT;
                }
                internalBuffersMustBeUpdated |= (torqueLimit != mdlOptions_->joints.torqueLimit);
            }
            if (boost::get<bool_t>(jointOptionsHolder.at("enableMotorInertia")))
            {
                vectorN_t & motorInertia = boost::get<vectorN_t>(jointOptionsHolder.at("motorInertia"));
                if((int32_t) motorsVelocityIdx_.size() != motorInertia.size())
                {
                    std::cout << "Error - Model::setOptions - Wrong vector size for 'motorInertia'." << std::endl;
                    return result_t::ERROR_BAD_INPUT;
                }
                if(!(motorInertia.array() >= 0.0).all())
                {
                    std::cout << "Error - Model::setOptions - Every values in 'motorInertia' must be positive." << std::endl;
                    return result_t::ERROR_BAD_INPUT;
                }
                internalBuffersMustBeUpdated |= (motorInertia != mdlOptions_->joints.motorInertia);
            }

            // Check if the flexible model and its associated proxies must be regenerated
            configHolder_t & dynOptionsHolder =
                boost::get<configHolder_t>(mdlOptions.at("dynamics"));
            bool_t const & enableFlexibleModel = boost::get<bool_t>(dynOptionsHolder.at("enableFlexibleModel"));
            flexibilityConfig_t const & flexibilityConfig =
                boost::get<flexibilityConfig_t>(dynOptionsHolder.at("flexibilityConfig"));

            if(mdlOptions_
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
            refreshModelProxies();
        }

        return result_t::SUCCESS;
    }

    bool_t const & Model::getIsInitialized(void) const
    {
        return isInitialized_;
    }

    bool_t const & Model::getIsTelemetryConfigured(void) const
    {
        return isTelemetryConfigured_;
    }

    std::string const & Model::getUrdfPath(void) const
    {
        return urdfPath_;
    }

    bool_t const & Model::getHasFreeflyer(void) const
    {
        return hasFreeflyer_;
    }

    std::unordered_map<std::string, std::vector<std::string> > Model::getSensorsNames(void) const
    {
        std::unordered_map<std::string, std::vector<std::string> > sensorNames;
        for (sensorsGroupHolder_t::value_type const & sensorGroup : sensorsGroupHolder_)
        {
            for (sensorsHolder_t::value_type const & sensor : sensorGroup.second)
            {
                sensorNames[sensorGroup.first].push_back(sensor.first);
            }
        }
        return sensorNames;
    }

    result_t Model::loadUrdfModel(std::string const & urdfPath,
                                  bool_t      const & hasFreeflyer)
    {
        if (!std::ifstream(urdfPath.c_str()).good())
        {
            std::cout << "Error - Model::loadUrdfModel - The URDF file does not exist. Impossible to load it." << std::endl;
            return result_t::ERROR_BAD_INPUT;
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
            std::cout << "Error - Model::loadUrdfModel - Something is wrong with the URDF. Impossible to build a model from it." << std::endl;
            return result_t::ERROR_BAD_INPUT;
        }

        return result_t::SUCCESS;
    }

    void Model::getSensorsData(sensorsDataMap_t & data) const
    {
        data.clear();
        for (auto & sensorGroupIt : sensorsGroupHolder_)
        {
            sensorDataTypeMap_t dataType;
            for (auto & sensorIt : sensorGroupIt.second)
            {
                dataType.emplace(sensorIt.first,
                                 sensorIt.second->getId(),
                                 sensorIt.second->get());
            }

            data.emplace(std::piecewise_construct,
                         std::forward_as_tuple(sensorGroupIt.first),
                         std::forward_as_tuple(std::move(dataType)));
        }
    }

    matrixN_t Model::getSensorsData(std::string const & sensorType) const
    {
        return sensorsGroupHolder_.at(sensorType).begin()->second->getAll();
    }

    vectorN_t const & Model::getSensorData(std::string const & sensorType,
                                           std::string const & sensorName) const
    {
        return *sensorsGroupHolder_.at(sensorType).at(sensorName)->get();
    }

    void Model::setSensorsData(float64_t const & t,
                               vectorN_t const & q,
                               vectorN_t const & v,
                               vectorN_t const & a,
                               vectorN_t const & u)
    {
        for (sensorsGroupHolder_t::value_type const & sensorGroup : sensorsGroupHolder_)
        {
            if (!sensorGroup.second.empty())
            {
                sensorGroup.second.begin()->second->setAll(t, q, v, a, u); // Access static member of the sensor Group through the first instance
            }
        }
    }

    void Model::updateTelemetry(void)
    {
        for (sensorsGroupHolder_t::value_type const & sensorGroup : sensorsGroupHolder_)
        {
            if (!sensorGroup.second.empty())
            {
                sensorGroup.second.begin()->second->updateTelemetryAll(); // Access static member of the sensor Group through the first instance
            }
        }
    }

    result_t Model::getLock(std::unique_ptr<LockGuardLocal> & lock)
    {
        if (mutexLocal_.isLocked())
        {
            std::cout << "Error - Model::getLock - Model already locked. Please release the current lock first." << std::endl;
            return result_t::ERROR_GENERIC;
        }

        lock = std::move(std::make_unique<LockGuardLocal>(mutexLocal_));

        return result_t::SUCCESS;
    }

    bool_t const & Model::getIsLocked(void) const
    {
        return mutexLocal_.isLocked();
    }

    std::vector<std::string> const & Model::getContactFramesNames(void) const
    {
        return contactFramesNames_;
    }

    std::vector<int32_t> const & Model::getContactFramesIdx(void) const
    {
        return contactFramesIdx_;
    }

    std::vector<std::string> const & Model::getMotorsNames(void) const
    {
        return motorsNames_;
    }

    std::vector<int32_t> const & Model::getMotorsModelIdx(void) const
    {
        return motorsModelIdx_;
    }

    std::vector<int32_t> const & Model::getMotorsPositionIdx(void) const
    {
        return motorsPositionIdx_;
    }

    std::vector<int32_t> const & Model::getMotorsVelocityIdx(void) const
    {
        return motorsVelocityIdx_;
    }

    std::vector<std::string> const & Model::getPositionFieldNames(void) const
    {
        return positionFieldNames_;
    }

    vectorN_t const & Model::getPositionLimitMin(void) const
    {
        return positionLimitMin_;
    }

    vectorN_t const & Model::getPositionLimitMax(void) const
    {
        return positionLimitMax_;
    }

    std::vector<std::string> const & Model::getVelocityFieldNames(void) const
    {
        return velocityFieldNames_;
    }

    vectorN_t const & Model::getVelocityLimit(void) const
    {
        return velocityLimit_;
    }

    vectorN_t const & Model::getTorqueLimit(void) const
    {
        return torqueLimit_;
    }

    vectorN_t Model::getMotorInertia(void) const
    {
        vectorN_t motorInertia(motorsVelocityIdx_.size());
        for (uint32_t i=0; i < motorsVelocityIdx_.size(); ++i)
        {
            motorInertia[i] = pncModel_.rotorInertia[motorsVelocityIdx_[i]];
        }
        return motorInertia;
    }

    std::vector<std::string> const & Model::getAccelerationFieldNames(void) const
    {
        return accelerationFieldNames_;
    }

    std::vector<std::string> const & Model::getMotorTorqueFieldNames(void) const
    {
        return motorTorqueFieldNames_;
    }

    std::vector<std::string> const & Model::getRigidJointsNames(void) const
    {
        return rigidJointsNames_;
    }

    std::vector<int32_t> const & Model::getRigidJointsModelIdx(void) const
    {
        return rigidJointsModelIdx_;
    }

    std::vector<int32_t> const & Model::getRigidJointsPositionIdx(void) const
    {
        return rigidJointsPositionIdx_;
    }

    std::vector<int32_t> const & Model::getRigidJointsVelocityIdx(void) const
    {
        return rigidJointsVelocityIdx_;
    }

    std::vector<std::string> const & Model::getFlexibleJointsNames(void) const
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

    std::vector<int32_t> const & Model::getFlexibleJointsModelIdx(void) const
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

    uint32_t const & Model::nq(void) const
    {
        return nq_;
    }

    uint32_t const & Model::nv(void) const
    {
        return nv_;
    }

    uint32_t const & Model::nx(void) const
    {
        return nx_;
    }
}
