
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
    positionFieldNames_(),
    velocityFieldNames_(),
    accelerationFieldNames_(),
    motorTorqueFieldNames_(),
    pncModelFlexibleOrig_(),
    sensorsDataHolder_(),
    nq_(0),
    nv_(0),
    nx_(0)
    {
        setOptions(getDefaultOptions());
    }

    Model::~Model(void)
    {
        // Empty.
    }

    result_t Model::initialize(std::string              const & urdfPath,
                               std::vector<std::string> const & contactFramesNames,
                               std::vector<std::string> const & motorsNames,
                               bool                     const & hasFreeflyer)
    {
        result_t returnCode = result_t::SUCCESS;

        // Remove all sensors, if any
        sensorsGroupHolder_.clear();
        sensorsDataHolder_.clear();
        sensorTelemetryOptions_.clear();

        // Initialize the URDF model
        returnCode = loadUrdfModel(urdfPath, hasFreeflyer);
        isInitialized_ = true;

        if (returnCode == result_t::SUCCESS)
        {
            motorsNames_ = motorsNames;
            contactFramesNames_ = contactFramesNames;
            contactForces_ = pinocchio::container::aligned_vector<pinocchio::Force>(
                contactFramesNames_.size(),
                pinocchio::Force::Zero());

            // Backup the original model and data
            pncModelRigidOrig_ = pncModel_;
            pncDataRigidOrig_ = pinocchio::Data(pncModelRigidOrig_);

            /* Get the list of joint names of the rigid model and
               remove the 'universe' and 'root' if any, since there
               are not actual joints. */
            rigidJointsNames_ = pncModelRigidOrig_.names;
            rigidJointsNames_.erase(rigidJointsNames_.begin()); // remove the 'universe'
            if (hasFreeflyer)
            {
                rigidJointsNames_.erase(rigidJointsNames_.begin()); // remove the 'root'
            }
        }

        // Create the flexible model and add biases to the dynamics properties of the model
        if (returnCode == result_t::SUCCESS)
        {
            returnCode = generateBiasedModel();
        }

        // Update the bounds if necessary
        if (returnCode == result_t::SUCCESS)
        {
            returnCode = setOptions(mdlOptionsHolder_);
        }

        // Set the initialization flag
        if (returnCode != result_t::SUCCESS)
        {
            isInitialized_ = false;
        }

        return returnCode;
    }

    void Model::reset(void)
    {
        if (isInitialized_)
        {
            /* Update the biases added to the dynamics properties of the model.
               It cannot throw an error. */
            generateBiasedModel();
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
            isTelemetryConfigured_ = true;
        }

        if (returnCode != result_t::SUCCESS)
        {
            isTelemetryConfigured_ = false;
        }

        return returnCode;
    }

    result_t Model::removeSensor(std::string const & sensorType,
                                 std::string const & sensorName)
    {
        result_t returnCode = result_t::SUCCESS;

        if (!isInitialized_)
        {
            std::cout << "Error - Model::removeSensor - Model not initialized." << std::endl;
            returnCode = result_t::ERROR_INIT_FAILED;
        }

        sensorsGroupHolder_t::iterator sensorGroupIt;
        if (returnCode == result_t::SUCCESS)
        {
            sensorGroupIt = sensorsGroupHolder_.find(sensorType);
            if (sensorGroupIt == sensorsGroupHolder_.end())
            {
                std::cout << "Error - Model::removeSensor - This type of sensor does not exist." << std::endl;
                returnCode = result_t::ERROR_BAD_INPUT;
            }
        }

        sensorsHolder_t::iterator sensorIt;
        if (returnCode == result_t::SUCCESS)
        {
            sensorIt = sensorGroupIt->second.find(sensorName);
            if (sensorIt == sensorGroupIt->second.end())
            {
                std::cout << "Error - Model::removeSensors - No sensor with this type and name exists." << std::endl;
                returnCode = result_t::ERROR_BAD_INPUT;
            }
        }

        if (returnCode == result_t::SUCCESS)
        {
            // Remove the sensor from its group
            sensorGroupIt->second.erase(sensorIt);

            // Remove the sensor group if there is no more sensors left.
            if (sensorGroupIt->second.empty())
            {
                sensorsGroupHolder_.erase(sensorType);
                sensorsDataHolder_.erase(sensorType);
                sensorTelemetryOptions_.erase(sensorType);
            }
        }

        return returnCode;
    }

    result_t Model::removeSensors(std::string const & sensorType)
    {
        result_t returnCode = result_t::SUCCESS;

        if (!isInitialized_)
        {
            std::cout << "Error - Model::removeSensors - Model not initialized." << std::endl;
            returnCode = result_t::ERROR_INIT_FAILED;
        }

        sensorsGroupHolder_t::iterator sensorGroupIt;
        if (returnCode == result_t::SUCCESS)
        {
            sensorGroupIt = sensorsGroupHolder_.find(sensorType);
            if (sensorGroupIt == sensorsGroupHolder_.end())
            {
                std::cout << "Error - Model::removeSensors - No sensor with this type exists." << std::endl;
                returnCode = result_t::ERROR_BAD_INPUT;
            }
        }

        if (returnCode == result_t::SUCCESS)
        {
            sensorsGroupHolder_.erase(sensorGroupIt);
            sensorsDataHolder_.erase(sensorType);
            sensorTelemetryOptions_.erase(sensorType);
        }

        return returnCode;
    }

    result_t Model::generateFlexibleModel(void)
    {
        result_t returnCode = result_t::SUCCESS;

        if (!isInitialized_)
        {
            std::cout << "Error - Model::generateFlexibleModel - Model not initialized." << std::endl;
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

    result_t Model::generateBiasedModel(void)
    {
        result_t returnCode = result_t::SUCCESS;

        if (!isInitialized_)
        {
            std::cout << "Error - Model::generateBiasedModel - Model not initialized." << std::endl;
            returnCode = result_t::ERROR_INIT_FAILED;
        }

        // Generate a new flexible model and associated variables if needed
        if (returnCode == result_t::SUCCESS)
        {
            if(pncModelFlexibleOrig_ == pinocchio::Model())
            {
                returnCode = generateFlexibleModel();
            }
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
        }

        // Extract some high level features of the rigid model
        if (returnCode == result_t::SUCCESS)
        {
            nq_ = pncModel_.nq;
            nv_ = pncModel_.nv;
            nx_ = nq_ + nv_;
        }

        // Extract some joint and frame indices in the model
        if (returnCode == result_t::SUCCESS)
        {
            getJointsModelIdx(pncModel_, rigidJointsNames_, rigidJointsModelIdx_);
            getJointsPositionIdx(pncModel_, rigidJointsNames_, rigidJointsPositionIdx_, false);
            getJointsVelocityIdx(pncModel_, rigidJointsNames_, rigidJointsVelocityIdx_, false);
            returnCode = getFramesIdx(pncModel_, contactFramesNames_, contactFramesIdx_);
        }
        if (returnCode == result_t::SUCCESS)
        {
            returnCode = getJointsModelIdx(pncModel_, motorsNames_, motorsModelIdx_);
        }
        if (returnCode == result_t::SUCCESS)
        {
            getJointsPositionIdx(pncModel_, motorsNames_, motorsPositionIdx_, true);
            getJointsVelocityIdx(pncModel_, motorsNames_, motorsVelocityIdx_, true);
        }

        /* Generate the fieldnames of the elements of the vectorial
           representation of the configuration, velocity, acceleration
           and motor torques. */
        if (returnCode == result_t::SUCCESS)
        {
            returnCode = generateFieldNames();
        }

        // Initialize Pinocchio Data internal state
        if (returnCode == result_t::SUCCESS)
        {
            pncData_ = pinocchio::Data(pncModel_);
            pinocchio::forwardKinematics(pncModel_,
                                         pncData_,
                                         vectorN_t::Zero(pncModel_.nq),
                                         vectorN_t::Zero(pncModel_.nv));
            pinocchio::framesForwardKinematics(pncModel_, pncData_);
        }

        // Update the position and velocity limits
        if (returnCode == result_t::SUCCESS)
        {
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

        // Update pncModel_.rotorInertia
        if(returnCode == result_t::SUCCESS)
        {
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

    result_t Model::generateFieldNames(void)
    {
        result_t returnCode = result_t::SUCCESS;

        if (!isInitialized_)
        {
            std::cout << "Error - Model::generateFieldNames - Model not initialized." << std::endl;
            returnCode = result_t::ERROR_INIT_FAILED;
        }

        if (returnCode == result_t::SUCCESS)
        {
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

                        returnCode = getJointTypeFromId(pncModel_, jointIdx, jointType);
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

            motorTorqueFieldNames_.clear();
            for (std::string const & jointName : removeFieldnamesSuffix(motorsNames_, "Joint"))
            {
                motorTorqueFieldNames_.emplace_back(JOINT_PREFIX_BASE + "Torque" + jointName);
            }
        }

        return returnCode;
    }

    result_t Model::getSensorsOptions(std::string    const & sensorType,
                                      configHolder_t       & sensorsOptions) const
    {
        result_t returnCode = result_t::SUCCESS;

        if (!isInitialized_)
        {
            std::cout << "Error - Model::getSensorsOptions - Model not initialized." << std::endl;
            returnCode = result_t::ERROR_INIT_FAILED;
        }

        sensorsGroupHolder_t::const_iterator sensorGroupIt;
        if (returnCode == result_t::SUCCESS)
        {
            sensorGroupIt = sensorsGroupHolder_.find(sensorType);
            if (sensorGroupIt == sensorsGroupHolder_.end())
            {
                std::cout << "Error - Model::getSensorsOptions - This type of sensor does not exist." << std::endl;
                returnCode = result_t::ERROR_BAD_INPUT;
            }
        }

        if (returnCode == result_t::SUCCESS)
        {
            sensorsOptions = configHolder_t();
            for (sensorsHolder_t::value_type const & sensor : sensorGroupIt->second)
            {
                sensorsOptions[sensor.first] = sensor.second->getOptions();
            }
        }

        return returnCode;
    }

    result_t Model::getSensorsOptions(configHolder_t & sensorsOptions) const
    {
        result_t returnCode = result_t::SUCCESS;

        if (!isInitialized_)
        {
            std::cout << "Error - Model::getSensorsOptions - Model not initialized." << std::endl;
            returnCode = result_t::ERROR_INIT_FAILED;
        }

        if (returnCode == result_t::SUCCESS)
        {
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
        }

        return returnCode;
    }

    result_t Model::getSensorOptions(std::string    const & sensorType,
                                     std::string    const & sensorName,
                                     configHolder_t       & sensorOptions) const
    {
        result_t returnCode = result_t::SUCCESS;

        if (!isInitialized_)
        {
            std::cout << "Error - Model::getSensorOptions - Model not initialized." << std::endl;
            returnCode = result_t::ERROR_INIT_FAILED;
        }

        sensorsGroupHolder_t::const_iterator sensorGroupIt;
        if (returnCode == result_t::SUCCESS)
        {
            sensorGroupIt = sensorsGroupHolder_.find(sensorType);
            if (sensorGroupIt == sensorsGroupHolder_.end())
            {
                std::cout << "Error - Model::getSensorOptions - This type of sensor does not exist." << std::endl;
                returnCode = result_t::ERROR_BAD_INPUT;
            }
        }

        sensorsHolder_t::const_iterator sensorIt;
        if (returnCode == result_t::SUCCESS)
        {
            sensorIt = sensorGroupIt->second.find(sensorName);
            if (sensorIt == sensorGroupIt->second.end())
            {
                std::cout << "Error - Model::getSensorOptions - No sensor with this type and name exists." << std::endl;
                returnCode = result_t::ERROR_BAD_INPUT;
            }
        }

        if (returnCode == result_t::SUCCESS)
        {
            sensorOptions = sensorIt->second->getOptions();
        }

        return returnCode;
    }

    result_t Model::setSensorOptions(std::string    const & sensorType,
                                     std::string    const & sensorName,
                                     configHolder_t const & sensorOptions)
    {
        result_t returnCode = result_t::SUCCESS;

        if (!isInitialized_)
        {
            std::cout << "Error - Model::setSensorOptions - Model not initialized." << std::endl;
            returnCode = result_t::ERROR_INIT_FAILED;
        }

        sensorsGroupHolder_t::iterator sensorGroupIt;
        if (returnCode == result_t::SUCCESS)
        {
            sensorGroupIt = sensorsGroupHolder_.find(sensorType);
            if (sensorGroupIt == sensorsGroupHolder_.end())
            {
                std::cout << "Error - Model::setSensorOptions - This type of sensor does not exist." << std::endl;
                returnCode = result_t::ERROR_BAD_INPUT;
            }
        }

        sensorsHolder_t::iterator sensorIt;
        if (returnCode == result_t::SUCCESS)
        {
            sensorIt = sensorGroupIt->second.find(sensorName);
            if (sensorIt == sensorGroupIt->second.end())
            {
                std::cout << "Error - Model::setSensorOptions - No sensor with this type and name exists." << std::endl;
                returnCode = result_t::ERROR_BAD_INPUT;
            }
        }

        if (returnCode == result_t::SUCCESS)
        {
            sensorIt->second->setOptions(sensorOptions);
        }

        return returnCode;
    }

    result_t Model::setSensorsOptions(std::string    const & sensorType,
                                      configHolder_t const & sensorsOptions)
    {
        result_t returnCode = result_t::SUCCESS;

        if (!isInitialized_)
        {
            std::cout << "Error - Model::setSensorsOptions - Model not initialized." << std::endl;
            returnCode = result_t::ERROR_INIT_FAILED;
        }

        sensorsGroupHolder_t::iterator sensorGroupIt;
        if (returnCode == result_t::SUCCESS)
        {
            sensorGroupIt = sensorsGroupHolder_.find(sensorType);
            if (sensorGroupIt == sensorsGroupHolder_.end())
            {
                std::cout << "Error - Model::setSensorsOptions - This type of sensor does not exist." << std::endl;
                returnCode = result_t::ERROR_BAD_INPUT;
            }
        }

        if (returnCode == result_t::SUCCESS)
        {
            for (sensorsHolder_t::value_type const & sensor : sensorGroupIt->second)
            {
                configHolder_t::const_iterator it = sensorsOptions.find(sensor.first);
                if (it != sensorsOptions.end())
                {
                    sensor.second->setOptions(boost::get<configHolder_t>(it->second));
                }
                else
                {
                    sensor.second->setOptionsAll(sensorsOptions);
                    break;
                }
            }
        }

        return returnCode;
    }

    result_t Model::setSensorsOptions(configHolder_t const & sensorsOptions)
    {
        result_t returnCode = result_t::SUCCESS;

        if (!isInitialized_)
        {
            std::cout << "Error - Model::setSensorsOptions - Model not initialized." << std::endl;
            returnCode = result_t::ERROR_INIT_FAILED;
        }

        if (returnCode == result_t::SUCCESS)
        {
            for (sensorsGroupHolder_t::value_type const & sensorGroup : sensorsGroupHolder_)
            {
                for (sensorsHolder_t::value_type const & sensor : sensorGroup.second)
                {
                    sensor.second->setOptions(boost::get<configHolder_t>(
                        boost::get<configHolder_t>(
                            sensorsOptions.at(sensorGroup.first)).at(sensor.first))); // TODO: missing check for sensor type and name availability
                }
            }
        }

        return returnCode;
    }

    result_t Model::getTelemetryOptions(configHolder_t & telemetryOptions) const
    {
        result_t returnCode = result_t::SUCCESS;

        if (!isInitialized_)
        {
            std::cout << "Error - Model::setSensorsOptions - Model not initialized." << std::endl;
            returnCode = result_t::ERROR_INIT_FAILED;
        }

        if (returnCode == result_t::SUCCESS)
        {
            telemetryOptions.clear();
            for (auto const & sensorGroupTelemetryOption : sensorTelemetryOptions_)
            {
                std::string optionTelemetryName = "enable" + sensorGroupTelemetryOption.first + "s";
                telemetryOptions[optionTelemetryName] = sensorGroupTelemetryOption.second;
            }
        }

        return returnCode;
    }

    result_t Model::setTelemetryOptions(configHolder_t const & telemetryOptions)
    {
        result_t returnCode = result_t::SUCCESS;

        if (!isInitialized_)
        {
            std::cout << "Error - Model::setTelemetryOptions - Model not initialized." << std::endl;
            returnCode = result_t::ERROR_INIT_FAILED;
        }

        for (auto & sensorGroupTelemetryOption : sensorTelemetryOptions_)
        {
            std::string optionTelemetryName = "enable" + sensorGroupTelemetryOption.first + "s";
            configHolder_t::const_iterator sensorTelemetryOptionIt =
                telemetryOptions.find(optionTelemetryName);
            if (sensorTelemetryOptionIt == telemetryOptions.end())
            {
                std::cout << "Error - Model::setTelemetryOptions - Missing field." << std::endl;
                returnCode = result_t::ERROR_GENERIC;
            }
            if (returnCode == result_t::SUCCESS)
            {
                sensorGroupTelemetryOption.second = boost::get<bool>(sensorTelemetryOptionIt->second);
            }
        }

        return returnCode;
    }

    configHolder_t Model::getOptions(void) const
    {
        return mdlOptionsHolder_;
    }

    result_t Model::setOptions(configHolder_t mdlOptions)
    {
        result_t returnCode = result_t::SUCCESS;

        if (isInitialized_)
        {
            // Make sure the user-defined position limit has the right dimension
            configHolder_t & jointOptionsHolder =
                boost::get<configHolder_t>(mdlOptions.at("joints"));
            if (!boost::get<bool>(jointOptionsHolder.at("positionLimitFromUrdf")))
            {
                vectorN_t & positionLimitMin = boost::get<vectorN_t>(jointOptionsHolder.at("positionLimitMin"));
                if((int32_t) rigidJointsPositionIdx_.size() != positionLimitMin.size())
                {
                    std::cout << "Error - Model::setOptions - Wrong vector size for 'positionLimitMin'." << std::endl;
                    returnCode = result_t::ERROR_BAD_INPUT;
                }
                vectorN_t & positionLimitMax = boost::get<vectorN_t>(jointOptionsHolder.at("positionLimitMax"));
                if((uint32_t) rigidJointsPositionIdx_.size() != positionLimitMax.size())
                {
                    std::cout << "Error - Model::setOptions - Wrong vector size for 'positionLimitMax'." << std::endl;
                    returnCode = result_t::ERROR_BAD_INPUT;
                }
            }
            if (!boost::get<bool>(jointOptionsHolder.at("velocityLimitFromUrdf")))
            {
                vectorN_t & velocityLimit = boost::get<vectorN_t>(jointOptionsHolder.at("velocityLimit"));
                if((int32_t) rigidJointsVelocityIdx_.size() != velocityLimit.size())
                {
                    std::cout << "Error - Model::setOptions - Wrong vector size for 'velocityLimit'." << std::endl;
                    returnCode = result_t::ERROR_BAD_INPUT;
                }
            }
            if (boost::get<bool>(jointOptionsHolder.at("enableMotorInertia")))
            {
                vectorN_t & motorInertia = boost::get<vectorN_t>(jointOptionsHolder.at("motorInertia"));
                if((int32_t) motorsVelocityIdx_.size() != motorInertia.size())
                {
                    std::cout << "Error - Model::setOptions - Wrong vector size for 'motorInertia'." << std::endl;
                    returnCode = result_t::ERROR_BAD_INPUT;
                }
                if(!(motorInertia.array() >= 0.0).all())
                {
                    std::cout << "Error - Model::setOptions - Every values in 'motorInertia' must be positive." << std::endl;
                    returnCode = result_t::ERROR_BAD_INPUT;
                }
            }
        }

        // Check if the flexible model and its associated variables must be regenerated
        bool isFlexibleModelInvalid = false;
        bool isModelInvalid = false;
        if (returnCode == result_t::SUCCESS)
        {
            if (isInitialized_)
            {
                configHolder_t & dynOptionsHolder =
                    boost::get<configHolder_t>(mdlOptions.at("dynamics"));
                bool const & enableFlexibleModel = boost::get<bool>(dynOptionsHolder.at("enableFlexibleModel"));
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
                    isModelInvalid = true;
                }
            }
        }

        if (returnCode == result_t::SUCCESS)
        {
            // Update the internal options
            mdlOptionsHolder_ = mdlOptions;

            // Create a fast struct accessor
            mdlOptions_ = std::make_unique<modelOptions_t const>(mdlOptionsHolder_);
        }

        if (returnCode == result_t::SUCCESS)
        {
            // Force flexible model regeneration if necessary
            if (isFlexibleModelInvalid)
            {
                pncModelFlexibleOrig_ = pinocchio::Model();
            }

            // Trigger biased model regeneration and info extraction if necessary
            if (isFlexibleModelInvalid || isModelInvalid)
            {
                reset();
            }
        }

        return returnCode;
    }

    bool const & Model::getIsInitialized(void) const
    {
        return isInitialized_;
    }

    bool const & Model::getIsTelemetryConfigured(void) const
    {
        return isTelemetryConfigured_;
    }

    std::string const & Model::getUrdfPath(void) const
    {
        return urdfPath_;
    }

    bool const & Model::getHasFreeflyer(void) const
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
                                  bool        const & hasFreeflyer)
    {
        result_t returnCode = result_t::SUCCESS;

        if (!std::ifstream(urdfPath.c_str()).good())
        {
            std::cout << "Error - Model::loadUrdfModel - The URDF file does not exist. Impossible to load it." << std::endl;
            returnCode = result_t::ERROR_BAD_INPUT;
        }
        urdfPath_ = urdfPath;
        hasFreeflyer_ = hasFreeflyer;

        if (returnCode == result_t::SUCCESS)
        {
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
                returnCode = result_t::ERROR_BAD_INPUT;
            }
        }

        return returnCode;
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
