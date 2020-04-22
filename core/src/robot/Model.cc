
#include <iostream>
#include <fstream>
#include <exception>

#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/frames.hpp"

#include "jiminy/core/Utilities.h"
#include "jiminy/core/Constants.h"

#include "jiminy/core/robot/Model.h"


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
    urdfPath_(),
    hasFreeflyer_(false),
    mdlOptionsHolder_(),
    contactFramesNames_(),
    contactFramesIdx_(),
    rigidJointsNames_(),
    rigidJointsModelIdx_(),
    rigidJointsPositionIdx_(),
    rigidJointsVelocityIdx_(),
    flexibleJointsNames_(),
    flexibleJointsModelIdx_(),
    positionLimitMin_(),
    positionLimitMax_(),
    velocityLimit_(),
    positionFieldnames_(),
    velocityFieldnames_(),
    accelerationFieldnames_(),
    pncModelFlexibleOrig_(),
    nq_(0),
    nv_(0),
    nx_(0)
    {
        setOptions(getDefaultModelOptions());
    }

    hresult_t Model::initialize(std::string const & urdfPath,
                                bool_t      const & hasFreeflyer)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

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
            /* Add biases to the dynamics properties of the model.
               Note that is also refresh all proxies automatically. */
            returnCode = generateModelBiased();
        }

        // Unset the initialization flag in case of failure
        if (returnCode != hresult_t::SUCCESS)
        {
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
    }

    hresult_t Model::addContactPoints(std::vector<std::string> const & frameNames)
    {
        if (!isInitialized_)
        {
            std::cout << "Error - Model::addContactPoints - Model not initialized." << std::endl;
            return hresult_t::ERROR_INIT_FAILED;
        }

        // Make sure that the frame list is not empty
        if (frameNames.empty())
        {
            std::cout << "Error - Model::addContactPoints - The list of frames must not be empty." << std::endl;
            return hresult_t::ERROR_BAD_INPUT;
        }

        // Make sure that no frame are duplicates
        if (checkDuplicates(frameNames))
        {
            std::cout << "Error - Model::addContactPoints - Some frames are duplicates." << std::endl;
            return hresult_t::ERROR_BAD_INPUT;
        }

        // Make sure that no contact point is associated with any of the frame in the list
        if (checkIntersection(contactFramesNames_, frameNames))
        {
            std::cout << "Error - Model::addContactPoints - At least one of the frame is already been associated with a contact point." << std::endl;
            return hresult_t::ERROR_BAD_INPUT;
        }

        // Make sure that all the frames exist
        for (std::string const & frame : frameNames)
        {
            if (!pncModel_.existFrame(frame))
            {
                std::cout << "Error - Model::addContactPoints - At least one of the frame does not exist." << std::endl;
                return hresult_t::ERROR_BAD_INPUT;
            }
        }

        // Add the list of frames to the set of contact points
        contactFramesNames_.insert(contactFramesNames_.end(), frameNames.begin(), frameNames.end());

        // Reset the contact force internal buffer
        contactForces_ = forceVector_t(contactFramesNames_.size(), pinocchio::Force::Zero());

        // Refresh proxies associated with the contact points only
        refreshContactsProxies();

        return hresult_t::SUCCESS;
    }

    hresult_t Model::removeContactPoints(std::vector<std::string> const & frameNames)
    {
        if (!isInitialized_)
        {
            std::cout << "Error - Model::removeContactPoints - Model not initialized." << std::endl;
            return hresult_t::ERROR_INIT_FAILED;
        }

        // Make sure that no frame are duplicates
        if (checkDuplicates(frameNames))
        {
            std::cout << "Error - Model::removeContactPoints - Some frames are duplicates." << std::endl;
            return hresult_t::ERROR_BAD_INPUT;
        }

        // Make sure that every frame in the list is associated with a contact point
        if (!checkInclusion(contactFramesNames_, frameNames))
        {
            std::cout << "Error - Model::removeContactPoints - At least one of the frame is not associated with any contact point." << std::endl;
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
        refreshContactsProxies();

        return hresult_t::SUCCESS;
    }

    hresult_t Model::generateModelFlexible(void)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (!isInitialized_)
        {
            std::cout << "Error - Model::generateModelFlexible - Model not initialized." << std::endl;
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
                        removeSuffix(jointName, "Joint") + FLEXIBLE_JOINT_SUFFIX;
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

    hresult_t Model::generateModelBiased(void)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (!isInitialized_)
        {
            std::cout << "Error - Model::generateModelBiased - Model not initialized." << std::endl;
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

        if (returnCode == hresult_t::SUCCESS)
        {
            // Initialize the internal proxies.
            returnCode = refreshProxies();
        }

        return returnCode;
    }

    hresult_t Model::refreshProxies(void)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (!isInitialized_)
        {
            std::cout << "Error - Model::refreshProxies - Model not initialized." << std::endl;
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
            positionFieldnames_.clear();
            positionFieldnames_.resize(nq_);
            velocityFieldnames_.clear();
            velocityFieldnames_.resize(nv_);
            accelerationFieldnames_.clear();
            accelerationFieldnames_.resize(nv_);
            std::vector<std::string> const & jointNames = pncModel_.names;
            std::vector<std::string> jointShortNames = removeSuffix(jointNames, "Joint");
            for (uint32_t i=0; i<jointNames.size(); ++i)
            {
                std::string const & jointName = jointNames[i];
                int32_t const jointIdx = pncModel_.getJointId(jointName);

                int32_t const idx_q = pncModel_.joints[jointIdx].idx_q();

                if (idx_q >= 0) // Otherwise the joint is not part of the vectorial representation
                {
                    int32_t const idx_v = pncModel_.joints[jointIdx].idx_v();

                    joint_t jointType;
                    std::string jointPrefix;
                    if (returnCode == hresult_t::SUCCESS)
                    {
                        returnCode = getJointTypeFromIdx(pncModel_, jointIdx, jointType);
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
                                  positionFieldnames_.begin() + idx_q);
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
                                  velocityFieldnames_.begin() + idx_v);
                        std::copy(jointAccelerationFieldnames.begin(),
                                  jointAccelerationFieldnames.end(),
                                  accelerationFieldnames_.begin() + idx_v);
                    }
                }
            }
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            // Get the joint position limits from the URDF or the user options
            positionLimitMin_ = vectorN_t::Constant(pncModel_.nq, -INF); // Do NOT use robot_->pncModel_.(lower|upper)PositionLimit
            positionLimitMax_ = vectorN_t::Constant(pncModel_.nq, +INF);
            for (int32_t i=0 ; i < pncModel_.njoints ; i++)
            {
                joint_t jointType(joint_t::NONE);
                getJointTypeFromIdx(pncModel_, i, jointType);
                // The "position" of spherical joints is bounded between -1.0 and 1.0 since it corresponds to normalized quaternions
                if (jointType == joint_t::SPHERICAL)
                {
                    uint32_t const & positionIdx = pncModel_.joints[i].idx_q();
                    positionLimitMin_.segment<4>(positionIdx).setConstant(-1.0);
                    positionLimitMax_.segment<4>(positionIdx).setConstant(+1.0);
                }
                if (jointType == joint_t::FREE)
                {
                    uint32_t const & positionIdx = pncModel_.joints[i].idx_q();
                    positionLimitMin_.segment<4>(positionIdx + 3).setConstant(-1.0);
                    positionLimitMax_.segment<4>(positionIdx + 3).setConstant(+1.0);
                }
            }

            if (mdlOptions_->joints.enablePositionLimit)
            {
                if (mdlOptions_->joints.positionLimitFromUrdf)
                {
                    for (int32_t & positionIdx : rigidJointsPositionIdx_)
                    {
                        positionLimitMin_[positionIdx] = pncModel_.lowerPositionLimit[positionIdx];
                        positionLimitMax_[positionIdx] = pncModel_.upperPositionLimit[positionIdx];
                    }
                }
                else
                {
                    for (uint32_t i=0; i < rigidJointsPositionIdx_.size(); i++)
                    {
                        positionLimitMin_[rigidJointsPositionIdx_[i]] = mdlOptions_->joints.positionLimitMin[i];
                        positionLimitMax_[rigidJointsPositionIdx_[i]] = mdlOptions_->joints.positionLimitMax[i];
                    }
                }
            }

            // Get the joint velocity limits from the URDF or the user options
            velocityLimit_ = vectorN_t::Constant(pncModel_.nv, +INF);
            if (mdlOptions_->joints.enableVelocityLimit)
            {
                if (mdlOptions_->joints.velocityLimitFromUrdf)
                {
                    for (int32_t & velocityIdx : rigidJointsVelocityIdx_)
                    {
                        velocityLimit_[velocityIdx] = pncModel_.velocityLimit[velocityIdx];
                    }
                }
                else
                {
                    for (uint32_t i=0; i < rigidJointsVelocityIdx_.size(); i++)
                    {
                        velocityLimit_[rigidJointsVelocityIdx_[i]] = mdlOptions_->joints.velocityLimit[i];
                    }
                }
            }
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            returnCode = refreshContactsProxies();
        }

        return returnCode;
    }

    hresult_t Model::refreshContactsProxies(void)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (!isInitialized_)
        {
            std::cout << "Error - Model::refreshContactsProxies - Model not initialized." << std::endl;
            returnCode = hresult_t::ERROR_INIT_FAILED;
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            // Extract the contact frames indices in the model
            getFramesIdx(pncModel_, contactFramesNames_, contactFramesIdx_);
        }

        return returnCode;
    }

    hresult_t Model::setOptions(configHolder_t modelOptions)
    {
        bool_t internalBuffersMustBeUpdated = false;
        bool_t isFlexibleModelInvalid = false;
        bool_t isCurrentModelInvalid = false;
        if (isInitialized_)
        {
            /* Check that the following user parameters has the right dimension,
               then update the required internal buffers to reflect changes, if any. */
            configHolder_t & jointOptionsHolder =
                boost::get<configHolder_t>(modelOptions.at("joints"));
            if (!boost::get<bool_t>(jointOptionsHolder.at("positionLimitFromUrdf")))
            {
                vectorN_t & jointsPositionLimitMin = boost::get<vectorN_t>(jointOptionsHolder.at("positionLimitMin"));
                if ((int32_t) rigidJointsPositionIdx_.size() != jointsPositionLimitMin.size())
                {
                    std::cout << "Error - Model::setOptions - Wrong vector size for 'positionLimitMin'." << std::endl;
                    return hresult_t::ERROR_BAD_INPUT;
                }
                vectorN_t jointsPositionLimitMinDiff = jointsPositionLimitMin - mdlOptions_->joints.positionLimitMin;
                internalBuffersMustBeUpdated |= (jointsPositionLimitMinDiff.array().abs() >= EPS).all();
                vectorN_t & jointsPositionLimitMax = boost::get<vectorN_t>(jointOptionsHolder.at("positionLimitMax"));
                if ((uint32_t) rigidJointsPositionIdx_.size() != jointsPositionLimitMax.size())
                {
                    std::cout << "Error - Model::setOptions - Wrong vector size for 'positionLimitMax'." << std::endl;
                    return hresult_t::ERROR_BAD_INPUT;
                }
                vectorN_t jointsPositionLimitMaxDiff = jointsPositionLimitMax - mdlOptions_->joints.positionLimitMax;
                internalBuffersMustBeUpdated |= (jointsPositionLimitMaxDiff.array().abs() >= EPS).all();
            }
            if (!boost::get<bool_t>(jointOptionsHolder.at("velocityLimitFromUrdf")))
            {
                vectorN_t & jointsVelocityLimit = boost::get<vectorN_t>(jointOptionsHolder.at("velocityLimit"));
                if ((int32_t) rigidJointsVelocityIdx_.size() != jointsVelocityLimit.size())
                {
                    std::cout << "Error - Model::setOptions - Wrong vector size for 'velocityLimit'." << std::endl;
                    return hresult_t::ERROR_BAD_INPUT;
                }
                vectorN_t jointsVelocityLimitDiff = jointsVelocityLimit - mdlOptions_->joints.velocityLimit;
                internalBuffersMustBeUpdated |= (jointsVelocityLimitDiff.array().abs() >= EPS).all();
            }

            // Check if the flexible model and its associated proxies must be regenerated
            configHolder_t & dynOptionsHolder =
                boost::get<configHolder_t>(modelOptions.at("dynamics"));
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
        mdlOptionsHolder_ = modelOptions;

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

    configHolder_t Model::getOptions(void) const
    {
        return mdlOptionsHolder_;
    }

    bool_t const & Model::getIsInitialized(void) const
    {
        return isInitialized_;
    }

    std::string const & Model::getUrdfPath(void) const
    {
        return urdfPath_;
    }

    bool_t const & Model::getHasFreeflyer(void) const
    {
        return hasFreeflyer_;
    }

    hresult_t Model::loadUrdfModel(std::string const & urdfPath,
                                   bool_t      const & hasFreeflyer)
    {
        if (!std::ifstream(urdfPath.c_str()).good())
        {
            std::cout << "Error - Model::loadUrdfModel - The URDF file does not exist. Impossible to load it." << std::endl;
            return hresult_t::ERROR_BAD_INPUT;
        }

        urdfPath_ = urdfPath;
        hasFreeflyer_ = hasFreeflyer;

        try
        {
            if (hasFreeflyer)
            {
                pinocchio::urdf::buildModel(urdfPath,
                                            pinocchio::JointModelFreeFlyer(),
                                            pncModel_);
            }
            else
            {
                pinocchio::urdf::buildModel(urdfPath, pncModel_);
            }
        }
        catch (std::exception& e)
        {
            std::cout << "Error - Model::loadUrdfModel - Something is wrong with the URDF. Impossible to build a model from it." << std::endl;
            return hresult_t::ERROR_BAD_INPUT;
        }

        return hresult_t::SUCCESS;
    }

    hresult_t Model::getFlexibleStateFromRigid(vectorN_t const & xRigid,
                                               vectorN_t       & xFlex) const
    {
        // Define some proxies
        uint32_t const & nqRigid = pncModelRigidOrig_.nq;
        uint32_t const & nvRigid = pncModelRigidOrig_.nv;
        uint32_t const & nqFlex = pncModelFlexibleOrig_.nq;
        uint32_t const & nvFlex = pncModelFlexibleOrig_.nv;

        // Check the size of the input state
        if (xRigid.size() != nqRigid + nvRigid)
        {
            std::cout << "Error - Model::getFlexibleStateFromRigid - Size of xRigid inconsistent with theoretical model." << std::endl;
            return hresult_t::ERROR_BAD_INPUT;
        }

        // Initialize the flexible state
        xFlex.resize(nqFlex + nvFlex);
        xFlex << pinocchio::neutral(pncModelFlexibleOrig_), vectorN_t::Zero(nvFlex);

        // Compute the flexible state based on the rigid state
        int32_t idxRigid = 0;
        int32_t idxFlex = 0;
        for (; idxRigid < pncModelRigidOrig_.njoints; idxFlex++)
        {
            std::string const & jointRigidName = pncModelRigidOrig_.names[idxRigid];
            std::string const & jointFlexName = pncModelFlexibleOrig_.names[idxFlex];
            if (jointRigidName == jointFlexName)
            {
                auto const & jointRigid = pncModelRigidOrig_.joints[idxRigid];
                auto const & jointFlex = pncModelFlexibleOrig_.joints[idxFlex];
                if (jointRigid.idx_q() >= 0)
                {
                    xFlex.segment(jointFlex.idx_q(), jointFlex.nq()) =
                        xRigid.segment(jointRigid.idx_q(), jointRigid.nq());
                    xFlex.segment(nqFlex + jointFlex.idx_v(), jointFlex.nv()) =
                        xRigid.segment(nqRigid + jointRigid.idx_v(), jointRigid.nv());
                }
                idxRigid++;
            }
        }

        return hresult_t::SUCCESS;
    }

    hresult_t Model::getRigidStateFromFlexible(vectorN_t const & xFlex,
                                               vectorN_t       & xRigid) const
    {
        // Define some proxies
        uint32_t const & nqRigid = pncModelRigidOrig_.nq;
        uint32_t const & nvRigid = pncModelRigidOrig_.nv;
        uint32_t const & nqFlex = pncModelFlexibleOrig_.nq;
        uint32_t const & nvFlex = pncModelFlexibleOrig_.nv;

        // Check the size of the input state
        if (xFlex.size() != nqFlex + nvFlex)
        {
            std::cout << "Error - Model::getFlexibleStateFromRigid - Size of xRigid inconsistent with theoretical model." << std::endl;
            return hresult_t::ERROR_BAD_INPUT;
        }

        // Initialize the flexible state
        xRigid.resize(nqRigid + nvRigid);
        xRigid << pinocchio::neutral(pncModelRigidOrig_), vectorN_t::Zero(nvRigid);

        // Compute the flexible state based on the rigid state
        int32_t idxRigid = 0;
        int32_t idxFlex = 0;
        for (; idxFlex < pncModelFlexibleOrig_.njoints; idxRigid++, idxFlex++)
        {
            std::string const & jointRigidName = pncModelRigidOrig_.names[idxRigid];
            std::string const & jointFlexName = pncModelRigidOrig_.names[idxFlex];
            if (jointRigidName == jointFlexName)
            {
                auto const & jointRigid = pncModelRigidOrig_.joints[idxRigid];
                auto const & jointFlex = pncModelFlexibleOrig_.joints[idxFlex];
                if (jointRigid.idx_q() >= 0)
                {
                    xRigid.segment(jointRigid.idx_q(), jointRigid.nq()) =
                        xFlex.segment(jointFlex.idx_q(), jointFlex.nq());
                    xRigid.segment(nqRigid + jointRigid.idx_v(), jointRigid.nv()) =
                        xFlex.segment(nqFlex + jointFlex.idx_v(), jointFlex.nv());
                }
            }
            else
            {
                idxFlex++;
            }
        }

        return hresult_t::SUCCESS;
    }

    std::vector<std::string> const & Model::getContactFramesNames(void) const
    {
        return contactFramesNames_;
    }

    std::vector<int32_t> const & Model::getContactFramesIdx(void) const
    {
        return contactFramesIdx_;
    }

    std::vector<std::string> const & Model::getPositionFieldnames(void) const
    {
        return positionFieldnames_;
    }

    vectorN_t const & Model::getPositionLimitMin(void) const
    {
        return positionLimitMin_;
    }

    vectorN_t const & Model::getPositionLimitMax(void) const
    {
        return positionLimitMax_;
    }

    std::vector<std::string> const & Model::getVelocityFieldnames(void) const
    {
        return velocityFieldnames_;
    }

    vectorN_t const & Model::getVelocityLimit(void) const
    {
        return velocityLimit_;
    }

    std::vector<std::string> const & Model::getAccelerationFieldnames(void) const
    {
        return accelerationFieldnames_;
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

    int32_t const & Model::nq(void) const
    {
        return nq_;
    }

    int32_t const & Model::nv(void) const
    {
        return nv_;
    }

    int32_t const & Model::nx(void) const
    {
        return nx_;
    }
}
