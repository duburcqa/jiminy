
#include <iostream>
#include <fstream>
#include <exception>

#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/center-of-mass.hpp"

#include "jiminy/core/Utilities.h"
#include "jiminy/core/Constants.h"

#include "jiminy/core/robot/Model.h"


namespace jiminy
{
    Model::Model(void) :
    pncModel_(),
    pncData_(pncModel_),
    pncGeometryModel_(),
    pncGeometryData_(nullptr),
    pncModelRigidOrig_(),
    pncDataRigidOrig_(pncModelRigidOrig_),
    mdlOptions_(nullptr),
    contactForces_(),
    isInitialized_(false),
    urdfPath_(),
    hasFreeflyer_(false),
    mdlOptionsHolder_(),
    collisionBodiesNames_(),
    contactFramesNames_(),
    collisionBodiesIdx_(),
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

            // Initialize Pinocchio data internal state, including
            // stuffs as simple as the mass of the bodies.
            pinocchio::forwardKinematics(pncModelRigidOrig_,
                                         pncDataRigidOrig_,
                                         pinocchio::neutral(pncModelRigidOrig_),
                                         vectorN_t::Zero(pncModelRigidOrig_.nv));
            pinocchio::updateFramePlacements(pncModelRigidOrig_, pncDataRigidOrig_);
            pinocchio::centerOfMass(pncModelRigidOrig_,
                                    pncDataRigidOrig_,
                                    pinocchio::neutral(pncModelRigidOrig_));

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
            // Update the biases added to the dynamics properties of the model
            generateModelBiased();
        }
    }

    hresult_t Model::addFrame(std::string    const & frameName,
                              std::string    const & parentBodyName,
                              pinocchio::SE3 const & framePlacement)
    {
        // Note that since it is not possible to add a frame to another frame,
        // the frame is added directly to the parent joint, thus relative transform
        // of the frame wrt the parent joint must be computed.
        hresult_t returnCode = hresult_t::SUCCESS;

        pinocchio::FrameType const frameType = pinocchio::FrameType::OP_FRAME;

        int32_t parentFrameId;

        // Add the frame to the the current model
        returnCode = getFrameIdx(pncModel_, parentBodyName, parentFrameId);
        if (returnCode == hresult_t::SUCCESS)
        {
            int32_t const & parentJointId = pncModel_.frames[parentFrameId].parent;
            pinocchio::SE3 const & parentFramePlacement = pncModel_.frames[parentFrameId].placement;
            pinocchio::SE3 const jointFramePlacement = parentFramePlacement.actInv(framePlacement);
            pinocchio::Frame const frame(frameName, parentJointId, parentFrameId, jointFramePlacement, frameType);
            pncModel_.addFrame(frame);
        }

        // Add the frame to the the original rigid model
        returnCode = getFrameIdx(pncModelRigidOrig_, parentBodyName, parentFrameId);
        if (returnCode == hresult_t::SUCCESS)
        {
            int32_t const & parentJointId = pncModelRigidOrig_.frames[parentFrameId].parent;
            pinocchio::SE3 const & parentFramePlacement = pncModelRigidOrig_.frames[parentFrameId].placement;
            pinocchio::SE3 const jointFramePlacement = parentFramePlacement.actInv(framePlacement);
            pinocchio::Frame const frame(frameName, parentJointId, parentFrameId, jointFramePlacement, frameType);
            pncModelRigidOrig_.addFrame(frame);
        }

        // Add the frame to the the original flexible model
        returnCode = getFrameIdx(pncModelFlexibleOrig_, parentBodyName, parentFrameId);
        if (returnCode == hresult_t::SUCCESS)
        {
            int32_t const & parentJointId = pncModelFlexibleOrig_.frames[parentFrameId].parent;
            pinocchio::SE3 const & parentFramePlacement = pncModelFlexibleOrig_.frames[parentFrameId].placement;
            pinocchio::SE3 const jointFramePlacement = parentFramePlacement.actInv(framePlacement);
            pinocchio::Frame const frame(frameName, parentJointId, parentFrameId, jointFramePlacement, frameType);
            pncModelFlexibleOrig_.addFrame(frame);
        }

        return returnCode;
    }

    hresult_t Model::removeFrame(std::string const & frameName)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        pinocchio::FrameType const frameType = pinocchio::FrameType::OP_FRAME;

        // Check that the frame can be removed from the current model.
        // If so, assuming it is also the case for the original models.
        int32_t frameId;
        returnCode = getFrameIdx(pncModelRigidOrig_, frameName, frameId);
        if (returnCode == hresult_t::SUCCESS)
        {
            if (pncModelRigidOrig_.frames[frameId].type != frameType)
            {
                std::cout << "Error - Model::removeFrame - Impossible to remove this frame. One should only remove frames added manually." << std::endl;
                returnCode = hresult_t::ERROR_BAD_INPUT;
            }
        }

        // Remove the frame from the the current model
        if (returnCode == hresult_t::SUCCESS)
        {
            pncModel_.frames.erase(pncModel_.frames.begin() + frameId);
            pncModel_.nframes--;
        }

        // Remove the frame from the the current model
        returnCode = getFrameIdx(pncModelRigidOrig_, frameName, frameId);
        if (returnCode == hresult_t::SUCCESS)
        {
            pncModelRigidOrig_.frames.erase(pncModelRigidOrig_.frames.begin() + frameId);
            pncModelRigidOrig_.nframes--;
        }

        // Remove the frame from the the current model
        returnCode = getFrameIdx(pncModelFlexibleOrig_, frameName, frameId);
        if (returnCode == hresult_t::SUCCESS)
        {
            pncModelFlexibleOrig_.frames.erase(pncModelFlexibleOrig_.frames.begin() + frameId);
            pncModelFlexibleOrig_.nframes--;
        }

        return returnCode;
    }

    hresult_t Model::addCollisionBodies(std::vector<std::string> const & bodyNames)
    {
        if (!isInitialized_)
        {
            std::cout << "Error - Model::addCollisionBodies - Model not initialized." << std::endl;
            return hresult_t::ERROR_INIT_FAILED;
        }

        // Make sure that the body list is not empty
        if (bodyNames.empty())
        {
            std::cout << "Error - Model::addCollisionBodies - The list of bodies must not be empty." << std::endl;
            return hresult_t::ERROR_BAD_INPUT;
        }

        // Make sure that no body are duplicates
        if (checkDuplicates(bodyNames))
        {
            std::cout << "Error - Model::addCollisionBodies - Some bodies are duplicates." << std::endl;
            return hresult_t::ERROR_BAD_INPUT;
        }

        // Make sure that there is no collision already associated with any of the bodies in the list
        if (checkIntersection(collisionBodiesNames_, bodyNames))
        {
            std::cout << "Error - Model::addCollisionBodies - At least one of the body is already been associated with a collision." << std::endl;
            return hresult_t::ERROR_BAD_INPUT;
        }

        // Make sure that all the bodies exist
        for (std::string const & name : bodyNames)
        {
            if (!pncModel_.existBodyName(name))
            {
                std::cout << "Error - Model::addCollisionBodies - At least one of the body does not exist." << std::endl;
                return hresult_t::ERROR_BAD_INPUT;
            }
        }

        // Make sure that one and only one geometry is associated with each body
        for (std::string const & name : bodyNames)
        {
            int32_t nChildGeom = 0;
            for (pinocchio::GeometryObject const & geom : pncGeometryModel_.geometryObjects)
            {
                if (pncModel_.frames[geom.parentFrame].name == name)
                {
                    nChildGeom++;
                }
            }
            if (nChildGeom != 1)
            {
                std::cout << "Error - Model::addCollisionBodies - Collision is only supported for bodies associated with one and only one geometry." << std::endl;
                return hresult_t::ERROR_BAD_INPUT;
            }
        }

        // Add the list of bodies to the set of collision bodies
        collisionBodiesNames_.insert(collisionBodiesNames_.end(), bodyNames.begin(), bodyNames.end());

        // Create the collision pairs and add them to the geometry model of the robot
        pinocchio::GeomIndex const & groundId = pncGeometryModel_.getGeometryId("ground");
        for (std::string const & name : bodyNames)
        {
            // Find the body id by looking at the first geometry having it for parent
            pinocchio::GeomIndex bodyId;
            for (uint32_t i=0; i<pncGeometryModel_.geometryObjects.size(); ++i)
            {
                pinocchio::GeometryObject const & geom = pncGeometryModel_.geometryObjects[i];
                if (pncModel_.frames[geom.parentFrame].name == name)
                {
                    bodyId = i;
                    break;
                }
            }

            /* Create and add the collision pair with the ground.
               Note that the ground must come first for the normal to be properly computed
               since the contact information only reports the normal of the second geometry
               wrt the world, which is the only one that is really interesting since the
               ground normal never changes for flat ground, as it is the case now. */
            pinocchio::CollisionPair const collisionPair(bodyId, groundId);
            pncGeometryModel_.addCollisionPair(collisionPair);

            // Refresh proxies associated with the collisions only
            refreshCollisionsProxies();
        }

        return hresult_t::SUCCESS;
    }

    hresult_t Model::removeCollisionBodies(std::vector<std::string> const & bodyNames)
    {
        if (!isInitialized_)
        {
            std::cout << "Error - Model::removeCollisionBodies - Model not initialized." << std::endl;
            return hresult_t::ERROR_INIT_FAILED;
        }

        // Make sure that no body are duplicates
        if (checkDuplicates(bodyNames))
        {
            std::cout << "Error - Model::removeCollisionBodies - Some bodies are duplicates." << std::endl;
            return hresult_t::ERROR_BAD_INPUT;
        }

        // Make sure that every body in the list is associated with a collision
        if (!checkInclusion(collisionBodiesNames_, bodyNames))
        {
            std::cout << "Error - Model::removeCollisionBodies - At least one of the body is not associated with any collision." << std::endl;
            return hresult_t::ERROR_BAD_INPUT;
        }

        // Remove the list of bodies from the set of collision bodies
        if (!bodyNames.empty())
        {
            eraseVector(collisionBodiesNames_, bodyNames);
        }
        else
        {
            collisionBodiesNames_.clear();
        }

        // Get the indices of the corresponding collision pairs in the geometry model of the robot and remove them
        pinocchio::GeomIndex const & groundId = pncGeometryModel_.getGeometryId("ground");
        for (std::string const & name : bodyNames)
        {
            // Find the body id by looking at the first geometry having it for parent
            pinocchio::GeomIndex bodyId;
            for (uint32_t i=0; i<pncGeometryModel_.geometryObjects.size(); ++i)
            {
                pinocchio::GeometryObject const & geom = pncGeometryModel_.geometryObjects[i];
                if (pncModel_.frames[geom.parentFrame].name == name)
                {
                    bodyId = i;
                    break;
                }
            }

            // Create and remove the collision pair with the ground
            pinocchio::CollisionPair const collisionPair(groundId, bodyId);
            pncGeometryModel_.removeCollisionPair(collisionPair);

            // Refresh proxies associated with the collisions only
            refreshCollisionsProxies();
        }

        return hresult_t::SUCCESS;
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

        // Make sure that there is no contact already associated with any of the frames in the list
        if (checkIntersection(contactFramesNames_, frameNames))
        {
            std::cout << "Error - Model::addContactPoints - At least one of the frame is already been associated with a contact." << std::endl;
            return hresult_t::ERROR_BAD_INPUT;
        }

        // Make sure that all the frames exist
        for (std::string const & name : frameNames)
        {
            if (!pncModel_.existFrame(name))
            {
                std::cout << "Error - Model::addContactPoints - At least one of the frame does not exist." << std::endl;
                return hresult_t::ERROR_BAD_INPUT;
            }
        }

        // Add the list of frames to the set of contact frames
        contactFramesNames_.insert(contactFramesNames_.end(), frameNames.begin(), frameNames.end());

        // Refresh proxies associated with the contact only
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

        // Make sure that every frame in the list is associated with a contact
        if (!checkInclusion(contactFramesNames_, frameNames))
        {
            std::cout << "Error - Model::removeContactPoints - At least one of the frame is not associated with any contact." << std::endl;
            return hresult_t::ERROR_BAD_INPUT;
        }

        // Remove the list of frames from the set of contact frames
        if (!frameNames.empty())
        {
            eraseVector(contactFramesNames_, frameNames);
        }
        else
        {
            contactFramesNames_.clear();
        }

        // Refresh proxies associated with the contact only
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

                // Add joints to model
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
                                         pinocchio::neutral(pncModel_),
                                         vectorN_t::Zero(pncModel_.nv));
            pinocchio::updateFramePlacements(pncModel_, pncData_);
            pinocchio::centerOfMass(pncModel_, pncData_,
                                    pinocchio::neutral(pncModel_));
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
            returnCode = refreshCollisionsProxies();
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            returnCode = refreshContactsProxies();
        }

        return returnCode;
    }

    hresult_t Model::refreshCollisionsProxies(void)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (!isInitialized_)
        {
            std::cout << "Error - Model::refreshCollisionsProxies - Model not initialized." << std::endl;
            returnCode = hresult_t::ERROR_INIT_FAILED;
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            // A new geometry data object must be instantiate after changing the collision pairs
            pncGeometryData_ = std::make_unique<pinocchio::GeometryData>(pncGeometryModel_);
            pinocchio::updateGeometryPlacements(pncModel_,
                                                pncData_,
                                                pncGeometryModel_,
                                                *pncGeometryData_);

            // Set the max number of contact points per collision pairs
            // Only a global collisionRequest is available for Pinocchio < 2.4.4, instead of one for each collision pair.
            # if PINOCCHIO_MINOR_VERSION >= 4 || PINOCCHIO_PATCH_VERSION >= 4
            for (hpp::fcl::CollisionRequest & collisionRequest : pncGeometryData_->collisionRequests)
            {
                collisionRequest.num_max_contacts = mdlOptions_->collisions.maxContactPointsPerBody;
            }
            #else
            pncGeometryData_->collisionRequest.num_max_contacts = mdlOptions_->collisions.maxContactPointsPerBody;
            #endif

            // Extract the contact frames indices in the model
            getFramesIdx(pncModel_, collisionBodiesNames_, collisionBodiesIdx_);
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
            // Reset the contact force internal buffer
            contactForces_ = forceVector_t(contactFramesNames_.size(), pinocchio::Force::Zero());

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
        bool_t isCollisionDataInvalid = false;
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

            // Check if the flexible model and its proxies must be regenerated
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

            // Check that the collisions options are valid
            configHolder_t & collisionOptionsHolder =
                boost::get<configHolder_t>(modelOptions.at("collisions"));
            uint32_t const & maxContactPointsPerBody = boost::get<uint32_t>(collisionOptionsHolder.at("maxContactPointsPerBody"));
            if (maxContactPointsPerBody < 1)
            {
                std::cout << "Error - Model::setOptions - The number of contact points by collision pair 'maxContactPointsPerBody' must be at least 1." << std::endl;
                return hresult_t::ERROR_BAD_INPUT;
            }
            if (mdlOptions_ && maxContactPointsPerBody != mdlOptions_->collisions.maxContactPointsPerBody)
            {
                isCollisionDataInvalid = true;
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
        else if (isCollisionDataInvalid)
        {
            // Update the collision data
            refreshCollisionsProxies();
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

        // Build the robot model
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

        // Build the robot geometry model
        pinocchio::urdf::buildGeom(pncModel_, urdfPath, pinocchio::COLLISION, pncGeometryModel_);

        // Replace the geometry object by their convex representation for efficiency
        #if PINOCCHIO_MINOR_VERSION >= 4 || PINOCCHIO_PATCH_VERSION >= 4
        for (uint32_t i=0; i<pncGeometryModel_.geometryObjects.size(); ++i)
        {
            hpp::fcl::BVHModelPtr_t bvh = boost::dynamic_pointer_cast<hpp::fcl::BVHModelBase>(pncGeometryModel_.geometryObjects[i].geometry);
            bvh->buildConvexHull(true);
            pncGeometryModel_.geometryObjects[i].geometry = bvh->convex;
        }
        #endif

        // Instantiate ground FCL box geometry, wrapped as a pinocchio collision geometry.
        // Note that half-space cannot be used for Shape-Shape collision because it has no
        // shape support. So a very large box is used instead. In the future, it could be
        // a more complex topological object, even a mesh would be supported.
        auto groudBox = boost::shared_ptr<hpp::fcl::CollisionGeometry>(new hpp::fcl::Box(1000.0, 1000.0, 2.0));

        // Create a Pinocchio Geometry object associated with the ground plan.
        // Its parent frame and parent joint are the universe. It is aligned with world frame,
        // and the top face is the actual ground surface.
        pinocchio::SE3 groundPose = pinocchio::SE3::Identity();
        groundPose.translation() = (vector3_t() << 0.0, 0.0, -1.0).finished();
        pinocchio::GeometryObject groundPlane("ground", 0, 0, groudBox, groundPose);

        // Add the ground plane pinocchio to the robot model
        pncGeometryModel_.addGeometryObject(groundPlane, pncModel_);

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
            std::string const & jointFlexName = pncModelFlexibleOrig_.names[idxFlex];
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

    std::vector<std::string> const & Model::getCollisionBodiesNames(void) const
    {
        return collisionBodiesNames_;
    }

    std::vector<std::string> const & Model::getContactFramesNames(void) const
    {
        return contactFramesNames_;
    }

    std::vector<int32_t> const & Model::getCollisionBodiesIdx(void) const
    {
        return collisionBodiesIdx_;
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
