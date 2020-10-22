
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
    meshPackageDirs_(),
    hasFreeflyer_(false),
    mdlOptionsHolder_(),
    collisionBodiesNames_(),
    contactFramesNames_(),
    collisionBodiesIdx_(),
    collisionPairsIdx_(),
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

    hresult_t Model::initialize(std::string              const & urdfPath,
                                bool_t                   const & hasFreeflyer,
                                std::vector<std::string> const & meshPackageDirs)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        // Initialize the URDF model
        returnCode = loadUrdfModel(urdfPath, hasFreeflyer, meshPackageDirs);
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

        // Check that no frame with the same name already exists.
        if (pncModelRigidOrig_.existFrame(frameName))
        {
            std::cout << "Error - Model::addFrame - A frame with the same name already exists." << std::endl;
            returnCode = hresult_t::ERROR_BAD_INPUT;
        }

        // Add the frame to the the original rigid model
        int32_t parentFrameId;
        pinocchio::FrameType const frameType = pinocchio::FrameType::OP_FRAME;
        if (returnCode == hresult_t::SUCCESS)
        {
            returnCode = getFrameIdx(pncModelRigidOrig_, parentBodyName, parentFrameId);
        }
        if (returnCode == hresult_t::SUCCESS)
        {
            int32_t const & parentJointId = pncModelRigidOrig_.frames[parentFrameId].parent;
            pinocchio::SE3 const & parentFramePlacement = pncModelRigidOrig_.frames[parentFrameId].placement;
            pinocchio::SE3 const jointFramePlacement = parentFramePlacement.act(framePlacement);
            pinocchio::Frame const frame(frameName, parentJointId, parentFrameId, jointFramePlacement, frameType);
            pncModelRigidOrig_.addFrame(frame);
        }

        /* Add the frame to the the original flexible model.
           It can no longer fail at this point. */
        if (returnCode == hresult_t::SUCCESS)
        {
            getFrameIdx(pncModelFlexibleOrig_, parentBodyName, parentFrameId);
            int32_t const & parentJointId = pncModelFlexibleOrig_.frames[parentFrameId].parent;
            pinocchio::SE3 const & parentFramePlacement = pncModelFlexibleOrig_.frames[parentFrameId].placement;
            pinocchio::SE3 const jointFramePlacement = parentFramePlacement.act(framePlacement);
            pinocchio::Frame const frame(frameName, parentJointId, parentFrameId, jointFramePlacement, frameType);
            pncModelFlexibleOrig_.addFrame(frame);
        }

        /* One must re-generate the model after adding a frame.
           Note that it is unecessary to call 'reset' since the proxies
           are still up-to-date, because the frame is added at the end
           of the vector. */
        if (returnCode == hresult_t::SUCCESS)
        {
            generateModelBiased();
        }

        return returnCode;
    }

    hresult_t Model::removeFrame(std::string const & frameName)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        /* Check that the frame can be safely removed from the original rigid model.
           If so, it is also the case for the original flexible models. */
        int32_t frameId;
        pinocchio::FrameType const frameType = pinocchio::FrameType::OP_FRAME;
        returnCode = getFrameIdx(pncModelRigidOrig_, frameName, frameId);
        if (returnCode == hresult_t::SUCCESS)
        {
            if (pncModelRigidOrig_.frames[frameId].type != frameType)
            {
                std::cout << "Error - Model::removeFrame - Impossible to remove this frame. One should only remove frames added manually." << std::endl;
                returnCode = hresult_t::ERROR_BAD_INPUT;
            }
        }
        if (returnCode == hresult_t::SUCCESS)
        {
            // Remove the frame from the the original rigid model
            pncModelRigidOrig_.frames.erase(pncModelRigidOrig_.frames.begin() + frameId);
            pncModelRigidOrig_.nframes--;

            // Remove the frame from the the original flexible model
            getFrameIdx(pncModelFlexibleOrig_, frameName, frameId);
            pncModelFlexibleOrig_.frames.erase(pncModelFlexibleOrig_.frames.begin() + frameId);
            pncModelFlexibleOrig_.nframes--;
        }

        // One must reset the model after removing a frame
        if (returnCode == hresult_t::SUCCESS)
        {
            reset();
        }

        return returnCode;
    }

    hresult_t Model::addCollisionBodies(std::vector<std::string> const & bodyNames,
                                        bool_t const & ignoreMeshes)
    {
        if (!isInitialized_)
        {
            std::cout << "Error - Model::addCollisionBodies - Model not initialized." << std::endl;
            return hresult_t::ERROR_INIT_FAILED;
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
            std::cout << "Error - Model::addCollisionBodies - At least one of the bodies is already been associated with a collision." << std::endl;
            return hresult_t::ERROR_BAD_INPUT;
        }

        // Make sure that all the bodies exist
        for (std::string const & name : bodyNames)
        {
            if (!pncModel_.existBodyName(name))
            {
                std::cout << "Error - Model::addCollisionBodies - At least one of the bodies does not exist." << std::endl;
                return hresult_t::ERROR_BAD_INPUT;
            }
        }

        // Make sure that at least one geometry is associated with each body
        for (std::string const & name : bodyNames)
        {
            bool_t hasGeometry = false;
            for (pinocchio::GeometryObject const & geom : pncGeometryModel_.geometryObjects)
            {
                bool_t const isGeomMesh = (geom.meshPath.find('/') != std::string::npos ||
                                           geom.meshPath.find('\\') != std::string::npos);
                if (!(ignoreMeshes && isGeomMesh) &&  // geom.meshPath is the geometry type if it is not an actual mesh
                    pncModel_.frames[geom.parentFrame].name == name)
                {
                    hasGeometry = true;
                    break;
                }
            }
            if (!hasGeometry)
            {
                std::cout << "Error - Model::addCollisionBodies - At least one of the bodies is not associated with any collision geometry of requested type." << std::endl;
                return hresult_t::ERROR_BAD_INPUT;
            }
        }

        // Add the list of bodies to the set of collision bodies
        collisionBodiesNames_.insert(collisionBodiesNames_.end(), bodyNames.begin(), bodyNames.end());

        // Create the collision pairs and add them to the geometry model of the robot
        pinocchio::GeomIndex const & groundId = pncGeometryModel_.getGeometryId("ground");
        for (std::string const & name : bodyNames)
        {
            // Find the geometries having the body for parent, and add a collision pair for each of them
            for (uint32_t i=0; i<pncGeometryModel_.geometryObjects.size(); ++i)
            {
                pinocchio::GeometryObject const & geom = pncGeometryModel_.geometryObjects[i];
                bool_t const isGeomMesh = (geom.meshPath.find('/') != std::string::npos ||
                                           geom.meshPath.find('\\') != std::string::npos);
                if (!(ignoreMeshes && isGeomMesh) &&
                    pncModel_.frames[geom.parentFrame].name == name)
                {
                    /* Create and add the collision pair with the ground.
                       Note that the ground always comes second for the normal to be
                       consistently compute wrt the ground instead of the body. */
                    pinocchio::CollisionPair const collisionPair(i, groundId);
                    pncGeometryModel_.addCollisionPair(collisionPair);
                }
            }
        }

        // Refresh proxies associated with the collisions only
        refreshCollisionsProxies();

        return hresult_t::SUCCESS;
    }

    hresult_t Model::removeCollisionBodies(std::vector<std::string> bodyNames)
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
            std::cout << "Error - Model::removeCollisionBodies - At least one of the bodies is not associated with any collision." << std::endl;
            return hresult_t::ERROR_BAD_INPUT;
        }

        /* Remove the list of bodies from the set of collision bodies, then
           remove the associated set of collision pairs for each of them. */
        if (bodyNames.empty())
        {
            bodyNames = collisionBodiesNames_;
        }

        for (uint32_t i=0; i<bodyNames.size(); ++i)
        {
            std::string const & bodyName = bodyNames[i];
            auto collisionBodiesNameIt = std::find(
                collisionBodiesNames_.begin(),
                collisionBodiesNames_.end(),
                bodyName);
            int32_t collisionBodiesNameIdx = std::distance(
                collisionBodiesNames_.begin(),
                collisionBodiesNameIt);
            collisionBodiesNames_.erase(collisionBodiesNameIt);
            collisionPairsIdx_.erase(collisionPairsIdx_.begin() + collisionBodiesNameIdx);
        }

        // Get the indices of the corresponding collision pairs in the geometry model of the robot and remove them
        pinocchio::GeomIndex const & groundId = pncGeometryModel_.getGeometryId("ground");
        for (std::string const & name : bodyNames)
        {
            // Find the geometries having the body for parent, and remove the collision pair for each of them
            for (uint32_t i=0; i<pncGeometryModel_.geometryObjects.size(); ++i)
            {
                pinocchio::GeometryObject const & geom = pncGeometryModel_.geometryObjects[i];
                if (pncModel_.frames[geom.parentFrame].name == name)
                {
                    // Create and remove the collision pair with the ground
                    pinocchio::CollisionPair const collisionPair(i, groundId);
                    pncGeometryModel_.removeCollisionPair(collisionPair);
                }
            }
        }

        // Refresh proxies associated with the collisions only
        refreshCollisionsProxies();

        return hresult_t::SUCCESS;
    }

    hresult_t Model::addContactPoints(std::vector<std::string> const & frameNames)
    {
        if (!isInitialized_)
        {
            std::cout << "Error - Model::addContactPoints - Model not initialized." << std::endl;
            return hresult_t::ERROR_INIT_FAILED;
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
            std::cout << "Error - Model::addContactPoints - At least one of the frames is already been associated with a contact." << std::endl;
            return hresult_t::ERROR_BAD_INPUT;
        }

        // Make sure that all the frames exist
        for (std::string const & name : frameNames)
        {
            if (!pncModel_.existFrame(name))
            {
                std::cout << "Error - Model::addContactPoints - At least one of the frames does not exist." << std::endl;
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
            std::cout << "Error - Model::removeContactPoints - At least one of the frames is not associated with any contact." << std::endl;
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
                    for (uint32_t i=0; i < rigidJointsPositionIdx_.size(); ++i)
                    {
                        positionLimitMin_[rigidJointsPositionIdx_[i]] = mdlOptions_->joints.positionLimitMin[i];
                        positionLimitMax_[rigidJointsPositionIdx_[i]] = mdlOptions_->joints.positionLimitMax[i];
                    }
                }
            }

            /* Overwrite the position bounds for some specific joint type, mainly
               due to quaternion normalization and cos/sin representation. */
            for (int32_t i=0 ; i < pncModel_.njoints ; ++i)
            {
                joint_t jointType(joint_t::NONE);
                getJointTypeFromIdx(pncModel_, i, jointType);

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
                if (jointType == joint_t::ROTARY_UNBOUNDED)
                {
                    uint32_t const & positionIdx = pncModel_.joints[i].idx_q();
                    positionLimitMin_.segment<2>(positionIdx).setConstant(-1.0);
                    positionLimitMax_.segment<2>(positionIdx).setConstant(+1.0);
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
                    for (uint32_t i=0; i < rigidJointsVelocityIdx_.size(); ++i)
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

            // Extract the indices of the collision pairs associated with each body
            collisionPairsIdx_.clear();
            for (std::string const & name : collisionBodiesNames_)
            {
                std::vector<int32_t> collisionPairsIdx;
                for (uint32_t i=0; i<pncGeometryModel_.collisionPairs.size(); ++i)
                {
                    pinocchio::CollisionPair const & pair = pncGeometryModel_.collisionPairs[i];
                    pinocchio::GeometryObject const & geom = pncGeometryModel_.geometryObjects[pair.first];
                    if (pncModel_.frames[geom.parentFrame].name == name)
                    {
                        collisionPairsIdx.push_back(i);
                    }
                }
                collisionPairsIdx_.push_back(std::move(collisionPairsIdx));
            }

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
            bool_t positionLimitFromUrdf = boost::get<bool_t>(jointOptionsHolder.at("positionLimitFromUrdf"));
            if (!positionLimitFromUrdf)
            {
                vectorN_t & jointsPositionLimitMin = boost::get<vectorN_t>(jointOptionsHolder.at("positionLimitMin"));
                if ((int32_t) rigidJointsPositionIdx_.size() != jointsPositionLimitMin.size())
                {
                    std::cout << "Error - Model::setOptions - Wrong vector size for 'positionLimitMin'." << std::endl;
                    return hresult_t::ERROR_BAD_INPUT;
                }
                auto jointsPositionLimitMinDiff = jointsPositionLimitMin - mdlOptions_->joints.positionLimitMin;
                internalBuffersMustBeUpdated |= (jointsPositionLimitMinDiff.array().abs() >= EPS).all();
                vectorN_t & jointsPositionLimitMax = boost::get<vectorN_t>(jointOptionsHolder.at("positionLimitMax"));
                if ((uint32_t) rigidJointsPositionIdx_.size() != jointsPositionLimitMax.size())
                {
                    std::cout << "Error - Model::setOptions - Wrong vector size for 'positionLimitMax'." << std::endl;
                    return hresult_t::ERROR_BAD_INPUT;
                }
                auto jointsPositionLimitMaxDiff = jointsPositionLimitMax - mdlOptions_->joints.positionLimitMax;
                internalBuffersMustBeUpdated |= (jointsPositionLimitMaxDiff.array().abs() >= EPS).all();
            }
            bool_t velocityLimitFromUrdf = boost::get<bool_t>(jointOptionsHolder.at("velocityLimitFromUrdf"));
            if (!velocityLimitFromUrdf)
            {
                vectorN_t & jointsVelocityLimit = boost::get<vectorN_t>(jointOptionsHolder.at("velocityLimit"));
                if ((int32_t) rigidJointsVelocityIdx_.size() != jointsVelocityLimit.size())
                {
                    std::cout << "Error - Model::setOptions - Wrong vector size for 'velocityLimit'." << std::endl;
                    return hresult_t::ERROR_BAD_INPUT;
                }
                auto jointsVelocityLimitDiff = jointsVelocityLimit - mdlOptions_->joints.velocityLimit;
                internalBuffersMustBeUpdated |= (jointsVelocityLimitDiff.array().abs() >= EPS).all();
            }

            // Check if the position or velocity limits have changed, and refresh proxies if so
            bool_t enablePositionLimit = boost::get<bool_t>(jointOptionsHolder.at("enablePositionLimit"));
            bool_t enableVelocityLimit = boost::get<bool_t>(jointOptionsHolder.at("enableVelocityLimit"));
            if (enablePositionLimit != mdlOptions_->joints.enablePositionLimit)
            {
                internalBuffersMustBeUpdated = true;
            }
            else if (enablePositionLimit && (positionLimitFromUrdf != mdlOptions_->joints.positionLimitFromUrdf))
            {
                internalBuffersMustBeUpdated = true;
            }
            else if (enableVelocityLimit != mdlOptions_->joints.enableVelocityLimit)
            {
                internalBuffersMustBeUpdated = true;
            }
            else if (enableVelocityLimit && (velocityLimitFromUrdf != mdlOptions_->joints.velocityLimitFromUrdf))
            {
                internalBuffersMustBeUpdated = true;
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

    std::vector<std::string> const & Model::getMeshPackageDirs(void) const
    {
        return meshPackageDirs_;
    }

    bool_t const & Model::getHasFreeflyer(void) const
    {
        return hasFreeflyer_;
    }

    hresult_t Model::loadUrdfModel(std::string              const & urdfPath,
                                   bool_t                   const & hasFreeflyer,
                                   std::vector<std::string>         meshPackageDirs)
    {
        if (!std::ifstream(urdfPath.c_str()).good())
        {
            std::cout << "Error - Model::loadUrdfModel - The URDF file does not exist. Impossible to load it." << std::endl;
            return hresult_t::ERROR_BAD_INPUT;
        }

        urdfPath_ = urdfPath;
        meshPackageDirs_ = meshPackageDirs;
        hasFreeflyer_ = hasFreeflyer;

        try
        {
            // Build robot physics model
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
            std::cout << "Error - Model::loadUrdfModel - Something is wrong with the URDF. Impossible to build a model from it.\n"
                      << "Raised by exception: " << e.what() << std::endl;
            return hresult_t::ERROR_BAD_INPUT;
        }

        try
        {
            // Build robot geometry model
            pinocchio::urdf::buildGeom(pncModel_, urdfPath, pinocchio::COLLISION, pncGeometryModel_, meshPackageDirs);
        }
        catch (std::exception& e)
        {
            std::cout << "Error - Model::loadUrdfModel - Something is wrong with the URDF. Impossible to load the collision geometries.\n"
                      << "Raised by exception: " << e.what() << std::endl;
            return hresult_t::ERROR_BAD_INPUT;
        }

        // Replace the mesh geometry object by its convex representation for efficiency
        #if PINOCCHIO_MINOR_VERSION >= 4 || PINOCCHIO_PATCH_VERSION >= 4
        for (uint32_t i=0; i<pncGeometryModel_.geometryObjects.size(); ++i)
        {
            hpp::fcl::BVHModelPtr_t bvh = boost::dynamic_pointer_cast<hpp::fcl::BVHModelBase>(pncGeometryModel_.geometryObjects[i].geometry);
            if (bvh)
            {
                // If the dynamic cast succeeded (bvh is not nullptr), it means that the object
                // actually derive from the BVH model (cloud points or triangles).
                bvh->buildConvexHull(true);
                pncGeometryModel_.geometryObjects[i].geometry = bvh->convex;
            }
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

    hresult_t Model::getFlexibleStateFromRigid(vectorN_t const & qRigid,
                                               vectorN_t const & vRigid,
                                               vectorN_t       & qFlex,
                                               vectorN_t       & vFlex) const
    {
        // Define some proxies
        uint32_t const & nqRigid = pncModelRigidOrig_.nq;
        uint32_t const & nvRigid = pncModelRigidOrig_.nv;
        uint32_t const & nvFlex = pncModelFlexibleOrig_.nv;

        // Check the size of the input state
        if (qRigid.size() != nqRigid || vRigid.size() != nvRigid)
        {
            std::cout << "Error - Model::getFlexibleStateFromRigid - Size of xRigid inconsistent with theoretical model." << std::endl;
            return hresult_t::ERROR_BAD_INPUT;
        }

        // Initialize the flexible state
        qFlex = pinocchio::neutral(pncModelFlexibleOrig_);
        vFlex = vectorN_t::Zero(nvFlex);

        // Compute the flexible state based on the rigid state
        int32_t idxRigid = 0;
        int32_t idxFlex = 0;
        for (; idxRigid < pncModelRigidOrig_.njoints; ++idxFlex)
        {
            std::string const & jointRigidName = pncModelRigidOrig_.names[idxRigid];
            std::string const & jointFlexName = pncModelFlexibleOrig_.names[idxFlex];
            if (jointRigidName == jointFlexName)
            {
                auto const & jointRigid = pncModelRigidOrig_.joints[idxRigid];
                auto const & jointFlex = pncModelFlexibleOrig_.joints[idxFlex];
                if (jointRigid.idx_q() >= 0)
                {
                    qFlex.segment(jointFlex.idx_q(), jointFlex.nq()) =
                        qRigid.segment(jointRigid.idx_q(), jointRigid.nq());
                    vFlex.segment(jointFlex.idx_v(), jointFlex.nv()) =
                        vRigid.segment(jointRigid.idx_v(), jointRigid.nv());
                }
                ++idxRigid;
            }
        }

        return hresult_t::SUCCESS;
    }

    hresult_t Model::getRigidStateFromFlexible(vectorN_t const & qFlex,
                                               vectorN_t const & vFlex,
                                               vectorN_t       & qRigid,
                                               vectorN_t       & vRigid) const
    {
        // Define some proxies
        uint32_t const & nvRigid = pncModelRigidOrig_.nv;
        uint32_t const & nqFlex = pncModelFlexibleOrig_.nq;
        uint32_t const & nvFlex = pncModelFlexibleOrig_.nv;

        // Check the size of the input state
        if (qFlex.size() != nqFlex || vFlex.size() != nvFlex)
        {
            std::cout << "Error - Model::getRigidStateFromFlexible - Size of xFlex inconsistent with flexible model." << std::endl;
            return hresult_t::ERROR_BAD_INPUT;
        }

        // Initialize the rigid state
        qRigid = pinocchio::neutral(pncModelRigidOrig_);
        vRigid = vectorN_t::Zero(nvRigid);

        // Compute the rigid state based on the flexible state
        int32_t idxRigid = 0;
        int32_t idxFlex = 0;
        for (; idxFlex < pncModelFlexibleOrig_.njoints; ++idxRigid, ++idxFlex)
        {
            std::string const & jointRigidName = pncModelRigidOrig_.names[idxRigid];
            std::string const & jointFlexName = pncModelFlexibleOrig_.names[idxFlex];
            if (jointRigidName == jointFlexName)
            {
                auto const & jointRigid = pncModelRigidOrig_.joints[idxRigid];
                auto const & jointFlex = pncModelFlexibleOrig_.joints[idxFlex];
                if (jointRigid.idx_q() >= 0)
                {
                    qRigid.segment(jointRigid.idx_q(), jointRigid.nq()) =
                        qFlex.segment(jointFlex.idx_q(), jointFlex.nq());
                    vRigid.segment(jointRigid.idx_v(), jointRigid.nv()) =
                        vFlex.segment(jointFlex.idx_v(), jointFlex.nv());
                }
            }
            else
            {
                ++idxFlex;
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

    std::vector<std::vector<int32_t> > const & Model::getCollisionPairsIdx(void) const
    {
        return collisionPairsIdx_;
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
