
#include <iostream>
#include <fstream>
#include <exception>

#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/center-of-mass.hpp"

#include <Eigen/Eigenvalues>

#include "urdf_parser/urdf_parser.h"

#include "jiminy/core/robot/PinocchioOverloadAlgorithms.h"
#include "jiminy/core/robot/AbstractConstraint.h"
#include "jiminy/core/robot/JointConstraint.h"
#include "jiminy/core/robot/SphereConstraint.h"
#include "jiminy/core/robot/FixedFrameConstraint.h"
#include "jiminy/core/Utilities.h"
#include "jiminy/core/Constants.h"

#include "jiminy/core/robot/Model.h"


namespace jiminy
{
    constraintsHolder_t::constraintsHolder_t(void) :
    boundJoints(),
    contactFrames(),
    collisionBodies(),
    registered()
    {
        // Empty on purpose
    }

    void constraintsHolder_t::clear(void)
    {
        boundJoints.clear();
        contactFrames.clear();
        collisionBodies.clear();
        registered.clear();
    }

    constraintsMap_t::iterator getImpl(constraintsMap_t & constraintsMap,
                                       std::string const & key)
    {
        return std::find_if(constraintsMap.begin(),
                            constraintsMap.end(),
                            [&key](auto const & constraintPair)
                            {
                                return constraintPair.first == key;
                            });
    }

    std::tuple<constraintsMap_t *, constraintsMap_t::iterator>
        constraintsHolder_t::find(std::string const & key,
                                  constraintsHolderType_t const & holderType)
    {
        constraintsMap_t * constraintsMapPtr = nullptr;  // Pointers are NOT initialized to nullptr by default
        constraintsMap_t::iterator constraintIt;
        if (holderType == constraintsHolderType_t::COLLISION_BODIES)
        {
            for (uint32_t i = 0; i < collisionBodies.size(); ++i)
            {
                constraintsMapPtr = &collisionBodies[i];
                constraintIt = getImpl(*constraintsMapPtr, key);
                if (constraintIt != constraintsMapPtr->end())
                {
                    break;
                }
            }
        }
        else
        {
            switch (holderType)
            {
            case constraintsHolderType_t::BOUNDS_JOINTS:
                constraintsMapPtr = &boundJoints;
                break;
            case constraintsHolderType_t::CONTACT_FRAMES:
                constraintsMapPtr = &contactFrames;
                break;
            case constraintsHolderType_t::USER:
            case constraintsHolderType_t::COLLISION_BODIES:
            default:
                constraintsMapPtr = &registered;
            }
            constraintIt = getImpl(*constraintsMapPtr, key);
        }

        return {constraintsMapPtr, constraintIt};
    }

    bool_t constraintsHolder_t::exist(std::string const & key,
                                      constraintsHolderType_t const & holderType) const
    {
        auto [constraintsMapPtr, constraintIt] = const_cast<constraintsHolder_t *>(this)->find(key, holderType);
        return (constraintsMapPtr && constraintIt != constraintsMapPtr->end());
    }

    bool_t constraintsHolder_t::exist(std::string const & key) const
    {
        for (constraintsHolderType_t const & holderType : constraintsHolderTypeRange)
        {
            if (exist(key, holderType))
            {
                return true;
            }
        }
        return false;
    }

    std::shared_ptr<AbstractConstraintBase> constraintsHolder_t::get(std::string const & key,
                                                                 constraintsHolderType_t const & holderType)
    {
        auto [constraintsMapPtr, constraintIt] = find(key, holderType);
        if (constraintsMapPtr && constraintIt != constraintsMapPtr->end())
        {
            return constraintIt->second;
        }
        return {};
    }

    std::shared_ptr<AbstractConstraintBase> constraintsHolder_t::get(std::string const & key)
    {
        std::shared_ptr<AbstractConstraintBase> constraint;
        for (constraintsHolderType_t const & holderType : constraintsHolderTypeRange)
        {
            constraint = get(key, holderType);
            if (constraint)
            {
                return constraint;
            }
        }
        return {};
    }

    void constraintsHolder_t::insert(constraintsMap_t const & constraintsMap,
                                     constraintsHolderType_t const & holderType)
    {
        switch (holderType)
        {
        case constraintsHolderType_t::BOUNDS_JOINTS:
            boundJoints.insert(boundJoints.end(), constraintsMap.begin(), constraintsMap.end());
            break;
        case constraintsHolderType_t::CONTACT_FRAMES:
            contactFrames.insert(contactFrames.end(), constraintsMap.begin(), constraintsMap.end());
            break;
        case constraintsHolderType_t::COLLISION_BODIES:
            collisionBodies.push_back(constraintsMap);
            break;
        case constraintsHolderType_t::USER:
        default:
            registered.insert(registered.end(), constraintsMap.begin(), constraintsMap.end());
        }
    }

    constraintsMap_t::iterator constraintsHolder_t::erase(std::string const & key,
                                                          constraintsHolderType_t const & holderType)
    {
        auto [constraintsMapPtr, constraintIt] = find(key, holderType);
        if (constraintsMapPtr && constraintIt != constraintsMapPtr->end())
        {
            return constraintsMapPtr->erase(constraintIt);
        }
        return constraintsMapPtr->end();
    }

    Model::Model(void) :
    pncModel_(),
    pncData_(),
    pncGeometryModel_(),
    pncGeometryData_(nullptr),
    pncModelRigidOrig_(),
    pncDataRigidOrig_(),
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
    constraintsHolder_(),
    constraintsMask_(0U),
    constraintsJacobian_(),
    constraintsDrift_(),
    positionLimitMin_(),
    positionLimitMax_(),
    velocityLimit_(),
    positionFieldnames_(),
    velocityFieldnames_(),
    accelerationFieldnames_(),
    pncModelFlexibleOrig_(),
    jointsAcceleration_(),
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

        // Clear existing constraints
        constraintsHolder_.clear();
        constraintsMask_ = 0U;
        constraintsJacobian_.resize(0, 0);
        constraintsDrift_.resize(0);
        jointsAcceleration_.clear();

        // Initialize the URDF model
        returnCode = loadUrdfModel(urdfPath, hasFreeflyer, meshPackageDirs);
        isInitialized_ = true;

        if (returnCode == hresult_t::SUCCESS)
        {
            // Backup the original model and data
            pncDataRigidOrig_ = pinocchio::Data(pncModelRigidOrig_);

            // Initialize Pinocchio data internal state, including basic
            // attributes such as the mass of each body.
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
            rigidJointsNames_.erase(rigidJointsNames_.begin());  // remove the 'universe'
            if (hasFreeflyer)
            {
                rigidJointsNames_.erase(rigidJointsNames_.begin());  // remove the 'root'
            }

            // Create the flexible model
            returnCode = generateModelFlexible();
        }

        /* Add biases to the dynamics properties of the model.
           Note that is also refresh all proxies automatically. */
        if (returnCode == hresult_t::SUCCESS)
        {
            returnCode = generateModelBiased();
        }

        /* Add joint constraints.
           It will be used later to enforce bounds limits eventually. */
        if (returnCode == hresult_t::SUCCESS)
        {
            constraintsMap_t jointConstraintsMap;
            jointConstraintsMap.reserve(rigidJointsNames_.size());
            for (std::string const & jointName : rigidJointsNames_)
            {
                jointConstraintsMap.emplace_back(jointName, std::make_shared<JointConstraint>(jointName));
            }
            addConstraints(jointConstraintsMap, constraintsHolderType_t::BOUNDS_JOINTS);  // It cannot fail at this point
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
        /* Do NOT reset the constraints, since it will be handled by the engine.
           It is necessary, since the current model state is required to
           initialize the constraints, at least for baumgarte stabilization.
           Indeed, the current frame position must be stored. */

        if (isInitialized_)
        {
            // Update the biases added to the dynamics properties of the model
            generateModelBiased();
        }
    }

    hresult_t Model::addFrame(std::string          const & frameName,
                              std::string          const & parentBodyName,
                              pinocchio::SE3       const & framePlacement,
                              pinocchio::FrameType const & frameType)
    {
        // Note that since it is not possible to add a frame to another frame,
        // the frame is added directly to the parent joint, thus relative transform
        // of the frame wrt the parent joint must be computed.
        hresult_t returnCode = hresult_t::SUCCESS;

        // Check that no frame with the same name already exists.
        if (pncModelRigidOrig_.existFrame(frameName))
        {
            PRINT_ERROR("A frame with the same name already exists.");
            returnCode = hresult_t::ERROR_BAD_INPUT;
        }

        // Check that parent frame exists
        int32_t parentFrameId = 0;
        if (returnCode == hresult_t::SUCCESS)
        {
            returnCode = getFrameIdx(pncModelRigidOrig_, parentBodyName, parentFrameId);
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            // Add the frame to the the original rigid model
            {
                int32_t const & parentJointId = pncModelRigidOrig_.frames[parentFrameId].parent;
                pinocchio::SE3 const & parentFramePlacement = pncModelRigidOrig_.frames[parentFrameId].placement;
                pinocchio::SE3 const jointFramePlacement = parentFramePlacement.act(framePlacement);
                pinocchio::Frame const frame(frameName, parentJointId, parentFrameId, jointFramePlacement, frameType);
                pncModelRigidOrig_.addFrame(frame);
                pncDataRigidOrig_ = pinocchio::Data(pncModelRigidOrig_);
            }

            // Add the frame to the the original flexible model
            {
                getFrameIdx(pncModelFlexibleOrig_, parentBodyName, parentFrameId);  // It can no longer fail at this point.
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
            generateModelBiased();
        }

        return returnCode;
    }

    hresult_t Model::addFrame(std::string    const & frameName,
                              std::string    const & parentBodyName,
                              pinocchio::SE3 const & framePlacement)
    {
        pinocchio::FrameType const frameType = pinocchio::FrameType::OP_FRAME;
        return addFrame(frameName, parentBodyName, framePlacement, frameType);
    }

    hresult_t Model::removeFrames(std::vector<std::string> const & frameNames)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        /* Check that the frame can be safely removed from the original rigid model.
           If so, it is also the case for the original flexible models. */
        for (std::string const & frameName : frameNames)
        {
            int32_t frameId;
            pinocchio::FrameType const frameType = pinocchio::FrameType::OP_FRAME;
            returnCode = getFrameIdx(pncModelRigidOrig_, frameName, frameId);
            if (returnCode == hresult_t::SUCCESS)
            {
                if (pncModelRigidOrig_.frames[frameId].type != frameType)
                {
                    PRINT_ERROR("Impossible to remove this frame. One should only remove frames added manually.");
                    returnCode = hresult_t::ERROR_BAD_INPUT;
                }
            }
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            for (std::string const & frameName : frameNames)
            {
                // Get the frame idx
                int32_t frameId;
                getFrameIdx(pncModelRigidOrig_, frameName, frameId);  // It cannot fail

                // Remove the frame from the the original rigid model
                pncModelRigidOrig_.frames.erase(pncModelRigidOrig_.frames.begin() + frameId);
                pncModelRigidOrig_.nframes--;

                // Remove the frame from the the original flexible model
                getFrameIdx(pncModelFlexibleOrig_, frameName, frameId);
                pncModelFlexibleOrig_.frames.erase(pncModelFlexibleOrig_.frames.begin() + frameId);
                pncModelFlexibleOrig_.nframes--;
            }

            // Regenerate rigid data
            pncDataRigidOrig_ = pinocchio::Data(pncModelRigidOrig_);

            // One must reset the model after removing a frame
            reset();
        }

        return returnCode;
    }

    hresult_t Model::removeFrame(std::string const & frameName)
    {
        return removeFrames({frameName});
    }

    hresult_t Model::addCollisionBodies(std::vector<std::string> const & bodyNames,
                                        bool_t const & ignoreMeshes)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (!isInitialized_)
        {
            PRINT_ERROR("Model not initialized.");
            return hresult_t::ERROR_INIT_FAILED;
        }

        if (bodyNames.empty())
        {
            return hresult_t::SUCCESS;  // Nothing to do. Returning early.
        }

        if (pncGeometryModel_.ngeoms == 0)  // If successfully loaded, the ground should be available
        {
            PRINT_ERROR("Collision geometry not available. Some collision meshes were probably not found.");
            return hresult_t::ERROR_INIT_FAILED;
        }

        // Make sure that no body are duplicates
        if (checkDuplicates(bodyNames))
        {
            PRINT_ERROR("Some bodies are duplicates.");
            return hresult_t::ERROR_BAD_INPUT;
        }

        // Make sure that there is no collision already associated with any of the bodies in the list
        if (checkIntersection(collisionBodiesNames_, bodyNames))
        {
            PRINT_ERROR("At least one of the bodies is already been associated with a collision.");
            return hresult_t::ERROR_BAD_INPUT;
        }

        // Make sure that all the bodies exist
        for (std::string const & name : bodyNames)
        {
            if (!pncModel_.existBodyName(name))
            {
                PRINT_ERROR("At least one of the bodies does not exist.");
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
                PRINT_ERROR("At least one of the bodies is not associated with any collision geometry of requested type.");
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
            constraintsMap_t collisionConstraintsMap;
            for (uint32_t i=0; i<pncGeometryModel_.geometryObjects.size(); ++i)
            {
                pinocchio::GeometryObject const & geom = pncGeometryModel_.geometryObjects[i];
                bool_t const isGeomMesh = (geom.meshPath.find('/') != std::string::npos ||
                                           geom.meshPath.find('\\') != std::string::npos);
                std::string const & frameName = pncModel_.frames[geom.parentFrame].name;
                if (!(ignoreMeshes && isGeomMesh) && frameName  == name)
                {
                    /* Create and add the collision pair with the ground.
                       Note that the ground always comes second for the normal to be
                       consistently compute wrt the ground instead of the body. */
                    pinocchio::CollisionPair const collisionPair(i, groundId);
                    pncGeometryModel_.addCollisionPair(collisionPair);

                    if (returnCode == hresult_t::SUCCESS)
                    {
                        // Add constraint associated with contact frame only if it is a sphere
                        hpp::fcl::CollisionGeometry const & shape = *geom.geometry;
                        if (shape.getNodeType() == hpp::fcl::GEOM_SPHERE)
                        {
                            /* Add dedicated frame
                               Note that 'BODY' type is used instead of default 'OP_FRAME' to
                               it clear it is not consider as manually added to the model, and
                               therefore cannot be deleted by the user. */
                            pinocchio::FrameType const frameType = pinocchio::FrameType::FIXED_JOINT;
                            returnCode = addFrame(geom.name, frameName, geom.placement, frameType);

                            // Add fixed frame constraint of bounded sphere
                            // hpp::fcl::Sphere const & sphere = static_cast<hpp::fcl::Sphere const &>(shape);
                            // collisionConstraintsMap.emplace_back(geom.name, std::make_shared<SphereConstraint>(
                            //     geom.name, sphere.radius));
                            collisionConstraintsMap.emplace_back(geom.name, std::make_shared<FixedFrameConstraint>(
                                geom.name, true, false));
                        }
                        else
                        {
                            collisionConstraintsMap.emplace_back(geom.name, nullptr);
                        }
                    }
                }
            }

            // Add constraints map, even if nullptr for geometry shape not supported
            if (returnCode == hresult_t::SUCCESS)
            {
                returnCode = addConstraints(collisionConstraintsMap, constraintsHolderType_t::COLLISION_BODIES);
            }
        }


        // Refresh proxies associated with the collisions only
        if (returnCode == hresult_t::SUCCESS)
        {
            refreshCollisionsProxies();
        }

        return returnCode;
    }

    hresult_t Model::removeCollisionBodies(std::vector<std::string> bodyNames)
    {
        if (!isInitialized_)
        {
            PRINT_ERROR("Model not initialized.");
            return hresult_t::ERROR_INIT_FAILED;
        }

        // Make sure that no body are duplicates
        if (checkDuplicates(bodyNames))
        {
            PRINT_ERROR("Some bodies are duplicates.");
            return hresult_t::ERROR_BAD_INPUT;
        }

        // Make sure that every body in the list is associated with a collision
        if (!checkInclusion(collisionBodiesNames_, bodyNames))
        {
            PRINT_ERROR("At least one of the bodies is not associated with any collision.");
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
        std::vector<std::string> collisionConstraintsNames;
        pinocchio::GeomIndex const & groundId = pncGeometryModel_.getGeometryId("ground");
        for (std::string const & name : bodyNames)
        {
            // Find the geometries having the body for parent, and remove the collision pair for each of them
            for (uint32_t i=0; i<pncGeometryModel_.geometryObjects.size(); ++i)
            {
                pinocchio::GeometryObject const & geom = pncGeometryModel_.geometryObjects[i];
                if (pncModel_.frames[geom.parentFrame].name == name)
                {
                    // Remove the collision pair with the ground
                    pinocchio::CollisionPair const collisionPair(i, groundId);
                    pncGeometryModel_.removeCollisionPair(collisionPair);

                    // Append collision geometry to the list of constraints to remove
                    if (constraintsHolder_.exist(geom.name,  constraintsHolderType_t::COLLISION_BODIES))
                    {
                        collisionConstraintsNames.emplace_back(geom.name);
                    }
                }
            }
        }

        // Remove the constraints and associated frames
        removeConstraints(collisionConstraintsNames, constraintsHolderType_t::COLLISION_BODIES);
        removeFrames(collisionConstraintsNames);

        // Refresh proxies associated with the collisions only
        refreshCollisionsProxies();

        return hresult_t::SUCCESS;
    }

    hresult_t Model::addContactPoints(std::vector<std::string> const & frameNames)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (!isInitialized_)
        {
            PRINT_ERROR("Model not initialized.");
            return hresult_t::ERROR_INIT_FAILED;
        }

        // Make sure that no frame are duplicates
        if (checkDuplicates(frameNames))
        {
            PRINT_ERROR("Some frames are duplicates.");
            return hresult_t::ERROR_BAD_INPUT;
        }

        // Make sure that there is no contact already associated with any of the frames in the list
        if (checkIntersection(contactFramesNames_, frameNames))
        {
            PRINT_ERROR("At least one of the frames is already been associated with a contact.");
            return hresult_t::ERROR_BAD_INPUT;
        }

        // Make sure that all the frames exist
        for (std::string const & name : frameNames)
        {
            if (!pncModel_.existFrame(name))
            {
                PRINT_ERROR("At least one of the frames does not exist.");
                return hresult_t::ERROR_BAD_INPUT;
            }
        }

        // Add the list of frames to the set of contact frames
        contactFramesNames_.insert(contactFramesNames_.end(), frameNames.begin(), frameNames.end());

        // Add constraint associated with contact frame
        constraintsMap_t frameConstraintsMap;
        frameConstraintsMap.reserve(frameNames.size());
        for (std::string const & frameName : frameNames)
        {
            frameConstraintsMap.emplace_back(frameName, std::make_shared<FixedFrameConstraint>(
                frameName, true, false));
        }
        returnCode = addConstraints(frameConstraintsMap, constraintsHolderType_t::CONTACT_FRAMES);

        // Refresh proxies associated with contacts and constraints
        if (returnCode == hresult_t::SUCCESS)
        {
            refreshContactsProxies();
        }

        return returnCode;
    }

    hresult_t Model::removeContactPoints(std::vector<std::string> const & frameNames)
    {
        if (!isInitialized_)
        {
            PRINT_ERROR("Model not initialized.");
            return hresult_t::ERROR_INIT_FAILED;
        }

        // Make sure that no frame are duplicates
        if (checkDuplicates(frameNames))
        {
            PRINT_ERROR("Some frames are duplicates.");
            return hresult_t::ERROR_BAD_INPUT;
        }

        // Make sure that every frame in the list is associated with a contact
        if (!checkInclusion(contactFramesNames_, frameNames))
        {
            PRINT_ERROR("At least one of the frames is not associated with any contact.");
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

        // Remove constraint associated with contact frame, disable by default
        for (std::string const & frameName : frameNames)
        {
            removeConstraint(frameName, constraintsHolderType_t::CONTACT_FRAMES);  // It cannot fail at this point
        }

        // Refresh proxies associated with contacts and constraints
        refreshContactsProxies();

        return hresult_t::SUCCESS;
    }

    hresult_t Model::addConstraints(constraintsMap_t const & constraintsMap,
                                    constraintsHolderType_t const & holderType)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        // Look for constraint in every constraint holders sequentially
        for (auto const & constraintPair : constraintsMap)
        {
            std::string const & constraintName = constraintPair.first;
            if (constraintsHolder_.exist(constraintName))
            {
                PRINT_ERROR("A constraint with name '", constraintName, "' already exists.");
                return hresult_t::ERROR_BAD_INPUT;
            }
        }

        // Attach constraint if not already exist
        for (auto & constraintPair : constraintsMap)
        {
            if (returnCode == hresult_t::SUCCESS)
            {
                returnCode = constraintPair.second->attach(shared_from_this());
            }
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            // Add them to constraints holder
            constraintsHolder_.insert(constraintsMap, holderType);

            // Disable internal constraint by default if internal
            if (holderType != constraintsHolderType_t::USER)
            {
                for (auto & constraintItem : constraintsMap)
                {
                    constraintItem.second->disable();
                }
            }

            // Required to resize constraintsJacobian_ to the right size
            returnCode = refreshConstraintsProxies();
        }

        return returnCode;
    }

    hresult_t Model::addConstraint(std::string const & constraintName,
                                   std::shared_ptr<AbstractConstraintBase> const & constraint,
                                   constraintsHolderType_t const & holderType)
    {
        return addConstraints({{constraintName, constraint}}, holderType);
    }

    hresult_t Model::addConstraint(std::string const & constraintName,
                                   std::shared_ptr<AbstractConstraintBase> const & constraint)
    {
        return addConstraint(constraintName, constraint, constraintsHolderType_t::USER);
    }

    hresult_t Model::removeConstraints(std::vector<std::string> const & constraintsNames,
                                       constraintsHolderType_t const & holderType)
    {
        // Make sure the constraints exists
        for (std::string const & constraintName : constraintsNames)
        {
            if (!constraintsHolder_.exist(constraintName, holderType))
            {
                if (holderType == constraintsHolderType_t::USER)
                {
                    PRINT_ERROR("No constraint with this name exists.");
                }
                else
                {
                    PRINT_ERROR("No internal constraint with this name exists.");
                }
                return hresult_t::ERROR_BAD_INPUT;
            }
        }

        // Remove every constraint sequentially
        for (std::string const & constraintName : constraintsNames)
        {
            // Lookup constraint
            auto [constraintsMapPtr, constraintIt] = constraintsHolder_.find(constraintName, holderType);

            // Detach the constraint
            constraintIt->second->detach();  // It cannot fail at this point

            // Remove the constraint from the holder
            constraintsMapPtr->erase(constraintIt);
        }

        // Required to resize constraintsJacobian_ to the right size.
        refreshConstraintsProxies();

        return hresult_t::SUCCESS;
    }

    hresult_t Model::removeConstraint(std::string const & constraintName,
                                      constraintsHolderType_t const & holderType)
    {
        return removeConstraints({constraintName}, holderType);
    }

    hresult_t Model::removeConstraint(std::string const & constraintName)
    {
        return removeConstraint(constraintName, constraintsHolderType_t::USER);
    }

    hresult_t Model::getConstraint(std::string const & constraintName,
                                   std::shared_ptr<AbstractConstraintBase> & constraint)
    {
        constraint = constraintsHolder_.get(constraintName);
        if (!constraint)
        {
            PRINT_ERROR("No constraint with this name exists.");
            return hresult_t::ERROR_BAD_INPUT;
        }
        return hresult_t::SUCCESS;
    }

    hresult_t Model::getConstraint(std::string const & constraintName,
                                   std::weak_ptr<AbstractConstraintBase const> & constraint) const
    {
        constraint = std::const_pointer_cast<AbstractConstraintBase const>(
            const_cast<constraintsHolder_t &>(constraintsHolder_).get(constraintName));
        if (!constraint.lock())
        {
            PRINT_ERROR("No constraint with this name exists.");
            return hresult_t::ERROR_BAD_INPUT;
        }
        return hresult_t::SUCCESS;
    }

    constraintsHolder_t Model::getConstraints(void)
    {
        return constraintsHolder_;
    }

    bool_t Model::existConstraint(std::string const & constraintName) const
    {
        return constraintsHolder_.exist(constraintName);
    }

    hresult_t Model::resetConstraints(vectorN_t const & q,
                                      vectorN_t const & v)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        constraintsHolder_.foreach(
            [&q, &v, &returnCode](std::shared_ptr<AbstractConstraintBase> const & constraint,
                                  constraintsHolderType_t const & /* holderType */)
            {
                if (!constraint)
                {
                    return;
                }

                if (returnCode == hresult_t::SUCCESS)
                {
                    returnCode = constraint->reset(q, v);
                }
            });

        if (returnCode == hresult_t::SUCCESS)
        {
            auto lambda = [](std::shared_ptr<AbstractConstraintBase> const & constraint,
                             constraintsHolderType_t const & /* holderType */)
                          {
                              if (constraint)
                              {
                                  constraint->disable();
                              }
                          };
            constraintsHolder_.foreach(constraintsHolderType_t::BOUNDS_JOINTS, lambda);
            constraintsHolder_.foreach(constraintsHolderType_t::CONTACT_FRAMES, lambda);
            constraintsHolder_.foreach(constraintsHolderType_t::COLLISION_BODIES, lambda);
        }

        return returnCode;
    }

    static pinocchio::Inertia convertFromUrdf(::urdf::Inertial const & Y)
    {
        ::urdf::Vector3 const & p = Y.origin.position;
        vector3_t const com(p.x, p.y, p.z);
        ::urdf::Rotation const & q = Y.origin.rotation;
        matrix3_t const R = Eigen::Quaterniond(q.w, q.x, q.y, q.z).matrix();
        matrix3_t I;
        I << Y.ixx, Y.ixy, Y.ixz,
             Y.ixy, Y.iyy, Y.iyz,
             Y.ixz, Y.iyz, Y.izz;
        return {Y.mass, com, R*I*R.transpose()};
    }

    static pinocchio::Inertia getChildBodyInertiaFromUrdf(std::string const & urdfPath,
                                                          std::string const & frameName)
    {
        ::urdf::ModelInterfaceSharedPtr urdfTree = ::urdf::parseURDFFile(urdfPath);
        ::urdf::JointConstSharedPtr joint = urdfTree->getJoint(frameName);
        std::string const & child_link_name = joint->child_link_name;
        ::urdf::LinkConstSharedPtr child_link = urdfTree->getLink(child_link_name);
        return convertFromUrdf(*child_link->inertial);
    }

    hresult_t Model::generateModelFlexible(void)
    {
        if (!isInitialized_)
        {
            PRINT_ERROR("Model not initialized.");
            return hresult_t::ERROR_INIT_FAILED;
        }

        flexibleJointsNames_.clear();
        flexibleJointsModelIdx_.clear();
        pncModelFlexibleOrig_ = pncModelRigidOrig_;
        for(flexibleJointData_t const & flexibleJoint : mdlOptions_->dynamics.flexibilityConfig)
        {
            // Check if joint name exists
            std::string const & frameName = flexibleJoint.frameName;
            if (!pncModel_.existFrame(frameName))
            {
                return hresult_t::ERROR_GENERIC;
            }

            // Add joint to model, differently depending on its type
            int32_t frameIdx;
            ::jiminy::getFrameIdx(pncModelFlexibleOrig_, frameName, frameIdx);
            std::string flexName = frameName + FLEXIBLE_JOINT_SUFFIX;
            if (pncModelFlexibleOrig_.frames[frameIdx].type == pinocchio::FIXED_JOINT)
            {
                // Get the child inertia from the urdf, since it cannot be recovered from the model
                // https://github.com/stack-of-tasks/pinocchio/issues/741
                pinocchio::Inertia const childInertia = getChildBodyInertiaFromUrdf(urdfPath_, frameName);

                // Insert flexible joint at fixed frame, splitting "composite" body inertia
                insertFlexibilityAtFixedFrameInModel(
                    pncModelFlexibleOrig_, frameName, childInertia, flexName);
            }
            else if (pncModelFlexibleOrig_.frames[frameIdx].type == pinocchio::JOINT)
            {
                insertFlexibilityBeforeJointInModel(pncModelFlexibleOrig_, frameName, flexName);
            }
            else
            {
                PRINT_ERROR("Flexible joint can only be inserted at fixed or joint frames.");
                return hresult_t::ERROR_GENERIC;
            }

            flexibleJointsNames_.emplace_back(flexName);
        }

        getJointsModelIdx(pncModelFlexibleOrig_,
                          flexibleJointsNames_,
                          flexibleJointsModelIdx_);

        return hresult_t::SUCCESS;
    }

    hresult_t Model::generateModelBiased(void)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (!isInitialized_)
        {
            PRINT_ERROR("Model not initialized.");
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

                // Add bias to com position
                float64_t const & comBiasStd = mdlOptions_->dynamics.centerOfMassPositionBodiesBiasStd;
                if (comBiasStd > EPS)
                {
                    vector3_t & comRelativePositionBody = pncModel_.inertias[jointIdx].lever();
                    comRelativePositionBody += randVectorNormal(3U, comBiasStd);
                }

                /* Add bias to body mass.
                   Note that it cannot be less than min(original mass, 1g) for numerical stability. */
                float64_t const & massBiasStd = mdlOptions_->dynamics.massBodiesBiasStd;
                if (massBiasStd > EPS)
                {
                    float64_t & massBody = pncModel_.inertias[jointIdx].mass();
                    massBody = std::max(massBody + randNormal(0.0, massBiasStd), std::min(massBody, 1.0e-3));
                }

                /* Add bias to inertia matrix of body.
                   To preserve positive semidefinite property after noise addition, the principal
                   axes and moments are computed from the original inertia matrix, then independent
                   gaussian distributed noise is added on each principal moments, and a random small
                   rotation is applied to the principal axes based on a randomly generated rotation
                   axis. Finally, the biased inertia matrix is obtained doing A @ diag(M) @ A.T.
                   If no bias, the original inertia matrix is recovered. */
                float64_t const & inertiaBiasStd = mdlOptions_->dynamics.inertiaBodiesBiasStd;
                if (inertiaBiasStd > EPS)
                {
                    pinocchio::Symmetric3 & inertiaBody = pncModel_.inertias[jointIdx].inertia();
                    Eigen::SelfAdjointEigenSolver<matrix3_t> solver(inertiaBody.matrix());
                    vector3_t inertiaBodyMoments = solver.eigenvalues();
                    matrix3_t inertiaBodyAxes = solver.eigenvectors();
                    vector3_t const randAxis = randVectorNormal(3U, inertiaBiasStd);
                    inertiaBodyAxes = inertiaBodyAxes * quaternion_t(pinocchio::exp3(randAxis));
                    inertiaBodyMoments += randVectorNormal(3U, inertiaBiasStd);
                    inertiaBody = pinocchio::Symmetric3((
                        inertiaBodyAxes * inertiaBodyMoments.asDiagonal() * inertiaBodyAxes.transpose()).eval());
                }

                // Add bias to relative body position (rotation excluded !)
                float64_t const & relativeBodyPosBiasStd = mdlOptions_->dynamics.relativePositionBodiesBiasStd;
                if (relativeBodyPosBiasStd > EPS)
                {
                    vector3_t & relativePositionBody = pncModel_.jointPlacements[jointIdx].translation();
                    relativePositionBody += randVectorNormal(3U, relativeBodyPosBiasStd);
                }
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

    void Model::computeConstraints(vectorN_t const & q,
                                   vectorN_t const & v)
    {
        /* Note that it is assumed that the kinematic quantities have been
           updated previously to be consistent with (q, v, a, u). If not, one
           is supposed to call  `pinocchio::forwardKinematics` before calling
           this method. */

        // Early return if no constraint is enabled
        if (!hasConstraint())
        {
            return;
        }

        /* Computing forward kinematics without acceleration to get the drift.
           Note that it will alter the actual joints spatial accelerations, so
           it is necessary to do a backup first to restore it later on. */
        jointsAcceleration_.swap(pncData_.a);
        pinocchio_overload::forwardKinematicsAcceleration(
            pncModel_, pncData_, vectorN_t::Zero(pncModel_.nv));

        // Compute joint jacobian manually since not done by engine for efficiency
        pinocchio::computeJointJacobians(pncModel_, pncData_, q);

        // Compute sequentially the jacobian and drift of each enabled constraint
        constraintsMask_ = 0U;
        constraintsHolder_.foreach(
            [&](std::shared_ptr<AbstractConstraintBase> const & constraint,
                constraintsHolderType_t const & /* holderType */)
            {
                // Skip constraint if disabled
                if (!constraint || !constraint->getIsEnabled())
                {
                    return;
                }

                // Compute constraint jacobian and drift
                uint32_t const constraintDimPrev = constraint->getDim();
                constraint->computeJacobianAndDrift(q, v);

                // Resize matrix if needed
                uint32_t const constraintDim = constraint->getDim();
                if (constraintDimPrev != constraintDim)
                {
                    constraintsJacobian_.conservativeResize(
                        constraintsJacobian_.rows() + constraintDim - constraintDimPrev,
                        Eigen::NoChange);
                    constraintsDrift_.conservativeResize(
                        constraintsDrift_.size() + constraintDim - constraintDimPrev);
                }

                // Update global jacobian and drift of all constraints
                constraintsJacobian_.block(constraintsMask_, 0, constraintDim, pncModel_.nv) = constraint->getJacobian();
                constraintsDrift_.segment(constraintsMask_, constraintDim) = constraint->getDrift();
                constraintsMask_ += constraintDim;
            });

        // Restore true acceleration
        jointsAcceleration_.swap(pncData_.a);
    }

    hresult_t Model::refreshProxies(void)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (!isInitialized_)
        {
            PRINT_ERROR("Model not initialized.");
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
            positionLimitMin_ = vectorN_t::Constant(pncModel_.nq, -INF);  // Do NOT use robot_->pncModel_.(lower|upper)PositionLimit
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
                    positionLimitMin_.segment<4>(positionIdx).setConstant(-1.0 - EPS);
                    positionLimitMax_.segment<4>(positionIdx).setConstant(+1.0 + EPS);
                }
                if (jointType == joint_t::FREE)
                {
                    uint32_t const & positionIdx = pncModel_.joints[i].idx_q();
                    positionLimitMin_.segment<4>(positionIdx + 3).setConstant(-1.0 - EPS);
                    positionLimitMax_.segment<4>(positionIdx + 3).setConstant(+1.0 + EPS);
                }
                if (jointType == joint_t::ROTARY_UNBOUNDED)
                {
                    uint32_t const & positionIdx = pncModel_.joints[i].idx_q();
                    positionLimitMin_.segment<2>(positionIdx).setConstant(-1.0 - EPS);
                    positionLimitMax_.segment<2>(positionIdx).setConstant(+1.0 + EPS);
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

        if (returnCode == hresult_t::SUCCESS)
        {
            returnCode = refreshConstraintsProxies();
        }

        return returnCode;
    }

    hresult_t Model::refreshCollisionsProxies(void)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (!isInitialized_)
        {
            PRINT_ERROR("Model not initialized.");
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
            PRINT_ERROR("Model not initialized.");
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

    hresult_t Model::refreshConstraintsProxies(void)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        // Initialize backup joint space acceleration
        jointsAcceleration_ = motionVector_t(pncData_.a.size(), pinocchio::Motion::Zero());

        uint32_t constraintSize = 0;
        constraintsHolder_.foreach(
            [&](std::shared_ptr<AbstractConstraintBase> const & constraint,
                constraintsHolderType_t const & /* holderType */)
            {
                // Early return if no constraint is defined (nullptr)
                if (!constraint)
                {
                    return;
                }

                if (returnCode == hresult_t::SUCCESS)
                {
                    // Reset constraint using neutral configuration and zero velocity
                    returnCode = constraint->reset(
                        pinocchio::neutral(pncModel_), vectorN_t::Zero(nv_));
                }

                if (returnCode == hresult_t::SUCCESS)
                {
                    // Call constraint on neutral position and zero velocity.
                    auto J = constraint->getJacobian();

                    // Check dimensions consistency
                    if (J.cols() != pncModel_.nv)
                    {
                        PRINT_ERROR("Model::refreshConstraintsProxies: constraint has "
                                    "inconsistent jacobian and drift (size mismatch).");
                        returnCode = hresult_t::ERROR_GENERIC;
                    }

                    // Store constraint size
                    if (returnCode == hresult_t::SUCCESS)
                    {
                        constraintSize += constraint->getDim();
                    }
                }
            });

        // Reset jacobian and drift to 0
        if (returnCode == hresult_t::SUCCESS)
        {
            constraintsMask_ = 0U;
            constraintsJacobian_ = matrixN_t::Zero(constraintSize, pncModel_.nv);
            constraintsDrift_ = vectorN_t::Zero(constraintSize);
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
                    PRINT_ERROR("Wrong vector size for 'positionLimitMin'.");
                    return hresult_t::ERROR_BAD_INPUT;
                }
                auto jointsPositionLimitMinDiff = jointsPositionLimitMin - mdlOptions_->joints.positionLimitMin;
                internalBuffersMustBeUpdated |= (jointsPositionLimitMinDiff.array().abs() >= EPS).all();
                vectorN_t & jointsPositionLimitMax = boost::get<vectorN_t>(jointOptionsHolder.at("positionLimitMax"));
                if ((uint32_t) rigidJointsPositionIdx_.size() != jointsPositionLimitMax.size())
                {
                    PRINT_ERROR("Wrong vector size for 'positionLimitMax'.");
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
                    PRINT_ERROR("Wrong vector size for 'velocityLimit'.");
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
                PRINT_ERROR("The number of contact points by collision pair 'maxContactPointsPerBody' must be at least 1.");
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
            PRINT_ERROR("The URDF file does not exist. Impossible to load it.");
            return hresult_t::ERROR_BAD_INPUT;
        }

        urdfPath_ = urdfPath;
        meshPackageDirs_ = meshPackageDirs;
        hasFreeflyer_ = hasFreeflyer;

        // Build robot physics model
        try
        {
            pncModelRigidOrig_ = pinocchio::Model();
            if (hasFreeflyer)
            {
                pinocchio::urdf::buildModel(urdfPath,
                                            pinocchio::JointModelFreeFlyer(),
                                            pncModelRigidOrig_);
            }
            else
            {
                pinocchio::urdf::buildModel(urdfPath, pncModelRigidOrig_);
            }
        }
        catch (std::exception const & e)
        {
            PRINT_ERROR("Something is wrong with the URDF. Impossible to build a model from it.\n"
                        "Raised from exception: ", e.what());
            return hresult_t::ERROR_BAD_INPUT;
        }

        // Build robot geometry model
        try
        {
            pncGeometryModel_ = pinocchio::GeometryModel();
            pinocchio::urdf::buildGeom(pncModelRigidOrig_,
                                       urdfPath,
                                       pinocchio::COLLISION,
                                       pncGeometryModel_,
                                       meshPackageDirs);
        }
        catch (std::exception const & e)
        {
            PRINT_WARNING("Something is wrong with the URDF. Impossible to load the collision geometries.");
            return hresult_t::SUCCESS;
        }

        // Replace the mesh geometry object by its convex representation for efficiency
        try
        {
            for (uint32_t i=0; i<pncGeometryModel_.geometryObjects.size(); ++i)
            {
                auto & geometry = pncGeometryModel_.geometryObjects[i].geometry;
                if (geometry->getObjectType() == hpp::fcl::OT_BVH)
                {
                    hpp::fcl::BVHModelPtr_t bvh = boost::static_pointer_cast<hpp::fcl::BVHModelBase>(geometry);
                    bvh->buildConvexHull(true);
                    geometry = bvh->convex;
                }
            }
        }
        catch (std::logic_error const & e)
        {
            PRINT_WARNING("hpp-fcl not built with qhull. Impossible to convert meshes to convex hulls.");
        }

        // Instantiate ground FCL box geometry, wrapped as a pinocchio collision geometry.
        // Note that half-space cannot be used for Shape-Shape collision because it has no
        // shape support. So a very large box is used instead. In the future, it could be
        // a more complex topological object, even a mesh would be supported.
        auto groudBox = hpp::fcl::CollisionGeometryPtr_t(new hpp::fcl::Box(1000.0, 1000.0, 2.0));

        // Create a Pinocchio Geometry object associated with the ground plan.
        // Its parent frame and parent joint are the universe. It is aligned with world frame,
        // and the top face is the actual ground surface.
        pinocchio::SE3 groundPose = pinocchio::SE3::Identity();
        groundPose.translation() = (vector3_t() << 0.0, 0.0, -1.0).finished();
        pinocchio::GeometryObject groundPlane("ground", 0, 0, groudBox, groundPose);

        // Add the ground plane pinocchio to the robot model
        pncGeometryModel_.addGeometryObject(groundPlane, pncModelRigidOrig_);

        return hresult_t::SUCCESS;
    }

    hresult_t Model::getFlexibleConfigurationFromRigid(vectorN_t const & qRigid,
                                                       vectorN_t       & qFlex) const
    {
        // Define some proxies
        uint32_t const & nqRigid = pncModelRigidOrig_.nq;

        // Check the size of the input state
        if (qRigid.size() != nqRigid)
        {
            PRINT_ERROR("Size of qRigid inconsistent with theoretical model.");
            return hresult_t::ERROR_BAD_INPUT;
        }

        // Initialize the flexible state
        qFlex = pinocchio::neutral(pncModelFlexibleOrig_);

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
                }
                ++idxRigid;
            }
        }

        return hresult_t::SUCCESS;
    }

    hresult_t Model::getRigidConfigurationFromFlexible(vectorN_t const & qFlex,
                                                       vectorN_t       & qRigid) const
    {
        // Define some proxies
        uint32_t const & nqFlex = pncModelFlexibleOrig_.nq;

        // Check the size of the input state
        if (qFlex.size() != nqFlex)
        {
            PRINT_ERROR("Size of qFlex inconsistent with flexible model.");
            return hresult_t::ERROR_BAD_INPUT;
        }

        // Initialize the rigid state
        qRigid = pinocchio::neutral(pncModelRigidOrig_);

        // Compute the rigid state based on the flexible state
        int32_t idxRigid = 0;
        int32_t idxFlex = 0;
        for (; idxFlex < pncModelFlexibleOrig_.njoints; ++idxFlex)
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
                }
                ++idxRigid;
            }
        }

        return hresult_t::SUCCESS;
    }

    hresult_t Model::getFlexibleVelocityFromRigid(vectorN_t const & vRigid,
                                                  vectorN_t       & vFlex) const
    {
        // Define some proxies
        uint32_t const & nvRigid = pncModelRigidOrig_.nv;
        uint32_t const & nvFlex = pncModelFlexibleOrig_.nv;

        // Check the size of the input state
        if (vRigid.size() != nvRigid)
        {
            PRINT_ERROR("Size of vRigid inconsistent with theoretical model.");
            return hresult_t::ERROR_BAD_INPUT;
        }

        // Initialize the flexible state
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
                    vFlex.segment(jointFlex.idx_v(), jointFlex.nv()) =
                        vRigid.segment(jointRigid.idx_v(), jointRigid.nv());
                }
                ++idxRigid;
            }
        }

        return hresult_t::SUCCESS;
    }

    hresult_t Model::getRigidVelocityFromFlexible(vectorN_t const & vFlex,
                                                  vectorN_t       & vRigid) const
    {
        // Define some proxies
        uint32_t const & nvRigid = pncModelRigidOrig_.nv;
        uint32_t const & nvFlex = pncModelFlexibleOrig_.nv;

        // Check the size of the input state
        if (vFlex.size() != nvFlex)
        {
            PRINT_ERROR("Size of vFlex inconsistent with flexible model.");
            return hresult_t::ERROR_BAD_INPUT;
        }

        // Initialize the rigid state
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

    /// \brief Get jacobian of the constraints.
    constMatrixBlock_t Model::getConstraintsJacobian(void) const
    {
        return constraintsJacobian_.topRows(constraintsMask_);
    }

    /// \brief Get drift of the constraints.
    constVectorBlock_t Model::getConstraintsDrift(void) const
    {
        return constraintsDrift_.head(constraintsMask_);
    }

    /// \brief Returns true if at least one constraint is active on the robot.
    bool_t Model::hasConstraint(void) const
    {
        bool_t hasConstraintEnabled = false;
        const_cast<constraintsHolder_t &>(constraintsHolder_).foreach(
            [&hasConstraintEnabled](std::shared_ptr<AbstractConstraintBase> const & constraint,
                                    constraintsHolderType_t const & /* holderType */)
            {
                if (constraint && constraint->getIsEnabled())
                {
                    hasConstraintEnabled = true;
                }
            });
        return hasConstraintEnabled;
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
