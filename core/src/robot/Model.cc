
#include <iostream>
#include <fstream>
#include <exception>

#include "pinocchio/spatial/symmetric3.hpp"                // `pinocchio::Symmetric3 `
#include "pinocchio/spatial/explog.hpp"                    // `pinocchio::exp3`
#include "pinocchio/spatial/se3.hpp"                       // `pinocchio::SE3`
#include "pinocchio/spatial/force.hpp"                     // `pinocchio::Force`
#include "pinocchio/spatial/motion.hpp"                    // `pinocchio::Motion`
#include "pinocchio/spatial/inertia.hpp"                   // `pinocchio::Inertia`
#include "pinocchio/multibody/joint/joint-free-flyer.hpp"  // `pinocchio::JointModelFreeFlyer`
#include "pinocchio/multibody/fwd.hpp"                     // `pinocchio::GeomIndex`
#include "pinocchio/multibody/model.hpp"                   // `pinocchio::Model`
#include "pinocchio/multibody/data.hpp"                    // `pinocchio::Data`
#include "pinocchio/multibody/geometry.hpp"                // `pinocchio::GeometryModel`, `pinocchio::GeometryData`
#include "pinocchio/multibody/fcl.hpp"                     // `pinocchio::GeometryObject`, `pinocchio::CollisionPair`
#include "pinocchio/algorithm/frames.hpp"                  // `pinocchio::Frame`, `pinocchio::FrameType`, `pinocchio::updateFramePlacements`
#include "pinocchio/algorithm/center-of-mass.hpp"          // `pinocchio::centerOfMass`
#include "pinocchio/algorithm/joint-configuration.hpp"     // `pinocchio::neutral`
#include "pinocchio/algorithm/kinematics.hpp"              // `pinocchio::forwardKinematics`
#include "pinocchio/algorithm/jacobian.hpp"                // `pinocchio::computeJointJacobians`
#include "pinocchio/algorithm/geometry.hpp"                // `pinocchio::updateGeometryPlacements`
#include "pinocchio/algorithm/cholesky.hpp"                // `pinocchio::cholesky::`

#include <Eigen/Eigenvalues>

#include "urdf_parser/urdf_parser.h"

#include "jiminy/core/robot/BasicSensors.h"
#include "jiminy/core/robot/PinocchioOverloadAlgorithms.h"
#include "jiminy/core/constraints/AbstractConstraint.h"
#include "jiminy/core/constraints/JointConstraint.h"
#include "jiminy/core/constraints/SphereConstraint.h"
#include "jiminy/core/constraints/FixedFrameConstraint.h"
#include "jiminy/core/utilities/Pinocchio.h"
#include "jiminy/core/utilities/Random.h"
#include "jiminy/core/utilities/Helpers.h"
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
            for (std::size_t i = 0; i < collisionBodies.size(); ++i)
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
        constraintsMap_t * constraintsMapPtr; constraintsMap_t::iterator constraintIt;
        std::tie(constraintsMapPtr, constraintIt) = const_cast<constraintsHolder_t *>(this)->find(key, holderType);
        return (constraintsMapPtr && constraintIt != constraintsMapPtr->end());
    }

    bool_t constraintsHolder_t::exist(std::string const & key) const
    {
        for (constraintsHolderType_t const & holderType : constraintsHolderTypesAll)
        {
            if (exist(key, holderType))
            {
                return true;
            }
        }
        return false;
    }

    std::shared_ptr<AbstractConstraintBase> constraintsHolder_t::get(std::string             const & key,
                                                                     constraintsHolderType_t const & holderType)
    {
        constraintsMap_t * constraintsMapPtr; constraintsMap_t::iterator constraintIt;
        std::tie(constraintsMapPtr, constraintIt) = find(key, holderType);
        if (constraintsMapPtr && constraintIt != constraintsMapPtr->end())
        {
            return constraintIt->second;
        }
        return {};
    }

    std::shared_ptr<AbstractConstraintBase> constraintsHolder_t::get(std::string const & key)
    {
        std::shared_ptr<AbstractConstraintBase> constraint;
        for (constraintsHolderType_t const & holderType : constraintsHolderTypesAll)
        {
            constraint = get(key, holderType);
            if (constraint)
            {
                break;
            }
        }
        return constraint;
    }

    void constraintsHolder_t::insert(constraintsMap_t        const & constraintsMap,
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

    constraintsMap_t::iterator constraintsHolder_t::erase(std::string             const & key,
                                                          constraintsHolderType_t const & holderType)
    {
        constraintsMap_t * constraintsMapPtr; constraintsMap_t::iterator constraintIt;
        std::tie(constraintsMapPtr, constraintIt) = find(key, holderType);
        if (constraintsMapPtr && constraintIt != constraintsMapPtr->end())
        {
            return constraintsMapPtr->erase(constraintIt);
        }
        return constraintsMapPtr->end();
    }

    Model::Model(void) :
    pncModelOrig_(),
    pncModel_(),
    collisionModelOrig_(),
    collisionModel_(),
    visualModelOrig_(),
    visualModel_(),
    pncDataOrig_(),
    pncData_(),
    collisionData_(nullptr),
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
    positionLimitMin_(),
    positionLimitMax_(),
    velocityLimit_(),
    positionFieldnames_(),
    velocityFieldnames_(),
    accelerationFieldnames_(),
    forceExternalFieldnames_(),
    pncModelFlexibleOrig_(),
    jointsAcceleration_(),
    nq_(0),
    nv_(0),
    nx_(0)
    {
        setOptions(getDefaultModelOptions());
    }

    hresult_t Model::initialize(pinocchio::Model         const & pncModel,
                                pinocchio::GeometryModel const & collisionModel,
                                pinocchio::GeometryModel const & visualModel)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (pncModel.nq == 0)
        {
            PRINT_ERROR("Pinocchio model must not be empty.");
            returnCode = hresult_t::ERROR_BAD_INPUT;
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            // Clear existing constraints
            constraintsHolder_.clear();
            jointsAcceleration_.clear();

            // Reset URDF info
            joint_t rootJointType;
            getJointTypeFromIdx(pncModel, 1, rootJointType);  // It cannot fail.
            urdfPath_ = "";
            hasFreeflyer_ = (rootJointType == joint_t::FREE);
            meshPackageDirs_.clear();

            // Set the models
            pncModelOrig_ = pncModel;
            collisionModelOrig_ = collisionModel;
            visualModelOrig_ = visualModel;

            // Add ground geometry object to collision model is not already available
            if (!collisionModelOrig_.existGeometryName("ground"))
            {
                // Instantiate ground FCL box geometry, wrapped as a pinocchio collision geometry.
                // Note that half-space cannot be used for Shape-Shape collision because it has no
                // shape support. So a very large box is used instead. In the future, it could be
                // a more complex topological object, even a mesh would be supported.
                auto groudBox = hpp::fcl::CollisionGeometryPtr_t(new hpp::fcl::Box(1000.0, 1000.0, 2.0));

                // Create a Pinocchio Geometry object associated with the ground plan.
                // Its parent frame and parent joint are the universe. It is aligned with world frame,
                // and the top face is the actual ground surface.
                pinocchio::SE3 groundPose = pinocchio::SE3::Identity();
                groundPose.translation() = - vector3_t::UnitZ();
                pinocchio::GeometryObject groundPlane("ground", 0, 0, groudBox, groundPose);

                // Add the ground plane pinocchio to the robot model
                collisionModelOrig_.addGeometryObject(groundPlane, pncModelOrig_);
            }

            // Backup the original model and data
            pncDataOrig_ = pinocchio::Data(pncModelOrig_);

            // Initialize Pinocchio data internal state, including basic
            // attributes such as the mass of each body.
            pinocchio::forwardKinematics(pncModelOrig_,
                                         pncDataOrig_,
                                         pinocchio::neutral(pncModelOrig_),
                                         vectorN_t::Zero(pncModelOrig_.nv));
            pinocchio::updateFramePlacements(pncModelOrig_, pncDataOrig_);
            pinocchio::centerOfMass(pncModelOrig_,
                                    pncDataOrig_,
                                    pinocchio::neutral(pncModelOrig_));

            /* Get the list of joint names of the rigid model and remove the 'universe'
               and 'root_joint' if any, since they are not actual joints. */
            rigidJointsNames_ = pncModelOrig_.names;
            rigidJointsNames_.erase(rigidJointsNames_.begin());  // remove 'universe'
            if (hasFreeflyer_)
            {
                rigidJointsNames_.erase(rigidJointsNames_.begin());  // remove 'root_joint'
            }

            // Create the flexible model
            returnCode = generateModelFlexible();
        }

        /* Add biases to the dynamics properties of the model.
           Note that is also refresh all proxies automatically. */
        if (returnCode == hresult_t::SUCCESS)
        {
            // Assume the model is fully initialized at this point
            isInitialized_ = true;
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

    hresult_t Model::initialize(std::string              const & urdfPath,
                                bool_t                   const & hasFreeflyer,
                                std::vector<std::string> const & meshPackageDirs)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        // Load new robot and collision models
        pinocchio::Model pncModel;
        pinocchio::GeometryModel pncCollisionModel;
        pinocchio::GeometryModel pncVisualModel;
        returnCode = buildModelsFromUrdf(
            urdfPath, hasFreeflyer, meshPackageDirs, pncModel, pncCollisionModel, pncVisualModel);

        // Initialize jiminy model
        if (returnCode == hresult_t::SUCCESS)
        {
            returnCode = initialize(pncModel, pncCollisionModel, pncVisualModel);
        }

        // Backup URDF info
        if (returnCode == hresult_t::SUCCESS)
        {
            urdfPath_ = urdfPath;
            meshPackageDirs_ = meshPackageDirs;
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
            /* Re-generate the true flexible model in case the original rigid
               model has been manually modified by the user. */
            generateModelFlexible();

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
        if (pncModelOrig_.existFrame(frameName))
        {
            PRINT_ERROR("A frame with the same name already exists.");
            returnCode = hresult_t::ERROR_BAD_INPUT;
        }

        // Check that parent frame exists
        frameIndex_t parentFrameId = 0;
        if (returnCode == hresult_t::SUCCESS)
        {
            returnCode = getFrameIdx(pncModelOrig_, parentBodyName, parentFrameId);
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            // Add the frame to the the original rigid model
            {
                jointIndex_t const & parentJointId = pncModelOrig_.frames[parentFrameId].parent;
                pinocchio::SE3 const & parentFramePlacement = pncModelOrig_.frames[parentFrameId].placement;
                pinocchio::SE3 const jointFramePlacement = parentFramePlacement.act(framePlacement);
                pinocchio::Frame const frame(frameName, parentJointId, parentFrameId, jointFramePlacement, frameType);
                pncModelOrig_.addFrame(frame);
                pncDataOrig_ = pinocchio::Data(pncModelOrig_);
            }

            // Add the frame to the the original flexible model
            {
                getFrameIdx(pncModelFlexibleOrig_, parentBodyName, parentFrameId);  // It can no longer fail at this point.
                jointIndex_t const & parentJointId = pncModelFlexibleOrig_.frames[parentFrameId].parent;
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
            frameIndex_t frameId;
            pinocchio::FrameType const frameType = pinocchio::FrameType::OP_FRAME;
            returnCode = getFrameIdx(pncModelOrig_, frameName, frameId);
            if (returnCode == hresult_t::SUCCESS)
            {
                if (pncModelOrig_.frames[frameId].type != frameType)
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
                frameIndex_t frameId;
                getFrameIdx(pncModelOrig_, frameName, frameId);  // It cannot fail

                // Remove the frame from the the original rigid model
                pncModelOrig_.frames.erase(std::next(
                    pncModelOrig_.frames.begin(), static_cast<uint32_t>(frameId)));
                pncModelOrig_.nframes--;

                // Remove the frame from the the original flexible model
                getFrameIdx(pncModelFlexibleOrig_, frameName, frameId);
                pncModelFlexibleOrig_.frames.erase(std::next(
                    pncModelFlexibleOrig_.frames.begin(), static_cast<uint32_t>(frameId)));
                pncModelFlexibleOrig_.nframes--;
            }

            // Regenerate rigid data
            pncDataOrig_ = pinocchio::Data(pncModelOrig_);

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

        if (collisionModelOrig_.ngeoms == 0)  // If successfully loaded, the ground should be available
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
            for (pinocchio::GeometryObject const & geom : collisionModelOrig_.geometryObjects)
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
        pinocchio::GeomIndex const & groundId = collisionModelOrig_.getGeometryId("ground");
        for (std::string const & name : bodyNames)
        {
            // Find the geometries having the body for parent, and add a collision pair for each of them
            constraintsMap_t collisionConstraintsMap;
            for (std::size_t i = 0; i < collisionModelOrig_.geometryObjects.size(); ++i)
            {
                if (returnCode == hresult_t::SUCCESS)
                {
                    pinocchio::GeometryObject const & geom = collisionModelOrig_.geometryObjects[i];
                    bool_t const isGeomMesh = (geom.meshPath.find('/') != std::string::npos ||
                                            geom.meshPath.find('\\') != std::string::npos);
                    std::string const & frameName = pncModel_.frames[geom.parentFrame].name;
                    if (!(ignoreMeshes && isGeomMesh) && frameName  == name)
                    {
                        // Add constraint associated with contact frame only if it is a sphere
                        hpp::fcl::CollisionGeometry const & shape = *geom.geometry;
                        if (shape.getNodeType() == hpp::fcl::GEOM_SPHERE)
                        {
                            /* Create and add the collision pair with the ground.
                            Note that the ground always comes second for the normal to be
                            consistently compute wrt the ground instead of the body. */
                            pinocchio::CollisionPair const collisionPair(i, groundId);
                            collisionModelOrig_.addCollisionPair(collisionPair);

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
                                geom.name, (Eigen::Matrix<bool_t, 6, 1>() << true, true, true, false, false, true).finished()));
                        }
                    }
                }
            }

            // Add constraints map
            if (returnCode == hresult_t::SUCCESS)
            {
                returnCode = addConstraints(collisionConstraintsMap, constraintsHolderType_t::COLLISION_BODIES);
            }
        }

        // Refresh proxies associated with the collisions only
        if (returnCode == hresult_t::SUCCESS)
        {
            refreshGeometryProxies();
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

        for (std::size_t i = 0; i < bodyNames.size(); ++i)
        {
            std::string const & bodyName = bodyNames[i];
            auto const collisionBodiesNameIt = std::find(
                collisionBodiesNames_.begin(),
                collisionBodiesNames_.end(),
                bodyName);
            std::ptrdiff_t const collisionBodiesNameIdx = std::distance(
                collisionBodiesNames_.begin(),
                collisionBodiesNameIt);
            collisionBodiesNames_.erase(collisionBodiesNameIt);
            collisionPairsIdx_.erase(collisionPairsIdx_.begin() + collisionBodiesNameIdx);
        }

        // Get the indices of the corresponding collision pairs in the geometry model of the robot and remove them
        std::vector<std::string> collisionConstraintsNames;
        pinocchio::GeomIndex const & groundId = collisionModelOrig_.getGeometryId("ground");
        for (std::string const & name : bodyNames)
        {
            // Find the geometries having the body for parent, and remove the collision pair for each of them
            for (std::size_t i = 0; i < collisionModelOrig_.geometryObjects.size(); ++i)
            {
                pinocchio::GeometryObject const & geom = collisionModelOrig_.geometryObjects[i];
                if (pncModel_.frames[geom.parentFrame].name == name)
                {
                    // Remove the collision pair with the ground
                    pinocchio::CollisionPair const collisionPair(i, groundId);
                    collisionModelOrig_.removeCollisionPair(collisionPair);

                    // Append collision geometry to the list of constraints to remove
                    if (constraintsHolder_.exist(geom.name, constraintsHolderType_t::COLLISION_BODIES))
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
        refreshGeometryProxies();

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
                frameName, (Eigen::Matrix<bool_t, 6, 1>() << true, true, true, false, false, true).finished()));
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

        /* Remove the constraint associated with contact frame, then
           remove the list of frames from the set of contact frames. */
        if (!frameNames.empty())
        {
            removeConstraints(frameNames, constraintsHolderType_t::CONTACT_FRAMES);  // It cannot fail at this point
            eraseVector(contactFramesNames_, frameNames);
        }
        else
        {
            removeConstraints(contactFramesNames_, constraintsHolderType_t::CONTACT_FRAMES);
            contactFramesNames_.clear();
        }

        // Refresh proxies associated with contacts and constraints
        refreshContactsProxies();

        return hresult_t::SUCCESS;
    }

    hresult_t Model::addConstraints(constraintsMap_t const & constraintsMap,
                                    constraintsHolderType_t const & holderType)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        // Check if constraint is properly defined and not already exists
        for (auto const & constraintPair : constraintsMap)
        {
            auto const & constraintName = std::get<0>(constraintPair);
            auto const & constraintPtr = std::get<1>(constraintPair);
            if (!constraintPtr)
            {
                PRINT_ERROR("Constraint with name '", constraintName, "' is unspecified.");
                returnCode = hresult_t::ERROR_BAD_INPUT;
            }
            if (constraintsHolder_.exist(constraintName))
            {
                PRINT_ERROR("A constraint with name '", constraintName, "' already exists.");
                returnCode = hresult_t::ERROR_BAD_INPUT;
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
            constraintsMap_t * constraintsMapPtr; constraintsMap_t::iterator constraintIt;
            std::tie(constraintsMapPtr, constraintIt) = constraintsHolder_.find(constraintName, holderType);

            // Detach the constraint
            constraintIt->second->detach();  // It cannot fail at this point

            // Remove the constraint from the holder
            constraintsMapPtr->erase(constraintIt);
        }

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
                if (returnCode == hresult_t::SUCCESS)
                {
                    returnCode = constraint->reset(q, v);
                }
            });

        if (returnCode == hresult_t::SUCCESS)
        {
            constraintsHolder_.foreach(
                std::array<constraintsHolderType_t, 3> {{
                    constraintsHolderType_t::BOUNDS_JOINTS,
                    constraintsHolderType_t::CONTACT_FRAMES,
                    constraintsHolderType_t::COLLISION_BODIES}},
                [](std::shared_ptr<AbstractConstraintBase> const & constraint,
                   constraintsHolderType_t const & /* holderType */)
                {
                    constraint->disable();
                });
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
        flexibleJointsNames_.clear();
        flexibleJointsModelIdx_.clear();
        pncModelFlexibleOrig_ = pncModelOrig_;
        for (flexibleJointData_t const & flexibleJoint : mdlOptions_->dynamics.flexibilityConfig)
        {
            // Check if joint name exists
            std::string const & frameName = flexibleJoint.frameName;
            if (!pncModelFlexibleOrig_.existFrame(frameName))
            {
                PRINT_ERROR("Frame '", frameName, "' does not exists. Impossible to insert flexible joint on it.");
                return hresult_t::ERROR_GENERIC;
            }

            // Add joint to model, differently depending on its type
            frameIndex_t frameIdx;
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

        // Add flexibility armuture-like inertia to the model
        for (flexibleJointData_t const & flexibleJoint : mdlOptions_->dynamics.flexibilityConfig)
        {
            std::string const & frameName = flexibleJoint.frameName;
            std::string flexName = frameName + FLEXIBLE_JOINT_SUFFIX;
            int32_t jointVelocityIdx;
            ::jiminy::getJointVelocityIdx(pncModelFlexibleOrig_, flexName, jointVelocityIdx);
            pncModelFlexibleOrig_.rotorInertia.segment<3>(jointVelocityIdx) = flexibleJoint.inertia;
        }

        // Compute flexible joint indices
        getJointsModelIdx(pncModelFlexibleOrig_, flexibleJointsNames_, flexibleJointsModelIdx_);

        return hresult_t::SUCCESS;
    }

    hresult_t Model::generateModelBiased(void)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        // Make sure the model is initialized
        if (!isInitialized_)
        {
            PRINT_ERROR("Model not initialized.");
            return hresult_t::ERROR_INIT_FAILED;
        }

        // Reset the robot either with the original rigid or flexible model
        if (returnCode == hresult_t::SUCCESS)
        {
            if (mdlOptions_->dynamics.enableFlexibleModel)
            {
                pncModel_ = pncModelFlexibleOrig_;
            }
            else
            {
                pncModel_ = pncModelOrig_;
            }

            for (std::string const & jointName : rigidJointsNames_)
            {
                jointIndex_t const & jointIdx = pncModel_.getJointId(jointName);

                // Add bias to com position
                float64_t const & comBiasStd = mdlOptions_->dynamics.centerOfMassPositionBodiesBiasStd;
                if (comBiasStd > EPS)
                {
                    vector3_t & comRelativePositionBody = pncModel_.inertias[jointIdx].lever();
                    comRelativePositionBody.array() *= 1.0 + randVectorNormal(3U, comBiasStd).array();
                }

                /* Add bias to body mass.
                   Note that it cannot be less than min(original mass, 1g) for numerical stability. */
                float64_t const & massBiasStd = mdlOptions_->dynamics.massBodiesBiasStd;
                if (massBiasStd > EPS)
                {
                    float64_t & massBody = pncModel_.inertias[jointIdx].mass();
                    massBody = std::max(massBody * (1.0 + randNormal(0.0, massBiasStd)),
                                        std::min(massBody, 1.0e-3));
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
                    inertiaBodyMoments.array() *= 1.0 + randVectorNormal(3U, inertiaBiasStd).array();
                    inertiaBody = pinocchio::Symmetric3((
                        inertiaBodyAxes * inertiaBodyMoments.asDiagonal() * inertiaBodyAxes.transpose()).eval());
                }

                // Add bias to relative body position (rotation excluded !)
                float64_t const & relativeBodyPosBiasStd = mdlOptions_->dynamics.relativePositionBodiesBiasStd;
                if (relativeBodyPosBiasStd > EPS)
                {
                    vector3_t & relativePositionBody = pncModel_.jointPlacements[jointIdx].translation();
                    relativePositionBody.array() *= 1.0 + randVectorNormal(3U, relativeBodyPosBiasStd).array();
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

            // Refresh internal proxies
            returnCode = refreshProxies();
        }

        return returnCode;
    }

    void Model::computeConstraints(vectorN_t const & q,
                                   vectorN_t const & v)
    {
        /* Note that it is assumed that the kinematic quantities have been
           updated previously to be consistent with (q, v, a, u). If not, one
           is supposed to call `pinocchio::forwardKinematics` before calling
           this method. */

        // Early return if no constraint is enabled
        if (!hasConstraints())
        {
            return;
        }

       // Compute joint jacobians manually since not done by engine for efficiency
        pinocchio::computeJointJacobians(pncModel_, pncData_);

        /* Compute inertia matrix, taking into account armature.
           Note that `crbaMinimal` is faster than `crba` as it also compute
           the joint jacobians as a by-product without having to call
           `computeJointJacobians` manually. However, it is less stable
           numerically, and it messes some variables (Ycrb[0] keeps accumulating
           and com[0] is "wrongly defined"). So using it must be avoided. */
        pinocchio_overload::crba(pncModel_, pncData_, q);

        // Compute the mass matrix decomposition, since it may be used for
        // constraint stabilization.
        pinocchio::cholesky::decompose(pncModel_, pncData_);

        /* Computing forward kinematics without acceleration to get the drift.
           Note that it will alter the actual joints spatial accelerations, so
           it is necessary to do a backup first to restore it later on. */
        jointsAcceleration_.swap(pncData_.a);
        pinocchio_overload::forwardKinematicsAcceleration(
            pncModel_, pncData_, vectorN_t::Zero(pncModel_.nv));

        // Compute sequentially the jacobian and drift of each enabled constraint
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
                constraint->computeJacobianAndDrift(q, v);
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

            /* Generate the fieldnames associated with the configuration vector,
               velocity, acceleration and external force vectors. */
            positionFieldnames_.clear();
            positionFieldnames_.resize(static_cast<std::size_t>(nq_));
            velocityFieldnames_.clear();
            velocityFieldnames_.resize(static_cast<std::size_t>(nv_));
            accelerationFieldnames_.clear();
            accelerationFieldnames_.resize(static_cast<std::size_t>(nv_));
            forceExternalFieldnames_.clear();
            forceExternalFieldnames_.resize(6U * (pncModel_.njoints - 1));
            for (std::size_t i = 1; i < pncModel_.joints.size(); ++i)
            {
                // Get joint name without "Joint" suffix, if any
                std::string jointShortName = removeSuffix(pncModel_.names[i], "Joint");

                // Get joint postion and velocity starting indices
                int32_t const idx_q = pncModel_.joints[i].idx_q();
                int32_t const idx_v = pncModel_.joints[i].idx_v();

                // Get joint prefix depending on its type
                joint_t jointType;
                std::string jointPrefix;
                if (returnCode == hresult_t::SUCCESS)
                {
                    returnCode = getJointTypeFromIdx(pncModel_, i, jointType);
                }
                if (returnCode == hresult_t::SUCCESS)
                {
                    if (jointType == joint_t::FREE)
                    {
                        // Discard the joint name for FREE joint type since it is unique if any
                        jointPrefix = FREE_FLYER_PREFIX_BASE_NAME;
                        jointShortName = "";
                    }
                    else
                    {
                        jointPrefix = JOINT_PREFIX_BASE;
                    }
                }

                // Get joint position and velocity suffices depending on its type
                std::vector<std::string> jointTypePositionSuffixes;
                std::vector<std::string> jointTypeVelocitySuffixes;
                if (returnCode == hresult_t::SUCCESS)
                {
                    returnCode = getJointTypePositionSuffixes(jointType,
                                                              jointTypePositionSuffixes);
                }
                if (returnCode == hresult_t::SUCCESS)
                {
                    returnCode = getJointTypeVelocitySuffixes(jointType,
                                                              jointTypeVelocitySuffixes);
                }

                if (returnCode == hresult_t::SUCCESS)
                {
                    // Define complete position fieldnames and backup them
                    std::vector<std::string> jointPositionFieldnames;
                    for (std::string const & suffix : jointTypePositionSuffixes)
                    {
                        jointPositionFieldnames.emplace_back(
                            jointPrefix + "Position" + jointShortName + suffix);
                    }
                    std::copy(jointPositionFieldnames.begin(),
                              jointPositionFieldnames.end(),
                              positionFieldnames_.begin() + idx_q);

                    // Define complete velocity and acceleration fieldnames and backup them
                    std::vector<std::string> jointVelocityFieldnames;
                    std::vector<std::string> jointAccelerationFieldnames;
                    for (std::string const & suffix : jointTypeVelocitySuffixes)
                    {
                        jointVelocityFieldnames.emplace_back(
                            jointPrefix + "Velocity" + jointShortName + suffix);
                        jointAccelerationFieldnames.emplace_back(
                            jointPrefix + "Acceleration" + jointShortName + suffix);
                    }
                    std::copy(jointVelocityFieldnames.begin(),
                              jointVelocityFieldnames.end(),
                              velocityFieldnames_.begin() + idx_v);
                    std::copy(jointAccelerationFieldnames.begin(),
                              jointAccelerationFieldnames.end(),
                              accelerationFieldnames_.begin() + idx_v);

                    // Define complete external force fieldnames and backup them
                    std::vector<std::string> jointForceExternalFieldnames;
                    for (std::string const & suffix : ForceSensor::fieldNames_)
                    {
                        jointForceExternalFieldnames.emplace_back(
                            jointPrefix + "ForceExternal" + jointShortName + suffix);
                    }
                    std::copy(jointForceExternalFieldnames.begin(),
                              jointForceExternalFieldnames.end(),
                              forceExternalFieldnames_.begin() + 6U * (i - 1));
                }
            }
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            // Get the joint position limits from the URDF or the user options
            positionLimitMin_.setConstant(pncModel_.nq, -INF);  // Do NOT use robot_->pncModel_.(lower|upper)PositionLimit
            positionLimitMax_.setConstant(pncModel_.nq, +INF);

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
                    for (std::size_t i = 0; i < rigidJointsPositionIdx_.size(); ++i)
                    {
                        uint32_t const & positionIdx = rigidJointsPositionIdx_[i];
                        positionLimitMin_[positionIdx] = mdlOptions_->joints.positionLimitMin[i];
                        positionLimitMax_[positionIdx] = mdlOptions_->joints.positionLimitMax[i];
                    }
                }
            }

            /* Overwrite the position bounds for some specific joint type, mainly
               due to quaternion normalization and cos/sin representation. */
            for (int32_t i = 0 ; i < pncModel_.njoints ; ++i)
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
            velocityLimit_.setConstant(pncModel_.nv, +INF);
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
                    for (std::size_t i = 0; i < rigidJointsVelocityIdx_.size(); ++i)
                    {
                        uint32_t const & velocityIdx = rigidJointsVelocityIdx_[i];
                        velocityLimit_[velocityIdx] = mdlOptions_->joints.velocityLimit[i];
                    }
                }
            }
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            returnCode = refreshGeometryProxies();
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

    hresult_t Model::refreshGeometryProxies(void)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (!isInitialized_)
        {
            PRINT_ERROR("Model not initialized.");
            returnCode = hresult_t::ERROR_INIT_FAILED;
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            // Restore collision and visual models
            collisionModel_ = collisionModelOrig_;
            visualModel_ = visualModelOrig_;

            // Update joint/frame fix for every geometry objects
            if (mdlOptions_->dynamics.enableFlexibleModel)
            {
                for (auto model : std::array<pinocchio::GeometryModel *, 2>{{&collisionModel_, &visualModel_}})
                {
                    for (pinocchio::GeometryObject & geom : model->geometryObjects)
                    {
                        geom.parentFrame = pncModel_.getFrameId(pncModelOrig_.frames[geom.parentFrame].name);
                        geom.parentJoint = pncModel_.getJointId(pncModelOrig_.names[geom.parentJoint]);
                    }
                }
            }

            // Update geometry data object after changing the collision pairs
            if (collisionData_.get())
            {
                // No object stored at this point, so created a new one
                *collisionData_ = pinocchio::GeometryData(collisionModel_);
            }
            else
            {
                /* Use copy assignment to avoid changing memory pointers, which would
                   result in dangling reference at Python-side. */
                collisionData_ = std::make_unique<pinocchio::GeometryData>(collisionModel_);
            }
            pinocchio::updateGeometryPlacements(pncModel_, pncData_, collisionModel_, *collisionData_);

            // Set the max number of contact points per collision pairs
            for (hpp::fcl::CollisionRequest & collisionRequest : collisionData_->collisionRequests)
            {
                collisionRequest.num_max_contacts = mdlOptions_->collisions.maxContactPointsPerBody;
            }

            // Extract the indices of the collision pairs associated with each body
            collisionPairsIdx_.clear();
            for (std::string const & name : collisionBodiesNames_)
            {
                std::vector<pairIndex_t> collisionPairsIdx;
                for (std::size_t i=0; i<collisionModel_.collisionPairs.size(); ++i)
                {
                    pinocchio::CollisionPair const & pair = collisionModel_.collisionPairs[i];
                    pinocchio::GeometryObject const & geom = collisionModel_.geometryObjects[pair.first];
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

        constraintsHolder_.foreach(
            [&](std::shared_ptr<AbstractConstraintBase> const & constraint,
                constraintsHolderType_t const & /* holderType */)
            {
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
                }
            });

        return returnCode;
    }

    hresult_t Model::setOptions(configHolder_t modelOptions)
    {
        bool_t internalBuffersMustBeUpdated = false;
        bool_t areModelsInvalid = false;
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
                if (rigidJointsPositionIdx_.size() != static_cast<uint32_t>(jointsPositionLimitMin.size()))
                {
                    PRINT_ERROR("Wrong vector size for 'positionLimitMin'.");
                    return hresult_t::ERROR_BAD_INPUT;
                }
                auto jointsPositionLimitMinDiff = jointsPositionLimitMin - mdlOptions_->joints.positionLimitMin;
                internalBuffersMustBeUpdated |= (jointsPositionLimitMinDiff.array().abs() >= EPS).all();
                vectorN_t & jointsPositionLimitMax = boost::get<vectorN_t>(jointOptionsHolder.at("positionLimitMax"));
                if (rigidJointsPositionIdx_.size() != static_cast<uint32_t>(jointsPositionLimitMax.size()))
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
                if (rigidJointsVelocityIdx_.size() != static_cast<uint32_t>(jointsVelocityLimit.size()))
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
            configHolder_t & dynOptionsHolder = boost::get<configHolder_t>(modelOptions.at("dynamics"));
            bool_t const & enableFlexibleModel = boost::get<bool_t>(dynOptionsHolder.at("enableFlexibleModel"));
            flexibilityConfig_t const & flexibilityConfig =
                boost::get<flexibilityConfig_t>(dynOptionsHolder.at("flexibilityConfig"));
            if (mdlOptions_
            && (flexibilityConfig.size() != mdlOptions_->dynamics.flexibilityConfig.size()
                || !std::equal(flexibilityConfig.begin(),
                               flexibilityConfig.end(),
                               mdlOptions_->dynamics.flexibilityConfig.begin())
                || enableFlexibleModel != mdlOptions_->dynamics.enableFlexibleModel))
            {
                areModelsInvalid = true;
            }
        }

        // Check that the collisions options are valid
        configHolder_t & collisionOptionsHolder = boost::get<configHolder_t>(modelOptions.at("collisions"));
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

        // Check that the model randomization parameters are valid
        configHolder_t & dynOptionsHolder = boost::get<configHolder_t>(modelOptions.at("dynamics"));
        for (auto const & field : std::array<std::string, 4>{{
                "inertiaBodiesBiasStd",
                "massBodiesBiasStd",
                "centerOfMassPositionBodiesBiasStd",
                "relativePositionBodiesBiasStd"}})
        {
            float64_t const & value = boost::get<float64_t>(dynOptionsHolder.at(field));
            if (0.9 < value || value < 0.0)
            {
                PRINT_ERROR("'" + field + "' must be positive, and lower than 0.9 to avoid physics issues.");
                return hresult_t::ERROR_BAD_INPUT;
            }
        }

        // Update the internal options
        mdlOptionsHolder_ = modelOptions;

        // Create a fast struct accessor
        mdlOptions_ = std::make_unique<modelOptions_t const>(mdlOptionsHolder_);

        if (areModelsInvalid)
        {
            // Trigger models regeneration
            reset();
        }
        else if (internalBuffersMustBeUpdated)
        {
            // Update the info extracted from the model
            refreshProxies();
        }
        else if (isCollisionDataInvalid)
        {
            // Update the visual and collision data
            refreshGeometryProxies();
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

    std::string const & Model::getName(void) const
    {
        return pncModelOrig_.name;
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

    hresult_t Model::getFlexibleConfigurationFromRigid(vectorN_t const & qRigid,
                                                       vectorN_t       & qFlex) const
    {
        // Define some proxies
        uint32_t const & nqRigid = pncModelOrig_.nq;

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
        for (; idxRigid < pncModelOrig_.njoints; ++idxFlex)
        {
            std::string const & jointRigidName = pncModelOrig_.names[idxRigid];
            std::string const & jointFlexName = pncModelFlexibleOrig_.names[idxFlex];
            if (jointRigidName == jointFlexName)
            {
                auto const & jointRigid = pncModelOrig_.joints[idxRigid];
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
        qRigid = pinocchio::neutral(pncModelOrig_);

        // Compute the rigid state based on the flexible state
        int32_t idxRigid = 0;
        int32_t idxFlex = 0;
        for (; idxFlex < pncModelFlexibleOrig_.njoints; ++idxFlex)
        {
            std::string const & jointRigidName = pncModelOrig_.names[idxRigid];
            std::string const & jointFlexName = pncModelFlexibleOrig_.names[idxFlex];
            if (jointRigidName == jointFlexName)
            {
                auto const & jointRigid = pncModelOrig_.joints[idxRigid];
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
        uint32_t const & nvRigid = pncModelOrig_.nv;
        uint32_t const & nvFlex = pncModelFlexibleOrig_.nv;

        // Check the size of the input state
        if (vRigid.size() != nvRigid)
        {
            PRINT_ERROR("Size of vRigid inconsistent with theoretical model.");
            return hresult_t::ERROR_BAD_INPUT;
        }

        // Initialize the flexible state
        vFlex.setZero(nvFlex);

        // Compute the flexible state based on the rigid state
        int32_t idxRigid = 0;
        int32_t idxFlex = 0;
        for (; idxRigid < pncModelOrig_.njoints; ++idxFlex)
        {
            std::string const & jointRigidName = pncModelOrig_.names[idxRigid];
            std::string const & jointFlexName = pncModelFlexibleOrig_.names[idxFlex];
            if (jointRigidName == jointFlexName)
            {
                auto const & jointRigid = pncModelOrig_.joints[idxRigid];
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
        uint32_t const & nvRigid = pncModelOrig_.nv;
        uint32_t const & nvFlex = pncModelFlexibleOrig_.nv;

        // Check the size of the input state
        if (vFlex.size() != nvFlex)
        {
            PRINT_ERROR("Size of vFlex inconsistent with flexible model.");
            return hresult_t::ERROR_BAD_INPUT;
        }

        // Initialize the rigid state
        vRigid.setZero(nvRigid);

        // Compute the rigid state based on the flexible state
        int32_t idxRigid = 0;
        int32_t idxFlex = 0;
        for (; idxFlex < pncModelFlexibleOrig_.njoints; ++idxRigid, ++idxFlex)
        {
            std::string const & jointRigidName = pncModelOrig_.names[idxRigid];
            std::string const & jointFlexName = pncModelFlexibleOrig_.names[idxFlex];
            if (jointRigidName == jointFlexName)
            {
                auto const & jointRigid = pncModelOrig_.joints[idxRigid];
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

    std::vector<frameIndex_t> const & Model::getCollisionBodiesIdx(void) const
    {
        return collisionBodiesIdx_;
    }

    std::vector<std::vector<pairIndex_t> > const & Model::getCollisionPairsIdx(void) const
    {
        return collisionPairsIdx_;
    }

    std::vector<frameIndex_t> const & Model::getContactFramesIdx(void) const
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

    std::vector<std::string> const & Model::getForceExternalFieldnames(void) const
    {
        return forceExternalFieldnames_;
    }

    std::vector<std::string> const & Model::getRigidJointsNames(void) const
    {
        return rigidJointsNames_;
    }

    std::vector<jointIndex_t> const & Model::getRigidJointsModelIdx(void) const
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

    std::vector<jointIndex_t> const & Model::getFlexibleJointsModelIdx(void) const
    {
        static std::vector<jointIndex_t> const flexibleJointsModelIdxEmpty {};
        if (mdlOptions_->dynamics.enableFlexibleModel)
        {
            return flexibleJointsModelIdx_;
        }
        else
        {
            return flexibleJointsModelIdxEmpty;
        }
    }

    /// \brief Returns true if at least one constraint is active on the robot.
    bool_t Model::hasConstraints(void) const
    {
        bool_t hasConstraintsEnabled = false;
        const_cast<constraintsHolder_t &>(constraintsHolder_).foreach(
            [&hasConstraintsEnabled](std::shared_ptr<AbstractConstraintBase> const & constraint,
                                     constraintsHolderType_t const & /* holderType */)
            {
                if (constraint->getIsEnabled())
                {
                    hasConstraintsEnabled = true;
                }
            });
        return hasConstraintsEnabled;
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
