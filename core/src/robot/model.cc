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
#include "pinocchio/multibody/geometry.hpp"  // `pinocchio::GeometryModel`, `pinocchio::GeometryData`
#include "pinocchio/multibody/fcl.hpp"  // `pinocchio::GeometryObject`, `pinocchio::CollisionPair`
#include "pinocchio/algorithm/frames.hpp"  // `pinocchio::Frame`, `pinocchio::FrameType`, `pinocchio::updateFramePlacements`
#include "pinocchio/algorithm/center-of-mass.hpp"       // `pinocchio::centerOfMass`
#include "pinocchio/algorithm/joint-configuration.hpp"  // `pinocchio::neutral`
#include "pinocchio/algorithm/kinematics.hpp"           // `pinocchio::forwardKinematics`
#include "pinocchio/algorithm/jacobian.hpp"             // `pinocchio::computeJointJacobians`
#include "pinocchio/algorithm/geometry.hpp"             // `pinocchio::updateGeometryPlacements`
#include "pinocchio/algorithm/cholesky.hpp"             // `pinocchio::cholesky::`

#include <Eigen/Eigenvalues>

#include "urdf_parser/urdf_parser.h"

#include "jiminy/core/hardware/basic_sensors.h"
#include "jiminy/core/robot/pinocchio_overload_algorithms.h"
#include "jiminy/core/constraints/abstract_constraint.h"
#include "jiminy/core/constraints/joint_constraint.h"
#include "jiminy/core/constraints/sphere_constraint.h"
#include "jiminy/core/constraints/frame_constraint.h"
#include "jiminy/core/utilities/pinocchio.h"
#include "jiminy/core/utilities/random.h"
#include "jiminy/core/utilities/helpers.h"

#include "jiminy/core/robot/model.h"


namespace jiminy
{
    void constraintsHolder_t::clear() noexcept
    {
        boundJoints.clear();
        contactFrames.clear();
        collisionBodies.clear();
        registered.clear();
    }

    constraintsMap_t::iterator getImpl(constraintsMap_t & constraintsMap, const std::string & key)
    {
        return std::find_if(constraintsMap.begin(),
                            constraintsMap.end(),
                            [&key](const auto & constraintPair)
                            { return constraintPair.first == key; });
    }

    std::pair<constraintsMap_t *, constraintsMap_t::iterator> constraintsHolder_t::find(
        const std::string & key, constraintsHolderType_t holderType)
    {
        // Pointers are NOT initialized to nullptr by default
        constraintsMap_t * constraintsMapPtr{nullptr};
        constraintsMap_t::iterator constraintIt;
        if (holderType == constraintsHolderType_t::COLLISION_BODIES)
        {
            for (constraintsMap_t & collisionBody : collisionBodies)
            {
                constraintsMapPtr = &collisionBody;
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

    bool constraintsHolder_t::exist(const std::string & key,
                                    constraintsHolderType_t holderType) const
    {
        const auto [constraintsMapPtr, constraintIt] =
            const_cast<constraintsHolder_t *>(this)->find(key, holderType);
        return (constraintsMapPtr && constraintIt != constraintsMapPtr->end());
    }

    bool constraintsHolder_t::exist(const std::string & key) const
    {
        for (constraintsHolderType_t holderType : constraintsHolderTypesAll)
        {
            if (exist(key, holderType))
            {
                return true;
            }
        }
        return false;
    }

    std::shared_ptr<AbstractConstraintBase> constraintsHolder_t::get(
        const std::string & key, constraintsHolderType_t holderType)
    {
        auto [constraintsMapPtr, constraintIt] = find(key, holderType);
        if (constraintsMapPtr && constraintIt != constraintsMapPtr->end())
        {
            return constraintIt->second;
        }
        return {};
    }

    std::shared_ptr<AbstractConstraintBase> constraintsHolder_t::get(const std::string & key)
    {
        std::shared_ptr<AbstractConstraintBase> constraint;
        for (constraintsHolderType_t holderType : constraintsHolderTypesAll)
        {
            constraint = get(key, holderType);
            if (constraint)
            {
                break;
            }
        }
        return constraint;
    }

    void constraintsHolder_t::insert(const constraintsMap_t & constraintsMap,
                                     constraintsHolderType_t holderType)
    {
        switch (holderType)
        {
        case constraintsHolderType_t::BOUNDS_JOINTS:
            boundJoints.insert(boundJoints.end(), constraintsMap.begin(), constraintsMap.end());
            break;
        case constraintsHolderType_t::CONTACT_FRAMES:
            contactFrames.insert(
                contactFrames.end(), constraintsMap.begin(), constraintsMap.end());
            break;
        case constraintsHolderType_t::COLLISION_BODIES:
            collisionBodies.push_back(constraintsMap);
            break;
        case constraintsHolderType_t::USER:
        default:
            registered.insert(registered.end(), constraintsMap.begin(), constraintsMap.end());
        }
    }

    constraintsMap_t::iterator constraintsHolder_t::erase(const std::string & key,
                                                          constraintsHolderType_t holderType)
    {
        auto [constraintsMapPtr, constraintIt] = find(key, holderType);
        if (constraintsMapPtr && constraintIt != constraintsMapPtr->end())
        {
            return constraintsMapPtr->erase(constraintIt);
        }
        return constraintsMapPtr->end();
    }

    Model::Model() noexcept
    {
        setOptions(getDefaultModelOptions());
    }

    hresult_t Model::initialize(const pinocchio::Model & pncModel,
                                const pinocchio::GeometryModel & collisionModel,
                                const pinocchio::GeometryModel & visualModel)
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
            JointModelType rootJointType;
            getJointTypeFromIdx(pncModel, 1, rootJointType);  // Cannot fail
            urdfPath_ = "";
            urdfData_ = "";
            hasFreeflyer_ = (rootJointType == JointModelType::FREE);
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
                // shape support. So a very large box is used instead. In the future, it could be a
                // more complex topological object, even a mesh would be supported.
                auto groudBox =
                    hpp::fcl::CollisionGeometryPtr_t(new hpp::fcl::Box(1000.0, 1000.0, 2.0));

                // Create a Pinocchio Geometry object associated with the ground plan.
                // Its parent frame and parent joint are the universe. It is aligned with world
                // frame, and the top face is the actual ground surface.
                pinocchio::SE3 groundPose = pinocchio::SE3::Identity();
                groundPose.translation() = -Eigen::Vector3d::UnitZ();
                pinocchio::GeometryObject groundPlane("ground", 0, 0, groudBox, groundPose);

                // Add the ground plane pinocchio to the robot model
                collisionModelOrig_.addGeometryObject(groundPlane, pncModelOrig_);
            }

            // Backup the original model and data
            pncDataOrig_ = pinocchio::Data(pncModelOrig_);

            /* Initialize Pinocchio data internal state.
               This includes "basic" attributes such as the mass of each body. */
            pinocchio::forwardKinematics(pncModelOrig_,
                                         pncDataOrig_,
                                         pinocchio::neutral(pncModelOrig_),
                                         Eigen::VectorXd::Zero(pncModelOrig_.nv));
            pinocchio::updateFramePlacements(pncModelOrig_, pncDataOrig_);
            pinocchio::centerOfMass(
                pncModelOrig_, pncDataOrig_, pinocchio::neutral(pncModelOrig_));

            /* Get the list of joint names of the rigid model and remove the 'universe' and
               'root_joint' if any, since they are not actual joints. */
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
            for (const std::string & jointName : rigidJointsNames_)
            {
                jointConstraintsMap.emplace_back(jointName,
                                                 std::make_shared<JointConstraint>(jointName));
            }
            addConstraints(jointConstraintsMap,
                           constraintsHolderType_t::BOUNDS_JOINTS);  // Cannot fail at this point
        }

        // Unset the initialization flag in case of failure
        if (returnCode != hresult_t::SUCCESS)
        {
            isInitialized_ = false;
        }

        return returnCode;
    }

    hresult_t Model::initialize(const std::string & urdfPath,
                                bool hasFreeflyer,
                                const std::vector<std::string> & meshPackageDirs,
                                bool loadVisualMeshes)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        // Load new robot and collision models
        pinocchio::Model pncModel;
        pinocchio::GeometryModel pncCollisionModel;
        pinocchio::GeometryModel pncVisualModel;
        returnCode = buildModelsFromUrdf(urdfPath,
                                         hasFreeflyer,
                                         meshPackageDirs,
                                         pncModel,
                                         pncCollisionModel,
                                         pncVisualModel,
                                         loadVisualMeshes);

        // Initialize jiminy model
        if (returnCode == hresult_t::SUCCESS)
        {
            returnCode = initialize(pncModel, pncCollisionModel, pncVisualModel);
        }

        // Backup URDF info
        if (returnCode == hresult_t::SUCCESS)
        {
            urdfPath_ = urdfPath;
            std::ifstream urdfFileStream(urdfPath_);
            urdfData_ = std::string(std::istreambuf_iterator<char>(urdfFileStream),
                                    std::istreambuf_iterator<char>());
            meshPackageDirs_ = meshPackageDirs;
        }

        return returnCode;
    }

    void Model::reset()
    {
        /* Do NOT reset the constraints, since it will be handled by the engine.
           It is necessary, since the current model state is required to initialize the
           constraints, at least for baumgarte stabilization. Indeed, the current frame position
           must be stored. */

        if (isInitialized_)
        {
            /* Re-generate the true flexible model in case the original rigid model has been
               manually modified by the user. */
            generateModelFlexible();

            // Update the biases added to the dynamics properties of the model
            generateModelBiased();
        }
    }

    hresult_t Model::addFrame(const std::string & frameName,
                              const std::string & parentBodyName,
                              const pinocchio::SE3 & framePlacement,
                              const pinocchio::FrameType & frameType)
    {
        /* The frame must be added directly to the parent joint because it is not possible to add a
           frame to another frame. This means that the relative transform of the frame wrt the
           parent joint must be computed. */
        hresult_t returnCode = hresult_t::SUCCESS;

        // Check that no frame with the same name already exists.
        if (pncModelOrig_.existFrame(frameName))
        {
            PRINT_ERROR("A frame with the same name already exists.");
            returnCode = hresult_t::ERROR_BAD_INPUT;
        }

        // Check that parent frame exists
        pinocchio::FrameIndex parentFrameId = 0;
        if (returnCode == hresult_t::SUCCESS)
        {
            returnCode = getFrameIdx(pncModelOrig_, parentBodyName, parentFrameId);
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            // Add the frame to the the original rigid model
            {
                pinocchio::JointIndex parentJointModelId =
                    pncModelOrig_.frames[parentFrameId].parent;
                const pinocchio::SE3 & parentFramePlacement =
                    pncModelOrig_.frames[parentFrameId].placement;
                const pinocchio::SE3 jointFramePlacement =
                    parentFramePlacement.act(framePlacement);
                const pinocchio::Frame frame(
                    frameName, parentJointModelId, parentFrameId, jointFramePlacement, frameType);
                pncModelOrig_.addFrame(frame);
                pncDataOrig_ = pinocchio::Data(pncModelOrig_);
            }

            // Add the frame to the the original flexible model
            {
                getFrameIdx(pncModelFlexibleOrig_,
                            parentBodyName,
                            parentFrameId);  // Cannot fail at this point
                pinocchio::JointIndex parentJointModelId =
                    pncModelFlexibleOrig_.frames[parentFrameId].parent;
                const pinocchio::SE3 & parentFramePlacement =
                    pncModelFlexibleOrig_.frames[parentFrameId].placement;
                const pinocchio::SE3 jointFramePlacement =
                    parentFramePlacement.act(framePlacement);
                const pinocchio::Frame frame(
                    frameName, parentJointModelId, parentFrameId, jointFramePlacement, frameType);
                pncModelFlexibleOrig_.addFrame(frame);
            }

            /* Backup the current rotor inertias and effort limits to restore them.
               Note that it is only necessary because 'reset' is not called for efficiency. It is
               reasonable to assume that no other fields have been overriden by derived classes
               such as Robot. */
            Eigen::VectorXd rotorInertia = pncModel_.rotorInertia;
            Eigen::VectorXd effortLimit = pncModel_.effortLimit;

            /* One must re-generate the model after adding a frame.
               Note that, since the added frame being the "last" of the model, the proxies are
               still up-to-date and therefore it is unecessary to call 'reset'. */
            generateModelBiased();

            // Restore the current rotor inertias and effort limits
            pncModel_.rotorInertia.swap(rotorInertia);
            pncModel_.effortLimit.swap(effortLimit);
        }

        return returnCode;
    }

    hresult_t Model::addFrame(const std::string & frameName,
                              const std::string & parentBodyName,
                              const pinocchio::SE3 & framePlacement)
    {
        const pinocchio::FrameType frameType = pinocchio::FrameType::OP_FRAME;
        return addFrame(frameName, parentBodyName, framePlacement, frameType);
    }

    hresult_t Model::removeFrames(const std::vector<std::string> & frameNames)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        /* Check that the frame can be safely removed from the original rigid model.
           If so, it is also the case for the original flexible models. */
        for (const std::string & frameName : frameNames)
        {
            pinocchio::FrameIndex frameId;
            const pinocchio::FrameType frameType = pinocchio::FrameType::OP_FRAME;
            returnCode = getFrameIdx(pncModelOrig_, frameName, frameId);
            if (returnCode == hresult_t::SUCCESS)
            {
                if (pncModelOrig_.frames[frameId].type != frameType)
                {
                    PRINT_ERROR("Impossible to remove this frame. One should only remove frames "
                                "added manually.");
                    returnCode = hresult_t::ERROR_BAD_INPUT;
                }
            }
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            for (const std::string & frameName : frameNames)
            {
                // Get the frame idx
                pinocchio::FrameIndex frameIdx;
                getFrameIdx(pncModelOrig_, frameName, frameIdx);  // Cannot fail at this point

                // Remove the frame from the the original rigid model
                pncModelOrig_.frames.erase(
                    std::next(pncModelOrig_.frames.begin(), static_cast<uint32_t>(frameIdx)));
                pncModelOrig_.nframes--;

                // Remove the frame from the the original flexible model
                getFrameIdx(pncModelFlexibleOrig_,
                            frameName,
                            frameIdx);  // Cannot fail at this point
                pncModelFlexibleOrig_.frames.erase(
                    std::next(pncModelFlexibleOrig_.frames.begin(), frameIdx));
                pncModelFlexibleOrig_.nframes--;
            }

            // Regenerate rigid data
            pncDataOrig_ = pinocchio::Data(pncModelOrig_);

            // One must reset the model after removing a frame
            reset();
        }

        return returnCode;
    }

    hresult_t Model::removeFrame(const std::string & frameName)
    {
        return removeFrames({frameName});
    }

    hresult_t Model::addCollisionBodies(const std::vector<std::string> & bodyNames,
                                        bool ignoreMeshes)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (!isInitialized_)
        {
            PRINT_ERROR("Model not initialized.");
            return hresult_t::ERROR_INIT_FAILED;
        }

        // Returning early if nothing to do
        if (bodyNames.empty())
        {
            return hresult_t::SUCCESS;
        }

        // If successfully loaded, the ground should be available
        if (collisionModelOrig_.ngeoms == 0)
        {
            PRINT_ERROR("Collision geometry not available. Some collision meshes were probably "
                        "not found.");
            return hresult_t::ERROR_INIT_FAILED;
        }

        // Make sure that no body are duplicates
        if (checkDuplicates(bodyNames))
        {
            PRINT_ERROR("Some bodies are duplicates.");
            return hresult_t::ERROR_BAD_INPUT;
        }

        // Make sure there is no collision already associated with any of the bodies in the list
        if (checkIntersection(collisionBodiesNames_, bodyNames))
        {
            PRINT_ERROR("At least one of the bodies is already been associated with a collision.");
            return hresult_t::ERROR_BAD_INPUT;
        }

        // Make sure that all the bodies exist
        for (const std::string & name : bodyNames)
        {
            if (!pncModel_.existBodyName(name))
            {
                PRINT_ERROR("At least one of the bodies does not exist.");
                return hresult_t::ERROR_BAD_INPUT;
            }
        }

        // Make sure that at least one geometry is associated with each body
        for (const std::string & name : bodyNames)
        {
            bool hasGeometry = false;
            for (const pinocchio::GeometryObject & geom : collisionModelOrig_.geometryObjects)
            {
                const bool isGeomMesh = (geom.meshPath.find('/') != std::string::npos ||
                                         geom.meshPath.find('\\') != std::string::npos);
                // geom.meshPath is the geometry type if it is not an actual mesh
                if (!(ignoreMeshes && isGeomMesh) &&
                    pncModel_.frames[geom.parentFrame].name == name)
                {
                    hasGeometry = true;
                    break;
                }
            }
            if (!hasGeometry)
            {
                PRINT_ERROR("At least one of the bodies is not associated with any collision "
                            "geometry of requested type.");
                return hresult_t::ERROR_BAD_INPUT;
            }
        }

        // Add the list of bodies to the set of collision bodies
        collisionBodiesNames_.insert(
            collisionBodiesNames_.end(), bodyNames.begin(), bodyNames.end());

        // Create the collision pairs and add them to the geometry model of the robot
        const pinocchio::GeomIndex & groundId = collisionModelOrig_.getGeometryId("ground");
        for (const std::string & name : bodyNames)
        {
            // Add a collision pair for all geometries having the body as parent
            constraintsMap_t collisionConstraintsMap;
            for (std::size_t i = 0; i < collisionModelOrig_.geometryObjects.size(); ++i)
            {
                if (returnCode == hresult_t::SUCCESS)
                {
                    const pinocchio::GeometryObject & geom =
                        collisionModelOrig_.geometryObjects[i];
                    const bool isGeomMesh = (geom.meshPath.find('/') != std::string::npos ||
                                             geom.meshPath.find('\\') != std::string::npos);
                    const std::string & frameName = pncModel_.frames[geom.parentFrame].name;
                    if (!(ignoreMeshes && isGeomMesh) && frameName == name)
                    {
                        // Add constraint associated with contact frame only if it is a sphere
                        const hpp::fcl::CollisionGeometry & shape = *geom.geometry;
                        if (shape.getNodeType() == hpp::fcl::GEOM_SPHERE)
                        {
                            /* Create and add the collision pair with the ground.
                               Note that the ground always comes second for the normal to be
                               consistently compute wrt the ground instead of the body. */
                            const pinocchio::CollisionPair collisionPair(i, groundId);
                            collisionModelOrig_.addCollisionPair(collisionPair);

                            /* Add dedicated frame.
                               Note that 'BODY' type is used instead of default 'OP_FRAME' to it
                               clear it is not consider as manually added to the model, and
                               therefore cannot be deleted by the user. */
                            const pinocchio::FrameType frameType =
                                pinocchio::FrameType::FIXED_JOINT;
                            returnCode = addFrame(geom.name, frameName, geom.placement, frameType);

                            // Add fixed frame constraint of bounded sphere
                            // const hpp::fcl::Sphere & sphere =
                            //     static_cast<const hpp::fcl::Sphere &>(shape);
                            // collisionConstraintsMap.emplace_back(
                            //     geom.name,
                            //     std::make_shared<SphereConstraint>(geom.name, sphere.radius));
                            collisionConstraintsMap.emplace_back(
                                geom.name,
                                std::make_shared<FrameConstraint>(
                                    geom.name,
                                    std::array<bool, 6>{{true, true, true, false, false, true}}));
                        }

                        // TODO: Add warning or error to notify that a geometry has been ignored
                    }
                }
            }

            // Add constraints map
            if (returnCode == hresult_t::SUCCESS)
            {
                returnCode = addConstraints(collisionConstraintsMap,
                                            constraintsHolderType_t::COLLISION_BODIES);
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

        /* Remove the list of bodies from the set of collision bodies, then remove the associated
           set of collision pairs for each of them. */
        if (bodyNames.empty())
        {
            bodyNames = collisionBodiesNames_;
        }

        for (std::size_t i = 0; i < bodyNames.size(); ++i)
        {
            const std::string & bodyName = bodyNames[i];
            const auto collisionBodiesNameIt =
                std::find(collisionBodiesNames_.begin(), collisionBodiesNames_.end(), bodyName);
            const std::ptrdiff_t collisionBodiesNameIdx =
                std::distance(collisionBodiesNames_.begin(), collisionBodiesNameIt);
            collisionBodiesNames_.erase(collisionBodiesNameIt);
            collisionPairsIdx_.erase(collisionPairsIdx_.begin() + collisionBodiesNameIdx);
        }

        // Get indices of corresponding collision pairs in geometry model of robot and remove them
        std::vector<std::string> collisionConstraintsNames;
        const pinocchio::GeomIndex & groundId = collisionModelOrig_.getGeometryId("ground");
        for (const std::string & name : bodyNames)
        {
            // Remove the collision pair for all the geometries having the body as parent
            for (std::size_t i = 0; i < collisionModelOrig_.geometryObjects.size(); ++i)
            {
                const pinocchio::GeometryObject & geom = collisionModelOrig_.geometryObjects[i];
                if (pncModel_.frames[geom.parentFrame].name == name)
                {
                    // Remove the collision pair with the ground
                    const pinocchio::CollisionPair collisionPair(i, groundId);
                    collisionModelOrig_.removeCollisionPair(collisionPair);

                    // Append collision geometry to the list of constraints to remove
                    if (constraintsHolder_.exist(geom.name,
                                                 constraintsHolderType_t::COLLISION_BODIES))
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

    hresult_t Model::addContactPoints(const std::vector<std::string> & frameNames)
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
        for (const std::string & name : frameNames)
        {
            if (!pncModel_.existFrame(name))
            {
                PRINT_ERROR("At least one of the frames does not exist.");
                return hresult_t::ERROR_BAD_INPUT;
            }
        }

        // Add the list of frames to the set of contact frames
        contactFramesNames_.insert(
            contactFramesNames_.end(), frameNames.begin(), frameNames.end());

        // Add constraint associated with contact frame
        constraintsMap_t frameConstraintsMap;
        frameConstraintsMap.reserve(frameNames.size());
        for (const std::string & frameName : frameNames)
        {
            frameConstraintsMap.emplace_back(
                frameName,
                std::make_shared<FrameConstraint>(
                    frameName, std::array<bool, 6>{{true, true, true, false, false, true}}));
        }
        returnCode = addConstraints(frameConstraintsMap, constraintsHolderType_t::CONTACT_FRAMES);

        // Refresh proxies associated with contacts and constraints
        if (returnCode == hresult_t::SUCCESS)
        {
            refreshContactsProxies();
        }

        return returnCode;
    }

    hresult_t Model::removeContactPoints(const std::vector<std::string> & frameNames)
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

        /* Remove the constraint associated with contact frame, then remove the list of frames from
           the set of contact frames. */
        if (!frameNames.empty())
        {
            removeConstraints(
                frameNames,
                constraintsHolderType_t::CONTACT_FRAMES);  // Cannot fail at this point
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

    hresult_t Model::addConstraints(const constraintsMap_t & constraintsMap,
                                    constraintsHolderType_t holderType)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        // Check if constraint is properly defined and not already exists
        for (const auto & [constraintName, constraintPtr] : constraintsMap)
        {
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

    hresult_t Model::addConstraint(const std::string & constraintName,
                                   const std::shared_ptr<AbstractConstraintBase> & constraint,
                                   constraintsHolderType_t holderType)
    {
        return addConstraints({{constraintName, constraint}}, holderType);
    }

    hresult_t Model::addConstraint(const std::string & constraintName,
                                   const std::shared_ptr<AbstractConstraintBase> & constraint)
    {
        return addConstraint(constraintName, constraint, constraintsHolderType_t::USER);
    }

    hresult_t Model::removeConstraints(const std::vector<std::string> & constraintsNames,
                                       constraintsHolderType_t holderType)
    {
        // Make sure the constraints exists
        for (const std::string & constraintName : constraintsNames)
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
        for (const std::string & constraintName : constraintsNames)
        {
            // Lookup constraint
            auto [constraintsMapPtr, constraintIt] =
                constraintsHolder_.find(constraintName, holderType);

            // Detach the constraint
            constraintIt->second->detach();  // Cannot fail at this point

            // Remove the constraint from the holder
            constraintsMapPtr->erase(constraintIt);
        }

        return hresult_t::SUCCESS;
    }

    hresult_t Model::removeConstraint(const std::string & constraintName,
                                      constraintsHolderType_t holderType)
    {
        return removeConstraints({constraintName}, holderType);
    }

    hresult_t Model::removeConstraint(const std::string & constraintName)
    {
        return removeConstraint(constraintName, constraintsHolderType_t::USER);
    }

    hresult_t Model::getConstraint(const std::string & constraintName,
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

    hresult_t Model::getConstraint(const std::string & constraintName,
                                   std::weak_ptr<const AbstractConstraintBase> & constraint) const
    {
        constraint = std::const_pointer_cast<const AbstractConstraintBase>(
            const_cast<constraintsHolder_t &>(constraintsHolder_).get(constraintName));
        if (!constraint.lock())
        {
            PRINT_ERROR("No constraint with this name exists.");
            return hresult_t::ERROR_BAD_INPUT;
        }
        return hresult_t::SUCCESS;
    }

    constraintsHolder_t Model::getConstraints()
    {
        return constraintsHolder_;
    }

    bool Model::existConstraint(const std::string & constraintName) const
    {
        return constraintsHolder_.exist(constraintName);
    }

    hresult_t Model::resetConstraints(const Eigen::VectorXd & q, const Eigen::VectorXd & v)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        constraintsHolder_.foreach(
            [&q, &v, &returnCode](const std::shared_ptr<AbstractConstraintBase> & constraint,
                                  constraintsHolderType_t /* holderType */)
            {
                if (returnCode == hresult_t::SUCCESS)
                {
                    returnCode = constraint->reset(q, v);
                }
            });

        if (returnCode == hresult_t::SUCCESS)
        {
            constraintsHolder_.foreach(
                std::array<constraintsHolderType_t, 3>{
                    {constraintsHolderType_t::BOUNDS_JOINTS,
                     constraintsHolderType_t::CONTACT_FRAMES,
                     constraintsHolderType_t::COLLISION_BODIES}},
                [](const std::shared_ptr<AbstractConstraintBase> & constraint,
                   constraintsHolderType_t /* holderType */) { constraint->disable(); });
        }

        return returnCode;
    }

    hresult_t Model::generateModelFlexible()
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        // Copy the original model
        pncModelFlexibleOrig_ = pncModelOrig_;

        // Check that the frames exist
        for (const FlexibleJointData & flexibleJoint : mdlOptions_->dynamics.flexibilityConfig)
        {
            const std::string & frameName = flexibleJoint.frameName;
            if (!pncModelOrig_.existFrame(frameName))
            {
                PRINT_ERROR("Frame '",
                            frameName,
                            "' does not exists. Impossible to insert flexible joint on it.");
                returnCode = hresult_t::ERROR_GENERIC;
                break;
            }
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            // Add all the flexible joints
            flexibleJointsNames_.clear();
            for (const FlexibleJointData & flexibleJoint : mdlOptions_->dynamics.flexibilityConfig)
            {
                // Extract some proxies
                const std::string & frameName = flexibleJoint.frameName;
                std::string flexName = frameName;
                pinocchio::FrameIndex frameIdx;
                getFrameIdx(pncModelFlexibleOrig_,
                            frameName,
                            frameIdx);  // Cannot fail at this point
                const pinocchio::Frame & frame = pncModelFlexibleOrig_.frames[frameIdx];

                // Add joint to model, differently depending on its type
                if (frame.type == pinocchio::FrameType::FIXED_JOINT)
                {
                    // Insert flexible joint at fixed frame, splitting "composite" body inertia
                    returnCode =
                        insertFlexibilityAtFixedFrameInModel(pncModelFlexibleOrig_, frameName);
                }
                else if (frame.type == pinocchio::FrameType::JOINT)
                {
                    flexName += FLEXIBLE_JOINT_SUFFIX;
                    insertFlexibilityBeforeJointInModel(pncModelFlexibleOrig_,
                                                        frameName,
                                                        flexName);  // Cannot fail at this point
                }
                else
                {
                    PRINT_ERROR("Flexible joint can only be inserted at fixed or joint frames.");
                    returnCode = hresult_t::ERROR_GENERIC;
                }
                if (returnCode == hresult_t::SUCCESS)
                {
                    flexibleJointsNames_.push_back(flexName);
                }
            }
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            // Compute flexible joint indices
            flexibleJointsModelIdx_.clear();
            getJointsModelIdx(
                pncModelFlexibleOrig_, flexibleJointsNames_, flexibleJointsModelIdx_);

            // Add flexibility armature-like inertia to the model
            for (std::size_t i = 0; i < flexibleJointsModelIdx_.size(); ++i)
            {
                const FlexibleJointData & flexibleJoint =
                    mdlOptions_->dynamics.flexibilityConfig[i];
                const pinocchio::JointModel & jmodel =
                    pncModelFlexibleOrig_.joints[flexibleJointsModelIdx_[i]];
                jmodel.jointVelocitySelector(pncModelFlexibleOrig_.rotorInertia) =
                    flexibleJoint.inertia;
            }

            // Check that the armature inertia is valid
            for (pinocchio::JointIndex flexibleJointModelIdx : flexibleJointsModelIdx_)
            {
                const pinocchio::Inertia & flexibleInertia =
                    pncModelFlexibleOrig_.inertias[flexibleJointModelIdx];
                const pinocchio::JointModel & jmodel =
                    pncModelFlexibleOrig_.joints[flexibleJointModelIdx];
                const Eigen::Vector3d inertiaDiag =
                    jmodel.jointVelocitySelector(pncModelFlexibleOrig_.rotorInertia) +
                    flexibleInertia.inertia().matrix().diagonal();
                if ((inertiaDiag.array() < 1e-5).any())
                {
                    PRINT_ERROR("The subtree diagonal inertia for flexibility joint ",
                                flexibleJointModelIdx,
                                " must be larger than 1e-5 for numerical stability: ",
                                inertiaDiag.transpose());
                    returnCode = hresult_t::ERROR_GENERIC;
                    break;
                }
            }
        }

        return returnCode;
    }

    hresult_t Model::generateModelBiased()
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        // Make sure the model is initialized
        if (!isInitialized_)
        {
            PRINT_ERROR("Model not initialized.");
            return hresult_t::ERROR_INIT_FAILED;
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
                pncModel_ = pncModelOrig_;
            }

            // Initially set effortLimit to zero systematically
            pncModel_.effortLimit.setZero();

            for (const std::string & jointName : rigidJointsNames_)
            {
                const pinocchio::JointIndex jointIdx = pncModel_.getJointId(jointName);

                // Add bias to com position
                const double comBiasStd = mdlOptions_->dynamics.centerOfMassPositionBodiesBiasStd;
                if (comBiasStd > EPS)
                {
                    Eigen::Vector3d & comRelativePositionBody =
                        pncModel_.inertias[jointIdx].lever();
                    comRelativePositionBody.array() *=
                        1.0 + randVectorNormal(3U, comBiasStd).array();
                }

                /* Add bias to body mass.
                   It cannot be less than min(original mass, 1g) for numerical stability. */
                const double massBiasStd = mdlOptions_->dynamics.massBodiesBiasStd;
                if (massBiasStd > EPS)
                {
                    double & massBody = pncModel_.inertias[jointIdx].mass();
                    massBody = std::max(massBody * (1.0 + randNormal(0.0, massBiasStd)),
                                        std::min(massBody, 1.0e-3));
                }

                /* Add bias to inertia matrix of body.
                   To preserve positive semi-definite property after noise addition, the principal
                   axes and moments are computed from the original inertia matrix, then independent
                   gaussian distributed noise is added on each principal moments, and a random
                   small rotation is applied to the principal axes based on a randomly generated
                   rotation axis. Finally, the biased inertia matrix is obtained doing
                   `A @ diag(M) @ A.T`. If no bias, the original inertia matrix is recovered. */
                const double inertiaBiasStd = mdlOptions_->dynamics.inertiaBodiesBiasStd;
                if (inertiaBiasStd > EPS)
                {
                    pinocchio::Symmetric3 & inertiaBody = pncModel_.inertias[jointIdx].inertia();
                    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(inertiaBody.matrix());
                    Eigen::Vector3d inertiaBodyMoments = solver.eigenvalues();
                    Eigen::Matrix3d inertiaBodyAxes = solver.eigenvectors();
                    const Eigen::Vector3d randAxis = randVectorNormal(3U, inertiaBiasStd);
                    inertiaBodyAxes =
                        inertiaBodyAxes * Eigen::Quaterniond(pinocchio::exp3(randAxis));
                    inertiaBodyMoments.array() *=
                        1.0 + randVectorNormal(3U, inertiaBiasStd).array();
                    inertiaBody =
                        pinocchio::Symmetric3((inertiaBodyAxes * inertiaBodyMoments.asDiagonal() *
                                               inertiaBodyAxes.transpose())
                                                  .eval());
                }

                // Add bias to relative body position (rotation excluded !)
                const double relativeBodyPosBiasStd =
                    mdlOptions_->dynamics.relativePositionBodiesBiasStd;
                if (relativeBodyPosBiasStd > EPS)
                {
                    Eigen::Vector3d & relativePositionBody =
                        pncModel_.jointPlacements[jointIdx].translation();
                    relativePositionBody.array() *=
                        1.0 + randVectorNormal(3U, relativeBodyPosBiasStd).array();
                }
            }

            // Initialize Pinocchio Data internal state
            pncData_ = pinocchio::Data(pncModel_);
            pinocchio::forwardKinematics(pncModel_,
                                         pncData_,
                                         pinocchio::neutral(pncModel_),
                                         Eigen::VectorXd::Zero(pncModel_.nv));
            pinocchio::updateFramePlacements(pncModel_, pncData_);
            pinocchio::centerOfMass(pncModel_, pncData_, pinocchio::neutral(pncModel_));

            // Refresh internal proxies
            returnCode = refreshProxies();
        }

        return returnCode;
    }

    void Model::computeConstraints(const Eigen::VectorXd & q, const Eigen::VectorXd & v)
    {
        /* Note that it is assumed that all kinematic quantities are consistent with (q, v, a, u).
           If not, one must call `pinocchio::forwardKinematics` before calling this method. */

        // Early return if no constraint is enabled
        if (!hasConstraints())
        {
            return;
        }

        /* Compute inertia matrix, taking into account armature.
           Note that `crbaMinimal` is faster than `crba` as it also compute the joint jacobians as
           a by-product without having to call `computeJointJacobians` manually. */
        pinocchio_overload::crba(pncModel_, pncData_, q);

        /* Computing forward kinematics without acceleration to get the drift.
           Note that it will alter the actual joints spatial accelerations, so it is necessary to
           do a backup first to restore it later on. */
        jointsAcceleration_.swap(pncData_.a);
        pncData_.a[0].setZero();
        for (int i = 1; i < pncModel_.njoints; ++i)
        {
            const auto & jmodel = pncModel_.joints[i];
            const auto & jdata = pncData_.joints[i];
            const pinocchio::JointIndex jointModelIdx = jmodel.id();
            const pinocchio::JointIndex parentJointModelIdx = pncModel_.parents[jointModelIdx];
            pncData_.a[jointModelIdx] = jdata.c() + pncData_.v[jointModelIdx].cross(jdata.v());
            if (parentJointModelIdx > 0)
            {
                pncData_.a[i] += pncData_.liMi[i].actInv(pncData_.a[parentJointModelIdx]);
            }
        }

        // Compute sequentially the jacobian and drift of each enabled constraint
        constraintsHolder_.foreach(
            [&](const std::shared_ptr<AbstractConstraintBase> & constraint,
                constraintsHolderType_t /* holderType */)
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

    hresult_t Model::refreshProxies()
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

            /* Generate the fieldnames associated with the configuration vector, velocity,
               acceleration and external force vectors. */
            logFieldnamesPosition_.clear();
            logFieldnamesPosition_.reserve(static_cast<std::size_t>(nq_));
            logFieldnamesVelocity_.clear();
            logFieldnamesVelocity_.reserve(static_cast<std::size_t>(nv_));
            logFieldnamesAcceleration_.clear();
            logFieldnamesAcceleration_.reserve(static_cast<std::size_t>(nv_));
            logFieldnamesForceExternal_.clear();
            logFieldnamesForceExternal_.reserve(6U * (pncModel_.njoints - 1));
            for (std::size_t i = 1; i < pncModel_.joints.size(); ++i)
            {
                // Get joint name without "Joint" suffix, if any
                std::string jointShortName{removeSuffix(pncModel_.names[i], "Joint")};

                // Get joint prefix depending on its type
                const JointModelType jointType{getJointType(pncModel_.joints[i])};
                std::string jointPrefix{JOINT_PREFIX_BASE};
                if (jointType == JointModelType::FREE)
                {
                    jointPrefix += FREE_FLYER_NAME;
                    jointShortName = "";
                }

                // Get joint position suffices depending on its type
                std::vector<std::string_view> jointTypePositionSuffixes{};
                std::vector<std::string_view> jointTypeVelocitySuffixes{};
                if (returnCode == hresult_t::SUCCESS)
                {
                    returnCode =
                        getJointTypePositionSuffixes(jointType, jointTypePositionSuffixes);
                }

                if (returnCode == hresult_t::SUCCESS)
                {
                    // Get joint velocity suffices depending on its type
                    getJointTypeVelocitySuffixes(
                        jointType, jointTypeVelocitySuffixes);  // Cannot fail at this point

                    // Define complete position fieldnames
                    for (const std::string_view & suffix : jointTypePositionSuffixes)
                    {
                        logFieldnamesPosition_.emplace_back(
                            toString(jointPrefix, "Position", jointShortName, suffix));
                    }

                    // Define complete velocity and acceleration fieldnames
                    for (const std::string_view & suffix : jointTypeVelocitySuffixes)
                    {
                        logFieldnamesVelocity_.emplace_back(
                            toString(jointPrefix, "Velocity", jointShortName, suffix));
                        logFieldnamesAcceleration_.emplace_back(
                            toString(jointPrefix, "Acceleration", jointShortName, suffix));
                    }

                    // Define complete external force fieldnames and backup them
                    std::vector<std::string> jointForceExternalFieldnames;
                    for (const std::string & suffix : ForceSensor::fieldnames_)
                    {
                        logFieldnamesForceExternal_.emplace_back(
                            toString(jointPrefix, "ForceExternal", jointShortName, suffix));
                    }
                }
            }
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            /* Get the joint position limits from the URDF or the user options.
               Do NOT use robot_->pncModel_.(lower|upper)PositionLimit. */
            positionLimitMin_.setConstant(pncModel_.nq, -INF);
            positionLimitMax_.setConstant(pncModel_.nq, +INF);

            if (mdlOptions_->joints.enablePositionLimit)
            {
                if (mdlOptions_->joints.positionLimitFromUrdf)
                {
                    for (Eigen::Index positionIdx : rigidJointsPositionIdx_)
                    {
                        positionLimitMin_[positionIdx] = pncModel_.lowerPositionLimit[positionIdx];
                        positionLimitMax_[positionIdx] = pncModel_.upperPositionLimit[positionIdx];
                    }
                }
                else
                {
                    for (std::size_t i = 0; i < rigidJointsPositionIdx_.size(); ++i)
                    {
                        Eigen::Index positionIdx = rigidJointsPositionIdx_[i];
                        positionLimitMin_[positionIdx] = mdlOptions_->joints.positionLimitMin[i];
                        positionLimitMax_[positionIdx] = mdlOptions_->joints.positionLimitMax[i];
                    }
                }
            }

            /* Overwrite the position bounds for some specific joint type, mainly due to quaternion
               normalization and cos/sin representation. */
            for (const auto & joint : pncModel_.joints)
            {
                Eigen::Index positionIdx, positionNq;
                switch (getJointType(joint))
                {
                case JointModelType::ROTARY_UNBOUNDED:
                case JointModelType::SPHERICAL:
                    positionIdx = joint.idx_q();
                    positionNq = joint.nq();
                    break;
                case JointModelType::FREE:
                    positionIdx = joint.idx_q() + 3;
                    positionNq = 4;
                case JointModelType::UNSUPPORTED:
                case JointModelType::LINEAR:
                case JointModelType::ROTARY:
                case JointModelType::PLANAR:
                case JointModelType::TRANSLATION:
                default:
                    continue;
                }
                positionLimitMin_.segment(positionIdx, positionNq).setConstant(-1.0 - EPS);
                positionLimitMax_.segment(positionIdx, positionNq).setConstant(+1.0 + EPS);
            }

            // Get the joint velocity limits from the URDF or the user options
            velocityLimit_.setConstant(pncModel_.nv, +INF);
            if (mdlOptions_->joints.enableVelocityLimit)
            {
                if (mdlOptions_->joints.velocityLimitFromUrdf)
                {
                    for (Eigen::Index & velocityIdx : rigidJointsVelocityIdx_)
                    {
                        velocityLimit_[velocityIdx] = pncModel_.velocityLimit[velocityIdx];
                    }
                }
                else
                {
                    for (std::size_t i = 0; i < rigidJointsVelocityIdx_.size(); ++i)
                    {
                        Eigen::Index velocityIdx = rigidJointsVelocityIdx_[i];
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

    hresult_t Model::refreshGeometryProxies()
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
                for (auto model :
                     std::array<pinocchio::GeometryModel *, 2>{{&collisionModel_, &visualModel_}})
                {
                    for (pinocchio::GeometryObject & geom : model->geometryObjects)
                    {
                        // Only the frame name remains unchanged no matter what
                        const pinocchio::Frame & frameOrig =
                            pncModelOrig_.frames[geom.parentFrame];
                        const std::string parentJointName = pncModelOrig_.names[frameOrig.parent];
                        pinocchio::FrameIndex frameIdx;
                        getFrameIdx(pncModel_,
                                    frameOrig.name,
                                    frameIdx);  // Cannot fail at this point
                        const pinocchio::Frame & frame = pncModel_.frames[frameIdx];
                        const pinocchio::JointIndex newParentModelIdx = frame.parent;
                        const pinocchio::JointIndex oldParentModelIdx =
                            pncModel_.getJointId(parentJointName);
                        geom.parentFrame = frameIdx;
                        geom.parentJoint = newParentModelIdx;

                        /* Compute the relative displacement between the new and old joint
                           placement wrt their common parent joint. */
                        pinocchio::SE3 geomPlacementRef = pinocchio::SE3::Identity();
                        for (pinocchio::JointIndex i = newParentModelIdx;
                             i > std::max(oldParentModelIdx, pinocchio::JointIndex{0});
                             i = pncModel_.parents[i])
                        {
                            geomPlacementRef = pncModel_.jointPlacements[i] * geomPlacementRef;
                        }
                        geom.placement = geomPlacementRef.actInv(geom.placement);
                    }
                }
            }

            /* Update geometry data object after changing the collision pairs
               Note that copy assignment is used to avoid changing memory pointers, which would
               result in dangling reference at Python-side. */
            collisionData_ = pinocchio::GeometryData(collisionModel_);
            pinocchio::updateGeometryPlacements(
                pncModel_, pncData_, collisionModel_, collisionData_);
            visualData_ = pinocchio::GeometryData(visualModel_);
            pinocchio::updateGeometryPlacements(pncModel_, pncData_, visualModel_, visualData_);

            // Set the max number of contact points per collision pairs
            for (hpp::fcl::CollisionRequest & collisionRequest : collisionData_.collisionRequests)
            {
                collisionRequest.num_max_contacts =
                    mdlOptions_->collisions.maxContactPointsPerBody;
            }

            // Extract the indices of the collision pairs associated with each body
            collisionPairsIdx_.clear();
            for (const std::string & name : collisionBodiesNames_)
            {
                std::vector<pinocchio::PairIndex> collisionPairsIdx;
                for (std::size_t i = 0; i < collisionModel_.collisionPairs.size(); ++i)
                {
                    const pinocchio::CollisionPair & pair = collisionModel_.collisionPairs[i];
                    const pinocchio::GeometryObject & geom =
                        collisionModel_.geometryObjects[pair.first];
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

    hresult_t Model::refreshContactsProxies()
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
            contactForces_ = ForceVector(contactFramesNames_.size(), pinocchio::Force::Zero());

            // Extract the contact frames indices in the model
            getFramesIdx(pncModel_, contactFramesNames_, contactFramesIdx_);
        }

        return returnCode;
    }

    hresult_t Model::refreshConstraintsProxies()
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        // Initialize backup joint space acceleration
        jointsAcceleration_ = MotionVector(pncData_.a.size(), pinocchio::Motion::Zero());

        constraintsHolder_.foreach(
            [&](const std::shared_ptr<AbstractConstraintBase> & constraint,
                constraintsHolderType_t /* holderType */)
            {
                if (returnCode == hresult_t::SUCCESS)
                {
                    // Reset constraint using neutral configuration and zero velocity
                    returnCode = constraint->reset(pinocchio::neutral(pncModel_),
                                                   Eigen::VectorXd::Zero(nv_));
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

    hresult_t Model::setOptions(GenericConfig modelOptions)
    {
        bool internalBuffersMustBeUpdated = false;
        bool areModelsInvalid = false;
        bool isCollisionDataInvalid = false;
        if (isInitialized_)
        {
            /* Check that the following user parameters has the right dimension, then update the
               required internal buffers to reflect changes, if any. */
            GenericConfig & jointOptionsHolder =
                boost::get<GenericConfig>(modelOptions.at("joints"));
            bool positionLimitFromUrdf =
                boost::get<bool>(jointOptionsHolder.at("positionLimitFromUrdf"));
            if (!positionLimitFromUrdf)
            {
                Eigen::VectorXd & jointsPositionLimitMin =
                    boost::get<Eigen::VectorXd>(jointOptionsHolder.at("positionLimitMin"));
                if (rigidJointsPositionIdx_.size() !=
                    static_cast<uint32_t>(jointsPositionLimitMin.size()))
                {
                    PRINT_ERROR("Wrong vector size for 'positionLimitMin'.");
                    return hresult_t::ERROR_BAD_INPUT;
                }
                Eigen::VectorXd & jointsPositionLimitMax =
                    boost::get<Eigen::VectorXd>(jointOptionsHolder.at("positionLimitMax"));
                if (rigidJointsPositionIdx_.size() !=
                    static_cast<uint32_t>(jointsPositionLimitMax.size()))
                {
                    PRINT_ERROR("Wrong vector size for 'positionLimitMax'.");
                    return hresult_t::ERROR_BAD_INPUT;
                }
                if (rigidJointsPositionIdx_.size() ==
                    static_cast<uint32_t>(mdlOptions_->joints.positionLimitMin.size()))
                {
                    auto jointsPositionLimitMinDiff =
                        jointsPositionLimitMin - mdlOptions_->joints.positionLimitMin;
                    internalBuffersMustBeUpdated |=
                        (jointsPositionLimitMinDiff.array().abs() >= EPS).all();
                    auto jointsPositionLimitMaxDiff =
                        jointsPositionLimitMax - mdlOptions_->joints.positionLimitMax;
                    internalBuffersMustBeUpdated |=
                        (jointsPositionLimitMaxDiff.array().abs() >= EPS).all();
                }
                else
                {
                    internalBuffersMustBeUpdated = true;
                }
            }
            bool velocityLimitFromUrdf =
                boost::get<bool>(jointOptionsHolder.at("velocityLimitFromUrdf"));
            if (!velocityLimitFromUrdf)
            {
                Eigen::VectorXd & jointsVelocityLimit =
                    boost::get<Eigen::VectorXd>(jointOptionsHolder.at("velocityLimit"));
                if (rigidJointsVelocityIdx_.size() !=
                    static_cast<uint32_t>(jointsVelocityLimit.size()))
                {
                    PRINT_ERROR("Wrong vector size for 'velocityLimit'.");
                    return hresult_t::ERROR_BAD_INPUT;
                }
                if (rigidJointsVelocityIdx_.size() ==
                    static_cast<uint32_t>(mdlOptions_->joints.velocityLimit.size()))
                {
                    auto jointsVelocityLimitDiff =
                        jointsVelocityLimit - mdlOptions_->joints.velocityLimit;
                    internalBuffersMustBeUpdated |=
                        (jointsVelocityLimitDiff.array().abs() >= EPS).all();
                }
                else
                {
                    internalBuffersMustBeUpdated = true;
                }
            }

            // Check if deformation points are all associated with different joints/frames
            GenericConfig & dynOptionsHolder =
                boost::get<GenericConfig>(modelOptions.at("dynamics"));
            const FlexibilityConfig & flexibilityConfig =
                boost::get<FlexibilityConfig>(dynOptionsHolder.at("flexibilityConfig"));
            std::set<std::string> flexibilityNames;
            std::transform(flexibilityConfig.begin(),
                           flexibilityConfig.end(),
                           std::inserter(flexibilityNames, flexibilityNames.begin()),
                           [](const FlexibleJointData & flexiblePoint) -> std::string
                           { return flexiblePoint.frameName; });
            if (flexibilityNames.size() != flexibilityConfig.size())
            {
                PRINT_ERROR(
                    "All joint or frame names in flexibility configuration must be unique.");
                return hresult_t::ERROR_BAD_INPUT;
            }
            if (std::find(flexibilityNames.begin(), flexibilityNames.end(), "universe") !=
                flexibilityNames.end())
            {
                PRINT_ERROR("No one can make the universe itself flexible.");
                return hresult_t::ERROR_BAD_INPUT;
            }
            for (const FlexibleJointData & flexibleJoint : flexibilityConfig)
            {
                if ((flexibleJoint.stiffness.array() < 0.0).any() ||
                    (flexibleJoint.damping.array() < 0.0).any() ||
                    (flexibleJoint.inertia.array() < 0.0).any())
                {
                    PRINT_ERROR(
                        "The stiffness, damping and inertia of flexibility must be positive.");
                    return hresult_t::ERROR_GENERIC;
                }
            }

            // Check if the position or velocity limits have changed, and refresh proxies if so
            bool enablePositionLimit =
                boost::get<bool>(jointOptionsHolder.at("enablePositionLimit"));
            bool enableVelocityLimit =
                boost::get<bool>(jointOptionsHolder.at("enableVelocityLimit"));
            if (enablePositionLimit != mdlOptions_->joints.enablePositionLimit)
            {
                internalBuffersMustBeUpdated = true;
            }
            else if (enablePositionLimit &&
                     (positionLimitFromUrdf != mdlOptions_->joints.positionLimitFromUrdf))
            {
                internalBuffersMustBeUpdated = true;
            }
            else if (enableVelocityLimit != mdlOptions_->joints.enableVelocityLimit)
            {
                internalBuffersMustBeUpdated = true;
            }
            else if (enableVelocityLimit &&
                     (velocityLimitFromUrdf != mdlOptions_->joints.velocityLimitFromUrdf))
            {
                internalBuffersMustBeUpdated = true;
            }

            // Check if the flexible model and its proxies must be regenerated
            bool enableFlexibleModel =
                boost::get<bool>(dynOptionsHolder.at("enableFlexibleModel"));
            if (mdlOptions_ &&
                (flexibilityConfig.size() != mdlOptions_->dynamics.flexibilityConfig.size() ||
                 !std::equal(flexibilityConfig.begin(),
                             flexibilityConfig.end(),
                             mdlOptions_->dynamics.flexibilityConfig.begin()) ||
                 enableFlexibleModel != mdlOptions_->dynamics.enableFlexibleModel))
            {
                areModelsInvalid = true;
            }
        }

        // Check that the collisions options are valid
        GenericConfig & collisionOptionsHolder =
            boost::get<GenericConfig>(modelOptions.at("collisions"));
        uint32_t maxContactPointsPerBody =
            boost::get<uint32_t>(collisionOptionsHolder.at("maxContactPointsPerBody"));
        if (maxContactPointsPerBody < 1)
        {
            PRINT_ERROR("The number of contact points by collision pair 'maxContactPointsPerBody' "
                        "must be at least 1.");
            return hresult_t::ERROR_BAD_INPUT;
        }
        if (mdlOptions_ &&
            maxContactPointsPerBody != mdlOptions_->collisions.maxContactPointsPerBody)
        {
            isCollisionDataInvalid = true;
        }

        // Check that the model randomization parameters are valid
        GenericConfig & dynOptionsHolder = boost::get<GenericConfig>(modelOptions.at("dynamics"));
        for (const auto & field : std::array<std::string, 4>{{"inertiaBodiesBiasStd",
                                                              "massBodiesBiasStd",
                                                              "centerOfMassPositionBodiesBiasStd",
                                                              "relativePositionBodiesBiasStd"}})
        {
            const double value = boost::get<double>(dynOptionsHolder.at(field));
            if (0.9 < value || value < 0.0)
            {
                PRINT_ERROR(
                    "'", field, "' must be positive, and lower than 0.9 to avoid physics issues.");
                return hresult_t::ERROR_BAD_INPUT;
            }
        }

        // Update the internal options
        mdlOptionsHolder_ = modelOptions;

        // Create a fast struct accessor
        mdlOptions_ = std::make_unique<const modelOptions_t>(mdlOptionsHolder_);

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

    GenericConfig Model::getOptions() const noexcept
    {
        return mdlOptionsHolder_;
    }

    bool Model::getIsInitialized() const
    {
        return isInitialized_;
    }

    const std::string & Model::getName() const
    {
        return pncModelOrig_.name;
    }

    const std::string & Model::getUrdfPath() const
    {
        return urdfPath_;
    }

    const std::string & Model::getUrdfAsString() const
    {
        return urdfData_;
    }

    const std::vector<std::string> & Model::getMeshPackageDirs() const
    {
        return meshPackageDirs_;
    }

    bool Model::getHasFreeflyer() const
    {
        return hasFreeflyer_;
    }

    hresult_t Model::getFlexibleConfigurationFromRigid(const Eigen::VectorXd & qRigid,
                                                       Eigen::VectorXd & qFlex) const
    {
        // Define some proxies
        int nqRigid = pncModelOrig_.nq;

        // Check the size of the input state
        if (qRigid.size() != nqRigid)
        {
            PRINT_ERROR("Size of qRigid inconsistent with theoretical model.");
            return hresult_t::ERROR_BAD_INPUT;
        }

        // Initialize the flexible state
        qFlex = pinocchio::neutral(pncModelFlexibleOrig_);

        // Compute the flexible state based on the rigid state
        int idxRigid = 0;
        int idxFlex = 0;
        for (; idxRigid < pncModelOrig_.njoints; ++idxFlex)
        {
            const std::string & jointRigidName = pncModelOrig_.names[idxRigid];
            const std::string & jointFlexName = pncModelFlexibleOrig_.names[idxFlex];
            if (jointRigidName == jointFlexName)
            {
                const auto & jointRigid = pncModelOrig_.joints[idxRigid];
                const auto & jointFlex = pncModelFlexibleOrig_.joints[idxFlex];
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

    hresult_t Model::getRigidConfigurationFromFlexible(const Eigen::VectorXd & qFlex,
                                                       Eigen::VectorXd & qRigid) const
    {
        // Define some proxies
        uint32_t nqFlex = pncModelFlexibleOrig_.nq;

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
        for (; idxRigid < pncModelOrig_.njoints; ++idxFlex)
        {
            const std::string & jointRigidName = pncModelOrig_.names[idxRigid];
            const std::string & jointFlexName = pncModelFlexibleOrig_.names[idxFlex];
            if (jointRigidName == jointFlexName)
            {
                const auto & jointRigid = pncModelOrig_.joints[idxRigid];
                const auto & jointFlex = pncModelFlexibleOrig_.joints[idxFlex];
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

    hresult_t Model::getFlexibleVelocityFromRigid(const Eigen::VectorXd & vRigid,
                                                  Eigen::VectorXd & vFlex) const
    {
        // Define some proxies
        uint32_t nvRigid = pncModelOrig_.nv;
        uint32_t nvFlex = pncModelFlexibleOrig_.nv;

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
            const std::string & jointRigidName = pncModelOrig_.names[idxRigid];
            const std::string & jointFlexName = pncModelFlexibleOrig_.names[idxFlex];
            if (jointRigidName == jointFlexName)
            {
                const auto & jointRigid = pncModelOrig_.joints[idxRigid];
                const auto & jointFlex = pncModelFlexibleOrig_.joints[idxFlex];
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

    hresult_t Model::getRigidVelocityFromFlexible(const Eigen::VectorXd & vFlex,
                                                  Eigen::VectorXd & vRigid) const
    {
        // Define some proxies
        uint32_t nvRigid = pncModelOrig_.nv;
        uint32_t nvFlex = pncModelFlexibleOrig_.nv;

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
        for (; idxRigid < pncModelOrig_.njoints; ++idxFlex)
        {
            const std::string & jointRigidName = pncModelOrig_.names[idxRigid];
            const std::string & jointFlexName = pncModelFlexibleOrig_.names[idxFlex];
            if (jointRigidName == jointFlexName)
            {
                const auto & jointRigid = pncModelOrig_.joints[idxRigid];
                const auto & jointFlex = pncModelFlexibleOrig_.joints[idxFlex];
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

    const std::vector<std::string> & Model::getCollisionBodiesNames() const
    {
        return collisionBodiesNames_;
    }

    const std::vector<std::string> & Model::getContactFramesNames() const
    {
        return contactFramesNames_;
    }

    const std::vector<pinocchio::FrameIndex> & Model::getCollisionBodiesIdx() const
    {
        return collisionBodiesIdx_;
    }

    const std::vector<std::vector<pinocchio::PairIndex>> & Model::getCollisionPairsIdx() const
    {
        return collisionPairsIdx_;
    }

    const std::vector<pinocchio::FrameIndex> & Model::getContactFramesIdx() const
    {
        return contactFramesIdx_;
    }

    const std::vector<std::string> & Model::getLogFieldnamesPosition() const
    {
        return logFieldnamesPosition_;
    }

    const Eigen::VectorXd & Model::getPositionLimitMin() const
    {
        return positionLimitMin_;
    }

    const Eigen::VectorXd & Model::getPositionLimitMax() const
    {
        return positionLimitMax_;
    }

    const std::vector<std::string> & Model::getLogFieldnamesVelocity() const
    {
        return logFieldnamesVelocity_;
    }

    const Eigen::VectorXd & Model::getVelocityLimit() const
    {
        return velocityLimit_;
    }

    const std::vector<std::string> & Model::getLogFieldnamesAcceleration() const
    {
        return logFieldnamesAcceleration_;
    }

    const std::vector<std::string> & Model::getLogFieldnamesForceExternal() const
    {
        return logFieldnamesForceExternal_;
    }

    const std::vector<std::string> & Model::getRigidJointsNames() const
    {
        return rigidJointsNames_;
    }

    const std::vector<pinocchio::JointIndex> & Model::getRigidJointsModelIdx() const
    {
        return rigidJointsModelIdx_;
    }

    const std::vector<Eigen::Index> & Model::getRigidJointsPositionIdx() const
    {
        return rigidJointsPositionIdx_;
    }

    const std::vector<Eigen::Index> & Model::getRigidJointsVelocityIdx() const
    {
        return rigidJointsVelocityIdx_;
    }

    const std::vector<std::string> & Model::getFlexibleJointsNames() const
    {
        static const std::vector<std::string> flexibleJointsNamesEmpty{};
        if (mdlOptions_->dynamics.enableFlexibleModel)
        {
            return flexibleJointsNames_;
        }
        else
        {
            return flexibleJointsNamesEmpty;
        }
    }

    const std::vector<pinocchio::JointIndex> & Model::getFlexibleJointsModelIdx() const
    {
        static const std::vector<pinocchio::JointIndex> flexibleJointsModelIdxEmpty{};
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
    bool Model::hasConstraints() const
    {
        bool hasConstraintsEnabled = false;
        const_cast<constraintsHolder_t &>(constraintsHolder_)
            .foreach(
                [&hasConstraintsEnabled](
                    const std::shared_ptr<AbstractConstraintBase> & constraint,
                    constraintsHolderType_t /* holderType */)
                {
                    if (constraint->getIsEnabled())
                    {
                        hasConstraintsEnabled = true;
                    }
                });
        return hasConstraintsEnabled;
    }

    Eigen::Index Model::nq() const
    {
        return nq_;
    }

    Eigen::Index Model::nv() const
    {
        return nv_;
    }

    Eigen::Index Model::nx() const
    {
        return nx_;
    }
}
