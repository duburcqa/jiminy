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
#include "jiminy/core/utilities/helpers.h"

#include "jiminy/core/robot/model.h"


namespace jiminy
{
    // ************************************** Constraints ************************************** //

    void ConstraintTree::clear() noexcept
    {
        boundJoints.clear();
        contactFrames.clear();
        collisionBodies.clear();
        registry.clear();
    }

    template<typename T>
    static auto findImpl(T && constraints, const std::string & key, ConstraintNodeType node)
    {
        // Determine return types based on argument constness
        constexpr bool isConst = std::is_const_v<std::remove_reference_t<T>>;
        using constraintMapT = std::conditional_t<isConst, const ConstraintMap, ConstraintMap>;
        using constraintIteratorT =
            std::conditional_t<isConst, ConstraintMap::const_iterator, ConstraintMap::iterator>;

        // Pointers are NOT initialized to nullptr by default
        constraintMapT * constraintMapPtr{nullptr};
        constraintIteratorT constraintIt{};
        if (node == ConstraintNodeType::COLLISION_BODIES)
        {
            for (auto & collisionBody : constraints.collisionBodies)
            {
                constraintMapPtr = &collisionBody;
                constraintIt = std::find_if(constraintMapPtr->begin(),
                                            constraintMapPtr->end(),
                                            [&key](const auto & constraintPair)
                                            { return constraintPair.first == key; });
                if (constraintIt != constraintMapPtr->end())
                {
                    break;
                }
            }
        }
        else
        {
            switch (node)
            {
            case ConstraintNodeType::BOUNDS_JOINTS:
                constraintMapPtr = &constraints.boundJoints;
                break;
            case ConstraintNodeType::CONTACT_FRAMES:
                constraintMapPtr = &constraints.contactFrames;
                break;
            case ConstraintNodeType::USER:
            case ConstraintNodeType::COLLISION_BODIES:
            default:
                constraintMapPtr = &constraints.registry;
            }
            constraintIt = std::find_if(constraintMapPtr->begin(),
                                        constraintMapPtr->end(),
                                        [&key](const auto & constraintPair)
                                        { return constraintPair.first == key; });
        }

        return std::make_pair(constraintMapPtr, constraintIt);
    }

    std::pair<ConstraintMap *, ConstraintMap::iterator> ConstraintTree::find(
        const std::string & key, ConstraintNodeType node)
    {
        return findImpl(*this, key, node);
    }

    std::pair<const ConstraintMap *, ConstraintMap::const_iterator> ConstraintTree::find(
        const std::string & key, ConstraintNodeType node) const
    {
        return findImpl(*this, key, node);
    }

    bool ConstraintTree::exist(const std::string & key, ConstraintNodeType node) const
    {
        const auto [constraintMapPtr, constraintIt] =
            const_cast<ConstraintTree *>(this)->find(key, node);
        return (constraintMapPtr && constraintIt != constraintMapPtr->cend());
    }

    bool ConstraintTree::exist(const std::string & key) const
    {
        for (ConstraintNodeType node : constraintNodeTypesAll)
        {
            if (exist(key, node))
            {
                return true;
            }
        }
        return false;
    }

    std::shared_ptr<AbstractConstraintBase> ConstraintTree::get(const std::string & key,
                                                                ConstraintNodeType node) const
    {
        auto [constraintMapPtr, constraintIt] = find(key, node);
        if (constraintMapPtr && constraintIt != constraintMapPtr->cend())
        {
            return constraintIt->second;
        }
        return {};
    }

    std::shared_ptr<AbstractConstraintBase> ConstraintTree::get(const std::string & key) const
    {
        std::shared_ptr<AbstractConstraintBase> constraint;
        for (ConstraintNodeType node : constraintNodeTypesAll)
        {
            constraint = get(key, node);
            if (constraint)
            {
                break;
            }
        }
        return constraint;
    }

    void ConstraintTree::insert(const ConstraintMap & constraintMap, ConstraintNodeType node)
    {
        switch (node)
        {
        case ConstraintNodeType::BOUNDS_JOINTS:
            boundJoints.insert(boundJoints.end(), constraintMap.begin(), constraintMap.end());
            break;
        case ConstraintNodeType::CONTACT_FRAMES:
            contactFrames.insert(contactFrames.end(), constraintMap.begin(), constraintMap.end());
            break;
        case ConstraintNodeType::COLLISION_BODIES:
            collisionBodies.push_back(constraintMap);
            break;
        case ConstraintNodeType::USER:
        default:
            registry.insert(registry.end(), constraintMap.begin(), constraintMap.end());
        }
    }

    ConstraintMap::iterator ConstraintTree::erase(const std::string & key, ConstraintNodeType node)
    {
        auto [constraintMapPtr, constraintIt] = find(key, node);
        if (constraintMapPtr && constraintIt != constraintMapPtr->end())
        {
            return constraintMapPtr->erase(constraintIt);
        }
        return constraintMapPtr->end();
    }

    // ***************************************** Model ***************************************** //

    Model::Model() noexcept
    {
        setOptions(getDefaultModelOptions());
    }

    void Model::initialize(const pinocchio::Model & pinocchioModel,
                           const std::optional<pinocchio::GeometryModel> & collisionModel,
                           const std::optional<pinocchio::GeometryModel> & visualModel)
    {
        if (pinocchioModel.nq == 0)
        {
            THROW_ERROR(std::invalid_argument, "Pinocchio model must not be empty.");
        }

        // Clear existing constraints
        constraints_.clear();
        jointSpatialAccelerations_.clear();

        // Reset URDF info
        JointModelType rootJointType = getJointTypeFromIndex(pinocchioModel, 1);
        urdfPath_ = "";
        urdfData_ = "";
        hasFreeflyer_ = (rootJointType == JointModelType::FREE);
        meshPackageDirs_.clear();

        // Set the models
        pinocchioModelOrig_ = pinocchioModel;
        collisionModelOrig_ = collisionModel.value_or(pinocchio::GeometryModel());
        visualModelOrig_ = visualModel.value_or(pinocchio::GeometryModel());

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
            collisionModelOrig_.addGeometryObject(groundPlane, pinocchioModelOrig_);
        }

        /* Re-allocate rigid data from scratch for the original rigid model.
           Note that the original rigid model is not used anywhere for simulation but is
           provided nonetheless to make life easier for end-users willing to perform
           computations on it rather than the actual simulation model. */
        pinocchioDataOrig_ = pinocchio::Data(pinocchioModelOrig_);

        /* Initialize Pinocchio data internal state.
           This includes "basic" attributes such as the mass of each body. */
        const Eigen::VectorXd qNeutralOrig = pinocchio::neutral(pinocchioModelOrig_);
        pinocchio::forwardKinematics(pinocchioModelOrig_,
                                     pinocchioDataOrig_,
                                     qNeutralOrig,
                                     Eigen::VectorXd::Zero(pinocchioModelOrig_.nv));
        pinocchio::updateFramePlacements(pinocchioModelOrig_, pinocchioDataOrig_);
        pinocchio::centerOfMass(pinocchioModelOrig_, pinocchioDataOrig_, qNeutralOrig);

        /* Get the list of joint names of the rigid model and remove the 'universe' and
           'root_joint' if any, since they are not actual joints. */
        rigidJointNames_ = pinocchioModelOrig_.names;
        rigidJointNames_.erase(rigidJointNames_.begin());  // remove 'universe'
        if (hasFreeflyer_)
        {
            rigidJointNames_.erase(rigidJointNames_.begin());  // remove 'root_joint'
        }

        // Create the flexible model
        generateModelFlexible();

        // Assuming the model is fully initialized at this point
        isInitialized_ = true;
        try
        {
            /* Add biases to the dynamics properties of the model.
            Note that is also refresh all proxies automatically. */
            generateModelBiased(std::random_device{});

            /* Add joint constraints.
            It will be used later to enforce bounds limits eventually. */
            ConstraintMap jointConstraintsMap;
            jointConstraintsMap.reserve(rigidJointNames_.size());
            for (const std::string & jointName : rigidJointNames_)
            {
                jointConstraintsMap.emplace_back(jointName,
                                                 std::make_shared<JointConstraint>(jointName));
            }
            addConstraints(jointConstraintsMap, ConstraintNodeType::BOUNDS_JOINTS);
        }
        catch (...)
        {
            // Unset the initialization flag in case of failure
            isInitialized_ = false;
            throw;
        }
    }

    void Model::initialize(const std::string & urdfPath,
                           bool hasFreeflyer,
                           const std::vector<std::string> & meshPackageDirs,
                           bool loadVisualMeshes)
    {
        // Load new robot and collision models
        pinocchio::Model pinocchioModel;
        pinocchio::GeometryModel pinocchioCollisionModel;
        pinocchio::GeometryModel pinocchioVisualModel;
        buildMultipleModelsFromUrdf(urdfPath,
                                    hasFreeflyer,
                                    meshPackageDirs,
                                    pinocchioModel,
                                    pinocchioCollisionModel,
                                    pinocchioVisualModel,
                                    loadVisualMeshes);

        // Initialize jiminy model
        initialize(pinocchioModel, pinocchioCollisionModel, pinocchioVisualModel);

        // Backup URDF info
        urdfPath_ = urdfPath;
        std::ifstream urdfFileStream(urdfPath_);
        urdfData_ = std::string(std::istreambuf_iterator<char>(urdfFileStream),
                                std::istreambuf_iterator<char>());
        meshPackageDirs_ = meshPackageDirs;
    }

    void Model::reset(const uniform_random_bit_generator_ref<uint32_t> & g)
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
            generateModelBiased(g);
        }
    }

    void Model::addFrame(const std::string & frameName,
                         const std::string & parentBodyName,
                         const pinocchio::SE3 & framePlacement,
                         const pinocchio::FrameType & frameType)
    {
        /* The frame must be added directly to the parent joint because it is not possible to add a
           frame to another frame. This means that the relative transform of the frame wrt the
           parent joint must be computed. */

        // Check that no frame with the same name already exists
        if (pinocchioModelOrig_.existFrame(frameName))
        {
            THROW_ERROR(std::invalid_argument, "Frame with same name already exists.");
        }

        // Add frame to original rigid model
        {
            const pinocchio::FrameIndex parentFrameIndex =
                getFrameIndex(pinocchioModelOrig_, parentBodyName);
            pinocchio::JointIndex parentJointIndex =
                pinocchioModelOrig_.frames[parentFrameIndex].parent;
            const pinocchio::SE3 & parentFramePlacement =
                pinocchioModelOrig_.frames[parentFrameIndex].placement;
            const pinocchio::SE3 jointFramePlacement = parentFramePlacement.act(framePlacement);
            const pinocchio::Frame frame(
                frameName, parentJointIndex, parentFrameIndex, jointFramePlacement, frameType);
            pinocchioModelOrig_.addFrame(frame);
            // TODO: Do NOT re-allocate from scratch but update existing data for efficiency
            pinocchioDataOrig_ = pinocchio::Data(pinocchioModelOrig_);
        }

        // Add frame to original flexible model
        {
            const pinocchio::FrameIndex parentFrameIndex =
                getFrameIndex(pncModelFlexibleOrig_, parentBodyName);
            pinocchio::JointIndex parentJointIndex =
                pncModelFlexibleOrig_.frames[parentFrameIndex].parent;
            const pinocchio::SE3 & parentFramePlacement =
                pncModelFlexibleOrig_.frames[parentFrameIndex].placement;
            const pinocchio::SE3 jointFramePlacement = parentFramePlacement.act(framePlacement);
            const pinocchio::Frame frame(
                frameName, parentJointIndex, parentFrameIndex, jointFramePlacement, frameType);
            pncModelFlexibleOrig_.addFrame(frame);
        }

        /* Backup the current rotor inertias and effort limits to restore them.
           Note that it is only necessary because 'reset' is not called for efficiency. It is
           reasonable to assume that no other fields have been overriden by derived classes
           such as Robot. */
        Eigen::VectorXd rotorInertia = pinocchioModel_.rotorInertia;
        Eigen::VectorXd effortLimit = pinocchioModel_.effortLimit;

        /* One must re-generate the model after adding a frame.
           Note that, since the added frame being the "last" of the model, the proxies are
           still up-to-date and therefore it is unecessary to call 'reset'. */
        generateModelBiased(std::random_device{});

        // Restore the current rotor inertias and effort limits
        pinocchioModel_.rotorInertia.swap(rotorInertia);
        pinocchioModel_.effortLimit.swap(effortLimit);
    }

    void Model::addFrame(const std::string & frameName,
                         const std::string & parentBodyName,
                         const pinocchio::SE3 & framePlacement)
    {
        const pinocchio::FrameType frameType = pinocchio::FrameType::OP_FRAME;
        return addFrame(frameName, parentBodyName, framePlacement, frameType);
    }

    void Model::removeFrames(const std::vector<std::string> & frameNames)
    {
        /* Check that the frame can be safely removed from the original rigid model.
           If so, it is also the case for the original flexible models. */
        for (const std::string & frameName : frameNames)
        {
            const pinocchio::FrameType frameType = pinocchio::FrameType::OP_FRAME;
            pinocchio::FrameIndex frameIndex = getFrameIndex(pinocchioModelOrig_, frameName);
            if (pinocchioModelOrig_.frames[frameIndex].type != frameType)
            {
                THROW_ERROR(std::logic_error, "Only frames manually added can be removed.");
            }
        }

        for (const std::string & frameName : frameNames)
        {
            // Remove frame from original rigid model
            {
                const pinocchio::FrameIndex frameIndex =
                    getFrameIndex(pinocchioModelOrig_, frameName);
                pinocchioModelOrig_.frames.erase(std::next(pinocchioModelOrig_.frames.begin(),
                                                           static_cast<uint32_t>(frameIndex)));
                pinocchioModelOrig_.nframes--;
            }

            // Remove frame from original flexible model
            {
                const pinocchio::FrameIndex frameIndex =
                    getFrameIndex(pncModelFlexibleOrig_, frameName);
                pncModelFlexibleOrig_.frames.erase(
                    std::next(pncModelFlexibleOrig_.frames.begin(), frameIndex));
                pncModelFlexibleOrig_.nframes--;
            }
        }

        // TODO: Do NOT re-allocate from scratch but update existing data for efficiency
        pinocchioDataOrig_ = pinocchio::Data(pinocchioModelOrig_);

        // One must reset the model after removing a frame
        reset(std::random_device{});
    }

    void Model::removeFrame(const std::string & frameName)
    {
        return removeFrames({frameName});
    }

    void Model::addCollisionBodies(const std::vector<std::string> & bodyNames, bool ignoreMeshes)
    {
        if (!isInitialized_)
        {
            THROW_ERROR(bad_control_flow, "Model not initialized.");
        }

        // Returning early if nothing to do
        if (bodyNames.empty())
        {
            return;
        }

        // If successfully loaded, the ground should be available
        if (collisionModelOrig_.ngeoms == 0)
        {
            THROW_ERROR(std::runtime_error,
                        "Collision geometry not available. Some collision meshes were "
                        "probably not found.");
        }

        // Make sure that no body are duplicates
        if (checkDuplicates(bodyNames))
        {
            THROW_ERROR(std::invalid_argument, "Duplicated bodies found.");
        }

        // Make sure there is no collision already associated with any of the bodies in the list
        if (checkIntersection(collisionBodyNames_, bodyNames))
        {
            THROW_ERROR(std::invalid_argument,
                        "At least one of the bodies already associated with a collision.");
        }

        // Make sure that all the bodies exist
        for (const std::string & name : bodyNames)
        {
            if (!pinocchioModel_.existBodyName(name))
            {
                THROW_ERROR(std::invalid_argument, "At least one of the bodies does not exist.");
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
                    pinocchioModel_.frames[geom.parentFrame].name == name)
                {
                    hasGeometry = true;
                    break;
                }
            }
            if (!hasGeometry)
            {
                THROW_ERROR(std::invalid_argument,
                            "At least one of the bodies not associated with any collision "
                            "geometry of requested type.");
            }
        }

        // Add the list of bodies to the set of collision bodies
        collisionBodyNames_.insert(collisionBodyNames_.end(), bodyNames.begin(), bodyNames.end());

        // Create the collision pairs and add them to the geometry model of the robot
        const pinocchio::GeomIndex & groundIndex = collisionModelOrig_.getGeometryId("ground");
        for (const std::string & name : bodyNames)
        {
            // Add a collision pair for all geometries having the body as parent
            ConstraintMap collisionConstraintsMap;
            for (std::size_t i = 0; i < collisionModelOrig_.geometryObjects.size(); ++i)
            {
                const pinocchio::GeometryObject & geom = collisionModelOrig_.geometryObjects[i];
                const bool isGeomMesh = (geom.meshPath.find('/') != std::string::npos ||
                                         geom.meshPath.find('\\') != std::string::npos);
                const std::string & frameName = pinocchioModel_.frames[geom.parentFrame].name;
                if (!(ignoreMeshes && isGeomMesh) && frameName == name)
                {
                    // Add constraint associated with contact frame only if it is a sphere
                    const hpp::fcl::CollisionGeometry & shape = *geom.geometry;
                    if (shape.getNodeType() == hpp::fcl::GEOM_SPHERE)
                    {
                        /* Create and add the collision pair with the ground.
                           Note that the ground always comes second for the normal to be
                           consistently compute wrt the ground instead of the body. */
                        const pinocchio::CollisionPair collisionPair(i, groundIndex);
                        collisionModelOrig_.addCollisionPair(collisionPair);

                        /* Add dedicated frame.
                           Note that 'BODY' type is used instead of default 'OP_FRAME' to it
                           clear it is not consider as manually added to the model, and
                           therefore cannot be deleted by the user. */
                        const pinocchio::FrameType frameType = pinocchio::FrameType::FIXED_JOINT;
                        addFrame(geom.name, frameName, geom.placement, frameType);

                        // Add fixed frame constraint of bounded sphere
                        // const hpp::fcl::Sphere & sphere =
                        //     static_cast<const hpp::fcl::Sphere &>(shape);
                        // collisionConstraintsMap.emplace_back(
                        //     geom.name,
                        //     std::make_shared<SphereConstraint>(geom.name, sphere.radius));
                        collisionConstraintsMap.emplace_back(
                            geom.name,
                            std::make_shared<FrameConstraint>(
                                geom.name, std::array{true, true, true, false, false, true}));
                    }

                    // TODO: Add warning or error to notify that a geometry has been ignored
                }
            }

            // Add constraints map
            addConstraints(collisionConstraintsMap, ConstraintNodeType::COLLISION_BODIES);
        }

        // Refresh proxies associated with the collisions only
        refreshGeometryProxies();
    }

    void Model::removeCollisionBodies(std::vector<std::string> bodyNames)
    {
        if (!isInitialized_)
        {
            THROW_ERROR(bad_control_flow, "Model not initialized.");
        }

        // Make sure that no body are duplicates
        if (checkDuplicates(bodyNames))
        {
            THROW_ERROR(std::invalid_argument, "Duplicated bodies found.");
        }

        // Make sure that every body in the list is associated with a collision
        if (!checkInclusion(collisionBodyNames_, bodyNames))
        {
            THROW_ERROR(std::invalid_argument,
                        "At least one of the bodies not associated with any collision.");
        }

        /* Remove the list of bodies from the set of collision bodies, then remove the associated
           set of collision pairs for each of them. */
        if (bodyNames.empty())
        {
            bodyNames = collisionBodyNames_;
        }

        for (const std::string & bodyName : bodyNames)
        {
            const auto collisionBodyNameIt =
                std::find(collisionBodyNames_.begin(), collisionBodyNames_.end(), bodyName);
            const std::ptrdiff_t collisionBodyIndex =
                std::distance(collisionBodyNames_.begin(), collisionBodyNameIt);
            collisionBodyNames_.erase(collisionBodyNameIt);
            collisionPairIndices_.erase(collisionPairIndices_.begin() + collisionBodyIndex);
        }

        // Get indices of corresponding collision pairs in geometry model of robot and remove them
        std::vector<std::string> collisionConstraintNames;
        const pinocchio::GeomIndex & groundIndex = collisionModelOrig_.getGeometryId("ground");
        for (const std::string & name : bodyNames)
        {
            // Remove the collision pair for all the geometries having the body as parent
            for (std::size_t i = 0; i < collisionModelOrig_.geometryObjects.size(); ++i)
            {
                const pinocchio::GeometryObject & geom = collisionModelOrig_.geometryObjects[i];
                if (pinocchioModel_.frames[geom.parentFrame].name == name)
                {
                    // Remove the collision pair with the ground
                    const pinocchio::CollisionPair collisionPair(i, groundIndex);
                    collisionModelOrig_.removeCollisionPair(collisionPair);

                    // Append collision geometry to the list of constraints to remove
                    if (constraints_.exist(geom.name, ConstraintNodeType::COLLISION_BODIES))
                    {
                        collisionConstraintNames.emplace_back(geom.name);
                    }
                }
            }
        }

        // Remove the constraints and associated frames
        removeConstraints(collisionConstraintNames, ConstraintNodeType::COLLISION_BODIES);
        removeFrames(collisionConstraintNames);

        // Refresh proxies associated with the collisions only
        refreshGeometryProxies();
    }

    void Model::addContactPoints(const std::vector<std::string> & frameNames)
    {
        if (!isInitialized_)
        {
            THROW_ERROR(bad_control_flow, "Model not initialized.");
        }

        // Make sure that no frame are duplicates
        if (checkDuplicates(frameNames))
        {
            THROW_ERROR(std::invalid_argument, "Duplicated frames found.");
        }

        // Make sure that there is no contact already associated with any of the frames in the list
        if (checkIntersection(contactFrameNames_, frameNames))
        {
            THROW_ERROR(std::invalid_argument,
                        "At least one of the frames already associated with a contact.");
        }

        // Make sure that all the frames exist
        for (const std::string & name : frameNames)
        {
            if (!pinocchioModel_.existFrame(name))
            {
                THROW_ERROR(std::invalid_argument, "At least one of the frames does not exist.");
            }
        }

        // Add the list of frames to the set of contact frames
        contactFrameNames_.insert(contactFrameNames_.end(), frameNames.begin(), frameNames.end());

        // Add constraint associated with contact frame
        ConstraintMap frameConstraintsMap;
        frameConstraintsMap.reserve(frameNames.size());
        for (const std::string & frameName : frameNames)
        {
            frameConstraintsMap.emplace_back(
                frameName,
                std::make_shared<FrameConstraint>(
                    frameName, std::array{true, true, true, false, false, true}));
        }
        addConstraints(frameConstraintsMap, ConstraintNodeType::CONTACT_FRAMES);

        // Refresh proxies associated with contacts and constraints
        refreshContactProxies();
    }

    void Model::removeContactPoints(const std::vector<std::string> & frameNames)
    {
        if (!isInitialized_)
        {
            THROW_ERROR(bad_control_flow, "Model not initialized.");
        }

        // Make sure that no frame are duplicates
        if (checkDuplicates(frameNames))
        {
            THROW_ERROR(std::invalid_argument, "Duplicated frames found.");
        }

        // Make sure that every frame in the list is associated with a contact
        if (!checkInclusion(contactFrameNames_, frameNames))
        {
            THROW_ERROR(std::invalid_argument,
                        "At least one of the frames not associated with a contact.");
        }

        /* Remove the constraint associated with contact frame, then remove the list of frames from
           the set of contact frames. */
        if (!frameNames.empty())
        {
            removeConstraints(frameNames, ConstraintNodeType::CONTACT_FRAMES);
            eraseVector(contactFrameNames_, frameNames);
        }
        else
        {
            removeConstraints(contactFrameNames_, ConstraintNodeType::CONTACT_FRAMES);
            contactFrameNames_.clear();
        }

        // Refresh proxies associated with contacts and constraints
        refreshContactProxies();
    }

    void Model::addConstraints(const ConstraintMap & constraintMap, ConstraintNodeType node)
    {
        // Check if constraint is properly defined and not already exists
        for (const auto & [constraintName, constraintPtr] : constraintMap)
        {
            if (!constraintPtr)
            {
                THROW_ERROR(std::invalid_argument,
                            "Constraint named '",
                            constraintName,
                            "' is undefined.");
            }
            if (constraints_.exist(constraintName))
            {
                THROW_ERROR(std::invalid_argument,
                            "A constraint named '",
                            constraintName,
                            "' already exists.");
            }
        }

        // Attach constraint if not already exist
        for (auto & constraintPair : constraintMap)
        {
            constraintPair.second->attach(shared_from_this());
        }

        // Add them to constraints holder
        constraints_.insert(constraintMap, node);

        // Disable internal constraint by default if internal
        if (node != ConstraintNodeType::USER)
        {
            for (auto & constraintItem : constraintMap)
            {
                constraintItem.second->disable();
            }
        }
    }

    void Model::addConstraint(const std::string & constraintName,
                              const std::shared_ptr<AbstractConstraintBase> & constraint,
                              ConstraintNodeType node)
    {
        return addConstraints({{constraintName, constraint}}, node);
    }

    void Model::addConstraint(const std::string & constraintName,
                              const std::shared_ptr<AbstractConstraintBase> & constraint)
    {
        return addConstraint(constraintName, constraint, ConstraintNodeType::USER);
    }

    void Model::removeConstraints(const std::vector<std::string> & constraintNames,
                                  ConstraintNodeType node)
    {
        // Make sure the constraints exists
        for (const std::string & constraintName : constraintNames)
        {
            if (!constraints_.exist(constraintName, node))
            {
                if (node == ConstraintNodeType::USER)
                {
                    THROW_ERROR(std::invalid_argument,
                                "No user-registered constraint with name '",
                                constraintName,
                                "' exists.");
                }
                THROW_ERROR(std::invalid_argument,
                            "No internal constraint with name '",
                            constraintName,
                            "' exists.");
            }
        }

        // Remove every constraint sequentially
        for (const std::string & constraintName : constraintNames)
        {
            // Lookup constraint
            auto [constraintMapPtr, constraintIt] = constraints_.find(constraintName, node);

            // Detach the constraint
            constraintIt->second->detach();

            // Remove the constraint from the holder
            constraintMapPtr->erase(constraintIt);
        }
    }

    void Model::removeConstraint(const std::string & constraintName, ConstraintNodeType node)
    {
        return removeConstraints({constraintName}, node);
    }

    void Model::removeConstraint(const std::string & constraintName)
    {
        return removeConstraint(constraintName, ConstraintNodeType::USER);
    }

    std::shared_ptr<AbstractConstraintBase> Model::getConstraint(
        const std::string & constraintName)
    {
        std::shared_ptr<AbstractConstraintBase> constraint = constraints_.get(constraintName);
        if (!constraint)
        {
            THROW_ERROR(
                std::invalid_argument, "No constraint with name '", constraintName, "' exists.");
        }
        return constraint;
    }

    std::weak_ptr<const AbstractConstraintBase> Model::getConstraint(
        const std::string & constraintName) const
    {
        std::weak_ptr<const AbstractConstraintBase> constraint =
            std::const_pointer_cast<const AbstractConstraintBase>(
                const_cast<ConstraintTree &>(constraints_).get(constraintName));
        if (!constraint.lock())
        {
            THROW_ERROR(
                std::invalid_argument, "No constraint with name '", constraintName, "' exists.");
        }
        return constraint;
    }

    const ConstraintTree & Model::getConstraints() const
    {
        return constraints_;
    }

    bool Model::existConstraint(const std::string & constraintName) const
    {
        return constraints_.exist(constraintName);
    }

    void Model::resetConstraints(const Eigen::VectorXd & q, const Eigen::VectorXd & v)
    {
        constraints_.foreach([&q, &v](const std::shared_ptr<AbstractConstraintBase> & constraint,
                                      ConstraintNodeType /* node */) { constraint->reset(q, v); });

        constraints_.foreach(std::array{ConstraintNodeType::BOUNDS_JOINTS,
                                        ConstraintNodeType::CONTACT_FRAMES,
                                        ConstraintNodeType::COLLISION_BODIES},
                             [](const std::shared_ptr<AbstractConstraintBase> & constraint,
                                ConstraintNodeType /* node */) { constraint->disable(); });
    }

    void Model::generateModelFlexible()
    {
        // Copy the original model
        pncModelFlexibleOrig_ = pinocchioModelOrig_;

        // Check that the frames exist
        for (const FlexibleJointData & flexibleJoint : modelOptions_->dynamics.flexibilityConfig)
        {
            const std::string & frameName = flexibleJoint.frameName;
            if (!pinocchioModelOrig_.existFrame(frameName))
            {
                THROW_ERROR(std::logic_error,
                            "Frame '",
                            frameName,
                            "' does not exists. Impossible to insert flexible joint on it.");
            }
        }

        // Add all the flexible joints
        flexibleJointNames_.clear();
        for (const FlexibleJointData & flexibleJoint : modelOptions_->dynamics.flexibilityConfig)
        {
            // Extract some proxies
            const std::string & frameName = flexibleJoint.frameName;
            std::string flexName = frameName;
            const pinocchio::FrameIndex frameIndex =
                getFrameIndex(pncModelFlexibleOrig_, frameName);
            const pinocchio::Frame & frame = pncModelFlexibleOrig_.frames[frameIndex];

            // Add joint to model, differently depending on its type
            if (frame.type == pinocchio::FrameType::FIXED_JOINT)
            {
                // Insert flexible joint at fixed frame, splitting "composite" body inertia
                insertFlexibilityAtFixedFrameInModel(pncModelFlexibleOrig_, frameName);
            }
            else if (frame.type == pinocchio::FrameType::JOINT)
            {
                flexName += FLEXIBLE_JOINT_SUFFIX;
                insertFlexibilityBeforeJointInModel(pncModelFlexibleOrig_, frameName, flexName);
            }
            else
            {
                THROW_ERROR(std::logic_error,
                            "Flexible joint can only be inserted at fixed or joint frames.");
            }
            flexibleJointNames_.push_back(flexName);
        }

        // Compute flexible joint indices
        flexibleJointIndices_ = getJointIndices(pncModelFlexibleOrig_, flexibleJointNames_);

        // Add flexibility armature-like inertia to the model
        for (std::size_t i = 0; i < flexibleJointIndices_.size(); ++i)
        {
            const FlexibleJointData & flexibleJoint = modelOptions_->dynamics.flexibilityConfig[i];
            const pinocchio::JointModel & jmodel =
                pncModelFlexibleOrig_.joints[flexibleJointIndices_[i]];
            jmodel.jointVelocitySelector(pncModelFlexibleOrig_.rotorInertia) =
                flexibleJoint.inertia;
        }

        // Check that the armature inertia is valid
        for (pinocchio::JointIndex flexibleJointIndex : flexibleJointIndices_)
        {
            const pinocchio::Inertia & flexibleInertia =
                pncModelFlexibleOrig_.inertias[flexibleJointIndex];
            const pinocchio::JointModel & jmodel =
                pncModelFlexibleOrig_.joints[flexibleJointIndex];
            const Eigen::Vector3d inertiaDiag =
                jmodel.jointVelocitySelector(pncModelFlexibleOrig_.rotorInertia) +
                flexibleInertia.inertia().matrix().diagonal();
            if ((inertiaDiag.array() < 1e-5).any())
            {
                THROW_ERROR(std::runtime_error,
                            "The subtree diagonal inertia for flexibility joint ",
                            flexibleJointIndex,
                            " must be larger than 1e-5 for numerical stability: ",
                            inertiaDiag.transpose());
            }
        }
    }

    void Model::generateModelBiased(const uniform_random_bit_generator_ref<uint32_t> & g)
    {
        // Make sure the model is initialized
        if (!isInitialized_)
        {
            THROW_ERROR(bad_control_flow, "Model not initialized.");
        }

        // Reset the robot either with the original rigid or flexible model
        if (modelOptions_->dynamics.enableFlexibleModel)
        {
            pinocchioModel_ = pncModelFlexibleOrig_;
        }
        else
        {
            pinocchioModel_ = pinocchioModelOrig_;
        }

        // Initially set effortLimit to zero systematically
        pinocchioModel_.effortLimit.setZero();

        for (const std::string & jointName : rigidJointNames_)
        {
            const pinocchio::JointIndex jointIndex =
                ::jiminy::getJointIndex(pinocchioModel_, jointName);

            // Add bias to com position
            const float comBiasStd =
                static_cast<float>(modelOptions_->dynamics.centerOfMassPositionBodiesBiasStd);
            if (comBiasStd > EPS)
            {
                Eigen::Vector3d & comRelativePositionBody =
                    pinocchioModel_.inertias[jointIndex].lever();
                comRelativePositionBody.array() *=
                    normal(3, 1, g, 1.0F, comBiasStd).array().cast<double>();
            }

            /* Add bias to body mass.
               It cannot be less than min(original mass, 1g) for numerical stability. */
            const float massBiasStd =
                static_cast<float>(modelOptions_->dynamics.massBodiesBiasStd);
            if (massBiasStd > EPS)
            {
                double & massBody = pinocchioModel_.inertias[jointIndex].mass();
                massBody =
                    std::max(massBody * normal(g, 1.0F, massBiasStd), std::min(massBody, 1.0e-3));
            }

            /* Add bias to inertia matrix of body.
               To preserve positive semi-definite property after noise addition, the principal
               axes and moments are computed from the original inertia matrix, then independent
               gaussian distributed noise is added on each principal moments, and a random
               small rotation is applied to the principal axes based on a randomly generated
               rotation axis. Finally, the biased inertia matrix is obtained doing
               `A @ diag(M) @ A.T`. If no bias, the original inertia matrix is recovered. */
            const float inertiaBiasStd =
                static_cast<float>(modelOptions_->dynamics.inertiaBodiesBiasStd);
            if (inertiaBiasStd > EPS)
            {
                pinocchio::Symmetric3 & inertiaBody =
                    pinocchioModel_.inertias[jointIndex].inertia();
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(inertiaBody.matrix());
                Eigen::Vector3d inertiaBodyMoments = solver.eigenvalues();
                Eigen::Matrix3d inertiaBodyAxes = solver.eigenvectors();
                const Eigen::Vector3d randAxis =
                    normal(3, 1, g, 0.0F, inertiaBiasStd).cast<double>();
                inertiaBodyAxes = inertiaBodyAxes * Eigen::Quaterniond(pinocchio::exp3(randAxis));
                inertiaBodyMoments.array() *=
                    normal(3, 1, g, 1.0F, inertiaBiasStd).array().cast<double>();
                inertiaBody =
                    pinocchio::Symmetric3((inertiaBodyAxes * inertiaBodyMoments.asDiagonal() *
                                           inertiaBodyAxes.transpose())
                                              .eval());
            }

            // Add bias to relative body position (rotation excluded !)
            const float relativeBodyPosBiasStd =
                static_cast<float>(modelOptions_->dynamics.relativePositionBodiesBiasStd);
            if (relativeBodyPosBiasStd > EPS)
            {
                Eigen::Vector3d & relativePositionBody =
                    pinocchioModel_.jointPlacements[jointIndex].translation();
                relativePositionBody.array() *=
                    normal(3, 1, g, 1.0F, relativeBodyPosBiasStd).array().cast<double>();
            }
        }

        // Re-allocate rigid data from scratch
        pinocchioData_ = pinocchio::Data(pinocchioModel_);

        // Initialize Pinocchio Data internal state
        const Eigen::VectorXd qNeutral = pinocchio::neutral(pinocchioModel_);
        pinocchio::forwardKinematics(
            pinocchioModel_, pinocchioData_, qNeutral, Eigen::VectorXd::Zero(pinocchioModel_.nv));
        pinocchio::updateFramePlacements(pinocchioModel_, pinocchioData_);
        pinocchio::centerOfMass(pinocchioModel_, pinocchioData_, qNeutral);

        // Refresh internal proxies
        refreshProxies();
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

        // Compute inertia matrix (taking into account rotor armatures) along with joint jacobians
        pinocchio_overload::crba(pinocchioModel_, pinocchioData_, q, false);

        /* Computing forward kinematics without acceleration to get the drift.
           Note that it will alter the actual joints spatial accelerations, so it is necessary to
           do a backup first to restore it later on. */
        jointSpatialAccelerations_.swap(pinocchioData_.a);
        pinocchioData_.a[0].setZero();
        for (int jointIndex = 1; jointIndex < pinocchioModel_.njoints; ++jointIndex)
        {
            const auto & jdata = pinocchioData_.joints[jointIndex];
            const pinocchio::JointIndex parentJointIndex = pinocchioModel_.parents[jointIndex];
            pinocchioData_.a[jointIndex] =
                jdata.c() + pinocchioData_.v[jointIndex].cross(jdata.v());
            if (parentJointIndex > 0)
            {
                pinocchioData_.a[jointIndex] +=
                    pinocchioData_.liMi[jointIndex].actInv(pinocchioData_.a[parentJointIndex]);
            }
        }

        // Compute sequentially the jacobian and drift of each enabled constraint
        constraints_.foreach(
            [&](const std::shared_ptr<AbstractConstraintBase> & constraint,
                ConstraintNodeType /* node */)
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
        jointSpatialAccelerations_.swap(pinocchioData_.a);
    }

    void Model::refreshProxies()
    {
        if (!isInitialized_)
        {
            THROW_ERROR(bad_control_flow, "Model not initialized.");
        }

        // Extract the dimensions of the configuration and velocity vectors
        nq_ = pinocchioModel_.nq;
        nv_ = pinocchioModel_.nv;
        nx_ = nq_ + nv_;

        // Extract some rigid joints indices in the model
        rigidJointIndices_ = getJointIndices(pinocchioModel_, rigidJointNames_);
        rigidJointPositionIndices_ =
            getJointsPositionIndices(pinocchioModel_, rigidJointNames_, false);
        rigidJointVelocityIndices_ =
            getJointsVelocityIndices(pinocchioModel_, rigidJointNames_, false);

        /* Generate the fieldnames associated with the configuration vector, velocity,
           acceleration and external force vectors. */
        logPositionFieldnames_.clear();
        logPositionFieldnames_.reserve(static_cast<std::size_t>(nq_));
        logVelocityFieldnames_.clear();
        logVelocityFieldnames_.reserve(static_cast<std::size_t>(nv_));
        logAccelerationFieldnames_.clear();
        logAccelerationFieldnames_.reserve(static_cast<std::size_t>(nv_));
        logForceExternalFieldnames_.clear();
        logForceExternalFieldnames_.reserve(6U * (pinocchioModel_.njoints - 1));
        for (std::size_t i = 1; i < pinocchioModel_.joints.size(); ++i)
        {
            // Get joint name without "Joint" suffix, if any
            std::string jointShortName{removeSuffix(pinocchioModel_.names[i], "Joint")};

            // Get joint prefix depending on its type
            const JointModelType jointType{getJointType(pinocchioModel_.joints[i])};
            std::string jointPrefix{JOINT_PREFIX_BASE};
            if (jointType == JointModelType::FREE)
            {
                jointPrefix += FREE_FLYER_NAME;
                jointShortName = "";
            }

            // Get joint position and velocity suffices depending on its type
            std::vector<std::string_view> jointTypePositionSuffixes =
                getJointTypePositionSuffixes(jointType);
            std::vector<std::string_view> jointTypeVelocitySuffixes =
                getJointTypeVelocitySuffixes(jointType);

            // Define complete position fieldnames
            for (const std::string_view & suffix : jointTypePositionSuffixes)
            {
                logPositionFieldnames_.emplace_back(
                    toString(jointPrefix, "Position", jointShortName, suffix));
            }

            // Define complete velocity and acceleration fieldnames
            for (const std::string_view & suffix : jointTypeVelocitySuffixes)
            {
                logVelocityFieldnames_.emplace_back(
                    toString(jointPrefix, "Velocity", jointShortName, suffix));
                logAccelerationFieldnames_.emplace_back(
                    toString(jointPrefix, "Acceleration", jointShortName, suffix));
            }

            // Define complete external force fieldnames and backup them
            std::vector<std::string> jointForceExternalFieldnames;
            for (const std::string & suffix : ForceSensor::fieldnames_)
            {
                logForceExternalFieldnames_.emplace_back(
                    toString(jointPrefix, "ForceExternal", jointShortName, suffix));
            }
        }

        /* Get the joint position limits from the URDF or the user options.
           Do NOT use robot_->pinocchioModel_.(lower|upper)PositionLimit. */
        positionLimitMin_.setConstant(pinocchioModel_.nq, -INF);
        positionLimitMax_.setConstant(pinocchioModel_.nq, +INF);

        if (modelOptions_->joints.enablePositionLimit)
        {
            if (modelOptions_->joints.positionLimitFromUrdf)
            {
                for (Eigen::Index positionIndex : rigidJointPositionIndices_)
                {
                    positionLimitMin_[positionIndex] =
                        pinocchioModel_.lowerPositionLimit[positionIndex];
                    positionLimitMax_[positionIndex] =
                        pinocchioModel_.upperPositionLimit[positionIndex];
                }
            }
            else
            {
                for (std::size_t i = 0; i < rigidJointPositionIndices_.size(); ++i)
                {
                    Eigen::Index positionIndex = rigidJointPositionIndices_[i];
                    positionLimitMin_[positionIndex] = modelOptions_->joints.positionLimitMin[i];
                    positionLimitMax_[positionIndex] = modelOptions_->joints.positionLimitMax[i];
                }
            }
        }

        /* Overwrite the position bounds for some specific joint type, mainly due to quaternion
           normalization and cos/sin representation. */
        Eigen::Index idx_q, nq;
        for (const auto & joint : pinocchioModel_.joints)
        {
            switch (getJointType(joint))
            {
            case JointModelType::ROTARY_UNBOUNDED:
            case JointModelType::SPHERICAL:
                idx_q = joint.idx_q();
                nq = joint.nq();
                break;
            case JointModelType::FREE:
                idx_q = joint.idx_q() + 3;
                nq = 4;
                break;
            case JointModelType::UNSUPPORTED:
            case JointModelType::LINEAR:
            case JointModelType::ROTARY:
            case JointModelType::PLANAR:
            case JointModelType::TRANSLATION:
            default:
                continue;
            }
            positionLimitMin_.segment(idx_q, nq).setConstant(-1.0 - EPS);
            positionLimitMax_.segment(idx_q, nq).setConstant(+1.0 + EPS);
        }

        // Get the joint velocity limits from the URDF or the user options
        velocityLimit_.setConstant(pinocchioModel_.nv, +INF);
        if (modelOptions_->joints.enableVelocityLimit)
        {
            if (modelOptions_->joints.velocityLimitFromUrdf)
            {
                for (Eigen::Index & velocityIndex : rigidJointVelocityIndices_)
                {
                    velocityLimit_[velocityIndex] = pinocchioModel_.velocityLimit[velocityIndex];
                }
            }
            else
            {
                for (std::size_t i = 0; i < rigidJointVelocityIndices_.size(); ++i)
                {
                    Eigen::Index velocityIndex = rigidJointVelocityIndices_[i];
                    velocityLimit_[velocityIndex] = modelOptions_->joints.velocityLimit[i];
                }
            }
        }

        refreshGeometryProxies();
        refreshContactProxies();
        refreshConstraintProxies();
    }

    void Model::refreshGeometryProxies()
    {
        if (!isInitialized_)
        {
            THROW_ERROR(bad_control_flow, "Model not initialized.");
        }

        // Restore collision and visual models
        collisionModel_ = collisionModelOrig_;
        visualModel_ = visualModelOrig_;

        // Update joint/frame fix for every geometry objects
        if (modelOptions_->dynamics.enableFlexibleModel &&
            !modelOptions_->dynamics.flexibilityConfig.empty())
        {
            for (pinocchio::GeometryModel * model : std::array{&collisionModel_, &visualModel_})
            {
                for (pinocchio::GeometryObject & geom : model->geometryObjects)
                {
                    // Only the frame name remains unchanged no matter what
                    const pinocchio::Frame & frameOrig =
                        pinocchioModelOrig_.frames[geom.parentFrame];
                    const std::string parentJointName =
                        pinocchioModelOrig_.names[frameOrig.parent];
                    pinocchio::FrameType frameType = static_cast<pinocchio::FrameType>(
                        pinocchio::FIXED_JOINT | pinocchio::BODY);
                    const pinocchio::FrameIndex frameIndex =
                        getFrameIndex(pinocchioModel_, frameOrig.name, frameType);
                    const pinocchio::Frame & frame = pinocchioModel_.frames[frameIndex];
                    const pinocchio::JointIndex newParentModelIndex = frame.parent;
                    const pinocchio::JointIndex oldParentModelIndex =
                        pinocchioModel_.getJointId(parentJointName);

                    geom.parentFrame = frameIndex;
                    geom.parentJoint = newParentModelIndex;

                    /* Compute the relative displacement between the new and old joint
                       placement wrt their common parent joint. */
                    pinocchio::SE3 geomPlacementRef = pinocchio::SE3::Identity();
                    for (pinocchio::JointIndex i = newParentModelIndex; i > oldParentModelIndex;
                         i = pinocchioModel_.parents[i])
                    {
                        geomPlacementRef = pinocchioModel_.jointPlacements[i] * geomPlacementRef;
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
            pinocchioModel_, pinocchioData_, collisionModel_, collisionData_);
        visualData_ = pinocchio::GeometryData(visualModel_);
        pinocchio::updateGeometryPlacements(
            pinocchioModel_, pinocchioData_, visualModel_, visualData_);

        // Set the max number of contact points per collision pairs
        for (hpp::fcl::CollisionRequest & collisionRequest : collisionData_.collisionRequests)
        {
            collisionRequest.num_max_contacts = modelOptions_->collisions.contactPointsPerBodyMax;
        }

        // Extract the indices of the collision pairs associated with each body
        collisionPairIndices_.clear();
        for (const std::string & name : collisionBodyNames_)
        {
            std::vector<pinocchio::PairIndex> collisionPairIndices;
            for (std::size_t i = 0; i < collisionModel_.collisionPairs.size(); ++i)
            {
                const pinocchio::CollisionPair & pair = collisionModel_.collisionPairs[i];
                const pinocchio::GeometryObject & geom =
                    collisionModel_.geometryObjects[pair.first];
                if (pinocchioModel_.frames[geom.parentFrame].name == name)
                {
                    collisionPairIndices.push_back(i);
                }
            }
            collisionPairIndices_.push_back(std::move(collisionPairIndices));
        }

        // Extract the contact frames indices in the model
        collisionBodyIndices_ = getFrameIndices(pinocchioModel_, collisionBodyNames_);
    }

    void Model::refreshContactProxies()
    {
        if (!isInitialized_)
        {
            THROW_ERROR(bad_control_flow, "Model not initialized.");
        }

        // Reset the contact force internal buffer
        contactForces_ = ForceVector(contactFrameNames_.size(), pinocchio::Force::Zero());

        // Extract the contact frames indices in the model
        contactFrameIndices_ = getFrameIndices(pinocchioModel_, contactFrameNames_);
    }

    void Model::refreshConstraintProxies()
    {
        // Initialize backup joint space acceleration
        jointSpatialAccelerations_ =
            MotionVector(pinocchioData_.a.size(), pinocchio::Motion::Zero());

        constraints_.foreach(
            [&](const std::shared_ptr<AbstractConstraintBase> & constraint,
                ConstraintNodeType /* node */)
            {
                // Reset constraint using neutral configuration and zero velocity
                constraint->reset(pinocchio::neutral(pinocchioModel_), Eigen::VectorXd::Zero(nv_));

                // Call constraint on neutral position and zero velocity.
                auto J = constraint->getJacobian();

                // Check dimensions consistency
                if (J.cols() != pinocchioModel_.nv)
                {
                    THROW_ERROR(std::logic_error,
                                "Constraint has inconsistent jacobian and drift (size mismatch).");
                }
            });
    }

    void Model::setOptions(GenericConfig modelOptions)
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
                if (rigidJointPositionIndices_.size() !=
                    static_cast<uint32_t>(jointsPositionLimitMin.size()))
                {
                    THROW_ERROR(std::invalid_argument,
                                "Wrong vector size for 'positionLimitMin'.");
                }
                Eigen::VectorXd & jointsPositionLimitMax =
                    boost::get<Eigen::VectorXd>(jointOptionsHolder.at("positionLimitMax"));
                if (rigidJointPositionIndices_.size() !=
                    static_cast<uint32_t>(jointsPositionLimitMax.size()))
                {
                    THROW_ERROR(std::invalid_argument,
                                "Wrong vector size for 'positionLimitMax'.");
                }
                if (rigidJointPositionIndices_.size() ==
                    static_cast<uint32_t>(modelOptions_->joints.positionLimitMin.size()))
                {
                    auto jointsPositionLimitMinDiff =
                        jointsPositionLimitMin - modelOptions_->joints.positionLimitMin;
                    internalBuffersMustBeUpdated |=
                        (jointsPositionLimitMinDiff.array().abs() >= EPS).all();
                    auto jointsPositionLimitMaxDiff =
                        jointsPositionLimitMax - modelOptions_->joints.positionLimitMax;
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
                if (rigidJointVelocityIndices_.size() !=
                    static_cast<uint32_t>(jointsVelocityLimit.size()))
                {
                    THROW_ERROR(std::invalid_argument, "Wrong vector size for 'velocityLimit'.");
                }
                if (rigidJointVelocityIndices_.size() ==
                    static_cast<uint32_t>(modelOptions_->joints.velocityLimit.size()))
                {
                    auto jointsVelocityLimitDiff =
                        jointsVelocityLimit - modelOptions_->joints.velocityLimit;
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
                THROW_ERROR(
                    std::invalid_argument,
                    "All joint or frame names in flexibility configuration must be unique.");
            }
            if (std::find(flexibilityNames.begin(), flexibilityNames.end(), "universe") !=
                flexibilityNames.end())
            {
                THROW_ERROR(std::invalid_argument,
                            "No one can make the universe itself flexible.");
            }
            for (const FlexibleJointData & flexibleJoint : flexibilityConfig)
            {
                if ((flexibleJoint.stiffness.array() < 0.0).any() ||
                    (flexibleJoint.damping.array() < 0.0).any() ||
                    (flexibleJoint.inertia.array() < 0.0).any())
                {
                    THROW_ERROR(std::invalid_argument,
                                "All stiffness, damping and inertia parameters of flexible "
                                "joints must be positive.");
                }
            }

            // Check if the position or velocity limits have changed, and refresh proxies if so
            bool enablePositionLimit =
                boost::get<bool>(jointOptionsHolder.at("enablePositionLimit"));
            bool enableVelocityLimit =
                boost::get<bool>(jointOptionsHolder.at("enableVelocityLimit"));
            if (enablePositionLimit != modelOptions_->joints.enablePositionLimit)
            {
                internalBuffersMustBeUpdated = true;
            }
            else if (enablePositionLimit &&
                     (positionLimitFromUrdf != modelOptions_->joints.positionLimitFromUrdf))
            {
                internalBuffersMustBeUpdated = true;
            }
            else if (enableVelocityLimit != modelOptions_->joints.enableVelocityLimit)
            {
                internalBuffersMustBeUpdated = true;
            }
            else if (enableVelocityLimit &&
                     (velocityLimitFromUrdf != modelOptions_->joints.velocityLimitFromUrdf))
            {
                internalBuffersMustBeUpdated = true;
            }

            // Check if the flexible model and its proxies must be regenerated
            bool enableFlexibleModel =
                boost::get<bool>(dynOptionsHolder.at("enableFlexibleModel"));
            if (modelOptions_ &&
                (flexibilityConfig.size() != modelOptions_->dynamics.flexibilityConfig.size() ||
                 !std::equal(flexibilityConfig.begin(),
                             flexibilityConfig.end(),
                             modelOptions_->dynamics.flexibilityConfig.begin()) ||
                 enableFlexibleModel != modelOptions_->dynamics.enableFlexibleModel))
            {
                areModelsInvalid = true;
            }
        }

        // Check that the collisions options are valid
        GenericConfig & collisionOptionsHolder =
            boost::get<GenericConfig>(modelOptions.at("collisions"));
        uint32_t contactPointsPerBodyMax =
            boost::get<uint32_t>(collisionOptionsHolder.at("contactPointsPerBodyMax"));
        if (contactPointsPerBodyMax < 1)
        {
            THROW_ERROR(std::invalid_argument,
                        "Number of contact points by collision pair "
                        "'contactPointsPerBodyMax' must be strictly larger than 0.");
        }
        if (modelOptions_ &&
            contactPointsPerBodyMax != modelOptions_->collisions.contactPointsPerBodyMax)
        {
            isCollisionDataInvalid = true;
        }

        // Check that the model randomization parameters are valid
        GenericConfig & dynOptionsHolder = boost::get<GenericConfig>(modelOptions.at("dynamics"));
        for (auto && field : std::array{"inertiaBodiesBiasStd",
                                        "massBodiesBiasStd",
                                        "centerOfMassPositionBodiesBiasStd",
                                        "relativePositionBodiesBiasStd"})
        {
            const double value = boost::get<double>(dynOptionsHolder.at(field));
            if (0.9 < value || value < 0.0)
            {
                THROW_ERROR(std::invalid_argument,
                            "'",
                            field,
                            "' must be positive, and lower than 0.9 to avoid physics issues.");
            }
        }

        // Update the internal options
        modelOptionsGeneric_ = modelOptions;

        // Create a fast struct accessor
        modelOptions_ = std::make_unique<const ModelOptions>(modelOptionsGeneric_);

        if (areModelsInvalid)
        {
            // Trigger models regeneration
            reset(std::random_device{});
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
    }

    GenericConfig Model::getOptions() const noexcept
    {
        return modelOptionsGeneric_;
    }

    bool Model::getIsInitialized() const
    {
        return isInitialized_;
    }

    const std::string & Model::getName() const
    {
        return pinocchioModelOrig_.name;
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

    void Model::getFlexiblePositionFromRigid(const Eigen::VectorXd & qRigid,
                                             Eigen::VectorXd & qFlex) const
    {
        // Define some proxies
        int nqRigid = pinocchioModelOrig_.nq;

        // Check the size of the input state
        if (qRigid.size() != nqRigid)
        {
            THROW_ERROR(std::invalid_argument,
                        "Size of qRigid inconsistent with theoretical model.");
        }

        // Initialize the flexible state
        qFlex = pinocchio::neutral(pncModelFlexibleOrig_);

        // Compute the flexible state based on the rigid state
        int idxRigid = 0;
        int idxFlex = 0;
        for (; idxRigid < pinocchioModelOrig_.njoints; ++idxFlex)
        {
            const std::string & jointRigidName = pinocchioModelOrig_.names[idxRigid];
            const std::string & jointFlexName = pncModelFlexibleOrig_.names[idxFlex];
            if (jointRigidName == jointFlexName)
            {
                const auto & jointRigid = pinocchioModelOrig_.joints[idxRigid];
                const auto & jointFlex = pncModelFlexibleOrig_.joints[idxFlex];
                if (jointRigid.idx_q() >= 0)
                {
                    qFlex.segment(jointFlex.idx_q(), jointFlex.nq()) =
                        qRigid.segment(jointRigid.idx_q(), jointRigid.nq());
                }
                ++idxRigid;
            }
        }
    }


    void Model::getFlexibleVelocityFromRigid(const Eigen::VectorXd & vRigid,
                                             Eigen::VectorXd & vFlex) const
    {
        // Define some proxies
        uint32_t nvRigid = pinocchioModelOrig_.nv;
        uint32_t nvFlex = pncModelFlexibleOrig_.nv;

        // Check the size of the input state
        if (vRigid.size() != nvRigid)
        {
            THROW_ERROR(std::invalid_argument,
                        "Size of vRigid inconsistent with theoretical model.");
        }

        // Initialize the flexible state
        vFlex.setZero(nvFlex);

        // Compute the flexible state based on the rigid state
        int32_t idxRigid = 0;
        int32_t idxFlex = 0;
        for (; idxRigid < pinocchioModelOrig_.njoints; ++idxFlex)
        {
            const std::string & jointRigidName = pinocchioModelOrig_.names[idxRigid];
            const std::string & jointFlexName = pncModelFlexibleOrig_.names[idxFlex];
            if (jointRigidName == jointFlexName)
            {
                const auto & jointRigid = pinocchioModelOrig_.joints[idxRigid];
                const auto & jointFlex = pncModelFlexibleOrig_.joints[idxFlex];
                if (jointRigid.idx_q() >= 0)
                {
                    vFlex.segment(jointFlex.idx_v(), jointFlex.nv()) =
                        vRigid.segment(jointRigid.idx_v(), jointRigid.nv());
                }
                ++idxRigid;
            }
        }
    }

    void Model::getRigidPositionFromFlexible(const Eigen::VectorXd & qFlex,
                                             Eigen::VectorXd & qRigid) const
    {
        // Define some proxies
        uint32_t nqFlex = pncModelFlexibleOrig_.nq;

        // Check the size of the input state
        if (qFlex.size() != nqFlex)
        {
            THROW_ERROR(std::invalid_argument, "Size of qFlex inconsistent with flexible model.");
        }

        // Initialize the rigid state
        qRigid = pinocchio::neutral(pinocchioModelOrig_);

        // Compute the rigid state based on the flexible state
        int32_t idxRigid = 0;
        int32_t idxFlex = 0;
        for (; idxRigid < pinocchioModelOrig_.njoints; ++idxFlex)
        {
            const std::string & jointRigidName = pinocchioModelOrig_.names[idxRigid];
            const std::string & jointFlexName = pncModelFlexibleOrig_.names[idxFlex];
            if (jointRigidName == jointFlexName)
            {
                const auto & jointRigid = pinocchioModelOrig_.joints[idxRigid];
                const auto & jointFlex = pncModelFlexibleOrig_.joints[idxFlex];
                if (jointRigid.idx_q() >= 0)
                {
                    qRigid.segment(jointRigid.idx_q(), jointRigid.nq()) =
                        qFlex.segment(jointFlex.idx_q(), jointFlex.nq());
                }
                ++idxRigid;
            }
        }
    }

    void Model::getRigidVelocityFromFlexible(const Eigen::VectorXd & vFlex,
                                             Eigen::VectorXd & vRigid) const
    {
        // Define some proxies
        uint32_t nvRigid = pinocchioModelOrig_.nv;
        uint32_t nvFlex = pncModelFlexibleOrig_.nv;

        // Check the size of the input state
        if (vFlex.size() != nvFlex)
        {
            THROW_ERROR(std::invalid_argument, "Size of vFlex inconsistent with flexible model.");
        }

        // Initialize the rigid state
        vRigid.setZero(nvRigid);

        // Compute the rigid state based on the flexible state
        int32_t idxRigid = 0;
        int32_t idxFlex = 0;
        for (; idxRigid < pinocchioModelOrig_.njoints; ++idxFlex)
        {
            const std::string & jointRigidName = pinocchioModelOrig_.names[idxRigid];
            const std::string & jointFlexName = pncModelFlexibleOrig_.names[idxFlex];
            if (jointRigidName == jointFlexName)
            {
                const auto & jointRigid = pinocchioModelOrig_.joints[idxRigid];
                const auto & jointFlex = pncModelFlexibleOrig_.joints[idxFlex];
                if (jointRigid.idx_q() >= 0)
                {
                    vRigid.segment(jointRigid.idx_v(), jointRigid.nv()) =
                        vFlex.segment(jointFlex.idx_v(), jointFlex.nv());
                }
                ++idxRigid;
            }
        }
    }

    const std::vector<std::string> & Model::getCollisionBodyNames() const
    {
        return collisionBodyNames_;
    }

    const std::vector<std::string> & Model::getContactFrameNames() const
    {
        return contactFrameNames_;
    }

    const std::vector<pinocchio::FrameIndex> & Model::getCollisionBodyIndices() const
    {
        return collisionBodyIndices_;
    }

    const std::vector<std::vector<pinocchio::PairIndex>> & Model::getCollisionPairIndices() const
    {
        return collisionPairIndices_;
    }

    const std::vector<pinocchio::FrameIndex> & Model::getContactFrameIndices() const
    {
        return contactFrameIndices_;
    }

    const std::vector<std::string> & Model::getLogPositionFieldnames() const
    {
        return logPositionFieldnames_;
    }

    const Eigen::VectorXd & Model::getPositionLimitMin() const
    {
        return positionLimitMin_;
    }

    const Eigen::VectorXd & Model::getPositionLimitMax() const
    {
        return positionLimitMax_;
    }

    const std::vector<std::string> & Model::getLogVelocityFieldnames() const
    {
        return logVelocityFieldnames_;
    }

    const Eigen::VectorXd & Model::getVelocityLimit() const
    {
        return velocityLimit_;
    }

    const std::vector<std::string> & Model::getLogAccelerationFieldnames() const
    {
        return logAccelerationFieldnames_;
    }

    const std::vector<std::string> & Model::getLogForceExternalFieldnames() const
    {
        return logForceExternalFieldnames_;
    }

    const std::vector<std::string> & Model::getRigidJointNames() const
    {
        return rigidJointNames_;
    }

    const std::vector<pinocchio::JointIndex> & Model::getRigidJointIndices() const
    {
        return rigidJointIndices_;
    }

    const std::vector<Eigen::Index> & Model::getRigidJointPositionIndices() const
    {
        return rigidJointPositionIndices_;
    }

    const std::vector<Eigen::Index> & Model::getRigidJointVelocityIndices() const
    {
        return rigidJointVelocityIndices_;
    }

    const std::vector<std::string> & Model::getFlexibleJointNames() const
    {
        static const std::vector<std::string> flexibleJointsNamesEmpty{};
        if (modelOptions_->dynamics.enableFlexibleModel)
        {
            return flexibleJointNames_;
        }
        else
        {
            return flexibleJointsNamesEmpty;
        }
    }

    const std::vector<pinocchio::JointIndex> & Model::getFlexibleJointIndices() const
    {
        static const std::vector<pinocchio::JointIndex> flexibleJointsModelIndexEmpty{};
        if (modelOptions_->dynamics.enableFlexibleModel)
        {
            return flexibleJointIndices_;
        }
        else
        {
            return flexibleJointsModelIndexEmpty;
        }
    }

    /// \brief Returns true if at least one constraint is active on the robot.
    bool Model::hasConstraints() const
    {
        bool hasConstraintsEnabled = false;
        const_cast<ConstraintTree &>(constraints_)
            .foreach(
                [&hasConstraintsEnabled](
                    const std::shared_ptr<AbstractConstraintBase> & constraint,
                    ConstraintNodeType /* node */)
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
