#include <fstream>
#include <exception>
#include <unordered_set>

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

#include <boost/serialization/nvp.hpp>

#include "urdf_parser/urdf_parser.h"

#include "jiminy/core/robot/pinocchio_overload_algorithms.h"
#include "jiminy/core/constraints/abstract_constraint.h"
#include "jiminy/core/constraints/joint_constraint.h"
#include "jiminy/core/constraints/sphere_constraint.h"
#include "jiminy/core/constraints/frame_constraint.h"
#include "jiminy/core/hardware/basic_sensors.h"
#include "jiminy/core/io/serialization.h"
#include "jiminy/core/utilities/helpers.h"
#include "jiminy/core/utilities/pinocchio.h"

#include "jiminy/core/robot/model.h"


namespace jiminy
{
    // ************************************** Constraints ************************************** //

    constexpr std::string_view getConstraintNodeName(ConstraintRegistryType type)
    {
        switch (type)
        {
        case ConstraintRegistryType::BOUNDS_JOINTS:
            return "BoundJoints";
        case ConstraintRegistryType::CONTACT_FRAMES:
            return "ContactFrames";
        case ConstraintRegistryType::COLLISION_BODIES:
            return "CollisionBodies";
        case ConstraintRegistryType::USER:
        default:
            return "User";
        }
    }

    template<typename T>
    static auto findImpl(T && constraints, const std::string & key, ConstraintRegistryType type)
    {
        // Determine return types based on argument constness
        constexpr bool isConst = std::is_const_v<std::remove_reference_t<T>>;
        using constraintMapT = std::conditional_t<isConst, const ConstraintMap, ConstraintMap>;
        using constraintIteratorT =
            std::conditional_t<isConst, ConstraintMap::const_iterator, ConstraintMap::iterator>;

        // Pointers are NOT initialized to nullptr by default
        constraintMapT * constraintMapPtr{nullptr};
        constraintIteratorT constraintIt{};
        if (type == ConstraintRegistryType::COLLISION_BODIES)
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
            switch (type)
            {
            case ConstraintRegistryType::BOUNDS_JOINTS:
                constraintMapPtr = &constraints.boundJoints;
                break;
            case ConstraintRegistryType::CONTACT_FRAMES:
                constraintMapPtr = &constraints.contactFrames;
                break;
            case ConstraintRegistryType::USER:
            case ConstraintRegistryType::COLLISION_BODIES:
            default:
                constraintMapPtr = &constraints.user;
            }
            constraintIt = std::find_if(constraintMapPtr->begin(),
                                        constraintMapPtr->end(),
                                        [&key](const auto & constraintPair)
                                        { return constraintPair.first == key; });
        }

        return std::make_pair(constraintMapPtr, constraintIt);
    }

    std::pair<ConstraintMap *, ConstraintMap::iterator> ConstraintTree::find(
        const std::string & key, ConstraintRegistryType type)
    {
        return findImpl(*this, key, type);
    }

    std::pair<const ConstraintMap *, ConstraintMap::const_iterator> ConstraintTree::find(
        const std::string & key, ConstraintRegistryType type) const
    {
        return findImpl(*this, key, type);
    }

    bool ConstraintTree::exist(const std::string & key, ConstraintRegistryType type) const
    {
        const auto [constraintMapPtr, constraintIt] = find(key, type);
        return (constraintMapPtr && constraintIt != constraintMapPtr->cend());
    }

    std::shared_ptr<AbstractConstraintBase> ConstraintTree::get(const std::string & key,
                                                                ConstraintRegistryType type) const
    {
        auto [constraintMapPtr, constraintIt] = find(key, type);
        if (constraintMapPtr && constraintIt == constraintMapPtr->end())
        {
            JIMINY_THROW(std::invalid_argument,
                         "No constraint named '",
                         key,
                         "' exists in registry '",
                         getConstraintNodeName(type),
                         "'.");
        }
        return constraintIt->second;
    }

    void ConstraintTree::insert(const ConstraintMap & constraintMap, ConstraintRegistryType type)
    {
        // Make sure that all constraints are properly defined and their names do not already exist
        std::unordered_set<std::string> keys;
        for (const auto & [constraintName, constraintPtr] : constraintMap)
        {
            if (!constraintPtr)
            {
                JIMINY_THROW(std::invalid_argument,
                             "Constraint named '",
                             constraintName,
                             "' is undefined.");
            }

            if (exist(constraintName, type))
            {
                JIMINY_THROW(std::invalid_argument,
                             "A constraint named '",
                             constraintName,
                             "' already exists in registry '",
                             getConstraintNodeName(type),
                             "'.");
            }

            keys.insert(constraintName);
        }

        // Make sure that all constraints have a unique name
        if (keys.size() != constraintMap.size())
        {
            JIMINY_THROW(std::invalid_argument,
                         "Impossible to insert constraints with duplicate names.");
        }

        switch (type)
        {
        case ConstraintRegistryType::BOUNDS_JOINTS:
            boundJoints.insert(boundJoints.end(), constraintMap.begin(), constraintMap.end());
            break;
        case ConstraintRegistryType::CONTACT_FRAMES:
            contactFrames.insert(contactFrames.end(), constraintMap.begin(), constraintMap.end());
            break;
        case ConstraintRegistryType::COLLISION_BODIES:
            collisionBodies.push_back(constraintMap);
            break;
        case ConstraintRegistryType::USER:
        default:
            user.insert(user.end(), constraintMap.begin(), constraintMap.end());
        }
    }

    ConstraintMap::iterator ConstraintTree::erase(const std::string & key,
                                                  ConstraintRegistryType type)
    {
        auto [constraintMapPtr, constraintIt] = find(key, type);
        if (constraintMapPtr && constraintIt == constraintMapPtr->end())
        {
            JIMINY_THROW(std::invalid_argument,
                         "No constraint named '",
                         key,
                         "' exists in registry '",
                         getConstraintNodeName(type),
                         "'.");
        }
        return constraintMapPtr->erase(constraintIt);
    }

    void ConstraintTree::clear() noexcept
    {
        boundJoints.clear();
        contactFrames.clear();
        collisionBodies.clear();
        user.clear();
    }

    // ***************************************** Model ***************************************** //

    Model::Model() noexcept
    {
        // Initialize options
        modelOptionsGeneric_ = getDefaultModelOptions();
        setOptions(getOptions());
    }

    Model::Model(const Model & other) :
    enable_shared_from_this()
    {
        *this = other;
    }

    Model & Model::operator=(const Model & other)
    {
        // Serialize the user-specified robot
        std::stringstream sstream;
        {
            stateful_binary_oarchive oa(sstream);
            const Model * const otherPtr = &other;
            oa << boost::serialization::make_nvp("px", otherPtr);
        }

        // De-serialize the user-specified robot then move assign to this robot
        {
            stateful_binary_iarchive ia(sstream);
            Model * robotPtr;
            ia >> boost::serialization::make_nvp("px", robotPtr);
            *this = std::move(*robotPtr);
        }

        return *this;
    }

    void initializePinocchioData(const pinocchio::Model & model, pinocchio::Data & data)
    {
        // Re-allocate Pinocchio Data from scratch
        data = pinocchio::Data(model);

        /* Initialize Pinocchio data internal state.
           This includes "basic" attributes such as the mass of each body. */
        const Eigen::VectorXd qNeutral = pinocchio::neutral(model);
        pinocchio::forwardKinematics(model, data, qNeutral, Eigen::VectorXd::Zero(model.nv));
        pinocchio::updateFramePlacements(model, data);
        pinocchio::centerOfMass(model, data, qNeutral);
    }

    void Model::initialize(const pinocchio::Model & pinocchioModel,
                           const std::optional<pinocchio::GeometryModel> & collisionModel,
                           const std::optional<pinocchio::GeometryModel> & visualModel)
    {
        // Make sure that the model is actually managed by a shared ptr
        try
        {
            (void)shared_from_this();
        }
        catch (std::bad_weak_ptr &)
        {
            JIMINY_THROW(bad_control_flow, "Model must be managed by a std::shared_ptr.");
        }

        // Make sure that the pinocchio model is valid
        if (pinocchioModel.nq == 0)
        {
            JIMINY_THROW(std::invalid_argument, "Pinocchio model must not be empty.");
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
        pinocchioModelTh_ = pinocchioModel;
        collisionModelTh_ = collisionModel.value_or(pinocchio::GeometryModel());
        visualModelTh_ = visualModel.value_or(pinocchio::GeometryModel());

        // Add ground geometry object to collision model is not already available
        if (!collisionModelTh_.existGeometryName("ground"))
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
            collisionModelTh_.addGeometryObject(groundPlane, pinocchioModelTh_);
        }

        /* Re-allocate data from scratch for the theoretical model.
           Note that the theoretical model is not used anywhere for simulation but is exposed
           nonetheless to make life easier for end-users willing to perform computations on it
           rather than the actual simulation model, which is supposed to be unknown. */
        initializePinocchioData(pinocchioModelTh_, pinocchioDataTh_);

        /* Get the list of joint names of the theoretical model, the 'universe' and
           'root_joint' excluded if any, since they are not mechanical joints. */
        mechanicalJointNames_ = pinocchioModelTh_.names;
        mechanicalJointNames_.erase(mechanicalJointNames_.begin());  // remove 'universe'
        if (hasFreeflyer_)
        {
            mechanicalJointNames_.erase(mechanicalJointNames_.begin());  // remove 'root_joint'
        }

        // Assuming the model is fully initialized at this point
        isInitialized_ = true;
        try
        {
            // Create the "extended" model
            generateModelExtended(std::random_device{});

            /* Add joint constraints.
               It will be used later to enforce bounds limits if requested. */
            ConstraintMap jointConstraintsMap;
            jointConstraintsMap.reserve(mechanicalJointNames_.size());
            for (const std::string & jointName : mechanicalJointNames_)
            {
                jointConstraintsMap.emplace_back(jointName,
                                                 std::make_shared<JointConstraint>(jointName));
            }
            addConstraints(jointConstraintsMap, ConstraintRegistryType::BOUNDS_JOINTS);
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

        // Make sure that the model is initialized
        if (!isInitialized_)
        {
            JIMINY_THROW(bad_control_flow, "Model not initialized.");
        }

        /* Re-generate the unbiased extended model and update bias added to the dynamics
            properties of the model.
            Note that re-generating the unbiased extended model is necessary since the
            theoretical model may have been manually modified by the user. */
        generateModelExtended(g);
    }

    void Model::addFrame(const std::string & frameName,
                         const std::string & parentBodyName,
                         const pinocchio::SE3 & framePlacement,
                         const pinocchio::FrameType & frameType)
    {
        /* The frame must be added directly to the parent joint because it is not possible to add a
           frame to another frame. This means that the relative transform of the frame wrt the
           parent joint must be computed. */

        // Make sure that no simulation is already running
        if (getIsLocked())
        {
            JIMINY_THROW(bad_control_flow,
                         "Model already locked, probably because a simulation is running. "
                         "Please stop it before adding frames.");
        }

        // Check that no frame with the same name already exists
        if (pinocchioModelTh_.existFrame(frameName))
        {
            JIMINY_THROW(std::invalid_argument, "Frame with same name already exists.");
        }

        // Add frame to theoretical model
        {
            const pinocchio::FrameIndex parentFrameIndex =
                getFrameIndex(pinocchioModelTh_, parentBodyName);
            pinocchio::JointIndex parentJointIndex =
                pinocchioModelTh_.frames[parentFrameIndex].parent;
            const pinocchio::SE3 & parentFramePlacement =
                pinocchioModelTh_.frames[parentFrameIndex].placement;
            const pinocchio::SE3 jointFramePlacement = parentFramePlacement.act(framePlacement);
            const pinocchio::Frame frame(
                frameName, parentJointIndex, parentFrameIndex, jointFramePlacement, frameType);
            pinocchioModelTh_.addFrame(frame);

            // TODO: Do NOT re-allocate from scratch but update existing data for efficiency
            initializePinocchioData(pinocchioModelTh_, pinocchioDataTh_);
        }

        /* Add frame to extended model.
           Note that, appending a frame to the model does not invalid proxies, and therefore it is
           unecessary to call 'reset'. */
        {
            const pinocchio::FrameIndex parentFrameIndex =
                getFrameIndex(pinocchioModel_, parentBodyName);
            pinocchio::JointIndex parentJointIndex =
                pinocchioModel_.frames[parentFrameIndex].parent;
            const pinocchio::SE3 & parentFramePlacement =
                pinocchioModel_.frames[parentFrameIndex].placement;
            const pinocchio::SE3 jointFramePlacement = parentFramePlacement.act(framePlacement);
            const pinocchio::Frame frame(frameName,
                                         parentJointIndex,
                                         parentFrameIndex,
                                         jointFramePlacement,
                                         pinocchio::FrameType::OP_FRAME);
            pinocchioModel_.addFrame(frame);

            // TODO: Do NOT re-allocate from scratch but update existing data for efficiency
            initializePinocchioData(pinocchioModel_, pinocchioData_);
        }
    }

    void Model::removeFrames(const std::vector<std::string> & frameNames,
                             const std::vector<pinocchio::FrameType> & filter)
    {
        // Make sure that no simulation is already running
        if (getIsLocked())
        {
            JIMINY_THROW(bad_control_flow,
                         "Model already locked, probably because a simulation is running. "
                         "Please stop it before removing frames.");
        }

        /* Check that the frame can be safely removed from the theoretical model.
           If so, then it holds true for the extended model. */
        if (!filter.empty())
        {
            for (const std::string & frameName : frameNames)
            {
                const pinocchio::FrameIndex frameIndex =
                    getFrameIndex(pinocchioModelTh_, frameName);
                const pinocchio::FrameType frameType = pinocchioModelTh_.frames[frameIndex].type;
                if (std::find(filter.begin(), filter.end(), frameType) != filter.end())
                {
                    JIMINY_THROW(std::logic_error,
                                 "Not allowed to remove frame '",
                                 frameName,
                                 "' of type '",
                                 frameType,
                                 "'.");
                }
            }
        }

        for (const std::string & frameName : frameNames)
        {
            // Remove frame from the theoretical model
            {
                const pinocchio::FrameIndex frameIndex =
                    getFrameIndex(pinocchioModelTh_, frameName);
                pinocchioModelTh_.frames.erase(
                    std::next(pinocchioModelTh_.frames.begin(), frameIndex));
                pinocchioModelTh_.nframes--;
            }

            // Remove frame from the extended model
            {
                const pinocchio::FrameIndex frameIndex = getFrameIndex(pinocchioModel_, frameName);
                pinocchioModel_.frames.erase(
                    std::next(pinocchioModel_.frames.begin(), frameIndex));
                pinocchioModel_.nframes--;
            }
        }

        // TODO: Do NOT re-allocate from scratch but update existing data for efficiency
        initializePinocchioData(pinocchioModel_, pinocchioData_);
        initializePinocchioData(pinocchioModelTh_, pinocchioDataTh_);
    }

    void Model::addFrame(const std::string & frameName,
                         const std::string & parentBodyName,
                         const pinocchio::SE3 & framePlacement)
    {
        return addFrame(frameName, parentBodyName, framePlacement, pinocchio::FrameType::OP_FRAME);
    }

    void Model::removeFrames(const std::vector<std::string> & frameNames)
    {
        removeFrames(frameNames, {pinocchio::FrameType::OP_FRAME});
    }

    void Model::addCollisionBodies(const std::vector<std::string> & bodyNames, bool ignoreMeshes)
    {
        // Make sure that the model is initialized
        if (!isInitialized_)
        {
            JIMINY_THROW(bad_control_flow, "Model not initialized.");
        }

        // Make sure that no simulation is already running
        if (getIsLocked())
        {
            JIMINY_THROW(bad_control_flow,
                         "Model already locked, probably because a simulation is running. "
                         "Please stop it before adding collision bodies.");
        }

        // Returning early if nothing to do
        if (bodyNames.empty())
        {
            return;
        }

        // If successfully loaded, the ground should be available
        if (collisionModelTh_.ngeoms == 0)
        {
            JIMINY_THROW(std::runtime_error,
                         "Collision geometry not available. Some collision meshes were "
                         "probably not found.");
        }

        // Make sure that no body are duplicates
        if (checkDuplicates(bodyNames))
        {
            JIMINY_THROW(std::invalid_argument, "Duplicated bodies found.");
        }

        // Make sure there is no collision already associated with any of the bodies in the list
        if (checkIntersection(collisionBodyNames_, bodyNames))
        {
            JIMINY_THROW(std::invalid_argument,
                         "At least one of the bodies already associated with a collision.");
        }

        // Make sure that all the bodies exist
        for (const std::string & name : bodyNames)
        {
            if (!pinocchioModelTh_.existBodyName(name))
            {
                JIMINY_THROW(std::invalid_argument, "At least one of the bodies does not exist.");
            }
        }

        /* Create the collision pairs and add them to the geometry model of the robot,
           after checking first that at least one geometry is associated with each body. */
        for (bool isDryRun : std::array{true, false})
        {
            const pinocchio::GeomIndex & groundIndex = collisionModelTh_.getGeometryId("ground");
            for (const std::string & name : bodyNames)
            {
                bool hasGeometry = false;
                ConstraintMap collisionConstraintsMap;

                // Add a collision pair for all geometries having the body as parent
                for (std::size_t i = 0; i < collisionModelTh_.geometryObjects.size(); ++i)
                {
                    const pinocchio::GeometryObject & geom = collisionModelTh_.geometryObjects[i];
                    // geom.meshPath is the geometry type if it is not an actual mesh
                    const bool isGeomMesh = (geom.meshPath.find('/') != std::string::npos ||
                                             geom.meshPath.find('\\') != std::string::npos);
                    const std::string & frameName = pinocchioModel_.frames[geom.parentFrame].name;
                    if (!(ignoreMeshes && isGeomMesh) && frameName == name)
                    {
                        // Add constraint associated with contact frame only if it is a sphere
                        const hpp::fcl::CollisionGeometry & shape = *geom.geometry;
                        if (shape.getNodeType() == hpp::fcl::GEOM_SPHERE)
                        {
                            // A valid geometry has been found
                            hasGeometry = true;

                            // Early return if dry run
                            if (isDryRun)
                            {
                                break;
                            }

                            /* Create and add the collision pair with the ground.
                            Note that the ground always comes second for the normal to be
                            consistently compute wrt the ground instead of the body. */
                            const pinocchio::CollisionPair collisionPair(i, groundIndex);
                            collisionModelTh_.addCollisionPair(collisionPair);

                            /* Add dedicated frame.
                            Note that 'FIXED_JOINT' type is used instead of default 'OP_FRAME' to
                            avoid considering it as manually added to the model, and therefore
                            prevent its deletion by the user. */
                            const pinocchio::FrameType frameType =
                                pinocchio::FrameType::FIXED_JOINT;
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
                        else
                        {
                            JIMINY_WARNING("Geometry object has been ignored because its type is "
                                           "unsupported for now.");
                        }
                    }
                }

                // Make sure that at least one geometry is associated with each body
                if (!hasGeometry)
                {
                    JIMINY_THROW(std::invalid_argument,
                                 "Body '",
                                 name,
                                 "' not associated with any collision "
                                 "geometry of a supported type.");
                }

                // Add constraints map if not dry run
                if (!isDryRun)
                {
                    addConstraints(collisionConstraintsMap,
                                   ConstraintRegistryType::COLLISION_BODIES);
                }
            }
        }

        // Add the list of bodies to the set of collision bodies
        collisionBodyNames_.insert(collisionBodyNames_.end(), bodyNames.begin(), bodyNames.end());

        // Refresh proxies associated with the collisions only
        refreshGeometryProxies();
    }

    void Model::removeCollisionBodies(std::vector<std::string> bodyNames)
    {
        // Make sure that the model is initialized
        if (!isInitialized_)
        {
            JIMINY_THROW(bad_control_flow, "Model not initialized.");
        }

        // Make sure that no simulation is already running
        if (getIsLocked())
        {
            JIMINY_THROW(bad_control_flow,
                         "Model already locked, probably because a simulation is running. "
                         "Please stop it before removing collision bodies.");
        }

        // Make sure that no body are duplicates
        if (checkDuplicates(bodyNames))
        {
            JIMINY_THROW(std::invalid_argument, "Duplicated bodies found.");
        }

        // Make sure that every body in the list is associated with a collision
        if (!checkInclusion(collisionBodyNames_, bodyNames))
        {
            JIMINY_THROW(std::invalid_argument,
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
        const pinocchio::GeomIndex & groundIndex = collisionModelTh_.getGeometryId("ground");
        for (const std::string & name : bodyNames)
        {
            // Remove the collision pair for all the geometries having the body as parent
            for (std::size_t i = 0; i < collisionModelTh_.geometryObjects.size(); ++i)
            {
                const pinocchio::GeometryObject & geom = collisionModelTh_.geometryObjects[i];
                if (pinocchioModel_.frames[geom.parentFrame].name == name)
                {
                    // Remove the collision pair with the ground
                    const pinocchio::CollisionPair collisionPair(i, groundIndex);
                    collisionModelTh_.removeCollisionPair(collisionPair);

                    // Append collision geometry to the list of constraints to remove
                    if (constraints_.exist(geom.name, ConstraintRegistryType::COLLISION_BODIES))
                    {
                        collisionConstraintNames.emplace_back(geom.name);
                    }
                }
            }
        }

        // Remove the constraints and associated frames
        removeConstraints(collisionConstraintNames, ConstraintRegistryType::COLLISION_BODIES);
        removeFrames(collisionConstraintNames, {pinocchio::FrameType::FIXED_JOINT});

        // Refresh proxies associated with the collisions only
        refreshGeometryProxies();
    }

    void Model::addContactPoints(const std::vector<std::string> & frameNames)
    {
        // Make sure that the model is initialized
        if (!isInitialized_)
        {
            JIMINY_THROW(bad_control_flow, "Model not initialized.");
        }

        // Make sure that no simulation is already running
        if (getIsLocked())
        {
            JIMINY_THROW(bad_control_flow,
                         "Model already locked, probably because a simulation is running. "
                         "Please stop it before adding contact points.");
        }

        // Make sure that no frame are duplicates
        if (checkDuplicates(frameNames))
        {
            JIMINY_THROW(std::invalid_argument, "Duplicated frames found.");
        }

        // Make sure that there is no contact already associated with any of the frames in the list
        if (checkIntersection(contactFrameNames_, frameNames))
        {
            JIMINY_THROW(std::invalid_argument,
                         "At least one of the frames already associated with a contact.");
        }

        // Make sure that all the frames exist
        for (const std::string & name : frameNames)
        {
            if (!pinocchioModelTh_.existFrame(name))
            {
                JIMINY_THROW(std::invalid_argument, "At least one of the frames does not exist.");
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
        addConstraints(frameConstraintsMap, ConstraintRegistryType::CONTACT_FRAMES);

        // Refresh proxies associated with contacts and constraints
        refreshContactProxies();
    }

    void Model::removeContactPoints(const std::vector<std::string> & frameNames)
    {
        // Make sure that the model is initialized
        if (!isInitialized_)
        {
            JIMINY_THROW(bad_control_flow, "Model not initialized.");
        }

        // Make sure that no simulation is already running
        if (getIsLocked())
        {
            JIMINY_THROW(bad_control_flow,
                         "Model already locked, probably because a simulation is running. "
                         "Please stop it before removing contact points.");
        }

        // Make sure that no frame are duplicates
        if (checkDuplicates(frameNames))
        {
            JIMINY_THROW(std::invalid_argument, "Duplicated frames found.");
        }

        // Make sure that every frame in the list is associated with a contact
        if (!checkInclusion(contactFrameNames_, frameNames))
        {
            JIMINY_THROW(std::invalid_argument,
                         "At least one of the frames not associated with a contact.");
        }

        /* Remove the constraint associated with contact frame, then remove the list of frames from
           the set of contact frames. */
        if (!frameNames.empty())
        {
            removeConstraints(frameNames, ConstraintRegistryType::CONTACT_FRAMES);
            eraseVector(contactFrameNames_, frameNames);
        }
        else
        {
            removeConstraints(contactFrameNames_, ConstraintRegistryType::CONTACT_FRAMES);
            contactFrameNames_.clear();
        }

        // Refresh proxies associated with contacts and constraints
        refreshContactProxies();
    }

    void Model::addConstraints(const ConstraintMap & constraintMap, ConstraintRegistryType type)
    {
        // Make sure that no simulation is already running
        if (getIsLocked())
        {
            JIMINY_THROW(bad_control_flow,
                         "Model already locked, probably because a simulation is running. "
                         "Please stop it before adding constraints.");
        }

        // Add constraints to registry
        constraints_.insert(constraintMap, type);

        // Attach constraint
        for (auto & constraintPair : constraintMap)
        {
            constraintPair.second->attach(shared_from_this());
        }

        // Loop over all constraints
        Eigen::VectorXd qNeutral = pinocchio::neutral(pinocchioModel_);
        for (auto & [constraintName, constraint] : constraintMap)
        {
            // Reset constraint using neutral configuration and zero velocity
            constraint->reset(qNeutral, Eigen::VectorXd::Zero(nv_));

            // Call constraint on neutral position and zero velocity.
            auto J = constraint->getJacobian();

            // Check dimensions consistency
            if (J.cols() != pinocchioModel_.nv)
            {
                JIMINY_THROW(std::logic_error,
                             "Constraint has inconsistent jacobian and drift (size mismatch).");
            }

            // Append log telemetry constraint fieldnames
            for (uint32_t j = 0; j < constraint->getSize(); ++j)
            {
                logConstraintFieldnames_.push_back(
                    toString("Constraint", getConstraintNodeName(type), constraintName, j));
            }

            // Disable constraint by default if internal
            if (type != ConstraintRegistryType::USER)
            {
                constraint->disable();
            }
        }
    }

    void Model::addConstraint(const std::string & constraintName,
                              const std::shared_ptr<AbstractConstraintBase> & constraint,
                              ConstraintRegistryType type)
    {
        return addConstraints({{constraintName, constraint}}, type);
    }

    void Model::addConstraint(const std::string & constraintName,
                              const std::shared_ptr<AbstractConstraintBase> & constraint)
    {
        return addConstraint(constraintName, constraint, ConstraintRegistryType::USER);
    }

    void Model::removeConstraints(const std::vector<std::string> & constraintNames,
                                  ConstraintRegistryType type)
    {
        // Make sure that no simulation is already running
        if (getIsLocked())
        {
            JIMINY_THROW(bad_control_flow,
                         "Model already locked, probably because a simulation is running. "
                         "Please stop it before removing constraints.");
        }

        // Make sure all constraints exist
        for (const std::string & constraintName : constraintNames)
        {
            if (!constraints_.exist(constraintName, type))
            {
                JIMINY_THROW(std::invalid_argument,
                             "No constraint named '",
                             constraintName,
                             "' exists in registry '",
                             getConstraintNodeName(type),
                             "'.");
            }
        }

        // Remove every constraint sequentially
        for (const std::string & constraintName : constraintNames)
        {
            // Lookup constraint
            auto [constraintMapPtr, constraintIt] = constraints_.find(constraintName, type);

            // Detach the constraint
            constraintIt->second->detach();

            // Remove log telemetry constraint fieldnames
            const std::string logConstraintFieldnameFirst =
                toString("Constraint", getConstraintNodeName(type), constraintName, 0);
            for (auto logConstraintFieldnamesIt = logConstraintFieldnames_.begin();
                 logConstraintFieldnamesIt != logConstraintFieldnames_.end();
                 ++logConstraintFieldnamesIt)
            {
                if (*logConstraintFieldnamesIt == logConstraintFieldnameFirst)
                {
                    logConstraintFieldnames_.erase(logConstraintFieldnamesIt,
                                                   logConstraintFieldnamesIt +
                                                       constraintIt->second->getSize());
                    break;
                }
            }

            // Remove the constraint from the holder
            constraintMapPtr->erase(constraintIt);
        }
    }

    void Model::removeConstraint(const std::string & constraintName, ConstraintRegistryType type)
    {
        return removeConstraints({constraintName}, type);
    }

    void Model::removeConstraint(const std::string & constraintName)
    {
        return removeConstraint(constraintName, ConstraintRegistryType::USER);
    }

    const ConstraintTree & Model::getConstraints() const
    {
        return constraints_;
    }

    bool Model::hasConstraints() const
    {
        bool hasConstraintsEnabled = false;
        const_cast<ConstraintTree &>(constraints_)
            .foreach(
                [&hasConstraintsEnabled](
                    const std::shared_ptr<AbstractConstraintBase> & constraint,
                    ConstraintRegistryType /* type */)
                {
                    if (constraint->getIsEnabled())
                    {
                        hasConstraintsEnabled = true;
                    }
                });
        return hasConstraintsEnabled;
    }

    void Model::resetConstraints(const Eigen::VectorXd & q, const Eigen::VectorXd & v)
    {
        // Make sure that no simulation is already running
        if (getIsLocked())
        {
            JIMINY_THROW(bad_control_flow,
                         "Model already locked, probably because a simulation is running. "
                         "Please stop it before resetting constraints.");
        }

        constraints_.foreach([&q, &v](const std::shared_ptr<AbstractConstraintBase> & constraint,
                                      ConstraintRegistryType /* type */)
                             { constraint->reset(q, v); });

        constraints_.foreach(std::array{ConstraintRegistryType::BOUNDS_JOINTS,
                                        ConstraintRegistryType::CONTACT_FRAMES,
                                        ConstraintRegistryType::COLLISION_BODIES},
                             [](const std::shared_ptr<AbstractConstraintBase> & constraint,
                                ConstraintRegistryType /* type */) { constraint->disable(); });
    }

    void Model::generateModelExtended(const uniform_random_bit_generator_ref<uint32_t> & g)
    {
        // Make sure the model is initialized
        if (!isInitialized_)
        {
            JIMINY_THROW(bad_control_flow, "Model not initialized.");
        }

        // Initialize the extended model
        initializeExtendedModel();

        // Add flexibility joints to the extended model if requested
        if (modelOptions_->dynamics.enableFlexibility)
        {
            addFlexibilityJointsToExtendedModel();
        }

        // Add biases to the dynamics properties of the model
        addBiasedToExtendedModel(g);

        // Refresh internal proxies
        refreshProxies();
    }

    void Model::initializeExtendedModel()
    {
        // Clear some proxies
        backlashJointIndices_.clear();
        backlashJointNames_.clear();
        flexibilityJointIndices_.clear();
        flexibilityJointNames_.clear();

        // Simply copy the theoretical model by default
        pinocchioModel_ = pinocchioModelTh_;

        // Set velocity and effort limits to infinity systematically as they are non-physical
        pinocchioModel_.effortLimit.setConstant(INF);
        pinocchioModel_.velocityLimit.setConstant(INF);
    }

    void Model::addFlexibilityJointsToExtendedModel()
    {
        // Check that the frames exist
        for (const FlexibilityJointConfig & flexibilityJoint :
             modelOptions_->dynamics.flexibilityConfig)
        {
            const std::string & frameName = flexibilityJoint.frameName;
            if (!pinocchioModelTh_.existFrame(frameName))
            {
                JIMINY_THROW(std::logic_error,
                             "Frame '",
                             frameName,
                             "' does not exists. Impossible to insert flexibility joint on it.");
            }
        }

        // Add all the flexibility joints
        for (const FlexibilityJointConfig & flexibilityJoint :
             modelOptions_->dynamics.flexibilityConfig)
        {
            // Extract some proxies
            const std::string & frameName = flexibilityJoint.frameName;
            std::string flexName = frameName;
            const pinocchio::FrameIndex frameIndex = getFrameIndex(pinocchioModel_, frameName);
            const pinocchio::Frame & frame = pinocchioModel_.frames[frameIndex];

            // Add joint to model, differently depending on its type
            if (frame.type == pinocchio::FrameType::FIXED_JOINT)
            {
                // Insert flexibility joint at fixed frame, splitting "composite" body inertia
                addFlexibilityJointAtFixedFrame(pinocchioModel_, frameName);
            }
            else if (frame.type == pinocchio::FrameType::JOINT)
            {
                flexName += FLEXIBLE_JOINT_SUFFIX;
                addFlexibilityJointBeforeMechanicalJoint(pinocchioModel_, frameName, flexName);
            }
            else
            {
                JIMINY_THROW(std::logic_error,
                             "Flexible joint can only be inserted at fixed or joint frames.");
            }
            flexibilityJointNames_.push_back(flexName);
        }

        // Compute flexibility joint indices
        std::vector<pinocchio::JointIndex> flexibilityJointIndices =
            getJointIndices(pinocchioModel_, flexibilityJointNames_);

        // Add flexibility armature-like inertia to the model
        for (std::size_t i = 0; i < flexibilityJointIndices.size(); ++i)
        {
            const FlexibilityJointConfig & flexibilityJoint =
                modelOptions_->dynamics.flexibilityConfig[i];
            const pinocchio::JointModel & jmodel =
                pinocchioModel_.joints[flexibilityJointIndices[i]];
            jmodel.jointVelocitySelector(pinocchioModel_.rotorInertia) = flexibilityJoint.inertia;
        }

        // Check that the armature inertia is valid
        for (pinocchio::JointIndex flexibilityJointIndex : flexibilityJointIndices)
        {
            const pinocchio::Inertia & flexibilityInertia =
                pinocchioModel_.inertias[flexibilityJointIndex];
            const pinocchio::JointModel & jmodel = pinocchioModel_.joints[flexibilityJointIndex];
            const Eigen::Vector3d inertiaDiag =
                jmodel.jointVelocitySelector(pinocchioModel_.rotorInertia) +
                flexibilityInertia.inertia().matrix().diagonal();
            if ((inertiaDiag.array() < 1e-5).any())
            {
                JIMINY_THROW(std::runtime_error,
                             "The subtree diagonal inertia for flexibility joint ",
                             flexibilityJointIndex,
                             " must be larger than 1e-5 for numerical stability: ",
                             inertiaDiag.transpose());
            }
        }
    }

    void Model::addBiasedToExtendedModel(const uniform_random_bit_generator_ref<uint32_t> & g)
    {
        for (const std::string & jointName : mechanicalJointNames_)
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

        // Re-allocate data to be consistent with new extended model
        initializePinocchioData(pinocchioModel_, pinocchioData_);
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
                ConstraintRegistryType /* type */)
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
        // Make sure that the model is initialized
        if (!isInitialized_)
        {
            JIMINY_THROW(bad_control_flow, "Model not initialized.");
        }

        // Extract the dimensions of the configuration and velocity vectors
        nq_ = pinocchioModel_.nq;
        nv_ = pinocchioModel_.nv;
        nx_ = nq_ + nv_;

        // Extract some mechanical joints indices in the model
        mechanicalJointIndices_ = getJointIndices(pinocchioModel_, mechanicalJointNames_);
        mechanicalJointPositionIndices_ =
            getJointsPositionIndices(pinocchioModel_, mechanicalJointNames_, false);
        mechanicalJointVelocityIndices_ =
            getJointsVelocityIndices(pinocchioModel_, mechanicalJointNames_, false);

        // Compute flexibility and backlash joint indices
        backlashJointIndices_ = getJointIndices(pinocchioModel_, backlashJointNames_);
        flexibilityJointIndices_ = getJointIndices(pinocchioModel_, flexibilityJointNames_);

        /* Generate the fieldnames associated with the configuration vector, velocity,
           acceleration and external force vectors. */
        logPositionFieldnames_.clear();
        logPositionFieldnames_.reserve(static_cast<std::size_t>(nq_));
        logVelocityFieldnames_.clear();
        logVelocityFieldnames_.reserve(static_cast<std::size_t>(nv_));
        logAccelerationFieldnames_.clear();
        logAccelerationFieldnames_.reserve(static_cast<std::size_t>(nv_));
        logEffortFieldnames_.clear();
        logEffortFieldnames_.reserve(static_cast<std::size_t>(nv_));
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
                logEffortFieldnames_.emplace_back(
                    toString(jointPrefix, "Effort", jointShortName, suffix));
            }

            // Define complete external force fieldnames and backup them
            for (const std::string & suffix : ForceSensor::fieldnames_)
            {
                logForceExternalFieldnames_.emplace_back(
                    toString(jointPrefix, "ForceExternal", jointShortName, suffix));
            }
        }

        // Re-initialize position limits
        Eigen::VectorXd positionLimitLower = Eigen::VectorXd::Constant(pinocchioModel_.nq, -INF);
        Eigen::VectorXd positionLimitUpper = Eigen::VectorXd::Constant(pinocchioModel_.nq, +INF);
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
            case JointModelType::LINEAR:
            case JointModelType::ROTARY:
            case JointModelType::PLANAR:
            case JointModelType::TRANSLATION:
            case JointModelType::UNSUPPORTED:
            default:
                continue;
            }
            positionLimitLower.segment(idx_q, nq).setConstant(-1.0 - EPS);
            positionLimitUpper.segment(idx_q, nq).setConstant(+1.0 + EPS);
        }

        // Copy backlash joint position limits as-is from extended model
        for (pinocchio::JointIndex jointIndex : backlashJointIndices_)
        {
            const Eigen::Index positionIndex = pinocchioModel_.idx_qs[jointIndex];
            positionLimitLower[positionIndex] = pinocchioModel_.lowerPositionLimit[positionIndex];
            positionLimitUpper[positionIndex] = pinocchioModel_.upperPositionLimit[positionIndex];
        }

        // Set mechanical joint position limits from theoretical model or user options
        if (modelOptions_->joints.positionLimitFromUrdf)
        {
            int jointIndexTh = 1;
            for (std::size_t i = 0; i < mechanicalJointIndices_.size(); ++i)
            {
                const pinocchio::JointIndex jointIndex = mechanicalJointIndices_[i];
                const std::string & jointName = pinocchioModel_.names[jointIndex];
                while (pinocchioModelTh_.names[jointIndexTh] != jointName)
                {
                    ++jointIndexTh;
                }
                const auto & joint = pinocchioModel_.joints[jointIndex];
                const auto & jointTh = pinocchioModelTh_.joints[jointIndexTh];
                positionLimitLower.segment(joint.idx_q(), joint.nq()) =
                    pinocchioModelTh_.lowerPositionLimit.segment(jointTh.idx_q(), jointTh.nq());
                positionLimitUpper.segment(joint.idx_q(), joint.nq()) =
                    pinocchioModelTh_.upperPositionLimit.segment(jointTh.idx_q(), jointTh.nq());
            }
        }
        else
        {
            for (std::size_t i = 0; i < mechanicalJointPositionIndices_.size(); ++i)
            {
                Eigen::Index positionIndex = mechanicalJointPositionIndices_[i];
                positionLimitLower[positionIndex] = modelOptions_->joints.positionLimitLower[i];
                positionLimitUpper[positionIndex] = modelOptions_->joints.positionLimitUpper[i];
            }
        }

        // Overwrite extended model position limits
        pinocchioModel_.lowerPositionLimit = positionLimitLower;
        pinocchioModel_.upperPositionLimit = positionLimitUpper;

        // Initialize backup joint space acceleration
        jointSpatialAccelerations_ =
            MotionVector(pinocchioData_.a.size(), pinocchio::Motion::Zero());

        // Refresh all other proxies
        refreshGeometryProxies();
        refreshContactProxies();
    }

    void Model::refreshGeometryProxies()
    {
        if (!isInitialized_)
        {
            JIMINY_THROW(bad_control_flow, "Model not initialized.");
        }

        // Restore collision and visual models
        collisionModel_ = collisionModelTh_;
        visualModel_ = visualModelTh_;

        // Update joint/frame for every geometry objects
        for (pinocchio::GeometryModel * model : std::array{&collisionModel_, &visualModel_})
        {
            for (pinocchio::GeometryObject & geom : model->geometryObjects)
            {
                // Only the frame name remains unchanged no matter what
                const pinocchio::Frame & frameOrig = pinocchioModelTh_.frames[geom.parentFrame];
                const std::string parentJointName = pinocchioModelTh_.names[frameOrig.parent];
                pinocchio::FrameType frameType =
                    static_cast<pinocchio::FrameType>(pinocchio::FIXED_JOINT | pinocchio::BODY);
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

        /* Update geometry data object after changing the collision pairs.
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
            JIMINY_THROW(bad_control_flow, "Model not initialized.");
        }

        // Reset the contact force internal buffer
        contactForces_ = ForceVector(contactFrameNames_.size(), pinocchio::Force::Zero());

        // Extract the contact frames indices in the model
        contactFrameIndices_ = getFrameIndices(pinocchioModel_, contactFrameNames_);
    }

    void Model::setOptions(const GenericConfig & modelOptions)
    {
        bool internalBuffersMustBeUpdated = false;
        bool isExtendedModelInvalid = false;
        bool isCollisionDataInvalid = false;
        if (isInitialized_)
        {
            /* Check that the following user parameters has the right dimension, then update the
               required internal buffers to reflect changes, if any. */
            const GenericConfig & jointOptionsHolder =
                boost::get<GenericConfig>(modelOptions.at("joints"));
            bool positionLimitFromUrdf =
                boost::get<bool>(jointOptionsHolder.at("positionLimitFromUrdf"));
            if (!positionLimitFromUrdf)
            {
                const Eigen::VectorXd & jointspositionLimitLower =
                    boost::get<Eigen::VectorXd>(jointOptionsHolder.at("positionLimitLower"));
                if (mechanicalJointPositionIndices_.size() !=
                    static_cast<uint32_t>(jointspositionLimitLower.size()))
                {
                    JIMINY_THROW(std::invalid_argument,
                                 "Wrong vector size for 'positionLimitLower'.");
                }
                const Eigen::VectorXd & jointspositionLimitUpper =
                    boost::get<Eigen::VectorXd>(jointOptionsHolder.at("positionLimitUpper"));
                if (mechanicalJointPositionIndices_.size() !=
                    static_cast<uint32_t>(jointspositionLimitUpper.size()))
                {
                    JIMINY_THROW(std::invalid_argument,
                                 "Wrong vector size for 'positionLimitUpper'.");
                }
                if (mechanicalJointPositionIndices_.size() ==
                    static_cast<uint32_t>(modelOptions_->joints.positionLimitLower.size()))
                {
                    auto jointspositionLimitLowerDiff =
                        jointspositionLimitLower - modelOptions_->joints.positionLimitLower;
                    internalBuffersMustBeUpdated |=
                        (jointspositionLimitLowerDiff.array().abs() >= EPS).all();
                    auto jointspositionLimitUpperDiff =
                        jointspositionLimitUpper - modelOptions_->joints.positionLimitUpper;
                    internalBuffersMustBeUpdated |=
                        (jointspositionLimitUpperDiff.array().abs() >= EPS).all();
                }
                else
                {
                    internalBuffersMustBeUpdated = true;
                }
            }

            // Check if deformation points are all associated with different joints/frames
            const GenericConfig & dynOptionsHolder =
                boost::get<GenericConfig>(modelOptions.at("dynamics"));
            const FlexibilityConfig & flexibilityConfig =
                boost::get<FlexibilityConfig>(dynOptionsHolder.at("flexibilityConfig"));
            std::set<std::string> flexibilityNames;
            std::transform(flexibilityConfig.begin(),
                           flexibilityConfig.end(),
                           std::inserter(flexibilityNames, flexibilityNames.begin()),
                           [](const FlexibilityJointConfig & flexibilityJoint) -> std::string
                           { return flexibilityJoint.frameName; });
            if (flexibilityNames.size() != flexibilityConfig.size())
            {
                JIMINY_THROW(
                    std::invalid_argument,
                    "All joint or frame names in flexibility configuration must be unique.");
            }
            if (std::find(flexibilityNames.begin(), flexibilityNames.end(), "universe") !=
                flexibilityNames.end())
            {
                JIMINY_THROW(std::invalid_argument,
                             "No one can make the universe itself flexibility.");
            }
            for (const FlexibilityJointConfig & flexibilityJoint : flexibilityConfig)
            {
                if ((flexibilityJoint.stiffness.array() < 0.0).any() ||
                    (flexibilityJoint.damping.array() < 0.0).any() ||
                    (flexibilityJoint.inertia.array() < 0.0).any())
                {
                    JIMINY_THROW(std::invalid_argument,
                                 "All stiffness, damping and inertia parameters of flexibility "
                                 "joints must be positive.");
                }
            }

            // Check if the extended model must be regenerated
            bool enableFlexibility = boost::get<bool>(dynOptionsHolder.at("enableFlexibility"));
            if (modelOptions_ &&
                ((enableFlexibility != modelOptions_->dynamics.enableFlexibility) ||
                 (enableFlexibility &&
                  ((flexibilityConfig.size() !=
                    modelOptions_->dynamics.flexibilityConfig.size()) ||
                   !std::equal(flexibilityConfig.begin(),
                               flexibilityConfig.end(),
                               modelOptions_->dynamics.flexibilityConfig.begin())))))
            {
                isExtendedModelInvalid = true;
            }
        }

        // Check that the collisions options are valid
        const GenericConfig & collisionOptionsHolder =
            boost::get<GenericConfig>(modelOptions.at("collisions"));
        uint32_t contactPointsPerBodyMax =
            boost::get<uint32_t>(collisionOptionsHolder.at("contactPointsPerBodyMax"));
        if (contactPointsPerBodyMax < 1)
        {
            JIMINY_THROW(std::invalid_argument,
                         "Number of contact points by collision pair "
                         "'contactPointsPerBodyMax' must be strictly larger than 0.");
        }
        if (modelOptions_ &&
            contactPointsPerBodyMax != modelOptions_->collisions.contactPointsPerBodyMax)
        {
            isCollisionDataInvalid = true;
        }

        // Check that the model randomization parameters are valid
        const GenericConfig & dynOptionsHolder =
            boost::get<GenericConfig>(modelOptions.at("dynamics"));
        for (auto && field : std::array{"inertiaBodiesBiasStd",
                                        "massBodiesBiasStd",
                                        "centerOfMassPositionBodiesBiasStd",
                                        "relativePositionBodiesBiasStd"})
        {
            const double value = boost::get<double>(dynOptionsHolder.at(field));
            if (0.9 < value || value < 0.0)
            {
                JIMINY_THROW(std::invalid_argument,
                             "'",
                             field,
                             "' must be positive, and lower than 0.9 to avoid physics issues.");
            }
        }

        // Update class-specific "strongly typed" accessor for fast and convenient access
        modelOptions_ = std::make_unique<const ModelOptions>(modelOptions);

        // Update inherited polymorphic accessor
        deepUpdate(modelOptionsGeneric_, modelOptions);

        if (isExtendedModelInvalid)
        {
            // Trigger extended regeneration
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

    const GenericConfig & Model::getOptions() const noexcept
    {
        return modelOptionsGeneric_;
    }

    bool Model::getIsInitialized() const
    {
        return isInitialized_;
    }

    const std::string & Model::getName() const
    {
        return pinocchioModelTh_.name;
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

    void Model::getExtendedPositionFromTheoretical(const Eigen::VectorXd & qTheoretical,
                                                   Eigen::VectorXd & qExtended) const
    {
        // Check the size of the input state
        if (qTheoretical.size() != pinocchioModelTh_.nq)
        {
            JIMINY_THROW(std::invalid_argument, "Input size inconsistent with theoretical model.");
        }

        // Initialize the returned extended configuration
        qExtended = pinocchio::neutral(pinocchioModel_);

        // Compute extended configuration from theoretical
        int theoreticalJointIndex = 1;
        int extendedJointIndex = 1;
        for (; theoreticalJointIndex < pinocchioModelTh_.njoints; ++extendedJointIndex)
        {
            const std::string & jointTheoreticalName =
                pinocchioModelTh_.names[theoreticalJointIndex];
            const std::string & jointExtendedName = pinocchioModel_.names[extendedJointIndex];
            if (jointTheoreticalName == jointExtendedName)
            {
                const auto & jointTheoretical = pinocchioModelTh_.joints[theoreticalJointIndex];
                const auto & jointExtended = pinocchioModel_.joints[extendedJointIndex];
                if (jointTheoretical.idx_q() >= 0)
                {
                    qExtended.segment(jointExtended.idx_q(), jointExtended.nq()) =
                        qTheoretical.segment(jointTheoretical.idx_q(), jointTheoretical.nq());
                }
                ++theoreticalJointIndex;
            }
        }
    }


    void Model::getExtendedVelocityFromTheoretical(const Eigen::VectorXd & vTheoretical,
                                                   Eigen::VectorXd & vExtended) const
    {
        // Check the size of the input state
        if (vTheoretical.size() != pinocchioModelTh_.nv)
        {
            JIMINY_THROW(std::invalid_argument, "Input size inconsistent with theoretical model.");
        }

        // Initialize the returned extended velocity
        vExtended.setZero(pinocchioModel_.nv);

        // Compute extended velocity from theoretical
        int32_t theoreticalJointIndex = 1;
        int32_t extendedJointIndex = 1;
        for (; theoreticalJointIndex < pinocchioModelTh_.njoints; ++extendedJointIndex)
        {
            const std::string & jointTheoreticalName =
                pinocchioModelTh_.names[theoreticalJointIndex];
            const std::string & jointExtendedName = pinocchioModel_.names[extendedJointIndex];
            if (jointTheoreticalName == jointExtendedName)
            {
                const auto & jointTheoretical = pinocchioModelTh_.joints[theoreticalJointIndex];
                const auto & jointExtended = pinocchioModel_.joints[extendedJointIndex];
                if (jointTheoretical.idx_q() >= 0)
                {
                    vExtended.segment(jointExtended.idx_v(), jointExtended.nv()) =
                        vTheoretical.segment(jointTheoretical.idx_v(), jointTheoretical.nv());
                }
                ++theoreticalJointIndex;
            }
        }
    }

    void Model::getTheoreticalPositionFromExtended(const Eigen::VectorXd & qExtended,
                                                   Eigen::VectorXd & qTheoretical) const
    {
        // Check the size of the input state
        if (qExtended.size() != pinocchioModel_.nq)
        {
            JIMINY_THROW(std::invalid_argument, "Input size inconsistent with extended model.");
        }

        // Initialize the returned theoretical configuration
        qTheoretical = pinocchio::neutral(pinocchioModelTh_);

        // Compute theoretical configuration from extended
        int32_t theoreticalJointIndex = 1;
        int32_t extendedJointIndex = 1;
        for (; theoreticalJointIndex < pinocchioModelTh_.njoints; ++extendedJointIndex)
        {
            const std::string & jointTheoreticalName =
                pinocchioModelTh_.names[theoreticalJointIndex];
            const std::string & jointExtendedName = pinocchioModel_.names[extendedJointIndex];
            if (jointTheoreticalName == jointExtendedName)
            {
                const auto & jointTheoretical = pinocchioModelTh_.joints[theoreticalJointIndex];
                const auto & jointExtended = pinocchioModel_.joints[extendedJointIndex];
                if (jointTheoretical.idx_q() >= 0)
                {
                    qTheoretical.segment(jointTheoretical.idx_q(), jointTheoretical.nq()) =
                        qExtended.segment(jointExtended.idx_q(), jointExtended.nq());
                }
                ++theoreticalJointIndex;
            }
        }
    }

    void Model::getTheoreticalVelocityFromExtended(const Eigen::VectorXd & vExtended,
                                                   Eigen::VectorXd & vTheoretical) const
    {
        // Check the size of the input state
        if (vExtended.size() != pinocchioModel_.nv)
        {
            JIMINY_THROW(std::invalid_argument, "Input size inconsistent with extended model.");
        }

        // Initialize the returned theoretical velocity
        vTheoretical.setZero(pinocchioModelTh_.nv);

        // Compute theoretical velocity from extended
        int32_t theoreticalJointIndex = 1;
        int32_t extendedJointIndex = 1;
        for (; theoreticalJointIndex < pinocchioModelTh_.njoints; ++extendedJointIndex)
        {
            const std::string & jointTheoreticalName =
                pinocchioModelTh_.names[theoreticalJointIndex];
            const std::string & jointExtendedName = pinocchioModel_.names[extendedJointIndex];
            if (jointTheoreticalName == jointExtendedName)
            {
                const auto & jointTheoretical = pinocchioModelTh_.joints[theoreticalJointIndex];
                const auto & jointExtended = pinocchioModel_.joints[extendedJointIndex];
                if (jointTheoretical.idx_q() >= 0)
                {
                    vTheoretical.segment(jointTheoretical.idx_v(), jointTheoretical.nv()) =
                        vExtended.segment(jointExtended.idx_v(), jointExtended.nv());
                }
                ++theoreticalJointIndex;
            }
        }
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

    const std::vector<std::string> & Model::getMechanicalJointNames() const
    {
        return mechanicalJointNames_;
    }

    const std::vector<pinocchio::JointIndex> & Model::getMechanicalJointIndices() const
    {
        return mechanicalJointIndices_;
    }

    const std::vector<Eigen::Index> & Model::getMechanicalJointPositionIndices() const
    {
        return mechanicalJointPositionIndices_;
    }

    const std::vector<Eigen::Index> & Model::getMechanicalJointVelocityIndices() const
    {
        return mechanicalJointVelocityIndices_;
    }

    const std::vector<std::string> & Model::getFlexibilityJointNames() const
    {
        return flexibilityJointNames_;
    }

    const std::vector<pinocchio::JointIndex> & Model::getFlexibilityJointIndices() const
    {
        return flexibilityJointIndices_;
    }

    const std::vector<std::string> & Model::getBacklashJointNames() const
    {
        return backlashJointNames_;
    }

    const std::vector<pinocchio::JointIndex> & Model::getBacklashJointIndices() const
    {
        return backlashJointIndices_;
    }

    const std::vector<std::string> & Model::getLogPositionFieldnames() const
    {
        return logPositionFieldnames_;
    }

    const std::vector<std::string> & Model::getLogVelocityFieldnames() const
    {
        return logVelocityFieldnames_;
    }

    const std::vector<std::string> & Model::getLogAccelerationFieldnames() const
    {
        return logAccelerationFieldnames_;
    }

    const std::vector<std::string> & Model::getLogEffortFieldnames() const
    {
        return logEffortFieldnames_;
    }

    const std::vector<std::string> & Model::getLogForceExternalFieldnames() const
    {
        return logForceExternalFieldnames_;
    }

    const std::vector<std::string> & Model::getLogConstraintFieldnames() const
    {
        return logConstraintFieldnames_;
    }

    std::unique_ptr<LockGuardLocal> Model::getLock()
    {
        // Make sure that the robot is not already locked
        if (mutexLocal_->isLocked())
        {
            JIMINY_THROW(bad_control_flow,
                         "Model already locked. Please release it first prior requesting lock.");
        }

        // Return lock
        return std::make_unique<LockGuardLocal>(*mutexLocal_);
    }

    bool Model::getIsLocked() const
    {
        return mutexLocal_->isLocked();
    }
}
