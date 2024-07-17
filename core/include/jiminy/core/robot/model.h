#ifndef JIMINY_MODEL_H
#define JIMINY_MODEL_H

#include <optional>

#include "pinocchio/spatial/fwd.hpp"         // `pinocchio::SE3`
#include "pinocchio/multibody/model.hpp"     // `pinocchio::Model`
#include "pinocchio/multibody/data.hpp"      // `pinocchio::Data`
#include "pinocchio/multibody/geometry.hpp"  // `pinocchio::GeometryModel`, `pinocchio::GeometryData`
#include "pinocchio/multibody/frame.hpp"  // `pinocchio::FrameType` (C-style enum cannot be forward declared)

#include "jiminy/core/fwd.h"


namespace jiminy
{
    inline constexpr std::string_view JOINT_PREFIX_BASE{"current"};
    inline constexpr std::string_view FREE_FLYER_NAME{"Freeflyer"};
    inline constexpr std::string_view FLEXIBLE_JOINT_SUFFIX{"Flexibility"};
    inline constexpr std::string_view BACKLASH_JOINT_SUFFIX{"Backlash"};

    class AbstractConstraintBase;
    class FrameConstraint;
    class JointConstraint;
    class MutexLocal;
    class LockGuardLocal;

    // ************************************** Constraints ************************************** //

    using ConstraintMap = static_map_t<std::string, std::shared_ptr<AbstractConstraintBase>>;

    enum class ConstraintRegistryType : uint8_t
    {
        CONTACT_FRAMES = 0,
        COLLISION_BODIES = 1,
        BOUNDS_JOINTS = 2,
        USER = 3
    };

    /* Note that the following ordering plays a critical role as it determines in which order
       `foreach` iterates over all the constraints. This has a directly effect on the solution
       found by 'PGS' constraint solvers. */
    inline constexpr std::array constraintNodeTypesAll{ConstraintRegistryType::BOUNDS_JOINTS,
                                                       ConstraintRegistryType::CONTACT_FRAMES,
                                                       ConstraintRegistryType::COLLISION_BODIES,
                                                       ConstraintRegistryType::USER};

    struct JIMINY_DLLAPI ConstraintTree
    {
    public:
        std::pair<ConstraintMap *, ConstraintMap::iterator> find(const std::string & key,
                                                                 ConstraintRegistryType type);
        std::pair<const ConstraintMap *, ConstraintMap::const_iterator> find(
            const std::string & key, ConstraintRegistryType type) const;

        bool exist(const std::string & key, ConstraintRegistryType type) const;

        std::shared_ptr<AbstractConstraintBase> get(const std::string & key,
                                                    ConstraintRegistryType type) const;

        void insert(const ConstraintMap & constraintMap, ConstraintRegistryType type);

        ConstraintMap::iterator erase(const std::string & key, ConstraintRegistryType type);

        void clear() noexcept;

        template<typename Function>
        void foreach(ConstraintRegistryType type, Function && func) const
        {
            if (type == ConstraintRegistryType::COLLISION_BODIES)
            {
                for (auto & constraintMap : collisionBodies)
                {
                    for (auto & constraintItem : constraintMap)
                    {
                        std::invoke(std::forward<Function>(func), constraintItem.second, type);
                    }
                }
            }
            else
            {
                const ConstraintMap * constraintMapPtr;
                switch (type)
                {
                case ConstraintRegistryType::BOUNDS_JOINTS:
                    constraintMapPtr = &boundJoints;
                    break;
                case ConstraintRegistryType::CONTACT_FRAMES:
                    constraintMapPtr = &contactFrames;
                    break;
                case ConstraintRegistryType::USER:
                    constraintMapPtr = &user;
                    break;
                case ConstraintRegistryType::COLLISION_BODIES:
                default:
                    constraintMapPtr = nullptr;
                }
                for (const auto & constraintItem : *constraintMapPtr)
                {
                    std::invoke(std::forward<Function>(func), constraintItem.second, type);
                }
            }
        }

        template<typename Function, std::size_t N>
        void foreach(const std::array<ConstraintRegistryType, N> & types, Function && func) const
        {
            for (ConstraintRegistryType type : types)
            {
                foreach(type, std::forward<Function>(func));
            }
        }

        template<typename Function>
        void foreach(Function && func) const
        {
            foreach(constraintNodeTypesAll, std::forward<Function>(func));
        }

    public:
        /// \brief Constraints registered by the engine to handle joint bounds.
        ConstraintMap boundJoints{};
        /// \brief Constraints registered by the engine to handle contact frames.
        ConstraintMap contactFrames{};
        /// \brief Constraints registered by the engine to handle collision bounds.
        std::vector<ConstraintMap> collisionBodies{};
        /// \brief Constraints explicitly registered by user.
        ConstraintMap user{};
    };

    // ***************************************** Model ***************************************** //

    class JIMINY_DLLAPI Model : public std::enable_shared_from_this<Model>
    {
    public:
        virtual GenericConfig getDefaultJointOptions()
        {
            GenericConfig config;
            config["positionLimitFromUrdf"] = true;
            config["positionLimitMin"] = Eigen::VectorXd{};
            config["positionLimitMax"] = Eigen::VectorXd{};

            return config;
        };

        virtual GenericConfig getDefaultDynamicsOptions()
        {
            // Add extra options or update default values
            GenericConfig config;
            config["inertiaBodiesBiasStd"] = 0.0;
            config["massBodiesBiasStd"] = 0.0;
            config["centerOfMassPositionBodiesBiasStd"] = 0.0;
            config["relativePositionBodiesBiasStd"] = 0.0;
            config["enableFlexibility"] = true;
            config["flexibilityConfig"] = FlexibilityConfig{};

            return config;
        };

        virtual GenericConfig getDefaultCollisionOptions()
        {
            // Add extra options or update default values
            GenericConfig config;
            /// \brief Max number of contact points per collision pairs.
            config["contactPointsPerBodyMax"] = 5U;

            return config;
        };

        virtual GenericConfig getDefaultModelOptions()
        {
            GenericConfig config;
            config["dynamics"] = getDefaultDynamicsOptions();
            config["joints"] = getDefaultJointOptions();
            config["collisions"] = getDefaultCollisionOptions();

            return config;
        };

        struct JointOptions
        {
            const bool positionLimitFromUrdf;
            /// \brief Min position limit of all the mechanical joints of the theoretical model.
            const Eigen::VectorXd positionLimitMin;
            const Eigen::VectorXd positionLimitMax;

            JointOptions(const GenericConfig & options) :
            positionLimitFromUrdf{boost::get<bool>(options.at("positionLimitFromUrdf"))},
            positionLimitMin{boost::get<Eigen::VectorXd>(options.at("positionLimitMin"))},
            positionLimitMax{boost::get<Eigen::VectorXd>(options.at("positionLimitMax"))}
            {
            }
        };

        struct DynamicsOptions
        {
            const double inertiaBodiesBiasStd;
            const double massBodiesBiasStd;
            const double centerOfMassPositionBodiesBiasStd;
            const double relativePositionBodiesBiasStd;
            const bool enableFlexibility;
            const FlexibilityConfig flexibilityConfig;

            DynamicsOptions(const GenericConfig & options) :
            inertiaBodiesBiasStd{boost::get<double>(options.at("inertiaBodiesBiasStd"))},
            massBodiesBiasStd{boost::get<double>(options.at("massBodiesBiasStd"))},
            centerOfMassPositionBodiesBiasStd{
                boost::get<double>(options.at("centerOfMassPositionBodiesBiasStd"))},
            relativePositionBodiesBiasStd{
                boost::get<double>(options.at("relativePositionBodiesBiasStd"))},
            enableFlexibility{boost::get<bool>(options.at("enableFlexibility"))},
            flexibilityConfig{boost::get<FlexibilityConfig>(options.at("flexibilityConfig"))}
            {
            }
        };

        struct CollisionOptions
        {
            const uint32_t contactPointsPerBodyMax;

            CollisionOptions(const GenericConfig & options) :
            contactPointsPerBodyMax{boost::get<uint32_t>(options.at("contactPointsPerBodyMax"))}
            {
            }
        };

        struct ModelOptions
        {
            const DynamicsOptions dynamics;
            const JointOptions joints;
            const CollisionOptions collisions;

            ModelOptions(const GenericConfig & options) :
            dynamics{boost::get<GenericConfig>(options.at("dynamics"))},
            joints{boost::get<GenericConfig>(options.at("joints"))},
            collisions{boost::get<GenericConfig>(options.at("collisions"))}
            {
            }
        };

    public:
        /* Manually enforcing memory alignment.

           Without it, head memory will not be properly allocated when de-serializing shared
           pointers of `Model`, because fixed-size `Eigen::matrix` objects of `pinocchio::Data`
           are not properly aligned as they should when AVX2 or higher is enabled.

           Note that customizing `boost::archive::detail::heap_allocation`, the heap allocator of
           `boost::serialization`, is not viable because the original deleter of the object will
           still be called upon destruction of the object, so the allocator and destructor must
           be consistent.

           The proposed workaround is based on `EIGEN_MAKE_ALIGNED_OPERATOR_NEW`.
           See: https://gitlab.com/libeigen/eigen/-/blob/master/Eigen/src/Core/util/Memory.h */
        EIGEN_DEVICE_FUNC void * operator new(std::size_t size)
        {
            return Eigen::internal::aligned_malloc(size);
        }

        EIGEN_DEVICE_FUNC void operator delete(void * ptr) EIGEN_NO_THROW
        {
            Eigen::internal::aligned_free(ptr);
        }

    public:
        explicit Model() noexcept;
        Model(const Model & other);
        Model & operator=(const Model & other);
        Model & operator=(Model && other) = default;
        virtual ~Model() = default;

        void initialize(
            const pinocchio::Model & pinocchioModel,
            const std::optional<pinocchio::GeometryModel> & collisionModel = std::nullopt,
            const std::optional<pinocchio::GeometryModel> & visualModel = std::nullopt);
        void initialize(const std::string & urdfPath,
                        bool hasFreeflyer = true,
                        const std::vector<std::string> & meshPackageDirs = {},
                        bool loadVisualMeshes = false);

        /// \brief Add a frame in the kinematic tree, attached to the frame of an existing body.
        ///
        /// \param[in] frameName Name of the frame to be added.
        /// \param[in] parentBodyName Name of the parent body frame.
        /// \param[in] framePlacement Frame placement wrt the parent body frame.
        void addFrame(const std::string & frameName,
                      const std::string & parentBodyName,
                      const pinocchio::SE3 & framePlacement);
        void removeFrames(const std::vector<std::string> & frameNames);

        void addCollisionBodies(const std::vector<std::string> & bodyNames,
                                bool ignoreMeshes = false);
        void removeCollisionBodies(std::vector<std::string> frameNames = {});  // Copy on purpose
        void addContactPoints(const std::vector<std::string> & frameNames);
        void removeContactPoints(const std::vector<std::string> & frameNames = {});

        /// \brief Add a kinematic constraint to the robot.
        ///
        /// \param[in] constraintName Unique name identifying the kinematic constraint.
        /// \param[in] constraint Constraint to add.
        void addConstraint(const std::string & constraintName,
                           const std::shared_ptr<AbstractConstraintBase> & constraint);

        /// \brief Remove a kinematic constraint form the system.
        ///
        /// \param[in] constraintName Unique name identifying the kinematic constraint.
        void removeConstraint(const std::string & constraintName);

        const ConstraintTree & getConstraints() const;

        /// \brief Returns true if at least one constraint is active on the robot.
        bool hasConstraints() const;

        void resetConstraints(const Eigen::VectorXd & q, const Eigen::VectorXd & v);

        /// \brief Compute jacobian and drift associated to all the constraints.
        ///
        /// \details The results are accessible using getConstraintsJacobian and
        ///          getConstraintsDrift.
        /// \note It is assumed frames forward kinematics has already been called.
        ///
        /// \param[in] q Joint position.
        /// \param[in] v Joint velocity.
        void computeConstraints(const Eigen::VectorXd & q, const Eigen::VectorXd & v);

        void setOptions(const GenericConfig & modelOptions);
        const GenericConfig & getOptions() const noexcept;

        /// \remark This method does not have to be called manually before running a simulation.
        ///         The Engine is taking care of it.
        virtual void reset(const uniform_random_bit_generator_ref<uint32_t> & g);

        bool getIsInitialized() const;
        const std::string & getName() const;
        const std::string & getUrdfPath() const;
        const std::string & getUrdfAsString() const;
        const std::vector<std::string> & getMeshPackageDirs() const;
        bool getHasFreeflyer() const;
        // Getters without 'get' prefix for consistency with pinocchio C++ API
        Eigen::Index nq() const;
        Eigen::Index nv() const;
        Eigen::Index nx() const;

        const std::vector<std::string> & getCollisionBodyNames() const;
        const std::vector<std::string> & getContactFrameNames() const;
        const std::vector<pinocchio::FrameIndex> & getCollisionBodyIndices() const;
        const std::vector<std::vector<pinocchio::PairIndex>> & getCollisionPairIndices() const;
        const std::vector<pinocchio::FrameIndex> & getContactFrameIndices() const;

        const std::vector<std::string> & getMechanicalJointNames() const;
        const std::vector<pinocchio::JointIndex> & getMechanicalJointIndices() const;
        const std::vector<Eigen::Index> & getMechanicalJointPositionIndices() const;
        const std::vector<Eigen::Index> & getMechanicalJointVelocityIndices() const;
        const std::vector<std::string> & getFlexibilityJointNames() const;
        const std::vector<pinocchio::JointIndex> & getFlexibilityJointIndices() const;
        const std::vector<std::string> & getBacklashJointNames() const;
        const std::vector<pinocchio::JointIndex> & getBacklashJointIndices() const;

        const std::vector<std::string> & getLogPositionFieldnames() const;
        const std::vector<std::string> & getLogVelocityFieldnames() const;
        const std::vector<std::string> & getLogAccelerationFieldnames() const;
        const std::vector<std::string> & getLogEffortFieldnames() const;
        const std::vector<std::string> & getLogForceExternalFieldnames() const;
        const std::vector<std::string> & getLogConstraintFieldnames() const;

        void getExtendedPositionFromTheoretical(const Eigen::VectorXd & qTheoretical,
                                                Eigen::VectorXd & qExtended) const;
        void getExtendedVelocityFromTheoretical(const Eigen::VectorXd & vTheoretical,
                                                Eigen::VectorXd & vExtended) const;
        void getTheoreticalPositionFromExtended(const Eigen::VectorXd & qExtended,
                                                Eigen::VectorXd & qTheoretical) const;
        void getTheoreticalVelocityFromExtended(const Eigen::VectorXd & vExtended,
                                                Eigen::VectorXd & vTheoretical) const;

        virtual std::unique_ptr<LockGuardLocal> getLock();
        bool getIsLocked() const;

    protected:
        void generateModelExtended(const uniform_random_bit_generator_ref<uint32_t> & g);

        virtual void initializeExtendedModel();
        void addFlexibilityJointsToExtendedModel();
        void addBiasedToExtendedModel(const uniform_random_bit_generator_ref<uint32_t> & g);

        void addFrame(const std::string & frameName,
                      const std::string & parentBodyName,
                      const pinocchio::SE3 & framePlacement,
                      const pinocchio::FrameType & frameType);
        void removeFrames(const std::vector<std::string> & frameNames,
                          const std::vector<pinocchio::FrameType> & filter);

        void addConstraint(const std::string & constraintName,
                           const std::shared_ptr<AbstractConstraintBase> & constraint,
                           ConstraintRegistryType type);
        void addConstraints(const ConstraintMap & constraintMap, ConstraintRegistryType type);
        void removeConstraint(const std::string & constraintName, ConstraintRegistryType type);
        void removeConstraints(const std::vector<std::string> & constraintNames,
                               ConstraintRegistryType type);

        void refreshGeometryProxies();
        void refreshContactProxies();
        virtual void refreshProxies();

    public:
        pinocchio::Model pinocchioModelTh_{};
        pinocchio::Model pinocchioModel_{};
        pinocchio::GeometryModel collisionModelTh_{};
        pinocchio::GeometryModel collisionModel_{};
        pinocchio::GeometryModel visualModelTh_{};
        pinocchio::GeometryModel visualModel_{};
        mutable pinocchio::Data pinocchioDataTh_{};
        mutable pinocchio::Data pinocchioData_{};
        mutable pinocchio::GeometryData collisionData_{};
        mutable pinocchio::GeometryData visualData_{};
        std::unique_ptr<const ModelOptions> modelOptions_{nullptr};
        /// \brief Buffer storing the contact forces.
        ForceVector contactForces_{};

    protected:
        bool isInitialized_{false};
        std::string urdfPath_{};
        std::string urdfData_{};
        std::vector<std::string> meshPackageDirs_{};
        bool hasFreeflyer_{false};
        GenericConfig modelOptionsGeneric_{};

        /// \brief Name of the collision bodies of the robot.
        std::vector<std::string> collisionBodyNames_{};
        /// \brief Name of the contact frames of the robot.
        std::vector<std::string> contactFrameNames_{};
        /// \brief Indices of the collision bodies in the frame list of the robot.
        std::vector<pinocchio::FrameIndex> collisionBodyIndices_{};
        /// \brief Indices of the collision pairs associated with each collision body.
        std::vector<std::vector<pinocchio::PairIndex>> collisionPairIndices_{};
        /// \brief Indices of the contact frames in the frame list of the robot.
        std::vector<pinocchio::FrameIndex> contactFrameIndices_{};
        /// \brief Name of the mechanical joints of the robot, ie all joints of the theoretical
        ///        excluding freeflyer if any.
        std::vector<std::string> mechanicalJointNames_{};
        /// \brief Index of the mechanical joints in the pinocchio robot.
        std::vector<pinocchio::JointIndex> mechanicalJointIndices_{};
        /// \brief All the indices of the mechanical joints in the configuration vector of the
        ///        robot, ie including all their respective degrees of freedom.
        std::vector<Eigen::Index> mechanicalJointPositionIndices_{};
        /// \brief All the indices of the mechanical joints in the velocity vector of the robot,
        ///        ie including all their respective degrees of freedom.
        std::vector<Eigen::Index> mechanicalJointVelocityIndices_{};
        /// \brief Name of the flexibility joints of the robot if enabled.
        std::vector<std::string> flexibilityJointNames_{};
        /// \brief Index of the flexibility joints in the pinocchio robot  if enabled.
        std::vector<pinocchio::JointIndex> flexibilityJointIndices_{};
        /// \brief Name of the backlash joints of the robot  if enabled.
        std::vector<std::string> backlashJointNames_{};
        /// \brief Index of the backlash joints in the pinocchio robot if enabled.
        std::vector<pinocchio::JointIndex> backlashJointIndices_{};

        /// \brief Store constraints.
        ConstraintTree constraints_{};

        /// \brief Fieldnames of the elements in the configuration vector of the model.
        std::vector<std::string> logPositionFieldnames_{};
        /// \brief Fieldnames of the elements in the velocity vector of the model.
        std::vector<std::string> logVelocityFieldnames_{};
        /// \brief Fieldnames of the elements in the acceleration vector of the model.
        std::vector<std::string> logAccelerationFieldnames_{};
        /// \brief Fieldnames of the elements in the effort vector of the model.
        std::vector<std::string> logEffortFieldnames_{};
        /// \brief Concatenated fieldnames of the external force applied at each joint of the
        ///        model, 'universe' excluded.
        std::vector<std::string> logForceExternalFieldnames_{};
        /// \brief Concatenated fieldnames of all the constraints.
        std::vector<std::string> logConstraintFieldnames_{};

    private:
        std::unique_ptr<MutexLocal> mutexLocal_{std::make_unique<MutexLocal>()};

        /// \brief Vector of joints acceleration corresponding to a copy of data.a.
        //         Used for computing constraints as a temporary buffer.
        MotionVector jointSpatialAccelerations_{};

        Eigen::Index nq_{0};
        Eigen::Index nv_{0};
        Eigen::Index nx_{0};
    };
}

#endif  // end of JIMINY_MODEL_H
