#ifndef JIMINY_MODEL_H
#define JIMINY_MODEL_H

#include "pinocchio/spatial/fwd.hpp"         // `pinocchio::SE3`
#include "pinocchio/multibody/model.hpp"     // `pinocchio::Model`
#include "pinocchio/multibody/data.hpp"      // `pinocchio::Data`
#include "pinocchio/multibody/geometry.hpp"  // `pinocchio::GeometryModel`, `pinocchio::GeometryData`
#include "pinocchio/multibody/frame.hpp"  // `pinocchio::FrameType` (C-style enum cannot be forward declared)

#include "jiminy/core/fwd.h"
#include "jiminy/core/utilities/random.h"  // `uniform_random_bit_generator_ref`


namespace jiminy
{
    inline constexpr std::string_view JOINT_PREFIX_BASE{"current"};
    inline constexpr std::string_view FREE_FLYER_NAME{"Freeflyer"};
    inline constexpr std::string_view FLEXIBLE_JOINT_SUFFIX{"Flexibility"};

    class AbstractConstraintBase;
    class FrameConstraint;
    class JointConstraint;

    using ConstraintMap = static_map_t<std::string, std::shared_ptr<AbstractConstraintBase>>;

    enum class JIMINY_DLLAPI ConstraintNodeType : uint8_t
    {
        BOUNDS_JOINTS = 0,
        CONTACT_FRAMES = 1,
        COLLISION_BODIES = 2,
        USER = 3
    };

    inline constexpr std::array constraintNodeTypesAll{ConstraintNodeType::BOUNDS_JOINTS,
                                                       ConstraintNodeType::CONTACT_FRAMES,
                                                       ConstraintNodeType::COLLISION_BODIES,
                                                       ConstraintNodeType::USER};

    struct JIMINY_DLLAPI ConstraintTree
    {
    public:
        void clear() noexcept;

        std::pair<ConstraintMap *, ConstraintMap::iterator> find(const std::string & key,
                                                                 ConstraintNodeType node);

        bool exist(const std::string & key) const;
        bool exist(const std::string & key, ConstraintNodeType node) const;

        std::shared_ptr<AbstractConstraintBase> get(const std::string & key);
        std::shared_ptr<AbstractConstraintBase> get(const std::string & key,
                                                    ConstraintNodeType node);

        void insert(const ConstraintMap & constraintMap, ConstraintNodeType node);

        ConstraintMap::iterator erase(const std::string & key, ConstraintNodeType node);

        template<typename Function>
        void foreach(ConstraintNodeType node, Function && func)
        {
            if (node == ConstraintNodeType::COLLISION_BODIES)
            {
                for (auto & constraintMap : collisionBodies)
                {
                    for (auto & constraintItem : constraintMap)
                    {
                        std::invoke(std::forward<Function>(func), constraintItem.second, node);
                    }
                }
            }
            else
            {
                ConstraintMap * constraintMapPtr;
                switch (node)
                {
                case ConstraintNodeType::BOUNDS_JOINTS:
                    constraintMapPtr = &boundJoints;
                    break;
                case ConstraintNodeType::CONTACT_FRAMES:
                    constraintMapPtr = &contactFrames;
                    break;
                case ConstraintNodeType::USER:
                    constraintMapPtr = &registry;
                    break;
                case ConstraintNodeType::COLLISION_BODIES:
                default:
                    constraintMapPtr = nullptr;
                }
                for (auto & constraintItem : *constraintMapPtr)
                {
                    std::invoke(std::forward<Function>(func), constraintItem.second, node);
                }
            }
        }

        template<typename Function, std::size_t N>
        void foreach(const std::array<ConstraintNodeType, N> & constraintsHolderTypes,
                     Function && func)
        {
            for (ConstraintNodeType node : constraintsHolderTypes)
            {
                foreach(node, std::forward<Function>(func));
            }
        }

        template<typename Function>
        void foreach(Function && func)
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
        ConstraintMap registry{};
    };

    class JIMINY_DLLAPI Model : public std::enable_shared_from_this<Model>
    {
    public:
        virtual GenericConfig getDefaultJointOptions()
        {
            GenericConfig config;
            config["enablePositionLimit"] = true;
            config["positionLimitFromUrdf"] = true;
            config["positionLimitMin"] = Eigen::VectorXd{};
            config["positionLimitMax"] = Eigen::VectorXd{};
            config["enableVelocityLimit"] = true;
            config["velocityLimitFromUrdf"] = true;
            config["velocityLimit"] = Eigen::VectorXd{};

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
            config["enableFlexibleModel"] = true;
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
            const bool enablePositionLimit;
            const bool positionLimitFromUrdf;
            /// \brief Min position limit of all the rigid joints, ie without freeflyer and
            ///        flexibility joints if any.
            const Eigen::VectorXd positionLimitMin;
            const Eigen::VectorXd positionLimitMax;
            const bool enableVelocityLimit;
            const bool velocityLimitFromUrdf;
            const Eigen::VectorXd velocityLimit;

            JointOptions(const GenericConfig & options) :
            enablePositionLimit{boost::get<bool>(options.at("enablePositionLimit"))},
            positionLimitFromUrdf{boost::get<bool>(options.at("positionLimitFromUrdf"))},
            positionLimitMin{boost::get<Eigen::VectorXd>(options.at("positionLimitMin"))},
            positionLimitMax{boost::get<Eigen::VectorXd>(options.at("positionLimitMax"))},
            enableVelocityLimit{boost::get<bool>(options.at("enableVelocityLimit"))},
            velocityLimitFromUrdf{boost::get<bool>(options.at("velocityLimitFromUrdf"))},
            velocityLimit{boost::get<Eigen::VectorXd>(options.at("velocityLimit"))}
            {
            }
        };

        struct DynamicsOptions
        {
            const double inertiaBodiesBiasStd;
            const double massBodiesBiasStd;
            const double centerOfMassPositionBodiesBiasStd;
            const double relativePositionBodiesBiasStd;
            const bool enableFlexibleModel;
            const FlexibilityConfig flexibilityConfig;

            DynamicsOptions(const GenericConfig & options) :
            inertiaBodiesBiasStd{boost::get<double>(options.at("inertiaBodiesBiasStd"))},
            massBodiesBiasStd{boost::get<double>(options.at("massBodiesBiasStd"))},
            centerOfMassPositionBodiesBiasStd{
                boost::get<double>(options.at("centerOfMassPositionBodiesBiasStd"))},
            relativePositionBodiesBiasStd{
                boost::get<double>(options.at("relativePositionBodiesBiasStd"))},
            enableFlexibleModel{boost::get<bool>(options.at("enableFlexibleModel"))},
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
        DISABLE_COPY(Model)

    public:
        explicit Model() noexcept;
        virtual ~Model() = default;

        void initialize(const pinocchio::Model & pinocchioModel,
                        const pinocchio::GeometryModel & collisionModel,
                        const pinocchio::GeometryModel & visualModel);
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
        void removeFrame(const std::string & frameName);
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

        /// \brief Pointer to the constraint referenced by constraintName
        ///
        /// \param[in] constraintName Name of the constraint to get.
        std::shared_ptr<AbstractConstraintBase> getConstraint(const std::string & constraintName);

        std::weak_ptr<const AbstractConstraintBase> getConstraint(
            const std::string & constraintName) const;

        // Copy on purpose
        ConstraintTree getConstraints();

        bool existConstraint(const std::string & constraintName) const;

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

        // Copy on purpose
        void setOptions(GenericConfig modelOptions);
        GenericConfig getOptions() const noexcept;

        /// \remark This method are not intended to be called manually. The Engine is taking care
        ///         of it.
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

        const std::vector<std::string> & getRigidJointNames() const;
        const std::vector<pinocchio::JointIndex> & getRigidJointIndices() const;
        const std::vector<Eigen::Index> & getRigidJointPositionIndices() const;
        const std::vector<Eigen::Index> & getRigidJointVelocityIndices() const;
        const std::vector<std::string> & getFlexibleJointNames() const;
        const std::vector<pinocchio::JointIndex> & getFlexibleJointIndices() const;

        const Eigen::VectorXd & getPositionLimitMin() const;
        const Eigen::VectorXd & getPositionLimitMax() const;
        const Eigen::VectorXd & getVelocityLimit() const;

        const std::vector<std::string> & getLogPositionFieldnames() const;
        const std::vector<std::string> & getLogVelocityFieldnames() const;
        const std::vector<std::string> & getLogAccelerationFieldnames() const;
        const std::vector<std::string> & getLogForceExternalFieldnames() const;

        void getFlexiblePositionFromRigid(const Eigen::VectorXd & qRigid,
                                          Eigen::VectorXd & qFlex) const;
        void getRigidPositionFromFlexible(const Eigen::VectorXd & qFlex,
                                          Eigen::VectorXd & qRigid) const;
        void getFlexibleVelocityFromRigid(const Eigen::VectorXd & vRigid,
                                          Eigen::VectorXd & vFlex) const;
        void getRigidVelocityFromFlexible(const Eigen::VectorXd & vFlex,
                                          Eigen::VectorXd & vRigid) const;

    protected:
        void generateModelFlexible();
        void generateModelBiased(const uniform_random_bit_generator_ref<uint32_t> & g);

        void addFrame(const std::string & frameName,
                      const std::string & parentBodyName,
                      const pinocchio::SE3 & framePlacement,
                      const pinocchio::FrameType & frameType);
        void removeFrames(const std::vector<std::string> & frameNames);

        void addConstraint(const std::string & constraintName,
                           const std::shared_ptr<AbstractConstraintBase> & constraint,
                           ConstraintNodeType node);
        void addConstraints(const ConstraintMap & constraintMap, ConstraintNodeType node);
        void removeConstraint(const std::string & constraintName, ConstraintNodeType node);
        void removeConstraints(const std::vector<std::string> & constraintNames,
                               ConstraintNodeType node);

        void refreshGeometryProxies();
        void refreshContactProxies();
        /// \brief Refresh the proxies of the kinematics constraints.
        void refreshConstraintProxies();
        virtual void refreshProxies();

    public:
        pinocchio::Model pinocchioModelOrig_{};
        pinocchio::Model pinocchioModel_{};
        pinocchio::GeometryModel collisionModelOrig_{};
        pinocchio::GeometryModel collisionModel_{};
        pinocchio::GeometryModel visualModelOrig_{};
        pinocchio::GeometryModel visualModel_{};
        mutable pinocchio::Data pinocchioDataOrig_{};
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
        /// \brief Name of the actual joints of the robot, not taking into account the freeflyer.
        std::vector<std::string> rigidJointNames_{};
        /// \brief Index of the actual joints in the pinocchio robot.
        std::vector<pinocchio::JointIndex> rigidJointIndices_{};
        /// \brief All the indices of the actual joints in the configuration vector of the robot
        ///        (ie including all the degrees of freedom).
        std::vector<Eigen::Index> rigidJointPositionIndices_{};
        /// \brief All the indices of the actual joints in the velocity vector of the robot (ie
        ///        including all the degrees of freedom).
        std::vector<Eigen::Index> rigidJointVelocityIndices_{};
        /// \brief Name of the flexibility joints of the robot regardless of whether the
        ///        flexibilities are enabled.
        std::vector<std::string> flexibleJointNames_{};
        /// \brief Index of the flexibility joints in the pinocchio robot regardless of whether the
        ///        flexibilities are enabled.
        std::vector<pinocchio::JointIndex> flexibleJointIndices_{};

        /// \brief Store constraints.
        ConstraintTree constraints_{};

        /// \brief Upper position limit of the whole configuration vector (INF for non-physical
        ///        joints, ie flexibility joints and freeflyer, if any).
        Eigen::VectorXd positionLimitMin_{};
        /// \brief Lower position limit of the whole configuration vector.
        Eigen::VectorXd positionLimitMax_{};
        /// \brief Maximum absolute velocity of the whole velocity vector.
        Eigen::VectorXd velocityLimit_{};

        /// \brief Fieldnames of the elements in the configuration vector of the model.
        std::vector<std::string> logPositionFieldnames_{};
        /// \brief Fieldnames of the elements in the velocity vector of the model.
        std::vector<std::string> logVelocityFieldnames_{};
        /// \brief Fieldnames of the elements in the acceleration vector of the model.
        std::vector<std::string> logAccelerationFieldnames_{};
        /// \brief Concatenated fieldnames of the external force applied at each joint of the
        ///        model, 'universe' excluded.
        std::vector<std::string> logForceExternalFieldnames_{};

    private:
        pinocchio::Model pncModelFlexibleOrig_{};
        /// \brief Vector of joints acceleration corresponding to a copy of data.a - temporary
        ///        buffer for computing constraints.
        MotionVector jointSpatialAccelerations_{};

        Eigen::Index nq_{0};
        Eigen::Index nv_{0};
        Eigen::Index nx_{0};
    };
}

#endif  // end of JIMINY_MODEL_H
