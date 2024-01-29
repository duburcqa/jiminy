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

    using constraintsMap_t = static_map_t<std::string, std::shared_ptr<AbstractConstraintBase>>;

    enum class JIMINY_DLLAPI constraintsHolderType_t : uint8_t
    {
        BOUNDS_JOINTS = 0,
        CONTACT_FRAMES = 1,
        COLLISION_BODIES = 2,
        USER = 3
    };

    const std::array<constraintsHolderType_t, 4> constraintsHolderTypesAll{
        {constraintsHolderType_t::BOUNDS_JOINTS,
         constraintsHolderType_t::CONTACT_FRAMES,
         constraintsHolderType_t::COLLISION_BODIES,
         constraintsHolderType_t::USER}};

    struct JIMINY_DLLAPI constraintsHolder_t
    {
    public:
        void clear() noexcept;

        std::pair<constraintsMap_t *, constraintsMap_t::iterator> find(
            const std::string & key, constraintsHolderType_t holderType);

        bool exist(const std::string & key) const;
        bool exist(const std::string & key, constraintsHolderType_t holderType) const;

        std::shared_ptr<AbstractConstraintBase> get(const std::string & key);
        std::shared_ptr<AbstractConstraintBase> get(const std::string & key,
                                                    constraintsHolderType_t holderType);

        void insert(const constraintsMap_t & constraintsMap, constraintsHolderType_t holderType);

        constraintsMap_t::iterator erase(const std::string & key,
                                         constraintsHolderType_t holderType);

        template<typename Function>
        void foreach(constraintsHolderType_t holderType, Function && lambda)
        {
            if (holderType == constraintsHolderType_t::COLLISION_BODIES)
            {
                for (auto & constraintsMap : collisionBodies)
                {
                    for (auto & constraintItem : constraintsMap)
                    {
                        std::invoke(
                            std::forward<Function>(lambda), constraintItem.second, holderType);
                    }
                }
            }
            else
            {
                constraintsMap_t * constraintsMapPtr;
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
                for (auto & constraintItem : *constraintsMapPtr)
                {
                    std::invoke(std::forward<Function>(lambda), constraintItem.second, holderType);
                }
            }
        }

        template<typename Function, std::size_t N>
        void foreach(std::array<constraintsHolderType_t, N> constraintsHolderTypes,
                     Function && lambda)
        {
            for (constraintsHolderType_t holderType : constraintsHolderTypes)
            {
                foreach(holderType, std::forward<Function>(lambda));
            }
        }

        template<typename Function>
        void foreach(Function && lambda)
        {
            foreach(constraintsHolderTypesAll, std::forward<Function>(lambda));
        }

    public:
        /// \brief Constraints registered by the engine to handle joint bounds.
        constraintsMap_t boundJoints{};
        /// \brief Constraints registered by the engine to handle contact frames.
        constraintsMap_t contactFrames{};
        /// \brief Constraints registered by the engine to handle collision bounds.
        std::vector<constraintsMap_t> collisionBodies{};
        /// \brief Constraints explicitly registered by user.
        constraintsMap_t registered{};
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
            config["maxContactPointsPerBody"] = 5U;

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

        struct jointOptions_t
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

            jointOptions_t(const GenericConfig & options) :
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

        struct dynamicsOptions_t
        {
            const double inertiaBodiesBiasStd;
            const double massBodiesBiasStd;
            const double centerOfMassPositionBodiesBiasStd;
            const double relativePositionBodiesBiasStd;
            const bool enableFlexibleModel;
            const FlexibilityConfig flexibilityConfig;

            dynamicsOptions_t(const GenericConfig & options) :
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

        struct collisionOptions_t
        {
            const uint32_t maxContactPointsPerBody;

            collisionOptions_t(const GenericConfig & options) :
            maxContactPointsPerBody{boost::get<uint32_t>(options.at("maxContactPointsPerBody"))}
            {
            }
        };

        struct modelOptions_t
        {
            const dynamicsOptions_t dynamics;
            const jointOptions_t joints;
            const collisionOptions_t collisions;

            modelOptions_t(const GenericConfig & options) :
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

        hresult_t initialize(const pinocchio::Model & pncModel,
                             const pinocchio::GeometryModel & collisionModel,
                             const pinocchio::GeometryModel & visualModel);
        hresult_t initialize(const std::string & urdfPath,
                             bool hasFreeflyer = true,
                             const std::vector<std::string> & meshPackageDirs = {},
                             bool loadVisualMeshes = false);

        /// \brief Add a frame in the kinematic tree, attached to the frame of an existing body.
        ///
        /// \param[in] frameName Name of the frame to be added.
        /// \param[in] parentBodyName Name of the parent body frame.
        /// \param[in] framePlacement Frame placement wrt the parent body frame.
        hresult_t addFrame(const std::string & frameName,
                           const std::string & parentBodyName,
                           const pinocchio::SE3 & framePlacement);
        hresult_t removeFrame(const std::string & frameName);
        hresult_t addCollisionBodies(const std::vector<std::string> & bodyNames,
                                     bool ignoreMeshes = false);
        hresult_t removeCollisionBodies(
            std::vector<std::string> frameNames = {});  // Copy on purpose
        hresult_t addContactPoints(const std::vector<std::string> & frameNames);
        hresult_t removeContactPoints(const std::vector<std::string> & frameNames = {});

        /// \brief Add a kinematic constraint to the robot.
        ///
        /// \param[in] constraintName Unique name identifying the kinematic constraint.
        /// \param[in] constraint Constraint to add.
        hresult_t addConstraint(const std::string & constraintName,
                                const std::shared_ptr<AbstractConstraintBase> & constraint);

        /// \brief Remove a kinematic constraint form the system.
        ///
        /// \param[in] constraintName Unique name identifying the kinematic constraint.
        hresult_t removeConstraint(const std::string & constraintName);

        /// \brief Pointer to the constraint referenced by constraintName
        ///
        /// \param[in] constraintName Name of the constraint to get.
        ///
        /// \return ERROR_BAD_INPUT if constraintName does not exist, SUCCESS otherwise.
        hresult_t getConstraint(const std::string & constraintName,
                                std::shared_ptr<AbstractConstraintBase> & constraint);

        hresult_t getConstraint(const std::string & constraintName,
                                std::weak_ptr<const AbstractConstraintBase> & constraint) const;

        // Copy on purpose
        constraintsHolder_t getConstraints();

        bool existConstraint(const std::string & constraintName) const;

        hresult_t resetConstraints(const Eigen::VectorXd & q, const Eigen::VectorXd & v);

        /// \brief Compute jacobian and drift associated to all the constraints.
        ///
        /// \details The results are accessible using getConstraintsJacobian and
        ///          getConstraintsDrift.
        /// \note It is assumed frames forward kinematics has already been called.
        ///
        /// \param[in] q Joint position.
        /// \param[in] v Joint velocity.
        void computeConstraints(const Eigen::VectorXd & q, const Eigen::VectorXd & v);

        /// \brief Returns true if at least one constraint is active on the robot.
        bool hasConstraints() const;

        // Copy on purpose
        hresult_t setOptions(GenericConfig modelOptions);
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

        const std::vector<std::string> & getCollisionBodiesNames() const;
        const std::vector<std::string> & getContactFramesNames() const;
        const std::vector<pinocchio::FrameIndex> & getCollisionBodiesIdx() const;
        const std::vector<std::vector<pinocchio::PairIndex>> & getCollisionPairsIdx() const;
        const std::vector<pinocchio::FrameIndex> & getContactFramesIdx() const;
        const std::vector<std::string> & getRigidJointsNames() const;
        const std::vector<pinocchio::JointIndex> & getRigidJointsModelIdx() const;
        const std::vector<Eigen::Index> & getRigidJointsPositionIdx() const;
        const std::vector<Eigen::Index> & getRigidJointsVelocityIdx() const;
        const std::vector<std::string> & getFlexibleJointsNames() const;
        const std::vector<pinocchio::JointIndex> & getFlexibleJointsModelIdx() const;

        const Eigen::VectorXd & getPositionLimitMin() const;
        const Eigen::VectorXd & getPositionLimitMax() const;
        const Eigen::VectorXd & getVelocityLimit() const;

        const std::vector<std::string> & getLogFieldnamesPosition() const;
        const std::vector<std::string> & getLogFieldnamesVelocity() const;
        const std::vector<std::string> & getLogFieldnamesAcceleration() const;
        const std::vector<std::string> & getLogFieldnamesForceExternal() const;

        hresult_t getFlexibleConfigurationFromRigid(const Eigen::VectorXd & qRigid,
                                                    Eigen::VectorXd & qFlex) const;
        hresult_t getRigidConfigurationFromFlexible(const Eigen::VectorXd & qFlex,
                                                    Eigen::VectorXd & qRigid) const;
        hresult_t getFlexibleVelocityFromRigid(const Eigen::VectorXd & vRigid,
                                               Eigen::VectorXd & vFlex) const;
        hresult_t getRigidVelocityFromFlexible(const Eigen::VectorXd & vFlex,
                                               Eigen::VectorXd & vRigid) const;

    protected:
        hresult_t generateModelFlexible();
        hresult_t generateModelBiased(const uniform_random_bit_generator_ref<uint32_t> & g);

        hresult_t addFrame(const std::string & frameName,
                           const std::string & parentBodyName,
                           const pinocchio::SE3 & framePlacement,
                           const pinocchio::FrameType & frameType);
        hresult_t removeFrames(const std::vector<std::string> & frameNames);

        hresult_t addConstraints(const constraintsMap_t & constraintsMap,
                                 constraintsHolderType_t holderType);
        hresult_t addConstraint(const std::string & constraintName,
                                const std::shared_ptr<AbstractConstraintBase> & constraint,
                                constraintsHolderType_t holderType);
        hresult_t removeConstraint(const std::string & constraintName,
                                   constraintsHolderType_t holderType);
        hresult_t removeConstraints(const std::vector<std::string> & constraintsNames,
                                    constraintsHolderType_t holderType);

        hresult_t refreshGeometryProxies();
        hresult_t refreshContactsProxies();
        /// \brief Refresh the proxies of the kinematics constraints.
        hresult_t refreshConstraintsProxies();
        virtual hresult_t refreshProxies();

    public:
        pinocchio::Model pncModelOrig_{};
        pinocchio::Model pncModel_{};
        pinocchio::GeometryModel collisionModelOrig_{};
        pinocchio::GeometryModel collisionModel_{};
        pinocchio::GeometryModel visualModelOrig_{};
        pinocchio::GeometryModel visualModel_{};
        pinocchio::Data pncDataOrig_{};
        mutable pinocchio::Data pncData_{};
        mutable pinocchio::GeometryData collisionData_{};
        mutable pinocchio::GeometryData visualData_{};
        std::unique_ptr<const modelOptions_t> mdlOptions_{nullptr};
        /// \brief Buffer storing the contact forces.
        ForceVector contactForces_{};

    protected:
        bool isInitialized_{false};
        std::string urdfPath_{};
        std::string urdfData_{};
        std::vector<std::string> meshPackageDirs_{};
        bool hasFreeflyer_{false};
        GenericConfig mdlOptionsHolder_{};

        /// \brief Name of the collision bodies of the robot.
        std::vector<std::string> collisionBodiesNames_{};
        /// \brief Name of the contact frames of the robot.
        std::vector<std::string> contactFramesNames_{};
        /// \brief Indices of the collision bodies in the frame list of the robot.
        std::vector<pinocchio::FrameIndex> collisionBodiesIdx_{};
        /// \brief Indices of the collision pairs associated with each collision body.
        std::vector<std::vector<pinocchio::PairIndex>> collisionPairsIdx_{};
        /// \brief Indices of the contact frames in the frame list of the robot.
        std::vector<pinocchio::FrameIndex> contactFramesIdx_{};
        /// \brief Name of the actual joints of the robot, not taking into account the freeflyer.
        std::vector<std::string> rigidJointsNames_{};
        /// \brief Index of the actual joints in the pinocchio robot.
        std::vector<pinocchio::JointIndex> rigidJointsModelIdx_{};
        /// \brief All the indices of the actual joints in the configuration vector of the robot
        ///        (ie including all the degrees of freedom).
        std::vector<Eigen::Index> rigidJointsPositionIdx_{};
        /// \brief All the indices of the actual joints in the velocity vector of the robot (ie
        ///        including all the degrees of freedom).
        std::vector<Eigen::Index> rigidJointsVelocityIdx_{};
        /// \brief Name of the flexibility joints of the robot regardless of whether the
        ///        flexibilities are enabled.
        std::vector<std::string> flexibleJointsNames_{};
        /// \brief Index of the flexibility joints in the pinocchio robot regardless of whether the
        ///        flexibilities are enabled.
        std::vector<pinocchio::JointIndex> flexibleJointsModelIdx_{};

        /// \brief Store constraints.
        constraintsHolder_t constraintsHolder_{};

        /// \brief Upper position limit of the whole configuration vector (INF for non-physical
        ///        joints, ie flexibility joints and freeflyer, if any).
        Eigen::VectorXd positionLimitMin_{};
        /// \brief Lower position limit of the whole configuration vector.
        Eigen::VectorXd positionLimitMax_{};
        /// \brief Maximum absolute velocity of the whole velocity vector.
        Eigen::VectorXd velocityLimit_{};

        /// \brief Fieldnames of the elements in the configuration vector of the model.
        std::vector<std::string> logFieldnamesPosition_{};
        /// \brief Fieldnames of the elements in the velocity vector of the model.
        std::vector<std::string> logFieldnamesVelocity_{};
        /// \brief Fieldnames of the elements in the acceleration vector of the model.
        std::vector<std::string> logFieldnamesAcceleration_{};
        /// \brief Concatenated fieldnames of the external force applied at each joint of the
        ///        model, 'universe' excluded.
        std::vector<std::string> logFieldnamesForceExternal_{};

    private:
        pinocchio::Model pncModelFlexibleOrig_{};
        /// \brief Vector of joints acceleration corresponding to a copy of data.a - temporary
        ///        buffer for computing constraints.
        MotionVector jointsAcceleration_{};

        Eigen::Index nq_{0};
        Eigen::Index nv_{0};
        Eigen::Index nx_{0};
    };
}

#endif  // end of JIMINY_MODEL_H
