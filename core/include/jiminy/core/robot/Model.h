#ifndef JIMINY_MODEL_H
#define JIMINY_MODEL_H

#include "pinocchio/spatial/fwd.hpp"         // `pinocchio::SE3`
#include "pinocchio/multibody/model.hpp"     // `pinocchio::Model`
#include "pinocchio/multibody/data.hpp"      // `pinocchio::Data`
#include "pinocchio/multibody/geometry.hpp"  // `pinocchio::GeometryModel`, `pinocchio::GeometryData`
#include "pinocchio/multibody/frame.hpp"     // `pinocchio::FrameType` (C-style enum cannot be forward declared)

#include "jiminy/core/Macros.h"
#include "jiminy/core/Types.h"


namespace jiminy
{
    class AbstractConstraintBase;
    class FixedFrameConstraint;
    class JointConstraint;

    template<typename DerivedConstraint>
    using constraintsMapTpl_t = static_map_t<std::string, std::shared_ptr<DerivedConstraint> >;
    using constraintsMap_t = constraintsMapTpl_t<AbstractConstraintBase>;

    enum class constraintsHolderType_t : uint8_t
    {
        BOUNDS_JOINTS = 0,
        CONTACT_FRAMES = 1,
        COLLISION_BODIES = 2,
        USER = 3
    };

    std::array<constraintsHolderType_t, 4> const constraintsHolderTypeRange = {{
        constraintsHolderType_t::BOUNDS_JOINTS,
        constraintsHolderType_t::CONTACT_FRAMES,
        constraintsHolderType_t::COLLISION_BODIES,
        constraintsHolderType_t::USER
    }};

    struct constraintsHolder_t
    {
    public:
        // Disable the copy of the class
        constraintsHolder_t(constraintsHolder_t const & constraintsHolder) = default;
        constraintsHolder_t & operator = (constraintsHolder_t const & other) = default;

        constraintsHolder_t(void);
        virtual ~constraintsHolder_t(void) = default;

        void clear(void);

        std::tuple<constraintsMap_t *, constraintsMap_t::iterator>
        find(std::string const & key,
             constraintsHolderType_t const & holderType);

        bool_t exist(std::string const & key) const;
        bool_t exist(std::string const & key,
                     constraintsHolderType_t const & holderType) const;

        std::shared_ptr<AbstractConstraintBase> get(std::string const & key);
        std::shared_ptr<AbstractConstraintBase> get(std::string const & key,
                                                constraintsHolderType_t const & holderType);

        void insert(constraintsMap_t const & constraintsMap,
                    constraintsHolderType_t const & holderType);

        constraintsMap_t::iterator erase(std::string const & key,
                                         constraintsHolderType_t const & holderType);

        template<typename Function>
        void foreach(constraintsHolderType_t const & holderType,
                     Function && lambda)
        {
            if (holderType == constraintsHolderType_t::COLLISION_BODIES)
            {
                for (auto & constraintsMap : collisionBodies)
                {
                    for (auto & constraintItem : constraintsMap)
                    {
                        std::forward<Function>(lambda)(constraintItem.second, holderType);
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
                    std::forward<Function>(lambda)(constraintItem.second, holderType);
                }
            }
        }

        template<typename Function>
        void foreach(Function && lambda)
        {
            for (constraintsHolderType_t const & holderType : constraintsHolderTypeRange)
            {
                foreach(holderType, std::forward<Function>(lambda));
            }
        }

    public:
        constraintsMap_t boundJoints;                   ///< Store internal constraints related to joint bounds
        constraintsMap_t contactFrames;                 ///< Store internal constraints related to contact frames
        std::vector<constraintsMap_t> collisionBodies;  ///< Store internal constraints related to collision bounds
        constraintsMap_t registered;                    ///< Store internal constraints registered by user
    };

    class Model: public std::enable_shared_from_this<Model>
    {
    public:
        virtual configHolder_t getDefaultJointOptions()
        {
            configHolder_t config;
            config["enablePositionLimit"] = true;
            config["positionLimitFromUrdf"] = true;
            config["positionLimitMin"] = vectorN_t();
            config["positionLimitMax"] = vectorN_t();
            config["enableVelocityLimit"] = true;
            config["velocityLimitFromUrdf"] = true;
            config["velocityLimit"] = vectorN_t();

            return config;
        };

        virtual configHolder_t getDefaultDynamicsOptions()
        {
            // Add extra options or update default values
            configHolder_t config;
            config["inertiaBodiesBiasStd"] = 0.0;
            config["massBodiesBiasStd"] = 0.0;
            config["centerOfMassPositionBodiesBiasStd"] = 0.0;
            config["relativePositionBodiesBiasStd"] = 0.0;
            config["enableFlexibleModel"] = true;
            config["flexibilityConfig"] = flexibilityConfig_t();

            return config;
        };

        virtual configHolder_t getDefaultCollisionOptions()
        {
            // Add extra options or update default values
            configHolder_t config;
            config["maxContactPointsPerBody"] = 5U;  // Max number of contact points per collision pairs

            return config;
        };

        virtual configHolder_t getDefaultModelOptions()
        {
            configHolder_t config;
            config["dynamics"] = getDefaultDynamicsOptions();
            config["joints"] = getDefaultJointOptions();
            config["collisions"] = getDefaultCollisionOptions();

            return config;
        };

        struct jointOptions_t
        {
            bool_t    const enablePositionLimit;
            bool_t    const positionLimitFromUrdf;
            vectorN_t const positionLimitMin;         ///< Min position limit of all the actual joints, namely without freeflyer and flexible joints if any
            vectorN_t const positionLimitMax;
            bool_t    const enableVelocityLimit;
            bool_t    const velocityLimitFromUrdf;
            vectorN_t const velocityLimit;

            jointOptions_t(configHolder_t const & options) :
            enablePositionLimit(boost::get<bool_t>(options.at("enablePositionLimit"))),
            positionLimitFromUrdf(boost::get<bool_t>(options.at("positionLimitFromUrdf"))),
            positionLimitMin(boost::get<vectorN_t>(options.at("positionLimitMin"))),
            positionLimitMax(boost::get<vectorN_t>(options.at("positionLimitMax"))),
            enableVelocityLimit(boost::get<bool_t>(options.at("enableVelocityLimit"))),
            velocityLimitFromUrdf(boost::get<bool_t>(options.at("velocityLimitFromUrdf"))),
            velocityLimit(boost::get<vectorN_t>(options.at("velocityLimit")))
            {
                // Empty.
            }
        };

        struct dynamicsOptions_t
        {
            float64_t           const inertiaBodiesBiasStd;
            float64_t           const massBodiesBiasStd;
            float64_t           const centerOfMassPositionBodiesBiasStd;
            float64_t           const relativePositionBodiesBiasStd;
            bool_t              const enableFlexibleModel;
            flexibilityConfig_t const flexibilityConfig;

            dynamicsOptions_t(configHolder_t const & options) :
            inertiaBodiesBiasStd(boost::get<float64_t>(options.at("inertiaBodiesBiasStd"))),
            massBodiesBiasStd(boost::get<float64_t>(options.at("massBodiesBiasStd"))),
            centerOfMassPositionBodiesBiasStd(boost::get<float64_t>(options.at("centerOfMassPositionBodiesBiasStd"))),
            relativePositionBodiesBiasStd(boost::get<float64_t>(options.at("relativePositionBodiesBiasStd"))),
            enableFlexibleModel(boost::get<bool_t>(options.at("enableFlexibleModel"))),
            flexibilityConfig(boost::get<flexibilityConfig_t>(options.at("flexibilityConfig")))
            {
                // Empty.
            }
        };

        struct collisionOptions_t
        {
            uint32_t const maxContactPointsPerBody;

            collisionOptions_t(configHolder_t const & options) :
            maxContactPointsPerBody(boost::get<uint32_t>(options.at("maxContactPointsPerBody")))
            {
                // Empty.
            }
        };

        struct modelOptions_t
        {
            dynamicsOptions_t const dynamics;
            jointOptions_t const joints;
            collisionOptions_t const collisions;

            modelOptions_t(configHolder_t const & options) :
            dynamics(boost::get<configHolder_t>(options.at("dynamics"))),
            joints(boost::get<configHolder_t>(options.at("joints"))),
            collisions(boost::get<configHolder_t>(options.at("collisions")))
            {
                // Empty.
            }
        };

    public:
        // Disable the copy of the class
        Model(Model const & robot) = delete;
        Model & operator = (Model const & other) = delete;

    public:
        Model(void);
        virtual ~Model(void) = default;

        auto shared_from_this() { return shared_from(this); }
        auto shared_from_this() const { return shared_from(this); }

        hresult_t initialize(std::string              const & urdfPath,
                             bool_t                   const & hasFreeflyer = true,
                             std::vector<std::string> const & meshPackageDirs = {});

        /// \brief Add a frame in the kinematic tree, attached to the frame of an existing body.
        ///
        /// \param[in] frameName        Name of the frame to be added
        /// \param[in] parentBodyName   Name of the parent body frame
        /// \param[in] framePlacement   Frame placement wrt the parent body frame
        hresult_t addFrame(std::string    const & frameName,
                           std::string    const & parentBodyName,
                           pinocchio::SE3 const & framePlacement);
        hresult_t removeFrame(std::string const & frameName);
        hresult_t addCollisionBodies(std::vector<std::string> const & bodyNames,
                                     bool_t const & ignoreMeshes = false);
        hresult_t removeCollisionBodies(std::vector<std::string> frameNames = {});  // Copy on purpose
        hresult_t addContactPoints(std::vector<std::string> const & frameNames);
        hresult_t removeContactPoints(std::vector<std::string> const & frameNames = {});

        /// \brief Add a kinematic constraint to the robot.
        ///
        /// \param[in] constraintName Unique name identifying the kinematic constraint.
        /// \param[in] constraint Constraint to add.
        hresult_t addConstraint(std::string const & constraintName,
                                std::shared_ptr<AbstractConstraintBase> const & constraint);

        /// \brief Remove a kinematic constraint form the system.
        ///
        /// \param[in] constraintName Unique name identifying the kinematic constraint.
        hresult_t removeConstraint(std::string const & constraintName);

        /// \brief Get a pointer to the constraint referenced by constraintName
        ///
        /// \param[in] constraintName Name of the constraint to get.
        /// \return ERROR_BAD_INPUT if constraintName does not exist, SUCCESS otherwise.
        hresult_t getConstraint(std::string const & constraintName,
                                std::shared_ptr<AbstractConstraintBase> & constraint);

        hresult_t getConstraint(std::string const & constraintName,
                                std::weak_ptr<AbstractConstraintBase const> & constraint) const;

        constraintsHolder_t getConstraints(void);  // Copy on purpose

        bool_t existConstraint(std::string const & constraintName) const;

        hresult_t resetConstraints(vectorN_t const & q,
                                   vectorN_t const & v);

        /// \brief Compute jacobian and drift associated to all the constraints.
        ///
        /// \details The results are accessible using getConstraintsJacobian and
        ///          getConstraintsDrift.
        /// \note  It is assumed frames forward kinematics has already been called.
        ///
        /// \param[in] q    Joint position.
        /// \param[in] v    Joint velocity.
        void computeConstraints(vectorN_t const & q,
                                vectorN_t const & v);

        /// \brief Get jacobian of the constraints.
        constMatrixBlock_t getConstraintsJacobian(void) const;

        /// \brief Get drift of the constraints.
        constVectorBlock_t getConstraintsDrift(void) const;

        /// \brief Returns true if at least one constraint is active on the robot.
        bool_t hasConstraint(void) const;

        hresult_t setOptions(configHolder_t modelOptions);  // Make a copy
        configHolder_t getOptions(void) const;

        /// This method are not intended to be called manually. The Engine is taking care of it.
        virtual void reset(void);

        bool_t const & getIsInitialized(void) const;
        std::string const & getUrdfPath(void) const;
        std::vector<std::string> const & getMeshPackageDirs(void) const;
        bool_t const & getHasFreeflyer(void) const;
        // Getters without 'get' prefix for consistency with pinocchio C++ API
        int32_t const & nq(void) const;
        int32_t const & nv(void) const;
        int32_t const & nx(void) const;

        std::vector<std::string> const & getCollisionBodiesNames(void) const;
        std::vector<std::string> const & getContactFramesNames(void) const;
        std::vector<int32_t> const & getCollisionBodiesIdx(void) const;
        std::vector<std::vector<int32_t> > const & getCollisionPairsIdx(void) const;
        std::vector<int32_t> const & getContactFramesIdx(void) const;
        std::vector<std::string> const & getRigidJointsNames(void) const;
        std::vector<int32_t> const & getRigidJointsModelIdx(void) const;
        std::vector<int32_t> const & getRigidJointsPositionIdx(void) const;
        std::vector<int32_t> const & getRigidJointsVelocityIdx(void) const;
        std::vector<std::string> const & getFlexibleJointsNames(void) const;
        std::vector<int32_t> const & getFlexibleJointsModelIdx(void) const;

        vectorN_t const & getPositionLimitMin(void) const;
        vectorN_t const & getPositionLimitMax(void) const;
        vectorN_t const & getVelocityLimit(void) const;

        std::vector<std::string> const & getPositionFieldnames(void) const;
        std::vector<std::string> const & getVelocityFieldnames(void) const;
        std::vector<std::string> const & getAccelerationFieldnames(void) const;

        hresult_t getFlexibleConfigurationFromRigid(vectorN_t const & qRigid,
                                                    vectorN_t       & qFlex) const;
        hresult_t getRigidConfigurationFromFlexible(vectorN_t const & qFlex,
                                                    vectorN_t       & qRigid) const;
        hresult_t getFlexibleVelocityFromRigid(vectorN_t const & vRigid,
                                               vectorN_t       & vFlex) const;
        hresult_t getRigidVelocityFromFlexible(vectorN_t const & vFlex,
                                               vectorN_t       & vRigid) const;

    protected:
        hresult_t loadUrdfModel(std::string              const & urdfPath,
                                bool_t                   const & hasFreeflyer,
                                std::vector<std::string>         meshPackageDirs);
        hresult_t generateModelFlexible(void);
        hresult_t generateModelBiased(void);

        hresult_t addFrame(std::string          const & frameName,
                           std::string          const & parentBodyName,
                           pinocchio::SE3       const & framePlacement,
                           pinocchio::FrameType const & frameType);
        hresult_t removeFrames(std::vector<std::string> const & frameNames);

        hresult_t addConstraints(constraintsMap_t const & constraintsMap,
                                 constraintsHolderType_t const & holderType);
        hresult_t addConstraint(std::string const & constraintName,
                                std::shared_ptr<AbstractConstraintBase> const & constraint,
                                constraintsHolderType_t const & holderType);
        hresult_t removeConstraint(std::string const & constraintName,
                                   constraintsHolderType_t const & holderType);
        hresult_t removeConstraints(std::vector<std::string> const & constraintsNames,
                                    constraintsHolderType_t const & holderType);

        hresult_t refreshCollisionsProxies(void);
        hresult_t refreshContactsProxies(void);
        /// \brief Refresh the proxies of the kinematics constraints.
        hresult_t refreshConstraintsProxies(void);
        virtual hresult_t refreshProxies(void);

    public:
        pinocchio::Model pncModel_;
        mutable pinocchio::Data pncData_;
        pinocchio::GeometryModel pncGeometryModel_;
        mutable std::unique_ptr<pinocchio::GeometryData> pncGeometryData_;  // Using smart ptr to avoid having to initialize it with an empty GeometryModel, which causes Pinocchio segfault at least up to v2.5.6
        pinocchio::Model pncModelRigidOrig_;
        pinocchio::Data pncDataRigidOrig_;
        std::unique_ptr<modelOptions_t const> mdlOptions_;
        forceVector_t contactForces_;                       ///< Buffer storing the contact forces

    protected:
        bool_t isInitialized_;
        std::string urdfPath_;
        std::vector<std::string> meshPackageDirs_;
        bool_t hasFreeflyer_;
        configHolder_t mdlOptionsHolder_;

        std::vector<std::string> collisionBodiesNames_;         ///< Name of the collision bodies of the robot
        std::vector<std::string> contactFramesNames_;           ///< Name of the contact frames of the robot
        std::vector<int32_t> collisionBodiesIdx_;               ///< Indices of the collision bodies in the frame list of the robot
        std::vector<std::vector<int32_t> > collisionPairsIdx_;  ///< Indices of the collision pairs associated with each collision body
        std::vector<int32_t> contactFramesIdx_;                 ///< Indices of the contact frames in the frame list of the robot
        std::vector<std::string> rigidJointsNames_;             ///< Name of the actual joints of the robot, not taking into account the freeflyer
        std::vector<int32_t> rigidJointsModelIdx_;              ///< Index of the actual joints in the pinocchio robot
        std::vector<int32_t> rigidJointsPositionIdx_;           ///< All the indices of the actual joints in the configuration vector of the robot (ie including all the degrees of freedom)
        std::vector<int32_t> rigidJointsVelocityIdx_;           ///< All the indices of the actual joints in the velocity vector of the robot (ie including all the degrees of freedom)
        std::vector<std::string> flexibleJointsNames_;          ///< Name of the flexibility joints of the robot regardless of whether the flexibilities are enable
        std::vector<int32_t> flexibleJointsModelIdx_;           ///< Index of the flexibility joints in the pinocchio robot regardless of whether the flexibilities are enable

        constraintsHolder_t constraintsHolder_;                 ///< Store constraints
        uint32_t constraintsMask_;                              ///< Mask used to filter out disable constraints from full jacobian and drift
        matrixN_t constraintsJacobian_;                         ///< Matrix holding the jacobian of the constraints
        vectorN_t constraintsDrift_;                            ///< Vector holding the drift of the constraints

        vectorN_t positionLimitMin_;                            ///< Upper position limit of the whole configuration vector (INF for non-physical joints, ie flexibility joints and freeflyer, if any)
        vectorN_t positionLimitMax_;                            ///< Lower position limit of the whole configuration vector (INF for non-physical joints, ie flexibility joints and freeflyer, if any)
        vectorN_t velocityLimit_;                               ///< Maximum absolute velocity of the whole velocity vector (INF for non-physical joints, ie flexibility joints and freeflyer, if any)

        std::vector<std::string> positionFieldnames_;           ///< Fieldnames of the elements in the configuration vector of the rigid robot
        std::vector<std::string> velocityFieldnames_;           ///< Fieldnames of the elements in the velocity vector of the rigid robot
        std::vector<std::string> accelerationFieldnames_;       ///< Fieldnames of the elements in the acceleration vector of the rigid robot

    private:
        pinocchio::Model pncModelFlexibleOrig_;
        motionVector_t jointsAcceleration_;                     ///< Vector of joints acceleration corresponding to a copy of data.a - temporary buffer for computing constraints.

        int32_t nq_;
        int32_t nv_;
        int32_t nx_;
    };
}

#endif //end of JIMINY_MODEL_H
