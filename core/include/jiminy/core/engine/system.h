


#ifndef JIMINY_SYSTEM_H
#define JIMINY_SYSTEM_H

#include <set>

#include "jiminy/core/fwd.h"
#include "jiminy/core/robot/model.h"


namespace jiminy
{
    class Robot;
    class AbstractConstraintSolver;
    class AbstractConstraintBase;
    class AbstractController;
    class LockGuardLocal;


    // External force functors
    using ForceProfileFunctor = std::function<pinocchio::Force(
        float64_t /*t*/, const Eigen::VectorXd & /*q*/, const Eigen::VectorXd & /*v*/)>;

    struct JIMINY_DLLAPI ForceProfile
    {
    public:
        ForceProfile() = default;
        ForceProfile(const std::string & frameNameIn,
                     pinocchio::FrameIndex frameIdxIn,
                     float64_t updatePeriodIn,
                     const ForceProfileFunctor & forceFctIn);

    public:
        std::string frameName;
        pinocchio::FrameIndex frameIdx;
        float64_t updatePeriod;
        pinocchio::Force forcePrev;
        ForceProfileFunctor forceFct;
    };

    struct JIMINY_DLLAPI ForceImpulse
    {
    public:
        ForceImpulse() = default;
        ForceImpulse(const std::string & frameNameIn,
                     pinocchio::FrameIndex frameIdxIn,
                     float64_t tIn,
                     float64_t dtIn,
                     const pinocchio::Force & FIn);

    public:
        std::string frameName;
        pinocchio::FrameIndex frameIdx;
        float64_t t;
        float64_t dt;
        pinocchio::Force F;
    };

    using ForceCouplingFunctor = std::function<pinocchio::Force(float64_t /*t*/,
                                                                const Eigen::VectorXd & /*q_1*/,
                                                                const Eigen::VectorXd & /*v_1*/,
                                                                const Eigen::VectorXd & /*q_2*/,
                                                                const Eigen::VectorXd & /*v_2*/)>;

    struct ForceCoupling
    {
    public:
        ForceCoupling() = default;
        ForceCoupling(const std::string & systemName1In,
                      int32_t systemIdx1In,
                      const std::string & systemName2In,
                      int32_t systemIdx2In,
                      const std::string & frameName1In,
                      pinocchio::FrameIndex frameIdx1In,
                      const std::string & frameName2In,
                      pinocchio::FrameIndex frameIdx2In,
                      const ForceCouplingFunctor & forceFctIn);

    public:
        std::string systemName1;
        int32_t systemIdx1;
        std::string systemName2;
        int32_t systemIdx2;
        std::string frameName1;
        pinocchio::FrameIndex frameIdx1;
        std::string frameName2;
        pinocchio::FrameIndex frameIdx2;
        ForceCouplingFunctor forceFct;
    };

    using ForceProfileRegister = std::vector<ForceProfile>;
    using ForceImpulseRegister = std::vector<ForceImpulse>;

    // Early termination callback functor
    using CallbackFunctor = std::function<bool_t(
        float64_t /*t*/, const Eigen::VectorXd & /*q*/, const Eigen::VectorXd & /*v*/)>;

    struct JIMINY_DLLAPI systemHolder_t
    {
    public:
        systemHolder_t();
        systemHolder_t(const std::string & systemNameIn,
                       std::shared_ptr<Robot> robotIn,
                       std::shared_ptr<AbstractController> controllerIn,
                       CallbackFunctor callbackFctIn);
        systemHolder_t(const systemHolder_t & other) = default;
        systemHolder_t(systemHolder_t && other) = default;
        systemHolder_t & operator=(const systemHolder_t & other) = default;
        systemHolder_t & operator=(systemHolder_t && other) = default;
        ~systemHolder_t() = default;

    public:
        std::string name;
        std::shared_ptr<Robot> robot;
        std::shared_ptr<AbstractController> controller;
        CallbackFunctor callbackFct;
    };

    struct JIMINY_DLLAPI systemState_t
    {
    public:
        // Non-default constructor to be considered initialized even if not
        systemState_t();

        hresult_t initialize(const Robot & robot);
        bool_t getIsInitialized() const;

        void clear();

    public:
        Eigen::VectorXd q;
        Eigen::VectorXd v;
        Eigen::VectorXd a;
        Eigen::VectorXd command;
        Eigen::VectorXd u;
        Eigen::VectorXd uMotor;
        Eigen::VectorXd uInternal;
        Eigen::VectorXd uCustom;
        ForceVector fExternal;

    private:
        bool_t isInitialized_;
    };

    struct JIMINY_DLLAPI systemDataHolder_t
    {
    public:
        DISABLE_COPY(systemDataHolder_t)

        /* Must move all definitions in source files to avoid compilation failure due to incomplete
           destructor for objects managed by `unique_ptr` member variable with MSVC compiler.
           See: https://stackoverflow.com/a/9954553
                https://developercommunity.visualstudio.com/t/unique-ptr-cant-delete-an-incomplete-type/1371585
        */
        explicit systemDataHolder_t();
        explicit systemDataHolder_t(systemDataHolder_t &&);
        systemDataHolder_t & operator=(systemDataHolder_t &&);
        ~systemDataHolder_t();

    public:
        std::unique_ptr<LockGuardLocal> robotLock;

        ForceProfileRegister forcesProfile;
        ForceImpulseRegister forcesImpulse;
        /// \brief Ordered list without repetitions of all the start/end times of the forces.
        std::set<float64_t> forcesImpulseBreaks;
        /// \brief Time of the next breakpoint associated with the impulse forces.
        std::set<float64_t>::const_iterator forcesImpulseBreakNextIt;
        /// \brief Set of flags tracking whether each force is active.
        ///
        /// \details This flag is used to handle t-, t+ properly. Without it, it is impossible to
        ///          determine at time t if the force is active or not.
        std::vector<bool_t> forcesImpulseActive;

        uint32_t successiveSolveFailed;
        std::unique_ptr<AbstractConstraintSolver> constraintSolver;
        /// \brief Store copy of constraints register for fast access.
        constraintsHolder_t constraintsHolder;
        /// \brief Contact forces for each contact frames in local frame.
        ForceVector contactFramesForces;
        /// \brief Contact forces for each geometries of each collision bodies in local frame.
        vector_aligned_t<ForceVector> collisionBodiesForces;
        /// \brief Jacobian of the joints in local frame. Used for computing `data.u`.
        std::vector<Matrix6Xd> jointsJacobians;

        std::vector<std::string> logFieldnamesPosition;
        std::vector<std::string> logFieldnamesVelocity;
        std::vector<std::string> logFieldnamesAcceleration;
        std::vector<std::string> logFieldnamesForceExternal;
        std::vector<std::string> logFieldnamesCommand;
        std::vector<std::string> logFieldnamesMotorEffort;
        std::string logFieldnameEnergy;

        /// \brief Internal buffer with the state for the integration loop.
        systemState_t state;
        /// \brief Internal state for the integration loop at the end of the previous iteration.
        systemState_t statePrev;
    };
}

#endif  // end of JIMINY_STEPPERS_H
