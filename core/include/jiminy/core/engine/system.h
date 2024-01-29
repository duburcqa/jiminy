


#ifndef JIMINY_SYSTEM_H
#define JIMINY_SYSTEM_H

#include <set>

#include "jiminy/core/fwd.h"
#include "jiminy/core/robot/model.h"


namespace jiminy
{
    class Robot;
    class AbstractConstraintSolver;
    class AbstractController;
    class LockGuardLocal;

    // External force functors
    using ForceProfileFunctor = std::function<pinocchio::Force(
        double /*t*/, const Eigen::VectorXd & /*q*/, const Eigen::VectorXd & /*v*/)>;

    struct JIMINY_DLLAPI ForceProfile
    {
    public:
        // FIXME: Designated aggregate initialization without constructors when moving to C++20
        explicit ForceProfile() = default;
        explicit ForceProfile(const std::string & frameNameIn,
                              pinocchio::FrameIndex frameIdxIn,
                              double updatePeriodIn,
                              const ForceProfileFunctor & forceFctIn) noexcept;

    public:
        std::string frameName{};
        pinocchio::FrameIndex frameIdx{0};
        double updatePeriod{0.0};
        pinocchio::Force forcePrev{pinocchio::Force::Zero()};
        ForceProfileFunctor forceFct{};
    };

    struct JIMINY_DLLAPI ForceImpulse
    {
    public:
        // FIXME: Designated aggregate initialization without constructors when moving to C++20
        explicit ForceImpulse() = default;
        explicit ForceImpulse(const std::string & frameNameIn,
                              pinocchio::FrameIndex frameIdxIn,
                              double tIn,
                              double dtIn,
                              const pinocchio::Force & FIn) noexcept;


    public:
        std::string frameName{};
        pinocchio::FrameIndex frameIdx{0};
        double t{0.0};
        double dt{0.0};
        pinocchio::Force F{};
    };

    using ForceCouplingFunctor = std::function<pinocchio::Force(double /*t*/,
                                                                const Eigen::VectorXd & /*q_1*/,
                                                                const Eigen::VectorXd & /*v_1*/,
                                                                const Eigen::VectorXd & /*q_2*/,
                                                                const Eigen::VectorXd & /*v_2*/)>;

    struct ForceCoupling
    {
    public:
        // FIXME: Designated aggregate initialization without constructors when moving to C++20
        explicit ForceCoupling() = default;
        explicit ForceCoupling(const std::string & systemName1In,
                               std::ptrdiff_t systemIdx1In,
                               const std::string & systemName2In,
                               std::ptrdiff_t systemIdx2In,
                               const std::string & frameName1In,
                               pinocchio::FrameIndex frameIdx1In,
                               const std::string & frameName2In,
                               pinocchio::FrameIndex frameIdx2In,
                               const ForceCouplingFunctor & forceFctIn) noexcept;

    public:
        std::string systemName1{};
        std::ptrdiff_t systemIdx1{-1};
        std::string systemName2{};
        std::ptrdiff_t systemIdx2{-1};
        std::string frameName1{};
        pinocchio::FrameIndex frameIdx1{0};
        std::string frameName2{};
        pinocchio::FrameIndex frameIdx2{0};
        ForceCouplingFunctor forceFct{};
    };

    using ForceProfileRegister = std::vector<ForceProfile>;
    using ForceImpulseRegister = std::vector<ForceImpulse>;
    using ForceCouplingRegister = std::vector<ForceCoupling>;

    // Early termination callback functor
    using CallbackFunctor = std::function<bool(
        double /*t*/, const Eigen::VectorXd & /*q*/, const Eigen::VectorXd & /*v*/)>;

    struct JIMINY_DLLAPI systemHolder_t
    {
        std::string name{};
        std::shared_ptr<Robot> robot{nullptr};
        std::shared_ptr<AbstractController> controller{nullptr};
        CallbackFunctor callbackFct{+[](double /* t */,
                                        const Eigen::VectorXd & /* q */,
                                        const Eigen::VectorXd & /* v */) -> bool
                                    {
                                        return false;
                                    }};
    };

    struct JIMINY_DLLAPI systemState_t
    {
    public:
        hresult_t initialize(const Robot & robot);
        bool getIsInitialized() const;

        void clear();

    public:
        Eigen::VectorXd q{};
        Eigen::VectorXd v{};
        Eigen::VectorXd a{};
        Eigen::VectorXd command{};
        Eigen::VectorXd u{};
        Eigen::VectorXd uMotor{};
        Eigen::VectorXd uInternal{};
        Eigen::VectorXd uCustom{};
        ForceVector fExternal{};

    private:
        bool isInitialized_{false};
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
        std::unique_ptr<LockGuardLocal> robotLock{nullptr};

        ForceProfileRegister forcesProfile{};
        ForceImpulseRegister forcesImpulse{};
        /// \brief Ordered list without repetitions of all the start/end times of the forces.
        std::set<double> forcesImpulseBreaks{};
        /// \brief Time of the next breakpoint associated with the impulse forces.
        std::set<double>::const_iterator forcesImpulseBreakNextIt{};
        /// \brief Set of flags tracking whether each force is active.
        ///
        /// \details This flag is used to handle t-, t+ properly. Without it, it is impossible to
        ///          determine at time t if the force is active or not.
        std::vector<bool> forcesImpulseActive{};

        uint32_t successiveSolveFailed{0};
        std::unique_ptr<AbstractConstraintSolver> constraintSolver{nullptr};
        /// \brief Store copy of constraints register for fast access.
        constraintsHolder_t constraintsHolder{};
        /// \brief Contact forces for each contact frames in local frame.
        ForceVector contactFramesForces{};
        /// \brief Contact forces for each geometries of each collision bodies in local frame.
        vector_aligned_t<ForceVector> collisionBodiesForces{};
        /// \brief Jacobian of the joints in local frame. Used for computing `data.u`.
        std::vector<Matrix6Xd> jointsJacobians{};

        std::vector<std::string> logFieldnamesPosition{};
        std::vector<std::string> logFieldnamesVelocity{};
        std::vector<std::string> logFieldnamesAcceleration{};
        std::vector<std::string> logFieldnamesForceExternal{};
        std::vector<std::string> logFieldnamesCommand{};
        std::vector<std::string> logFieldnamesMotorEffort{};
        std::string logFieldnameEnergy{};

        /// \brief Internal buffer with the state for the integration loop.
        systemState_t state{};
        /// \brief Internal state for the integration loop at the end of the previous iteration.
        systemState_t statePrev{};
    };
}

#endif  // end of JIMINY_STEPPERS_H
