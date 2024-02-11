


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

    // ******************************** External force functors ******************************** //

    using ProfileForceFunction = std::function<pinocchio::Force(
        double /*t*/, const Eigen::VectorXd & /*q*/, const Eigen::VectorXd & /*v*/)>;

    struct JIMINY_DLLAPI ProfileForce
    {
    public:
        // FIXME: Designated aggregate initialization without constructors when moving to C++20
        explicit ProfileForce() = default;
        explicit ProfileForce(const std::string & frameNameIn,
                              pinocchio::FrameIndex frameIndexIn,
                              double updatePeriodIn,
                              const ProfileForceFunction & forceFuncIn) noexcept;

    public:
        std::string frameName{};
        pinocchio::FrameIndex frameIndex{0};
        double updatePeriod{0.0};
        pinocchio::Force force{pinocchio::Force::Zero()};
        ProfileForceFunction func{};
    };

    struct JIMINY_DLLAPI ImpulseForce
    {
    public:
        // FIXME: Designated aggregate initialization without constructors when moving to C++20
        explicit ImpulseForce() = default;
        explicit ImpulseForce(const std::string & frameNameIn,
                              pinocchio::FrameIndex frameIndexIn,
                              double tIn,
                              double dtIn,
                              const pinocchio::Force & forceIn) noexcept;


    public:
        std::string frameName{};
        pinocchio::FrameIndex frameIndex{0};
        double t{0.0};
        double dt{0.0};
        pinocchio::Force force{};
    };

    using CouplingForceFunction =
        std::function<pinocchio::Force(double /* t */,
                                       const Eigen::VectorXd & /* q1 */,
                                       const Eigen::VectorXd & /* v1 */,
                                       const Eigen::VectorXd & /* q2 */,
                                       const Eigen::VectorXd & /* v2 */)>;

    struct CouplingForce
    {
    public:
        // FIXME: Designated aggregate initialization without constructors when moving to C++20
        explicit CouplingForce() = default;
        explicit CouplingForce(const std::string & systemName1In,
                               std::ptrdiff_t systemIndex1In,
                               const std::string & systemName2In,
                               std::ptrdiff_t systemIndex2In,
                               const std::string & frameName1In,
                               pinocchio::FrameIndex frameIndex1In,
                               const std::string & frameName2In,
                               pinocchio::FrameIndex frameIndex2In,
                               const CouplingForceFunction & forceFunIn) noexcept;

    public:
        std::string systemName1{};
        std::ptrdiff_t systemIndex1{-1};
        std::string systemName2{};
        std::ptrdiff_t systemIndex2{-1};
        std::string frameName1{};
        pinocchio::FrameIndex frameIndex1{0};
        std::string frameName2{};
        pinocchio::FrameIndex frameIndex2{0};
        CouplingForceFunction func{};
    };

    using ProfileForceVector = std::vector<ProfileForce>;
    using ImpulseForceVector = std::vector<ImpulseForce>;
    using CouplingForceVector = std::vector<CouplingForce>;

    // Early termination callback functor
    using AbortSimulationFunction = std::function<bool(
        double /*t*/, const Eigen::VectorXd & /*q*/, const Eigen::VectorXd & /*v*/)>;

    struct JIMINY_DLLAPI System
    {
        std::string name{};
        std::shared_ptr<Robot> robot{nullptr};
        std::shared_ptr<AbstractController> controller{nullptr};
        AbortSimulationFunction callback{[](double /* t */,
                                            const Eigen::VectorXd & /* q */,
                                            const Eigen::VectorXd & /* v */) -> bool
                                         {
                                             return false;
                                         }};
    };

    // ************************************** System state ************************************* //

    struct JIMINY_DLLAPI SystemState
    {
    public:
        void initialize(const Robot & robot);
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

    struct JIMINY_DLLAPI SystemData
    {
    public:
        DISABLE_COPY(SystemData)

        /* Must move all definitions in source files to avoid compilation failure due to incomplete
           destructor for objects managed by `unique_ptr` member variable with MSVC compiler.
           See: https://stackoverflow.com/a/9954553
                https://developercommunity.visualstudio.com/t/unique-ptr-cant-delete-an-incomplete-type/1371585
        */
        explicit SystemData();
        explicit SystemData(SystemData &&);
        SystemData & operator=(SystemData &&);
        ~SystemData();

    public:
        std::unique_ptr<LockGuardLocal> robotLock{nullptr};

        ProfileForceVector profileForces{};
        ImpulseForceVector impulseForces{};
        /// \brief Sorted list without repetitions of all the start/stop times of impulse forces.
        std::set<double> impulseForceBreakpoints{};
        /// \brief Time of the next breakpoint associated with the impulse forces.
        std::set<double>::const_iterator impulseForceBreakpointNextIt{};
        /// \brief Set of flags tracking whether each force is active.
        ///
        /// \details This flag is used to handle t-, t+ properly. Without it, it is impossible to
        ///          determine at time t if the force is active or not.
        std::vector<bool> isImpulseForceActiveVec{};

        uint32_t successiveSolveFailed{0};
        std::unique_ptr<AbstractConstraintSolver> constraintSolver{nullptr};
        /// \brief Store copy of constraints register for fast access.
        ConstraintTree constraints{};
        /// \brief Contact forces for each contact frames in local frame.
        ForceVector contactFrameForces{};
        /// \brief Contact forces for each geometries of each collision bodies in local frame.
        vector_aligned_t<ForceVector> collisionBodiesForces{};
        /// \brief Jacobian of the joints in local frame. Used for computing `data.u`.
        std::vector<Matrix6Xd> jointJacobians{};

        std::vector<std::string> logPositionFieldnames{};
        std::vector<std::string> logVelocityFieldnames{};
        std::vector<std::string> logAccelerationFieldnames{};
        std::vector<std::string> logForceExternalFieldnames{};
        std::vector<std::string> logCommandFieldnames{};
        std::vector<std::string> logMotorEffortFieldnames{};
        std::string logEnergyFieldname{};

        /// \brief Internal buffer with the state for the integration loop.
        SystemState state{};
        /// \brief Internal state for the integration loop at the end of the previous iteration.
        SystemState statePrev{};
    };
}

#endif  // end of JIMINY_STEPPERS_H
