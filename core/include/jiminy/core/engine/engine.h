#ifndef JIMINY_ENGINE_MULTIROBOT_H
#define JIMINY_ENGINE_MULTIROBOT_H

#include <set>
#include <optional>
#include <functional>

#include "jiminy/core/fwd.h"
#include "jiminy/core/utilities/helpers.h"  // `Timer`
#include "jiminy/core/utilities/random.h"   // `PCG32`


namespace jiminy
{
    inline constexpr std::string_view ENGINE_TELEMETRY_NAMESPACE{""};

    enum class ContactModelType : uint8_t
    {
        UNSUPPORTED = 0,
        SPRING_DAMPER = 1,
        CONSTRAINT = 2
    };

    enum class ConstraintSolverType : uint8_t
    {
        UNSUPPORTED = 0,
        PGS = 1  // Projected Gauss-Seidel
    };

    const std::map<std::string, ContactModelType> CONTACT_MODELS_MAP{
        {"spring_damper", ContactModelType::SPRING_DAMPER},
        {"constraint", ContactModelType::CONSTRAINT},
    };

    const std::map<std::string, ConstraintSolverType> CONSTRAINT_SOLVERS_MAP{
        {"PGS", ConstraintSolverType::PGS}};

    const std::set<std::string> STEPPERS{"euler_explicit", "runge_kutta_4", "runge_kutta_dopri"};

    class Robot;
    class AbstractConstraintSolver;
    class AbstractConstraintBase;
    class AbstractController;
    class AbstractStepper;
    class LockGuardLocal;
    class TelemetryData;
    class TelemetryRecorder;
    class TelemetrySender;
    struct LogData;

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
        explicit CouplingForce(const std::string & robotName1In,
                               std::ptrdiff_t robotIndex1In,
                               const std::string & robotName2In,
                               std::ptrdiff_t robotIndex2In,
                               const std::string & frameName1In,
                               pinocchio::FrameIndex frameIndex1In,
                               const std::string & frameName2In,
                               pinocchio::FrameIndex frameIndex2In,
                               const CouplingForceFunction & forceFunIn) noexcept;

    public:
        std::string robotName1{};
        std::ptrdiff_t robotIndex1{-1};
        std::string robotName2{};
        std::ptrdiff_t robotIndex2{-1};
        std::string frameName1{};
        pinocchio::FrameIndex frameIndex1{0};
        std::string frameName2{};
        pinocchio::FrameIndex frameIndex2{0};
        CouplingForceFunction func{};
    };

    using ProfileForceVector = std::vector<ProfileForce>;
    using ImpulseForceVector = std::vector<ImpulseForce>;
    using CouplingForceVector = std::vector<CouplingForce>;

    // ************************************** Robot state ************************************** //

    struct JIMINY_DLLAPI RobotState
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
        Eigen::VectorXd uTransmission{};
        Eigen::VectorXd uInternal{};
        Eigen::VectorXd uCustom{};
        ForceVector fExternal{};

    private:
        bool isInitialized_{false};
    };

    // *************************************** Robot data ************************************** //

    struct JIMINY_DLLAPI RobotData
    {
    public:
        JIMINY_DISABLE_COPY(RobotData)

        /* Must move all definitions in source files to avoid compilation failure due to incomplete
           destructor for objects managed by `unique_ptr` member variable with MSVC compiler.
           See: https://stackoverflow.com/a/9954553
                https://developercommunity.visualstudio.com/t/unique-ptr-cant-delete-an-incomplete-type/1371585
        */
        explicit RobotData();
        explicit RobotData(RobotData &&);
        RobotData & operator=(RobotData &&);
        ~RobotData();

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
        /// \brief Contact forces for each contact frames in local frame.
        ForceVector contactFrameForces{};
        /// \brief Contact forces for each geometries of each collision bodies in local frame.
        vector_aligned_t<ForceVector> collisionBodiesForces{};
        /// \brief Jacobian of the joints in local frame. Used for computing `data.u`.
        std::vector<Matrix6Xd> jointJacobians{};

        std::vector<std::string> logPositionFieldnames{};
        std::vector<std::string> logVelocityFieldnames{};
        std::vector<std::string> logAccelerationFieldnames{};
        std::vector<std::string> logEffortFieldnames{};
        std::vector<std::string> logForceExternalFieldnames{};
        std::vector<std::string> logConstraintFieldnames{};
        std::vector<std::string> logCommandFieldnames{};
        std::string logEnergyFieldname{};

        /// \brief Internal buffer with the state for the integration loop.
        RobotState state{};
        /// \brief Internal state for the integration loop at the end of the previous iteration.
        RobotState statePrev{};
    };

    // ************************************* Stepper state ************************************* //

    struct JIMINY_DLLAPI StepperState
    {
    public:
        void reset(double dtInit,
                   const std::vector<Eigen::VectorXd> & qSplitInit,
                   const std::vector<Eigen::VectorXd> & vSplitInit,
                   const std::vector<Eigen::VectorXd> & aSplitInit)
        {
            iter = 0U;
            iterFailed = 0U;
            t = 0.0;
            tPrev = 0.0;
            dt = dtInit;
            dtLargest = dtInit;
            dtLargestPrev = dtInit;
            tError = 0.0;
            qSplit = qSplitInit;
            vSplit = vSplitInit;
            aSplit = aSplitInit;
        }

    public:
        uint32_t iter{0U};
        uint32_t iterFailed{0U};
        double t{0.0};
        double tPrev{0.0};
        /// \brief Internal buffer used for Kahan algorithm storing the residual sum of errors.
        double tError{0.0};
        double dt{INF};
        double dtLargest{INF};
        double dtLargestPrev{INF};
        std::vector<Eigen::VectorXd> qSplit{};
        std::vector<Eigen::VectorXd> vSplit{};
        std::vector<Eigen::VectorXd> aSplit{};
    };

    // ************************************ Engine *********************************** //

    // Early termination callback functor
    using AbortSimulationFunction = std::function<bool()>;

    class JIMINY_DLLAPI Engine
    {
    public:
        GenericConfig getDefaultConstraintOptions()
        {
            GenericConfig config;
            config["solver"] = std::string{"PGS"};  // ["PGS",]
            /// \brief Relative inverse damping wrt diagonal of J.Minv.J.t.
            ///
            /// \details 0.0 enforces the minimum absolute regularizer.
            config["regularization"] = 1.0e-3;
            config["successiveSolveFailedMax"] = 100U;

            return config;
        };

        GenericConfig getDefaultContactOptions()
        {
            GenericConfig config;
            config["model"] = std::string{"constraint"};  // ["spring_damper", "constraint"]
            config["stiffness"] = 1.0e6;
            config["damping"] = 2.0e3;
            config["friction"] = 1.0;
            config["torsion"] = 0.0;
            config["transitionEps"] = 1.0e-3;       // [m]
            config["transitionVelocity"] = 1.0e-2;  // [m.s-1]
            config["stabilizationFreq"] = 20.0;     // [s-1]: 0.0 to disable

            return config;
        };

        GenericConfig getDefaultWorldOptions()
        {
            GenericConfig config;
            config["gravity"] = (Eigen::VectorXd(6) << 0.0, 0.0, -9.81, 0.0, 0.0, 0.0).finished();
            config["groundProfile"] = HeightmapFunction(
                [](const Eigen::Vector2d & /* xy */,
                   double & height,
                   std::optional<Eigen::Ref<Eigen::Vector3d>> normal) -> void
                {
                    height = 0.0;
                    if (normal.has_value())
                    {
                        normal.value() = Eigen::Vector3d::UnitZ();
                    }
                });

            return config;
        };

        GenericConfig getDefaultStepperOptions()
        {
            GenericConfig config;
            config["verbose"] = false;
            config["randomSeedSeq"] = VectorX<uint32_t>::Zero(1).eval();
            /// \details Must be either "runge_kutta_dopri", "runge_kutta_4" or "euler_explicit".
            config["odeSolver"] = std::string{"runge_kutta_dopri"};
            config["tolAbs"] = 1.0e-5;
            config["tolRel"] = 1.0e-4;
            config["dtMax"] = SIMULATION_MAX_TIMESTEP;
            config["dtRestoreThresholdRel"] = 0.2;
            config["successiveIterFailedMax"] = 1000U;
            config["iterMax"] = 0U;   // <= 0: disable
            config["timeout"] = 0.0;  // <= 0.0: disable
            config["sensorsUpdatePeriod"] = 0.0;
            config["controllerUpdatePeriod"] = 0.0;
            config["logInternalStepperSteps"] = false;

            return config;
        };

        GenericConfig getDefaultTelemetryOptions()
        {
            GenericConfig config;
            config["isPersistent"] = false;
            config["enableConfiguration"] = true;
            config["enableVelocity"] = true;
            config["enableAcceleration"] = true;
            config["enableForceExternal"] = false;
            config["enableConstraint"] = false;
            config["enableCommand"] = false;
            config["enableEffort"] = false;
            config["enableEnergy"] = false;
            return config;
        };

        GenericConfig getDefaultEngineOptions()
        {
            GenericConfig config;
            config["telemetry"] = getDefaultTelemetryOptions();
            config["stepper"] = getDefaultStepperOptions();
            config["world"] = getDefaultWorldOptions();
            config["constraints"] = getDefaultConstraintOptions();
            config["contacts"] = getDefaultContactOptions();

            return config;
        };

        struct ConstraintOptions
        {
            const std::string solver;
            const double regularization;
            const uint32_t successiveSolveFailedMax;

            ConstraintOptions(const GenericConfig & options) :
            solver{boost::get<std::string>(options.at("solver"))},
            regularization{boost::get<double>(options.at("regularization"))},
            successiveSolveFailedMax{boost::get<uint32_t>(options.at("successiveSolveFailedMax"))}
            {
            }
        };

        struct ContactOptions
        {
            const std::string model;
            const double stiffness;
            const double damping;
            const double friction;
            const double torsion;
            const double transitionEps;
            const double transitionVelocity;
            const double stabilizationFreq;

            ContactOptions(const GenericConfig & options) :
            model{boost::get<std::string>(options.at("model"))},
            stiffness{boost::get<double>(options.at("stiffness"))},
            damping{boost::get<double>(options.at("damping"))},
            friction{boost::get<double>(options.at("friction"))},
            torsion{boost::get<double>(options.at("torsion"))},
            transitionEps{boost::get<double>(options.at("transitionEps"))},
            transitionVelocity{boost::get<double>(options.at("transitionVelocity"))},
            stabilizationFreq{boost::get<double>(options.at("stabilizationFreq"))}
            {
            }
        };

        struct WorldOptions
        {
            const Eigen::VectorXd gravity;
            const HeightmapFunction groundProfile;

            WorldOptions(const GenericConfig & options) :
            gravity{boost::get<Eigen::VectorXd>(options.at("gravity"))},
            groundProfile{boost::get<HeightmapFunction>(options.at("groundProfile"))}
            {
            }
        };

        struct StepperOptions
        {
            const bool verbose;
            const VectorX<uint32_t> randomSeedSeq;
            const std::string odeSolver;
            const double tolAbs;
            const double tolRel;
            const double dtMax;
            const double dtRestoreThresholdRel;
            const uint32_t successiveIterFailedMax;
            const uint32_t iterMax;
            const double timeout;
            const double sensorsUpdatePeriod;
            const double controllerUpdatePeriod;
            const bool logInternalStepperSteps;

            StepperOptions(const GenericConfig & options) :
            verbose{boost::get<bool>(options.at("verbose"))},
            randomSeedSeq{boost::get<VectorX<uint32_t>>(options.at("randomSeedSeq"))},
            odeSolver{boost::get<std::string>(options.at("odeSolver"))},
            tolAbs{boost::get<double>(options.at("tolAbs"))},
            tolRel{boost::get<double>(options.at("tolRel"))},
            dtMax{boost::get<double>(options.at("dtMax"))},
            dtRestoreThresholdRel{boost::get<double>(options.at("dtRestoreThresholdRel"))},
            successiveIterFailedMax{boost::get<uint32_t>(options.at("successiveIterFailedMax"))},
            iterMax{boost::get<uint32_t>(options.at("iterMax"))},
            timeout{boost::get<double>(options.at("timeout"))},
            sensorsUpdatePeriod{boost::get<double>(options.at("sensorsUpdatePeriod"))},
            controllerUpdatePeriod{boost::get<double>(options.at("controllerUpdatePeriod"))},
            logInternalStepperSteps{boost::get<bool>(options.at("logInternalStepperSteps"))}
            {
            }
        };

        struct TelemetryOptions
        {
            const bool isPersistent;
            const bool enableConfiguration;
            const bool enableVelocity;
            const bool enableAcceleration;
            const bool enableForceExternal;
            const bool enableConstraint;
            const bool enableCommand;
            const bool enableEffort;
            const bool enableEnergy;

            TelemetryOptions(const GenericConfig & options) :
            isPersistent{boost::get<bool>(options.at("isPersistent"))},
            enableConfiguration{boost::get<bool>(options.at("enableConfiguration"))},
            enableVelocity{boost::get<bool>(options.at("enableVelocity"))},
            enableAcceleration{boost::get<bool>(options.at("enableAcceleration"))},
            enableForceExternal{boost::get<bool>(options.at("enableForceExternal"))},
            enableConstraint{boost::get<bool>(options.at("enableConstraint"))},
            enableCommand{boost::get<bool>(options.at("enableCommand"))},
            enableEffort{boost::get<bool>(options.at("enableEffort"))},
            enableEnergy{boost::get<bool>(options.at("enableEnergy"))}
            {
            }
        };

        struct EngineOptions
        {
            const TelemetryOptions telemetry;
            const StepperOptions stepper;
            const WorldOptions world;
            const ConstraintOptions constraints;
            const ContactOptions contacts;

            EngineOptions(const GenericConfig & options) :
            telemetry{boost::get<GenericConfig>(options.at("telemetry"))},
            stepper{boost::get<GenericConfig>(options.at("stepper"))},
            world{boost::get<GenericConfig>(options.at("world"))},
            constraints{boost::get<GenericConfig>(options.at("constraints"))},
            contacts{boost::get<GenericConfig>(options.at("contacts"))}
            {
            }
        };

    public:
        JIMINY_DISABLE_COPY(Engine)

    public:
        explicit Engine() noexcept;
        ~Engine();

        void addRobot(std::shared_ptr<Robot> robot);
        void removeRobot(const std::string & robotName);

        /// \brief Add a force linking both robots together.
        ///
        /// \details This function registers a callback function that links both robots by a given
        ///          force. This function must return the force that the second robots applies to
        ///          the first robot, in the global frame of the first frame, i.e. expressed at
        ///          the origin of the first frame, in word coordinates.
        ///
        /// \param[in] robotName1 Name of the robot receiving the force.
        /// \param[in] robotName2 Name of the robot applying the force.
        /// \param[in] frameName1 Frame on the first robot where the force is applied.
        /// \param[in] frameName2 Frame on the second robot where the opposite force is applied.
        /// \param[in] forceFunc Callback function returning the force that robotName2 applies on
        ///                      robotName1, in the global frame of frameName1.
        void registerCouplingForce(const std::string & robotName1,
                                   const std::string & robotName2,
                                   const std::string & frameName1,
                                   const std::string & frameName2,
                                   const CouplingForceFunction & forceFunc);
        void registerViscoelasticDirectionalCouplingForce(const std::string & robotName1,
                                                          const std::string & robotName2,
                                                          const std::string & frameName1,
                                                          const std::string & frameName2,
                                                          double stiffness,
                                                          double damping,
                                                          double restLength = 0.0);
        void registerViscoelasticDirectionalCouplingForce(const std::string & robotName,
                                                          const std::string & frameName1,
                                                          const std::string & frameName2,
                                                          double stiffness,
                                                          double damping,
                                                          double restLength = 0.0);

        /// \brief 6-DoFs spring-damper coupling force modelling a flexible bushing.
        ///
        /// \details The spring-damper forces are computed at a frame being the linear
        ///          interpolation (according on double-geodesic) between frame 1 and 2 by a factor
        ///          alpha. In particular, alpha = 0.0, 0.5, and 1.0 corresponds to frame 1,
        ///          halfway point, and frame 2.
        ///
        /// \see See official drake documentation:
        ///      https://drake.mit.edu/doxygen_cxx/classdrake_1_1multibody_1_1_linear_bushing_roll_pitch_yaw.html
        void registerViscoelasticCouplingForce(const std::string & robotName1,
                                               const std::string & robotName2,
                                               const std::string & frameName1,
                                               const std::string & frameName2,
                                               const Vector6d & stiffness,
                                               const Vector6d & damping,
                                               double alpha = 0.5);
        void registerViscoelasticCouplingForce(const std::string & robotName,
                                               const std::string & frameName1,
                                               const std::string & frameName2,
                                               const Vector6d & stiffness,
                                               const Vector6d & damping,
                                               double alpha = 0.5);
        void removeCouplingForces(const std::string & robotName1, const std::string & robotName2);
        void removeCouplingForces(const std::string & robotName);
        void removeCouplingForces();

        const CouplingForceVector & getCouplingForces() const;

        void removeAllForces();

        /// \brief Reset the engine and all the robots, which includes hardware and controller for
        ///        each of them.
        ///
        /// \details This method is made to be called in between simulations, to allow registering
        ///          of new variables to the telemetry, and reset the random number generators.
        ///
        /// \param[in] resetRandomNumbers Whether to reset the random number generators.
        /// \param[in] removeAllForce Whether to remove registered external forces.
        void reset(bool resetRandomNumbers = false, bool removeAllForce = false);

        /// \brief Start the simulation
        ///
        /// \warning This function calls `reset` internally only if necessary, namely if it was not
        ///          done manually at some point after stopping the previous simulation if any.
        ///
        /// \param[in] qInit Initial configuration of every robots.
        /// \param[in] vInit Initial velocity of every robots.
        /// \param[in] aInit Initial acceleration of every robots.
        ///                  Optional: Zero by default.
        void start(
            const std::map<std::string, Eigen::VectorXd> & qInit,
            const std::map<std::string, Eigen::VectorXd> & vInit,
            const std::optional<std::map<std::string, Eigen::VectorXd>> & aInit = std::nullopt);

        void start(const Eigen::VectorXd & qInit,
                   const Eigen::VectorXd & vInit,
                   const std::optional<Eigen::VectorXd> & aInit = std::nullopt,
                   bool isStateTheoretical = false);

        /// \brief Integrate robot dynamics from current state for a given duration.
        ///
        /// \details Internally, the integrator may perform multiple steps inside in the interval.
        ///
        /// \param[in] stepSize Duration for which to integrate. -1 for default duration, ie until
        ///                     next breakpoint if any, or 'engine_options["stepper"]["dtMax"]'.
        void step(double stepSize = -1);

        /// \brief Stop the simulation.
        ///
        /// \details It releases the lock on the robot and the telemetry, so that it is possible
        ///          again to update the robot (for example to update the options, add or remove
        ///          sensors...) and to register new variables or forces.
        void stop();

        /// \brief Run a simulation of duration tEnd, starting at xInit.
        ///
        /// \param[in] tEnd Duration of the simulation.
        /// \param[in] qInit Initial configuration of every robots, i.e. at t=0.0.
        /// \param[in] vInit Initial velocity of every robots, i.e. at t=0.0.
        /// \param[in] aInit Initial acceleration of every robots, i.e. at t=0.0.
        ///                  Optional: Zero by default.
        /// \param[in] callback Callable that can be specified to abort simulation. It will be
        ///                     evaluated after every simulation step. Abort if false is returned.
        void simulate(
            double tEnd,
            const std::map<std::string, Eigen::VectorXd> & qInit,
            const std::map<std::string, Eigen::VectorXd> & vInit,
            const std::optional<std::map<std::string, Eigen::VectorXd>> & aInit = std::nullopt,
            const AbortSimulationFunction & callback = []() { return true; });

        void simulate(
            double tEnd,
            const Eigen::VectorXd & qInit,
            const Eigen::VectorXd & vInit,
            const std::optional<Eigen::VectorXd> & aInit = std::nullopt,
            bool isStateTheoretical = false,
            const AbortSimulationFunction & callback = []() { return true; });

        /// \brief Apply an impulse force on a frame for a given duration at the desired time.
        ///
        /// \warning The force must be given in the world frame.
        void registerImpulseForce(const std::string & robotName,
                                  const std::string & frameName,
                                  double t,
                                  double dt,
                                  const pinocchio::Force & force);
        /// \brief Apply an external force profile on a frame.
        ///
        /// \details It can be either time-continuous or discrete. The force can be time- and
        ///          state-dependent, and must be given in the world frame.
        void registerProfileForce(const std::string & robotName,
                                  const std::string & frameName,
                                  const ProfileForceFunction & forceFunc,
                                  double updatePeriod = 0.0);

        void removeImpulseForces(const std::string & robotName);
        void removeProfileForces(const std::string & robotName);
        void removeImpulseForces();
        void removeProfileForces();

        const ImpulseForceVector & getImpulseForces(const std::string & robotName) const;
        const ProfileForceVector & getProfileForces(const std::string & robotName) const;

        void setOptions(const GenericConfig & engineOptions);
        const GenericConfig & getOptions() const noexcept;
        /// \brief Set the options of the engine and all the robots.
        ///
        /// \param[in] simulationOptions Dictionary gathering all the options. See
        ///                              `getSimulationOptions` for details about the hierarchy.
        void setSimulationOptions(const GenericConfig & simulationOptions);
        /// \brief Get the options of the engine and all the robots.
        ///
        /// \details The key 'engine' maps to the engine options, whereas `robot.name` maps to the
        ///          individual options of each robot for multi-robot simulations, 'robot' for
        ///          single-robot simulations.
        GenericConfig getSimulationOptions() const noexcept;

        bool getIsTelemetryConfigured() const;
        std::shared_ptr<Robot> getRobot(const std::string & robotName);
        std::ptrdiff_t getRobotIndex(const std::string & robotName) const;
        const RobotState & getRobotState(const std::string & robotName) const;
        const StepperState & getStepperState() const;
        const bool & getIsSimulationRunning() const;  // return const reference on purpose
        static double getSimulationDurationMax();
        static double getTelemetryTimeUnit();

        static void computeForwardKinematics(std::shared_ptr<Robot> & robot,
                                             const Eigen::VectorXd & q,
                                             const Eigen::VectorXd & v,
                                             const Eigen::VectorXd & a);
        void computeRobotsDynamics(double t,
                                   const std::vector<Eigen::VectorXd> & qSplit,
                                   const std::vector<Eigen::VectorXd> & vSplit,
                                   std::vector<Eigen::VectorXd> & aSplit,
                                   bool isStateUpToDate = false);

    protected:
        void configureTelemetry();
        void updateTelemetry();

        void syncStepperStateWithRobots();
        void syncRobotsStateWithStepper(bool isStateUpToDate = false);


        /// \brief Compute the force resulting from ground contact on a given body.
        ///
        /// \param[in] robot Robot for which to perform computation.
        /// \param[in] collisionPairIndex Index of the collision pair associated with the body.
        ///
        /// \returns Contact force, at parent joint, in the local frame.
        void computeContactDynamicsAtBody(
            const std::shared_ptr<Robot> & robot,
            const pinocchio::PairIndex & collisionPairIndex,
            const std::shared_ptr<AbstractConstraintBase> & contactConstraint,
            pinocchio::Force & fextLocal) const;

        /// \brief Compute the force resulting from ground contact on a given frame.
        ///
        /// \param[in] robot Robot for which to perform computation.
        /// \param[in] frameIndex Index of the frame in contact.
        ///
        /// \returns Contact force, at parent joint, in the local frame.
        void computeContactDynamicsAtFrame(
            const std::shared_ptr<Robot> & robot,
            pinocchio::FrameIndex frameIndex,
            const std::shared_ptr<AbstractConstraintBase> & collisionConstraint,
            pinocchio::Force & fextLocal) const;

        /// \brief Compute the ground reaction force for a given normal direction and depth.
        pinocchio::Force computeContactDynamics(const Eigen::Vector3d & nGround,
                                                double depth,
                                                const Eigen::Vector3d & vContactInWorld) const;

        void computeCommand(std::shared_ptr<Robot> & robot,
                            double t,
                            const Eigen::VectorXd & q,
                            const Eigen::VectorXd & v,
                            Eigen::VectorXd & command);
        void computeInternalDynamics(const std::shared_ptr<Robot> & robot,
                                     double t,
                                     const Eigen::VectorXd & q,
                                     const Eigen::VectorXd & v,
                                     Eigen::VectorXd & uInternal) const;
        void computeCollisionForces(const std::shared_ptr<Robot> & robot,
                                    RobotData & robotData,
                                    ForceVector & fext,
                                    bool isStateUpToDate = false) const;
        void computeExternalForces(const std::shared_ptr<Robot> & robot,
                                   RobotData & robotData,
                                   double t,
                                   const Eigen::VectorXd & q,
                                   const Eigen::VectorXd & v,
                                   ForceVector & fext);
        void computeCouplingForces(double t,
                                   const std::vector<Eigen::VectorXd> & qSplit,
                                   const std::vector<Eigen::VectorXd> & vSplit);
        void computeAllTerms(double t,
                             const std::vector<Eigen::VectorXd> & qSplit,
                             const std::vector<Eigen::VectorXd> & vSplit,
                             bool isStateUpToDate = false);

        /// \brief Compute robot acceleration from current robot state.
        ///
        /// \details This function performs forward dynamics computation, either with kinematic
        ///          constraints (using Lagrange multiplier for computing the forces) or
        ///          unconstrained (aba).
        ///
        /// \param[in] robot Robot for which to compute the dynamics.
        /// \param[in] q Joint position.
        /// \param[in] v Joint velocity.
        /// \param[in] u Joint effort.
        /// \param[in] fext External forces applied on the robot.
        ///
        /// \return Robot acceleration.
        const Eigen::VectorXd & computeAcceleration(std::shared_ptr<Robot> & robot,
                                                    RobotData & robotData,
                                                    const Eigen::VectorXd & q,
                                                    const Eigen::VectorXd & v,
                                                    const Eigen::VectorXd & u,
                                                    ForceVector & fext,
                                                    bool isStateUpToDate = false,
                                                    bool ignoreBounds = false);

    public:
        std::shared_ptr<const LogData> getLog();

        static LogData readLog(const std::string & filename, const std::string & format);

        void writeLog(const std::string & filename, const std::string & format);

    private:
        template<typename Scalar,
                 int Options,
                 template<typename, int>
                 class JointCollectionTpl,
                 typename ConfigVectorType,
                 typename TangentVectorType>
        inline Scalar
        kineticEnergy(const pinocchio::ModelTpl<Scalar, Options, JointCollectionTpl> & model,
                      pinocchio::DataTpl<Scalar, Options, JointCollectionTpl> & data,
                      const Eigen::MatrixBase<ConfigVectorType> & q,
                      const Eigen::MatrixBase<TangentVectorType> & v,
                      bool update_kinematics);
        template<typename Scalar,
                 int Options,
                 template<typename, int>
                 class JointCollectionTpl,
                 typename ConfigVectorType,
                 typename TangentVectorType1,
                 typename TangentVectorType2,
                 typename ForceDerived>
        inline const typename pinocchio::DataTpl<Scalar, Options, JointCollectionTpl>::
            TangentVectorType &
            rnea(const pinocchio::ModelTpl<Scalar, Options, JointCollectionTpl> & model,
                 pinocchio::DataTpl<Scalar, Options, JointCollectionTpl> & data,
                 const Eigen::MatrixBase<ConfigVectorType> & q,
                 const Eigen::MatrixBase<TangentVectorType1> & v,
                 const Eigen::MatrixBase<TangentVectorType2> & a,
                 const vector_aligned_t<ForceDerived> & fext);
        template<typename Scalar,
                 int Options,
                 template<typename, int>
                 class JointCollectionTpl,
                 typename ConfigVectorType,
                 typename TangentVectorType1,
                 typename TangentVectorType2,
                 typename ForceDerived>
        inline const typename pinocchio::DataTpl<Scalar, Options, JointCollectionTpl>::
            TangentVectorType &
            aba(const pinocchio::ModelTpl<Scalar, Options, JointCollectionTpl> & model,
                pinocchio::DataTpl<Scalar, Options, JointCollectionTpl> & data,
                const Eigen::MatrixBase<ConfigVectorType> & q,
                const Eigen::MatrixBase<TangentVectorType1> & v,
                const Eigen::MatrixBase<TangentVectorType2> & tau,
                const vector_aligned_t<ForceDerived> & fext);

    public:
        std::unique_ptr<const EngineOptions> engineOptions_{nullptr};
        std::vector<std::shared_ptr<Robot>> robots_{};

    protected:
        bool isTelemetryConfigured_{false};
        bool isSimulationRunning_{false};
        mutable bool areSimulationOptionsRefreshed_{false};
        mutable GenericConfig simulationOptionsGeneric_{};
        PCG32 generator_;

    private:
        Timer timer_{};
        ContactModelType contactModel_{ContactModelType::UNSUPPORTED};
        std::unique_ptr<TelemetrySender> telemetrySender_;
        std::shared_ptr<TelemetryData> telemetryData_;
        std::unique_ptr<TelemetryRecorder> telemetryRecorder_;
        std::unique_ptr<AbstractStepper> stepper_{nullptr};
        double stepperUpdatePeriod_{INF};
        StepperState stepperState_{};
        vector_aligned_t<RobotData> robotDataVec_{};
        CouplingForceVector couplingForces_{};
        vector_aligned_t<ForceVector> contactForcesPrev_{};
        vector_aligned_t<ForceVector> fPrev_{};
        vector_aligned_t<MotionVector> aPrev_{};
        std::vector<double> energy_{};
        std::shared_ptr<LogData> logData_{nullptr};
    };
}

#endif  // JIMINY_ENGINE_MULTIROBOT_H
