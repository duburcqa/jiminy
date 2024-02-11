#ifndef JIMINY_ENGINE_MULTIROBOT_H
#define JIMINY_ENGINE_MULTIROBOT_H

#include <optional>
#include <functional>

#include "jiminy/core/fwd.h"
#include "jiminy/core/utilities/helpers.h"  // `Timer`
#include "jiminy/core/utilities/random.h"   // `PCG32`
#include "jiminy/core/engine/system.h"


namespace jiminy
{
    inline constexpr std::string_view ENGINE_TELEMETRY_NAMESPACE{"HighLevelController"};

    enum class JIMINY_DLLAPI ContactModelType : uint8_t
    {
        UNSUPPORTED = 0,
        SPRING_DAMPER = 1,
        CONSTRAINT = 2
    };

    enum class JIMINY_DLLAPI ConstraintSolverType : uint8_t
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

    const std::set<std::string> STEPPERS{"euler_explicit", "runge_kutta_4", "runge_kutta_dopri5"};

    class Robot;
    class AbstractConstraintBase;
    class AbstractController;
    class AbstractStepper;
    class TelemetryData;
    class TelemetryRecorder;
    class TelemetrySender;
    struct LogData;

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
        double dt{0.0};
        double dtLargest{0.0};
        double dtLargestPrev{0.0};
        std::vector<Eigen::VectorXd> qSplit{};
        std::vector<Eigen::VectorXd> vSplit{};
        std::vector<Eigen::VectorXd> aSplit{};
    };

    class JIMINY_DLLAPI EngineMultiRobot
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

        GenericConfig getDefaultJointOptions()
        {
            GenericConfig config;
            config["boundStiffness"] = 1.0e7;
            config["boundDamping"] = 1.0e4;

            return config;
        };

        GenericConfig getDefaultWorldOptions()
        {
            GenericConfig config;
            config["gravity"] = (Eigen::VectorXd(6) << 0.0, 0.0, -9.81, 0.0, 0.0, 0.0).finished();
            config["groundProfile"] = HeightmapFunction(
                [](const Eigen::Vector2d & /* xy */,
                   double & height,
                   Eigen::Vector3d & normal) -> void
                {
                    height = 0.0;
                    normal = Eigen::Vector3d::UnitZ();
                });

            return config;
        };

        GenericConfig getDefaultStepperOptions()
        {
            GenericConfig config;
            config["verbose"] = false;
            config["randomSeedSeq"] = VectorX<uint32_t>::Zero(1).eval();
            /// \details Must be either "runge_kutta_dopri5", "runge_kutta_4" or "euler_explicit".
            config["odeSolver"] = std::string{"runge_kutta_dopri5"};
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
            config["enableCommand"] = true;
            config["enableMotorEffort"] = true;
            config["enableEnergy"] = true;
            return config;
        };

        GenericConfig getDefaultEngineOptions()
        {
            GenericConfig config;
            config["telemetry"] = getDefaultTelemetryOptions();
            config["stepper"] = getDefaultStepperOptions();
            config["world"] = getDefaultWorldOptions();
            config["joints"] = getDefaultJointOptions();
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

        struct JointOptions
        {
            const double boundStiffness;
            const double boundDamping;

            JointOptions(const GenericConfig & options) :
            boundStiffness{boost::get<double>(options.at("boundStiffness"))},
            boundDamping{boost::get<double>(options.at("boundDamping"))}
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
            const bool enableCommand;
            const bool enableMotorEffort;
            const bool enableEnergy;

            TelemetryOptions(const GenericConfig & options) :
            isPersistent{boost::get<bool>(options.at("isPersistent"))},
            enableConfiguration{boost::get<bool>(options.at("enableConfiguration"))},
            enableVelocity{boost::get<bool>(options.at("enableVelocity"))},
            enableAcceleration{boost::get<bool>(options.at("enableAcceleration"))},
            enableForceExternal{boost::get<bool>(options.at("enableForceExternal"))},
            enableCommand{boost::get<bool>(options.at("enableCommand"))},
            enableMotorEffort{boost::get<bool>(options.at("enableMotorEffort"))},
            enableEnergy{boost::get<bool>(options.at("enableEnergy"))}
            {
            }
        };

        struct EngineOptions
        {
            const TelemetryOptions telemetry;
            const StepperOptions stepper;
            const WorldOptions world;
            const JointOptions joints;
            const ConstraintOptions constraints;
            const ContactOptions contacts;

            EngineOptions(const GenericConfig & options) :
            telemetry{boost::get<GenericConfig>(options.at("telemetry"))},
            stepper{boost::get<GenericConfig>(options.at("stepper"))},
            world{boost::get<GenericConfig>(options.at("world"))},
            joints{boost::get<GenericConfig>(options.at("joints"))},
            constraints{boost::get<GenericConfig>(options.at("constraints"))},
            contacts{boost::get<GenericConfig>(options.at("contacts"))}
            {
            }
        };

    public:
        DISABLE_COPY(EngineMultiRobot)

    public:
        explicit EngineMultiRobot() noexcept;
        ~EngineMultiRobot();

        void addSystem(const std::string & systemName,
                       std::shared_ptr<Robot> robot,
                       std::shared_ptr<AbstractController> controller,
                       const AbortSimulationFunction & callback);
        void addSystem(const std::string & systemName,
                       std::shared_ptr<Robot> robot,
                       const AbortSimulationFunction & callback);
        void removeSystem(const std::string & systemName);

        void setController(const std::string & systemName,
                           std::shared_ptr<AbstractController> controller);

        /// \brief Add a force linking both systems together.
        ///
        /// \details This function registers a callback function that links both systems by a given
        ///          force. This function must return the force that the second systems applies to
        ///          the first system, in the global frame of the first frame, i.e. expressed at
        ///          the origin of the first frame, in word coordinates.
        ///
        /// \param[in] systemName1 Name of the system receiving the force.
        /// \param[in] systemName2 Name of the system applying the force.
        /// \param[in] frameName1 Frame on the first system where the force is applied.
        /// \param[in] frameName2 Frame on the second system where the opposite force is applied.
        /// \param[in] forceFunc Callback function returning the force that systemName2 applies on
        ///                      systemName1, in the global frame of frameName1.
        void registerCouplingForce(const std::string & systemName1,
                                   const std::string & systemName2,
                                   const std::string & frameName1,
                                   const std::string & frameName2,
                                   const CouplingForceFunction & forceFunc);
        void registerViscoelasticDirectionalCouplingForce(const std::string & systemName1,
                                                          const std::string & systemName2,
                                                          const std::string & frameName1,
                                                          const std::string & frameName2,
                                                          double stiffness,
                                                          double damping,
                                                          double restLength = 0.0);
        void registerViscoelasticDirectionalCouplingForce(const std::string & systemName,
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
        void registerViscoelasticCouplingForce(const std::string & systemName1,
                                               const std::string & systemName2,
                                               const std::string & frameName1,
                                               const std::string & frameName2,
                                               const Vector6d & stiffness,
                                               const Vector6d & damping,
                                               double alpha = 0.5);
        void registerViscoelasticCouplingForce(const std::string & systemName,
                                               const std::string & frameName1,
                                               const std::string & frameName2,
                                               const Vector6d & stiffness,
                                               const Vector6d & damping,
                                               double alpha = 0.5);
        void removeCouplingForces(const std::string & systemName1,
                                  const std::string & systemName2);
        void removeCouplingForces(const std::string & systemName);
        void removeCouplingForces();

        const CouplingForceVector & getCouplingForces() const;

        void removeAllForces();

        /// \brief Reset engine.
        ///
        /// \details This function resets the engine, the robot and the controller.
        ///          This method is made to be called in between simulations, to allow registering
        ///          of new variables to the telemetry, and reset the random number generators.
        ///
        /// \param[in] resetRandomNumbers Whether to reset the random number generators.
        /// \param[in] removeAllForce Whether to remove registered external forces.
        void reset(bool resetRandomNumbers = false, bool removeAllForce = false);

        /// \brief Reset the engine and compute initial state.
        ///
        /// \warning This function does NOT reset the engine, robot and controller. It is up to
        ///          the user to do so, by calling `reset` method first.
        ///
        /// \param[in] qInit Initial configuration of every system.
        /// \param[in] vInit Initial velocity of every system.
        /// \param[in] aInit Initial acceleration of every system.
        ///                  Optional: Zero by default.
        void start(
            const std::map<std::string, Eigen::VectorXd> & qInit,
            const std::map<std::string, Eigen::VectorXd> & vInit,
            const std::optional<std::map<std::string, Eigen::VectorXd>> & aInit = std::nullopt);

        /// \brief Integrate system dynamics from current state for a given duration.
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
        /// \param[in] tEnd End time, i.e. amount of time to simulate.
        /// \param[in] qInit Initial configuration of every system, i.e. at t=0.0.
        /// \param[in] vInit Initial velocity of every system, i.e. at t=0.0.
        /// \param[in] aInit Initial acceleration of every system, i.e. at t=0.0.
        ///                  Optional: Zero by default.
        void simulate(
            double tEnd,
            const std::map<std::string, Eigen::VectorXd> & qInit,
            const std::map<std::string, Eigen::VectorXd> & vInit,
            const std::optional<std::map<std::string, Eigen::VectorXd>> & aInit = std::nullopt);

        /// \brief Apply an impulse force on a frame for a given duration at the desired time.
        ///
        /// \warning The force must be given in the world frame.
        void registerImpulseForce(const std::string & systemName,
                                  const std::string & frameName,
                                  double t,
                                  double dt,
                                  const pinocchio::Force & force);
        /// \brief Apply an external force profile on a frame.
        ///
        /// \details It can be either time-continuous or discrete. The force can be time- and
        ///          state-dependent, and must be given in the world frame.
        void registerProfileForce(const std::string & systemName,
                                  const std::string & frameName,
                                  const ProfileForceFunction & forceFunc,
                                  double updatePeriod = 0.0);

        void removeImpulseForces(const std::string & systemName);
        void removeProfileForces(const std::string & systemName);
        void removeImpulseForces();
        void removeProfileForces();

        const ImpulseForceVector & getImpulseForces(const std::string & systemName) const;
        const ProfileForceVector & getProfileForces(const std::string & systemName) const;

        GenericConfig getOptions() const noexcept;
        void setOptions(const GenericConfig & engineOptions);
        bool getIsTelemetryConfigured() const;
        std::vector<std::string> getSystemNames() const;
        System & getSystem(const std::string & systemName);
        std::ptrdiff_t getSystemIndex(const std::string & systemName) const;
        const SystemState & getSystemState(const std::string & systemName) const;
        const StepperState & getStepperState() const;
        const bool & getIsSimulationRunning() const;  // return const reference on purpose
        static double getSimulationDurationMax();
        static double getTelemetryTimeUnit();

        static void computeForwardKinematics(System & system,
                                             const Eigen::VectorXd & q,
                                             const Eigen::VectorXd & v,
                                             const Eigen::VectorXd & a);
        void computeSystemsDynamics(double t,
                                    const std::vector<Eigen::VectorXd> & qSplit,
                                    const std::vector<Eigen::VectorXd> & vSplit,
                                    std::vector<Eigen::VectorXd> & aSplit,
                                    bool isStateUpToDate = false);

    protected:
        void configureTelemetry();
        void updateTelemetry();

        void syncStepperStateWithSystems();
        void syncSystemsStateWithStepper(bool isStateUpToDate = false);


        /// \brief Compute the force resulting from ground contact on a given body.
        ///
        /// \param[in] system System for which to perform computation.
        /// \param[in] collisionPairIndex Index of the collision pair associated with the body.
        ///
        /// \returns Contact force, at parent joint, in the local frame.
        void computeContactDynamicsAtBody(
            const System & system,
            const pinocchio::PairIndex & collisionPairIndex,
            std::shared_ptr<AbstractConstraintBase> & contactConstraint,
            pinocchio::Force & fextLocal) const;

        /// \brief Compute the force resulting from ground contact on a given frame.
        ///
        /// \param[in] system System for which to perform computation.
        /// \param[in] frameIndex Index of the frame in contact.
        ///
        /// \returns Contact force, at parent joint, in the local frame.
        void computeContactDynamicsAtFrame(
            const System & system,
            pinocchio::FrameIndex frameIndex,
            std::shared_ptr<AbstractConstraintBase> & collisionConstraint,
            pinocchio::Force & fextLocal) const;

        /// \brief Compute the ground reaction force for a given normal direction and depth.
        pinocchio::Force computeContactDynamics(const Eigen::Vector3d & nGround,
                                                double depth,
                                                const Eigen::Vector3d & vContactInWorld) const;

        void computeCommand(System & system,
                            double t,
                            const Eigen::VectorXd & q,
                            const Eigen::VectorXd & v,
                            Eigen::VectorXd & command);
        void computeInternalDynamics(const System & system,
                                     SystemData & systemData,
                                     double t,
                                     const Eigen::VectorXd & q,
                                     const Eigen::VectorXd & v,
                                     Eigen::VectorXd & uInternal) const;
        void computeCollisionForces(const System & system,
                                    SystemData & systemData,
                                    ForceVector & fext,
                                    bool isStateUpToDate = false) const;
        void computeExternalForces(const System & system,
                                   SystemData & systemData,
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

        /// \brief Compute system acceleration from current system state.
        ///
        /// \details This function performs forward dynamics computation, either with kinematic
        ///          constraints (using Lagrange multiplier for computing the forces) or
        ///          unconstrained (aba).
        ///
        /// \param[in] system System for which to compute the dynamics.
        /// \param[in] q Joint position.
        /// \param[in] v Joint velocity.
        /// \param[in] u Joint effort.
        /// \param[in] fext External forces applied on the system.
        ///
        /// \return System acceleration.
        const Eigen::VectorXd & computeAcceleration(System & system,
                                                    SystemData & systemData,
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
        std::vector<System> systems_{};

    protected:
        bool isTelemetryConfigured_{false};
        bool isSimulationRunning_{false};
        GenericConfig engineOptionsGeneric_{};
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
        vector_aligned_t<SystemData> systemDataVec_{};
        CouplingForceVector couplingForces_{};
        vector_aligned_t<ForceVector> contactForcesPrev_{};
        vector_aligned_t<ForceVector> fPrev_{};
        vector_aligned_t<MotionVector> aPrev_{};
        std::vector<double> energy_{};
        std::shared_ptr<LogData> logData_{nullptr};
    };
}

#endif  // JIMINY_ENGINE_MULTIROBOT_H
