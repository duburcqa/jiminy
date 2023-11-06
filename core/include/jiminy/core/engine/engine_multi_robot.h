#ifndef JIMINY_ENGINE_MULTIROBOT_H
#define JIMINY_ENGINE_MULTIROBOT_H

#include <functional>

#include "jiminy/core/macros.h"
#include "jiminy/core/types.h"
#include "jiminy/core/constants.h"
#include "jiminy/core/telemetry/telemetry_sender.h"
#include "jiminy/core/engine/system.h"


namespace jiminy
{
    const std::string ENGINE_TELEMETRY_NAMESPACE("HighLevelController");

    enum class JIMINY_DLLAPI contactModel_t : uint8_t
    {
        NONE = 0,
        SPRING_DAMPER = 1,
        CONSTRAINT = 2
    };

    enum class JIMINY_DLLAPI constraintSolver_t : uint8_t
    {
        NONE = 0,
        PGS = 1  // Projected Gauss-Seidel
    };

    const std::map<std::string, contactModel_t> CONTACT_MODELS_MAP{
        {"spring_damper", contactModel_t::SPRING_DAMPER},
        {   "constraint",    contactModel_t::CONSTRAINT},
    };

    const std::map<std::string, constraintSolver_t> CONSTRAINT_SOLVERS_MAP{
        {"PGS", constraintSolver_t::PGS}
    };

    const std::set<std::string> STEPPERS{"euler_explicit", "runge_kutta_4", "runge_kutta_dopri5"};

    class Timer;
    class Robot;
    class AbstractConstraintBase;
    class AbstractController;
    class AbstractStepper;
    class TelemetryData;
    class TelemetryRecorder;
    struct logData_t;

    using forceCouplingRegister_t = std::vector<forceCoupling_t>;

    struct JIMINY_DLLAPI stepperState_t
    {
    public:
        void reset(const float64_t & dtInit,
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
        uint32_t iter;
        uint32_t iterFailed;
        float64_t t;
        float64_t tPrev;
        /// \brief Internal buffer used for Kahan algorithm storing the residual sum of errors.
        float64_t tError;
        float64_t dt;
        float64_t dtLargest;
        float64_t dtLargestPrev;
        std::vector<Eigen::VectorXd> qSplit;
        std::vector<Eigen::VectorXd> vSplit;
        std::vector<Eigen::VectorXd> aSplit;
    };

    class JIMINY_DLLAPI EngineMultiRobot
    {
    public:
        configHolder_t getDefaultConstraintOptions()
        {
            configHolder_t config;
            config["solver"] = std::string("PGS");  // ["PGS",]
            /// \brief Relative inverse damping wrt diagonal of J.Minv.J.t.
            ///
            /// \details 0.0 enforces the minimum absolute regularizer.
            config["regularization"] = 1.0e-3;
            config["successiveSolveFailedMax"] = 100U;

            return config;
        };

        configHolder_t getDefaultContactOptions()
        {
            configHolder_t config;
            config["model"] = std::string("constraint");  // ["spring_damper", "constraint"]
            config["stiffness"] = 1.0e6;
            config["damping"] = 2.0e3;
            config["friction"] = 1.0;
            config["torsion"] = 0.0;
            config["transitionEps"] = 1.0e-3;       // [m]
            config["transitionVelocity"] = 1.0e-2;  // [m.s-1]
            config["stabilizationFreq"] = 20.0;     // [s-1]: 0.0 to disable

            return config;
        };

        configHolder_t getDefaultJointOptions()
        {
            configHolder_t config;
            config["boundStiffness"] = 1.0e7;
            config["boundDamping"] = 1.0e4;

            return config;
        };

        configHolder_t getDefaultWorldOptions()
        {
            configHolder_t config;
            config["gravity"] = (Eigen::VectorXd(6) << 0.0, 0.0, -9.81, 0.0, 0.0, 0.0).finished();
            config["groundProfile"] = heightmapFunctor_t(
                [](const Eigen::Vector3d & /* pos */) -> std::pair<float64_t, Eigen::Vector3d> {
                    return {0.0, Eigen::Vector3d::UnitZ()};
                });

            return config;
        };

        configHolder_t getDefaultStepperOptions()
        {
            configHolder_t config;
            config["verbose"] = false;
            config["randomSeed"] = 0U;
            /// \details Must be either "runge_kutta_dopri5", "runge_kutta_4" or "euler_explicit".
            config["odeSolver"] = std::string("runge_kutta_dopri5");
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

        configHolder_t getDefaultTelemetryOptions()
        {
            configHolder_t config;
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

        configHolder_t getDefaultEngineOptions()
        {
            configHolder_t config;
            config["telemetry"] = getDefaultTelemetryOptions();
            config["stepper"] = getDefaultStepperOptions();
            config["world"] = getDefaultWorldOptions();
            config["joints"] = getDefaultJointOptions();
            config["constraints"] = getDefaultConstraintOptions();
            config["contacts"] = getDefaultContactOptions();

            return config;
        };

        struct constraintOptions_t
        {
            const std::string solver;
            const float64_t regularization;
            const uint32_t successiveSolveFailedMax;

            constraintOptions_t(const configHolder_t & options) :
            solver(boost::get<std::string>(options.at("solver"))),
            regularization(boost::get<float64_t>(options.at("regularization"))),
            successiveSolveFailedMax(boost::get<uint32_t>(options.at("successiveSolveFailedMax")))
            {
            }
        };

        struct contactOptions_t
        {
            const std::string model;
            const float64_t stiffness;
            const float64_t damping;
            const float64_t friction;
            const float64_t torsion;
            const float64_t transitionEps;
            const float64_t transitionVelocity;
            const float64_t stabilizationFreq;

            contactOptions_t(const configHolder_t & options) :
            model(boost::get<std::string>(options.at("model"))),
            stiffness(boost::get<float64_t>(options.at("stiffness"))),
            damping(boost::get<float64_t>(options.at("damping"))),
            friction(boost::get<float64_t>(options.at("friction"))),
            torsion(boost::get<float64_t>(options.at("torsion"))),
            transitionEps(boost::get<float64_t>(options.at("transitionEps"))),
            transitionVelocity(boost::get<float64_t>(options.at("transitionVelocity"))),
            stabilizationFreq(boost::get<float64_t>(options.at("stabilizationFreq")))
            {
            }
        };

        struct jointOptions_t
        {
            const float64_t boundStiffness;
            const float64_t boundDamping;

            jointOptions_t(const configHolder_t & options) :
            boundStiffness(boost::get<float64_t>(options.at("boundStiffness"))),
            boundDamping(boost::get<float64_t>(options.at("boundDamping")))
            {
            }
        };

        struct worldOptions_t
        {
            const Eigen::VectorXd gravity;
            const heightmapFunctor_t groundProfile;

            worldOptions_t(const configHolder_t & options) :
            gravity(boost::get<Eigen::VectorXd>(options.at("gravity"))),
            groundProfile(boost::get<heightmapFunctor_t>(options.at("groundProfile")))
            {
            }
        };

        struct stepperOptions_t
        {
            const bool_t verbose;
            const uint32_t randomSeed;
            const std::string odeSolver;
            const float64_t tolAbs;
            const float64_t tolRel;
            const float64_t dtMax;
            const float64_t dtRestoreThresholdRel;
            const uint32_t successiveIterFailedMax;
            const uint32_t iterMax;
            const float64_t timeout;
            const float64_t sensorsUpdatePeriod;
            const float64_t controllerUpdatePeriod;
            const bool_t logInternalStepperSteps;

            stepperOptions_t(const configHolder_t & options) :
            verbose(boost::get<bool_t>(options.at("verbose"))),
            randomSeed(boost::get<uint32_t>(options.at("randomSeed"))),
            odeSolver(boost::get<std::string>(options.at("odeSolver"))),
            tolAbs(boost::get<float64_t>(options.at("tolAbs"))),
            tolRel(boost::get<float64_t>(options.at("tolRel"))),
            dtMax(boost::get<float64_t>(options.at("dtMax"))),
            dtRestoreThresholdRel(boost::get<float64_t>(options.at("dtRestoreThresholdRel"))),
            successiveIterFailedMax(boost::get<uint32_t>(options.at("successiveIterFailedMax"))),
            iterMax(boost::get<uint32_t>(options.at("iterMax"))),
            timeout(boost::get<float64_t>(options.at("timeout"))),
            sensorsUpdatePeriod(boost::get<float64_t>(options.at("sensorsUpdatePeriod"))),
            controllerUpdatePeriod(boost::get<float64_t>(options.at("controllerUpdatePeriod"))),
            logInternalStepperSteps(boost::get<bool_t>(options.at("logInternalStepperSteps")))
            {
            }
        };

        struct telemetryOptions_t
        {
            const bool_t isPersistent;
            const bool_t enableConfiguration;
            const bool_t enableVelocity;
            const bool_t enableAcceleration;
            const bool_t enableForceExternal;
            const bool_t enableCommand;
            const bool_t enableMotorEffort;
            const bool_t enableEnergy;

            telemetryOptions_t(const configHolder_t & options) :
            isPersistent(boost::get<bool_t>(options.at("isPersistent"))),
            enableConfiguration(boost::get<bool_t>(options.at("enableConfiguration"))),
            enableVelocity(boost::get<bool_t>(options.at("enableVelocity"))),
            enableAcceleration(boost::get<bool_t>(options.at("enableAcceleration"))),
            enableForceExternal(boost::get<bool_t>(options.at("enableForceExternal"))),
            enableCommand(boost::get<bool_t>(options.at("enableCommand"))),
            enableMotorEffort(boost::get<bool_t>(options.at("enableMotorEffort"))),
            enableEnergy(boost::get<bool_t>(options.at("enableEnergy")))
            {
            }
        };

        struct engineOptions_t
        {
            const telemetryOptions_t telemetry;
            const stepperOptions_t stepper;
            const worldOptions_t world;
            const jointOptions_t joints;
            const constraintOptions_t constraints;
            const contactOptions_t contacts;

            engineOptions_t(const configHolder_t & options) :
            telemetry(boost::get<configHolder_t>(options.at("telemetry"))),
            stepper(boost::get<configHolder_t>(options.at("stepper"))),
            world(boost::get<configHolder_t>(options.at("world"))),
            joints(boost::get<configHolder_t>(options.at("joints"))),
            constraints(boost::get<configHolder_t>(options.at("constraints"))),
            contacts(boost::get<configHolder_t>(options.at("contacts")))
            {
            }
        };

    public:
        DISABLE_COPY(EngineMultiRobot)

    public:
        EngineMultiRobot();
        virtual ~EngineMultiRobot();

        hresult_t addSystem(const std::string & systemName,
                            std::shared_ptr<Robot> robot,
                            std::shared_ptr<AbstractController> controller,
                            callbackFunctor_t callbackFct);
        hresult_t addSystem(const std::string & systemName,
                            std::shared_ptr<Robot> robot,
                            callbackFunctor_t callbackFct);
        hresult_t removeSystem(const std::string & systemName);

        hresult_t setController(const std::string & systemName,
                                std::shared_ptr<AbstractController> controller);

        /// \brief Add a force linking both systems together.
        ///
        /// \details This function registers a callback function forceFct that links both systems
        ///          by a given force. This function must return the force that the second systems
        ///          applies to the first system, in the global frame of the first frame, i.e.
        ///          expressed at the origin of the first frame, in word coordinates.
        ///
        /// \param[in] systemName1 Name of the system receiving the force.
        /// \param[in] systemName2 Name of the system applying the force.
        /// \param[in] frameName1 Frame on the first system where the force is applied.
        /// \param[in] frameName2 Frame on the second system where the opposite force is applied.
        /// \param[in] forceFct Callback function returning the force that systemName2 applies on
        ///                     systemName1, in the global frame of frameName1.
        hresult_t registerForceCoupling(const std::string & systemName1,
                                        const std::string & systemName2,
                                        const std::string & frameName1,
                                        const std::string & frameName2,
                                        forceCouplingFunctor_t forceFct);
        hresult_t registerViscoelasticDirectionalForceCoupling(const std::string & systemName1,
                                                               const std::string & systemName2,
                                                               const std::string & frameName1,
                                                               const std::string & frameName2,
                                                               const float64_t & stiffness,
                                                               const float64_t & damping,
                                                               const float64_t & restLength = 0.0);
        hresult_t registerViscoelasticDirectionalForceCoupling(const std::string & systemName,
                                                               const std::string & frameName1,
                                                               const std::string & frameName2,
                                                               const float64_t & stiffness,
                                                               const float64_t & damping,
                                                               const float64_t & restLength = 0.0);

        /// \brief 6-DoFs spring-damper coupling force modelling a flexible bushing.
        ///
        /// \details The spring-damper forces are computed at a frame being the linear
        ///          interpolation (according on double-geodesic) between frame 1 and 2 by a factor
        ///          alpha. In particular, alpha = 0.0, 0.5, and 1.0 corresponds to frame 1,
        ///          halfway point, and frame 2.
        ///
        /// \see See official drake documentation:
        ///      https://drake.mit.edu/doxygen_cxx/classdrake_1_1multibody_1_1_linear_bushing_roll_pitch_yaw.html
        hresult_t registerViscoelasticForceCoupling(const std::string & systemName1,
                                                    const std::string & systemName2,
                                                    const std::string & frameName1,
                                                    const std::string & frameName2,
                                                    const Vector6d & stiffness,
                                                    const Vector6d & damping,
                                                    const float64_t & alpha = 0.5);
        hresult_t registerViscoelasticForceCoupling(const std::string & systemName,
                                                    const std::string & frameName1,
                                                    const std::string & frameName2,
                                                    const Vector6d & stiffness,
                                                    const Vector6d & damping,
                                                    const float64_t & alpha = 0.5);
        hresult_t removeForcesCoupling(const std::string & systemName1,
                                       const std::string & systemName2);
        hresult_t removeForcesCoupling(const std::string & systemName);
        hresult_t removeForcesCoupling();

        const forceCouplingRegister_t & getForcesCoupling() const;

        hresult_t removeAllForces();

        /// \brief Reset engine.
        ///
        /// \details This function resets the engine, the robot and the controller.
        ///          This method is made to be called in between simulations, to allow registering
        ///          of new variables to the telemetry, and reset the random number generators.
        ///
        /// \param[in] resetRandomNumbers Whether to reset the random number generators.
        /// \param[in] removeAllForce Whether to remove registered external forces.
        void reset(const bool_t & resetRandomNumbers = false,
                   const bool_t & removeAllForce = false);

        /// \brief Reset the engine and compute initial state.
        ///
        /// \warning This function does NOT reset the engine, robot and controller. It is up to
        ///          the user to do so, by calling `reset` method first.
        ///
        /// \param[in] qInit Initial configuration of every system.
        /// \param[in] vInit Initial velocity of every system.
        /// \param[in] aInit Initial acceleration of every system.
        ///                  Optional: Zero by default.
        hresult_t start(
            const std::map<std::string, Eigen::VectorXd> & qInit,
            const std::map<std::string, Eigen::VectorXd> & vInit,
            const std::optional<std::map<std::string, Eigen::VectorXd>> & aInit = std::nullopt);

        /// \brief Integrate system from current state for a duration equal to stepSize
        ///
        /// \details This function performs a single 'integration step', in the sense that only the
        ///          endpoint is added to the log. The integrator object is allowed to perform
        ///          multiple steps inside of this interval.
        ///
        /// \remarks One may specify a negative timestep to use the default update value.
        ///
        /// \param[in] stepSize Duration for which to integrate ; set to negative value to use
        /// default update value.
        hresult_t step(float64_t stepSize = -1);

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
        hresult_t simulate(
            const float64_t & tEnd,
            const std::map<std::string, Eigen::VectorXd> & qInit,
            const std::map<std::string, Eigen::VectorXd> & vInit,
            const std::optional<std::map<std::string, Eigen::VectorXd>> & aInit = std::nullopt);

        /// \brief Apply an impulse force on a frame for a given duration at the desired time.
        ///
        /// \warning The force must be given in the world frame.
        hresult_t registerForceImpulse(const std::string & systemName,
                                       const std::string & frameName,
                                       const float64_t & t,
                                       const float64_t & dt,
                                       const pinocchio::Force & F);
        /// \brief Apply an external force profile on a frame.
        ///
        /// \details It can be either time-continuous or discrete. The force can be time- and
        ///          state-dependent, and must be given in the world frame.
        hresult_t registerForceProfile(const std::string & systemName,
                                       const std::string & frameName,
                                       const forceProfileFunctor_t & forceFct,
                                       const float64_t & updatePeriod = 0.0);

        hresult_t removeForcesImpulse(const std::string & systemName);
        hresult_t removeForcesProfile(const std::string & systemName);
        hresult_t removeForcesImpulse();
        hresult_t removeForcesProfile();

        hresult_t getForcesImpulse(const std::string & systemName,
                                   const forceImpulseRegister_t *& forcesImpulsePtr) const;
        hresult_t getForcesProfile(const std::string & systemName,
                                   const forceProfileRegister_t *& forcesProfilePtr) const;

        configHolder_t getOptions() const;
        hresult_t setOptions(const configHolder_t & engineOptions);
        bool_t getIsTelemetryConfigured() const;
        std::vector<std::string> getSystemsNames() const;
        hresult_t getSystemIdx(const std::string & systemName, int32_t & systemIdx) const;
        hresult_t getSystem(const std::string & systemName, systemHolder_t *& system);
        hresult_t getSystemState(const std::string & systemName,
                                 const systemState_t *& systemState) const;
        const stepperState_t & getStepperState() const;
        const bool_t & getIsSimulationRunning() const;
        static float64_t getMaxSimulationDuration();
        static float64_t getTelemetryTimeUnit();

        static void computeForwardKinematics(systemHolder_t & system,
                                             const Eigen::VectorXd & q,
                                             const Eigen::VectorXd & v,
                                             const Eigen::VectorXd & a);
        hresult_t computeSystemsDynamics(const float64_t & t,
                                         const std::vector<Eigen::VectorXd> & qSplit,
                                         const std::vector<Eigen::VectorXd> & vSplit,
                                         std::vector<Eigen::VectorXd> & aSplit,
                                         const bool_t & isStateUpToDate = false);

    protected:
        hresult_t configureTelemetry();
        void updateTelemetry();

        void syncStepperStateWithSystems();
        void syncSystemsStateWithStepper(const bool_t & isStateUpToDate = false);


        /// \brief Compute the force resulting from ground contact on a given body.
        ///
        /// \param[in] system System for which to perform computation.
        /// \param[in] collisionPairIdx Index of the collision pair associated with the body.
        ///
        /// \returns Contact force, at parent joint, in the local frame.
        void computeContactDynamicsAtBody(
            const systemHolder_t & system,
            const pairIndex_t & collisionPairIdx,
            std::shared_ptr<AbstractConstraintBase> & contactConstraint,
            pinocchio::Force & fextLocal) const;

        /// \brief Compute the force resulting from ground contact on a given frame.
        ///
        /// \param[in] system System for which to perform computation.
        /// \param[in] frameIdx Index of the frame in contact.
        ///
        /// \returns Contact force, at parent joint, in the local frame.
        void computeContactDynamicsAtFrame(
            const systemHolder_t & system,
            const frameIndex_t & frameIdx,
            std::shared_ptr<AbstractConstraintBase> & collisionConstraint,
            pinocchio::Force & fextLocal) const;

        /// \brief Compute the ground reaction force for a given normal direction and depth.
        pinocchio::Force computeContactDynamics(const Eigen::Vector3d & nGround,
                                                const float64_t & depth,
                                                const Eigen::Vector3d & vContactInWorld) const;

        void computeCommand(systemHolder_t & system,
                            const float64_t & t,
                            const Eigen::VectorXd & q,
                            const Eigen::VectorXd & v,
                            Eigen::VectorXd & command);
        void computeInternalDynamics(const systemHolder_t & system,
                                     systemDataHolder_t & systemData,
                                     const float64_t & t,
                                     const Eigen::VectorXd & q,
                                     const Eigen::VectorXd & v,
                                     Eigen::VectorXd & uInternal) const;
        void computeCollisionForces(const systemHolder_t & system,
                                    systemDataHolder_t & systemData,
                                    forceVector_t & fext,
                                    const bool_t & isStateUpToDate = false) const;
        void computeExternalForces(const systemHolder_t & system,
                                   systemDataHolder_t & systemData,
                                   const float64_t & t,
                                   const Eigen::VectorXd & q,
                                   const Eigen::VectorXd & v,
                                   forceVector_t & fext);
        void computeForcesCoupling(const float64_t & t,
                                   const std::vector<Eigen::VectorXd> & qSplit,
                                   const std::vector<Eigen::VectorXd> & vSplit);
        void computeAllTerms(const float64_t & t,
                             const std::vector<Eigen::VectorXd> & qSplit,
                             const std::vector<Eigen::VectorXd> & vSplit,
                             const bool_t & isStateUpToDate = false);

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
        const Eigen::VectorXd & computeAcceleration(systemHolder_t & system,
                                                    systemDataHolder_t & systemData,
                                                    const Eigen::VectorXd & q,
                                                    const Eigen::VectorXd & v,
                                                    const Eigen::VectorXd & u,
                                                    forceVector_t & fext,
                                                    const bool_t & isStateUpToDate = false,
                                                    const bool_t & ignoreBounds = false);

    public:
        hresult_t getLog(std::shared_ptr<const logData_t> & logData);

        static hresult_t readLog(
            const std::string & filename, const std::string & format, logData_t & logData);

        hresult_t writeLog(const std::string & filename, const std::string & format);

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
                      const bool_t & update_kinematics);
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
        std::unique_ptr<const engineOptions_t> engineOptions_;
        std::vector<systemHolder_t> systems_;

    protected:
        bool_t isTelemetryConfigured_;
        bool_t isSimulationRunning_;
        configHolder_t engineOptionsHolder_;

    private:
        std::unique_ptr<Timer> timer_;
        contactModel_t contactModel_;
        TelemetrySender telemetrySender_;
        std::shared_ptr<TelemetryData> telemetryData_;
        std::unique_ptr<TelemetryRecorder> telemetryRecorder_;
        std::unique_ptr<AbstractStepper> stepper_;
        float64_t stepperUpdatePeriod_;
        stepperState_t stepperState_;
        vector_aligned_t<systemDataHolder_t> systemsDataHolder_;
        forceCouplingRegister_t forcesCoupling_;
        vector_aligned_t<forceVector_t> contactForcesPrev_;
        vector_aligned_t<forceVector_t> fPrev_;
        vector_aligned_t<motionVector_t> aPrev_;
        std::vector<float64_t> energy_;
        std::shared_ptr<logData_t> logData_;
    };
}

#endif  // JIMINY_ENGINE_MULTIROBOT_H
