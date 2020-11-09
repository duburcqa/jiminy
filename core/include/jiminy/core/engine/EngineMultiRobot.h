#ifndef JIMINY_ENGINE_MULTIROBOT_H
#define JIMINY_ENGINE_MULTIROBOT_H

#include <functional>

#include "jiminy/core/telemetry/TelemetrySender.h"
#include "jiminy/core/Utilities.h"
#include "jiminy/core/Types.h"
#include "jiminy/core/Constants.h"

#include "jiminy/core/engine/System.h"


namespace jiminy
{
    std::string const ENGINE_OBJECT_NAME("HighLevelController");

    std::set<std::string> const STEPPERS {
        "runge_kutta_4",
        "runge_kutta_dopri5",
        "explicit_euler"
    };

    float64_t const CONSTRAINT_INVERSION_DAMPING = 1.0e-12; ///< Damping factor used to perform matrix pseudo-inverse
                                                            /// when computing forward dynamics with constraints.

    class Robot;
    class AbstractController;
    class AbstractStepper;
    class TelemetryData;
    class TelemetryRecorder;
    struct logData_t;

    using forceCouplingRegister_t = std::vector<forceCoupling_t>;

    struct stepperState_t
    {
    public:
        stepperState_t(void) :
        iter(0U),
        iterFailed(0U),
        t(0.0),
        tPrev(0.0),
        tError(0.0),
        dt(0.0),
        dtLargest(0.0),
        dtLargestPrev(0.0),
        qSplit(),
        vSplit(),
        aSplit()
        {
            // Empty on purpose.
        }

        void reset(float64_t              const & dtInit,
                   std::vector<vectorN_t> const & qSplitInit,
                   std::vector<vectorN_t> const & vSplitInit)
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
            aSplit.clear();
            aSplit.reserve(vSplitInit.size());
            std::transform(vSplitInit.begin(), vSplitInit.end(),
                           std::back_inserter(aSplit),
                           [](auto const & v) -> vectorN_t
                           {
                               return vectorN_t::Zero(v.size());
                           });
        }

    public:
        uint32_t iter;
        uint32_t iterFailed;
        float64_t t;
        float64_t tPrev;
        float64_t tError; ///< Internal buffer used for Kahan algorithm storing the residual sum of errors
        float64_t dt;
        float64_t dtLargest;
        float64_t dtLargestPrev;
        std::vector<vectorN_t> qSplit;
        std::vector<vectorN_t> vSplit;
        std::vector<vectorN_t> aSplit;
    };

    class EngineMultiRobot
    {
    public:
        configHolder_t getDefaultContactOptions()
        {
            configHolder_t config;
            config["frictionViscous"] = 0.8;
            config["frictionDry"] = 1.0;
            config["frictionStictionVel"] = 1.0e-2;
            config["frictionStictionRatio"] = 0.5;
            config["stiffness"] = 1.0e6;
            config["damping"] = 2.0e3;
            config["transitionEps"] = 1.0e-3; // [m]

            return config;
        };

        configHolder_t getDefaultJointOptions()
        {
            configHolder_t config;
            config["boundStiffness"] = 1.0e5;
            config["boundDamping"] = 2.0e3;
            config["transitionPositionEps"] = 2.0e-3; // [rad] 2.0e-3 ~= 0.1 degrees
            config["transitionVelocityEps"] = 1.0e+1; // [rad.s-1]

            return config;
        };

        configHolder_t getDefaultWorldOptions()
        {
            configHolder_t config;
            config["gravity"] = (vectorN_t(6) << 0.0, 0.0, -9.81, 0.0, 0.0, 0.0).finished();
            config["groundProfile"] = heatMapFunctor_t(
                [](vector3_t const & pos) -> std::pair <float64_t, vector3_t>
                {
                    return {0.0, (vector3_t() << 0.0, 0.0, 1.0).finished()};
                });

            return config;
        };

        configHolder_t getDefaultStepperOptions()
        {
            configHolder_t config;
            config["verbose"] = false;
            config["randomSeed"] = 0U;
            config["odeSolver"] = std::string("runge_kutta_dopri5"); // ["runge_kutta_dopri5", "runge_kutta_4", "explicit_euler"]
            config["tolAbs"] = 1.0e-5;
            config["tolRel"] = 1.0e-4;
            config["dtMax"] = SIMULATION_MAX_TIMESTEP;
            config["dtRestoreThresholdRel"] = 0.2;
            config["successiveIterFailedMax"] = 1000U;
            config["iterMax"] = -1; // <= 0: disable
            config["timeout"] = 0.0; // <= 0.0: disable
            config["sensorsUpdatePeriod"] = 0.0;
            config["controllerUpdatePeriod"] = 0.0;
            config["logInternalStepperSteps"] = false;

            return config;
        };

        configHolder_t getDefaultTelemetryOptions()
        {
            configHolder_t config;
            config["enableConfiguration"] = true;
            config["enableVelocity"] = true;
            config["enableAcceleration"] = true;
            config["enableEffort"] = true;
            config["enableEnergy"] = true;
            config["timeUnit"] = 1.0e9;
            return config;
        };

        configHolder_t getDefaultEngineOptions()
        {
            configHolder_t config;
            config["telemetry"] = getDefaultTelemetryOptions();
            config["stepper"] = getDefaultStepperOptions();
            config["world"] = getDefaultWorldOptions();
            config["joints"] = getDefaultJointOptions();
            config["contacts"] = getDefaultContactOptions();

            return config;
        };

        struct contactOptions_t
        {
            float64_t const frictionViscous;
            float64_t const frictionDry;
            float64_t const frictionStictionVel;
            float64_t const frictionStictionRatio;
            float64_t const stiffness;
            float64_t const damping;
            float64_t const transitionEps;

            contactOptions_t(configHolder_t const & options) :
            frictionViscous(boost::get<float64_t>(options.at("frictionViscous"))),
            frictionDry(boost::get<float64_t>(options.at("frictionDry"))),
            frictionStictionVel(boost::get<float64_t>(options.at("frictionStictionVel"))),
            frictionStictionRatio(boost::get<float64_t>(options.at("frictionStictionRatio"))),
            stiffness(boost::get<float64_t>(options.at("stiffness"))),
            damping(boost::get<float64_t>(options.at("damping"))),
            transitionEps(boost::get<float64_t>(options.at("transitionEps")))
            {
                // Empty.
            }
        };

        struct jointOptions_t
        {
            float64_t const boundStiffness;
            float64_t const boundDamping;
            float64_t const transitionPositionEps;
            float64_t const transitionVelocityEps;

            jointOptions_t(configHolder_t const & options) :
            boundStiffness(boost::get<float64_t>(options.at("boundStiffness"))),
            boundDamping(boost::get<float64_t>(options.at("boundDamping"))),
            transitionPositionEps(boost::get<float64_t>(options.at("transitionPositionEps"))),
            transitionVelocityEps(boost::get<float64_t>(options.at("transitionVelocityEps")))
            {
                // Empty.
            }
        };

        struct worldOptions_t
        {
            vectorN_t const gravity;
            heatMapFunctor_t const groundProfile;

            worldOptions_t(configHolder_t const & options) :
            gravity(boost::get<vectorN_t>(options.at("gravity"))),
            groundProfile(boost::get<heatMapFunctor_t>(options.at("groundProfile")))
            {
                // Empty.
            }
        };

        struct stepperOptions_t
        {
            bool_t      const verbose;
            uint32_t    const randomSeed;
            std::string const odeSolver;
            float64_t   const tolAbs;
            float64_t   const tolRel;
            float64_t   const dtMax;
            float64_t   const dtRestoreThresholdRel;
            uint32_t    const successiveIterFailedMax;
            int32_t     const iterMax;
            float64_t   const timeout;
            float64_t   const sensorsUpdatePeriod;
            float64_t   const controllerUpdatePeriod;
            bool_t      const logInternalStepperSteps;

            stepperOptions_t(configHolder_t const & options) :
            verbose(boost::get<bool_t>(options.at("verbose"))),
            randomSeed(boost::get<uint32_t>(options.at("randomSeed"))),
            odeSolver(boost::get<std::string>(options.at("odeSolver"))),
            tolAbs(boost::get<float64_t>(options.at("tolAbs"))),
            tolRel(boost::get<float64_t>(options.at("tolRel"))),
            dtMax(boost::get<float64_t>(options.at("dtMax"))),
            dtRestoreThresholdRel(boost::get<float64_t>(options.at("dtRestoreThresholdRel"))),
            successiveIterFailedMax(boost::get<uint32_t>(options.at("successiveIterFailedMax"))),
            iterMax(boost::get<int32_t>(options.at("iterMax"))),
            timeout(boost::get<float64_t>(options.at("timeout"))),
            sensorsUpdatePeriod(boost::get<float64_t>(options.at("sensorsUpdatePeriod"))),
            controllerUpdatePeriod(boost::get<float64_t>(options.at("controllerUpdatePeriod"))),
            logInternalStepperSteps(boost::get<bool_t>(options.at("logInternalStepperSteps")))
            {
                // Empty.
            }
        };

        struct telemetryOptions_t
        {
            bool_t const enableConfiguration;
            bool_t const enableVelocity;
            bool_t const enableAcceleration;
            bool_t const enableEffort;
            bool_t const enableEnergy;
            float64_t const timeUnit;

            telemetryOptions_t(configHolder_t const & options) :
            enableConfiguration(boost::get<bool_t>(options.at("enableConfiguration"))),
            enableVelocity(boost::get<bool_t>(options.at("enableVelocity"))),
            enableAcceleration(boost::get<bool_t>(options.at("enableAcceleration"))),
            enableEffort(boost::get<bool_t>(options.at("enableEffort"))),
            enableEnergy(boost::get<bool_t>(options.at("enableEnergy"))),
            timeUnit(boost::get<float64_t>(options.at("timeUnit")))
            {
                // Empty.
            }
        };

        struct engineOptions_t
        {
            telemetryOptions_t const telemetry;
            stepperOptions_t   const stepper;
            worldOptions_t     const world;
            jointOptions_t     const joints;
            contactOptions_t   const contacts;

            engineOptions_t(configHolder_t const & options) :
            telemetry(boost::get<configHolder_t>(options.at("telemetry"))),
            stepper(boost::get<configHolder_t>(options.at("stepper"))),
            world(boost::get<configHolder_t>(options.at("world"))),
            joints(boost::get<configHolder_t>(options.at("joints"))),
            contacts(boost::get<configHolder_t>(options.at("contacts")))
            {
                // Empty.
            }
        };

    public:
        // Disable the copy of the class
        EngineMultiRobot(EngineMultiRobot const & engine) = delete;
        EngineMultiRobot & operator = (EngineMultiRobot const & other) = delete;

    public:
        EngineMultiRobot(void);
        ~EngineMultiRobot(void);

        hresult_t addSystem(std::string const & systemName,
                            std::shared_ptr<Robot> robot,
                            std::shared_ptr<AbstractController> controller,
                            callbackFunctor_t callbackFct);
        hresult_t addSystem(std::string const & systemName,
                            std::shared_ptr<Robot> robot,
                            callbackFunctor_t callbackFct);
        hresult_t removeSystem(std::string const & systemName);

        hresult_t setController(std::string const & systemName,
                                std::shared_ptr<AbstractController> controller);

        /// \brief Add a force linking both systems together
        ///
        /// \details This function registers a callback function forceFct that links
        ///          both systems by a given force. This function must return the
        ///          force that the second systems applies to the first system,
        ///          in the global frame of the first frame (i.e. expressed at the origin
        ///          of the first frame, in word coordinates).
        ///
        /// \param[in] systemName1 Name of the first system (the one receiving the force)
        /// \param[in] systemName2 Name of the second system (the one applying the force)
        /// \param[in] frameName1 Frame on the first system where the force is applied.
        /// \param[in] frameName2 Frame on the second system where
        ///                       (the opposite of) the force is applied.
        /// \param[in] forceFct Callback function returning the force that systemName2
        ///                     applies on systemName1, in the global frame of frameName1.
        hresult_t addCouplingForce(std::string const & systemName1,
                                   std::string const & systemName2,
                                   std::string const & frameName1,
                                   std::string const & frameName2,
                                   forceCouplingFunctor_t forceFct);
        hresult_t removeCouplingForces(std::string const & systemName1,
                                       std::string const & systemName2);
        hresult_t removeCouplingForces(std::string const & systemName);

        /// \brief Reset engine.
        ///
        /// \details This function resets the engine, the robot and the controller.
        ///          This method is made to be called in between simulations, to allow
        ///          registering of new variables to log, and reset the random number
        ///          generator.
        ///
        /// \param[in] resetDynamicForceRegister Whether or not to register the external force profiles applied
        ///                                      during the simulation.
        void reset(bool_t const & resetDynamicForceRegister = false);

        /// \brief Reset the engine and compute initial state.
        ///
        /// \details This function reset the engine, the robot and the controller, and update internal data
        ///          to match the given initial state.
        ///
        /// \param[in] qInit Initial configuration of every system.
        /// \param[in] vInit Initial velocity of every system.
        /// \param[in] resetRandomNumbers Whether or not to reset the random number generator.
        /// \param[in] resetDynamicForceRegister Whether or not to register the external force profiles applied
        ///                                      during the simulation.
        hresult_t start(std::map<std::string, vectorN_t> const & qInit,
                        std::map<std::string, vectorN_t> const & vInit,
                        bool_t const & resetRandomNumbers = false,
                        bool_t const & resetDynamicForceRegister = false);

        /// \brief Integrate system from current state for a duration equal to stepSize
        ///
        /// \details This function performs a single 'integration step', in the sense that only
        ///          the endpoint is added to the log. The integrator object is allowed to perform
        ///          multiple steps inside of this interval.
        ///          One may specify a negative timestep to use the default update value.
        ///
        /// \param[in] stepSize Duration for which to integrate ; set to negative value to use default update value.
        hresult_t step(float64_t stepSize = -1);

        /// \brief Stop the simulation.
        ///
        /// \details It releases the lock on the robot and the telemetry, so that
        ///          it is possible again to update the robot (for example to update
        ///          the options, add or remove sensors...) and to register new
        ///          variables or forces.
        void stop(void);

        /// \brief Run a simulation of duration tEnd, starting at xInit.
        ///
        /// \param[in] tEnd End time, i.e. amount of time to simulate.
        /// \param[in] qInit Initial configuration of every system, i.e. at t=0.0.
        /// \param[in] vInit Initial velocity of every system, i.e. at t=0.0.
        hresult_t simulate(float64_t const & tEnd,
                           std::map<std::string, vectorN_t> const & qInit,
                           std::map<std::string, vectorN_t> const & vInit);

        /// \brief Apply an impulse force on a frame for a given duration at the desired time.
        ///        The force must be given in the world frame.
        hresult_t registerForceImpulse(std::string      const & systemName,
                                       std::string      const & frameName,
                                       float64_t        const & t,
                                       float64_t        const & dt,
                                       pinocchio::Force const & F);

        /// \brief Apply an time-continuous external force on a frame.
        ///        The force can be time and state dependent, and must be given in the world frame.
        hresult_t registerForceProfile(std::string const & systemName,
                                       std::string const & frameName,
                                       forceProfileFunctor_t forceFct);

        configHolder_t getOptions(void) const;
        hresult_t setOptions(configHolder_t const & engineOptions);
        bool_t getIsTelemetryConfigured(void) const;
        std::vector<std::string> getSystemsNames(void) const;
        hresult_t getSystemIdx(std::string const & systemName,
                               int32_t           & systemIdx) const;
        hresult_t getSystem(std::string    const   & systemName,
                            systemHolder_t const * & system) const;
        hresult_t getSystem(std::string    const   & systemName,
                            systemHolder_t       * & system);
        hresult_t getSystemState(std::string   const   & systemName,
                                 systemState_t const * & systemState) const;
        stepperState_t const & getStepperState(void) const;
        bool_t const & getIsSimulationRunning(void) const;
        float64_t getMaxSimulationDuration(void) const;

        hresult_t computeSystemDynamics(float64_t              const & t,
                                        std::vector<vectorN_t> const & qSplit,
                                        std::vector<vectorN_t> const & vSplit,
                                        std::vector<vectorN_t>       & aSplit);

    protected:
        hresult_t configureTelemetry(void);
        void updateTelemetry(void);

        void syncStepperStateWithSystems(void);
        void syncSystemsStateWithStepper(void);

        void reset(bool_t const & resetRandomNumbers,
                   bool_t const & resetDynamicForceRegister);

        static void computeForwardKinematics(systemHolder_t  & system,
                                             vectorN_t const & q,
                                             vectorN_t const & v,
                                             vectorN_t const & a);

        /// \brief Compute the force resulting from ground contact on a given body.
        ///
        /// \param[in] system              System for which to perform computation.
        /// \param[in] collisionPairIdx    Id of the collision pair associated with the body
        /// \return Contact force, at parent joint, in the local frame.
        pinocchio::Force computeContactDynamicsAtBody(systemHolder_t const & system,
                                                      int32_t        const & collisionPairIdx) const;

        /// \brief Compute the force resulting from ground contact on a given frame.
        ///
        /// \param[in] system      System for which to perform computation.
        /// \param[in] frameIdx    Id of the frame in contact.
        /// \return Contact force, at parent joint, in the local frame.
        pinocchio::Force computeContactDynamicsAtFrame(systemHolder_t const & system,
                                                       int32_t        const & frameIdx) const;

        /// \brief Compute the force resulting from ground contact for a given normal direction and depth.
        pinocchio::Force computeContactDynamics(vector3_t const & nGround,
                                                float64_t const & depth,
                                                vector3_t const & vContactInWorld) const;

        void computeCommand(systemHolder_t  & system,
                            float64_t const & t,
                            vectorN_t const & q,
                            vectorN_t const & v,
                            vectorN_t       & u);
        void computeInternalDynamics(systemHolder_t  & system,
                                     float64_t const & t,
                                     vectorN_t const & q,
                                     vectorN_t const & v,
                                     vectorN_t       & u) const;
        void computeExternalForces(systemHolder_t     const & system,
                                   systemDataHolder_t const & systemData,
                                   float64_t          const & t,
                                   vectorN_t          const & q,
                                   vectorN_t          const & v,
                                   forceVector_t            & fext) const;
        void computeInternalForces(float64_t              const & t,
                                   std::vector<vectorN_t> const & qSplit,
                                   std::vector<vectorN_t> const & vSplit);
        void computeAllForces(float64_t              const & t,
                              std::vector<vectorN_t> const & qSplit,
                              std::vector<vectorN_t> const & vSplit);

        /// \brief Compute system acceleration from current system state.
        ///
        /// \details This function performs forward dynamics computation, either
        ///          with kinematic constraints (using Lagrange multiplier for computing the forces)
        ///          or unconstrained (aba).
        ///
        /// \param[in] system System for which to compute the dynamics.
        /// \param[in] q Joint position.
        /// \param[in] v Joint velocity.
        /// \param[in] u Joint effort.
        /// \param[in] fext External forces applied on the system.
        /// \return System acceleration.
        vectorN_t computeAcceleration(systemHolder_t       & system,
                                      vectorN_t      const & q,
                                      vectorN_t      const & v,
                                      vectorN_t      const & u,
                                      forceVector_t  const & fext);

    public:
        hresult_t getLogDataRaw(logData_t & logData);

        /// \brief Get the full logged content.
        ///
        /// \param[out] header      Header, vector of field names.
        /// \param[out] logMatrix   Corresponding data in the log file.
        hresult_t getLogData(std::vector<std::string> & header,
                             matrixN_t                & logMatrix);

        hresult_t writeLog(std::string const & filename,
                           std::string const & format = "binary");

        static hresult_t parseLogBinaryRaw(std::string const & filename,
                                           logData_t         & logData);
        static hresult_t parseLogBinary(std::string              const & filename,
                                        std::vector<std::string>       & header,
                                        matrixN_t                      & logMatrix);
    private:
        hresult_t writeLogCsv(std::string const & filename);
        hresult_t writeLogHdf5(std::string const & filename);

    private:
        template<typename Scalar, int Options, template<typename, int> class JointCollectionTpl,
                 typename ConfigVectorType, typename TangentVectorType>
        inline Scalar
        kineticEnergy(pinocchio::ModelTpl<Scalar,Options,JointCollectionTpl> const & model,
                      pinocchio::DataTpl<Scalar,Options,JointCollectionTpl>        & data,
                      Eigen::MatrixBase<ConfigVectorType>                    const & q,
                      Eigen::MatrixBase<TangentVectorType>                   const & v,
                      bool_t                                                 const & update_kinematics);
        template<typename Scalar, int Options, template<typename, int> class JointCollectionTpl,
                 typename ConfigVectorType, typename TangentVectorType1, typename TangentVectorType2,
                 typename ForceDerived>
        inline const typename pinocchio::DataTpl<Scalar,Options,JointCollectionTpl>::TangentVectorType &
        rnea(pinocchio::ModelTpl<Scalar,Options,JointCollectionTpl> const & model,
             pinocchio::DataTpl<Scalar,Options,JointCollectionTpl>        & data,
             Eigen::MatrixBase<ConfigVectorType>                    const & q,
             Eigen::MatrixBase<TangentVectorType1>                  const & v,
             Eigen::MatrixBase<TangentVectorType2>                  const & a,
             pinocchio::container::aligned_vector<ForceDerived>     const & fext);
        template<typename Scalar, int Options, template<typename, int> class JointCollectionTpl,
                 typename ConfigVectorType, typename TangentVectorType1, typename TangentVectorType2,
                 typename ForceDerived>
        inline const typename pinocchio::DataTpl<Scalar,Options,JointCollectionTpl>::TangentVectorType &
        aba(pinocchio::ModelTpl<Scalar,Options,JointCollectionTpl> const & model,
            pinocchio::DataTpl<Scalar,Options,JointCollectionTpl>        & data,
            Eigen::MatrixBase<ConfigVectorType>                    const & q,
            Eigen::MatrixBase<TangentVectorType1>                  const & v,
            Eigen::MatrixBase<TangentVectorType2>                  const & tau,
            pinocchio::container::aligned_vector<ForceDerived>     const & fext);

    public:
        std::unique_ptr<engineOptions_t const> engineOptions_;
        std::vector<systemHolder_t> systems_;

    protected:
        bool_t isTelemetryConfigured_;
        bool_t isSimulationRunning_;
        configHolder_t engineOptionsHolder_;

    private:
        Timer timer_;
        TelemetrySender telemetrySender_;
        std::shared_ptr<TelemetryData> telemetryData_;
        std::unique_ptr<TelemetryRecorder> telemetryRecorder_;
        std::unique_ptr<AbstractStepper> stepper_;
        float64_t stepperUpdatePeriod_;
        stepperState_t stepperState_;
        std::vector<systemDataHolder_t> systemsDataHolder_;
        forceCouplingRegister_t forcesCoupling_;
    };
}

#endif //end of JIMINY_ENGINE_MULTIROBOT_H
