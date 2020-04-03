#ifndef JIMINY_ENGINE_MULTIROBOT_H
#define JIMINY_ENGINE_MULTIROBOT_H

#include <functional>

#include "jiminy/core/telemetry/TelemetrySender.h"
#include "jiminy/core/Utilities.h"
#include "jiminy/core/Types.h"

#include <boost/numeric/odeint.hpp>
#include <boost/numeric/odeint/external/eigen/eigen.hpp>


namespace jiminy
{
    std::string const ENGINE_OBJECT_NAME("HighLevelController");

    std::set<std::string> const STEPPER_ALGORITHMS{"runge_kutta_dopri5",
                                                   "explicit_euler",
                                                   "bulirsch_stoer"};

    using namespace boost::numeric::odeint;

    class AbstractController;
    class TelemetryData;
    class TelemetryRecorder;
    class Robot;
    class systemDataHolder_t;
    class EngineMultiRobot;

    // Impossible to use function pointer since it does not support functors
    using forceProfileFunctor_t = std::function<pinocchio::Force(float64_t                   const & /*t*/,
                                                                 Eigen::Ref<vectorN_t const> const & /*q*/,
                                                                 Eigen::Ref<vectorN_t const> const & /*v*/)>;
    using forceCouplingFunctor_t = std::function<pinocchio::Force(float64_t                   const & /*t*/,
                                                                  Eigen::Ref<vectorN_t const> const & /*q_1*/,
                                                                  Eigen::Ref<vectorN_t const> const & /*v_1*/,
                                                                  Eigen::Ref<vectorN_t const> const & /*q_2*/,
                                                                  Eigen::Ref<vectorN_t const> const & /*v_2*/)>;
    using callbackFunctor_t = std::function<bool_t(float64_t const & /*t*/,
                                                   vectorN_t const & /*q*/,
                                                   vectorN_t const & /*v*/)>;

    struct forceImpulse_t
    {
    public:
        forceImpulse_t(void) = default;

        forceImpulse_t(std::string      const & frameNameIn,
                       float64_t        const & tIn,
                       float64_t        const & dtIn,
                       pinocchio::Force const & FIn) :
        frameName(frameNameIn),
        t(tIn),
        dt(dtIn),
        F(FIn)
        {
            // Empty on purpose
        }

        bool operator < (forceImpulse_t const & r) const
        {
            return (t < r.t);
        }

    public:
        std::string frameName;
        float64_t t;
        float64_t dt;
        pinocchio::Force F;
    } ;

    struct forceProfile_t
    {
    public:
        forceProfile_t(void) = default;

        forceProfile_t(std::string           const & frameNameIn,
                       int32_t               const & frameIdxIn,
                       forceProfileFunctor_t const & forceFctIn) :
        frameName(frameNameIn),
        frameIdx(frameIdxIn),
        forceFct(forceFctIn)
        {
            // Empty on purpose
        }

    public:
        std::string frameName;
        int32_t frameIdx;
        forceProfileFunctor_t forceFct;
    } ;

    struct forceCoupling_t
    {
    public:
        forceCoupling_t(void) = default;

        forceCoupling_t(std::string            const & systemName1In,
                        int32_t                const & systemIdx1In,
                        std::string            const & systemName2In,
                        int32_t                const & systemIdx2In,
                        std::string            const & frameName1In,
                        int32_t                const & frameIdx1In,
                        std::string            const & frameName2In,
                        int32_t                const & frameIdx2In,
                        forceCouplingFunctor_t const & forceFctIn) :
        systemName1(systemName1In),
        systemIdx1(systemIdx1In),
        systemName2(systemName2In),
        systemIdx2(systemIdx2In),
        frameName1(frameName1In),
        frameIdx1(frameIdx1In),
        frameName2(frameName2In),
        frameIdx2(frameIdx2In),
        forceFct(forceFctIn)
        {
            // Empty on purpose.
        }

    public:
        std::string systemName1;
        int32_t systemIdx1;
        std::string systemName2;
        int32_t systemIdx2;
        std::string frameName1;
        int32_t frameIdx1;
        std::string frameName2;
        int32_t frameIdx2;
        forceCouplingFunctor_t forceFct;
    };

    using forceImpulseRegister_t = std::set<forceImpulse_t>;  // Use a set to automatically sort the forces wrt. the application time
    using forceProfileRegister_t = std::vector<forceProfile_t>;
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
        x(),
        dxdt()
        {
            // Empty on purpose.
        }

        void reset(float64_t const & dtInit,
                   vectorN_t const & xInit)
        {
            iter = 0U;
            iterFailed = 0U;
            t = 0.0;
            tPrev = 0.0;
            dt = dtInit;
            dtLargest = dtInit;
            dtLargestPrev = dtInit;
            tError = 0.0;
            x = xInit;
            dxdt = vectorN_t::Zero(xInit.size());
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
        vectorN_t x;
        vectorN_t dxdt;
    };

    template<template<typename> class F = type_identity>
    using stateSplitRef_t = std::pair<std::vector<Eigen::Ref<typename F<vectorN_t>::type> >,
                                      std::vector<Eigen::Ref<typename F<vectorN_t>::type> > >;

    struct systemState_t
    {
    public:
        systemState_t(void) :
        q(),
        v(),
        qDot(),
        a(),
        u(),
        uCommand(),
        uMotor(),
        uInternal(),
        fExternal(),
        isInitialized_(false),
        robot_(nullptr)
        {
            // Empty on purpose.
        }

        void initialize(Robot const * robot);

        bool_t const & getIsInitialized(void) const
        {
            return isInitialized_;
        }

        systemState_t & operator = (systemState_t const & other) = default;
        systemState_t(systemState_t const & other) = default;
        systemState_t(systemState_t&& other) = default;
        ~systemState_t(void) = default;

    public:
        vectorN_t q;
        vectorN_t v;
        vectorN_t qDot;
        vectorN_t a;
        vectorN_t u;
        vectorN_t uCommand;
        vectorN_t uMotor;
        vectorN_t uInternal;
        forceVector_t fExternal;

    private:
        bool_t isInitialized_;
        Robot const * robot_;

    };

    struct systemDataHolder_t
    {
    public:
        friend EngineMultiRobot;

        systemDataHolder_t(void);

        systemDataHolder_t(std::string const & systemNameIn,
                           std::shared_ptr<Robot> robotIn,
                           std::shared_ptr<AbstractController> controllerIn,
                           callbackFunctor_t callbackFctIn);

    public:
        std::string name;
        std::shared_ptr<Robot> robot;
        std::shared_ptr<AbstractController> controller;
        callbackFunctor_t callbackFct;
        std::vector<std::string> positionFieldnames;
        std::vector<std::string> velocityFieldnames;
        std::vector<std::string> accelerationFieldnames;
        std::vector<std::string> motorTorqueFieldnames;
        std::string energyFieldname;

    private:
        std::unique_ptr<MutexLocal::LockGuardLocal> robotLock;
        systemState_t state;       ///< Internal buffer with the state for the integration loop
        systemState_t statePrev;   ///< Internal state for the integration loop at the end of the previous iteration
        forceImpulseRegister_t forcesImpulse;
        forceImpulseRegister_t::const_iterator forceImpulseNextIt;
        forceProfileRegister_t forcesProfile;
    };

    class EngineMultiRobot
    {
    protected:
        configHolder_t getDefaultContactOptions()
        {
            configHolder_t config;
            config["frictionViscous"] = 0.8;
            config["frictionDry"] = 1.0;
            config["dryFrictionVelEps"] = 1.0e-2;
            config["stiffness"] = 1.0e6;
            config["damping"] = 2.0e3;
            config["transitionEps"] = 1.0e-3;

            return config;
        };

        configHolder_t getDefaultJointOptions()
        {
            configHolder_t config;
            config["boundStiffness"] = 1.0e5;
            config["boundDamping"] = 1.0e4;
            config["boundTransitionEps"] = 1.0e-2; // about 0.55 degrees

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
            config["odeSolver"] = std::string("runge_kutta_dopri5"); // ["runge_kutta_dopri5", "explicit_euler", "bulirsch_stoer"]
            config["tolAbs"] = 1.0e-5;
            config["tolRel"] = 1.0e-4;
            config["dtMax"] = 1.0e-3;
            config["dtRestoreThresholdRel"] = 0.2;
            config["iterMax"] = 1000000; // -1: infinity
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
            config["enableTorque"] = true;
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
            config["contacts"] = getDefaultContactOptions();

            return config;
        };

    public:
        struct contactOptions_t
        {
            float64_t const frictionViscous;
            float64_t const frictionDry;
            float64_t const dryFrictionVelEps;
            float64_t const stiffness;
            float64_t const damping;
            float64_t const transitionEps;

            contactOptions_t(configHolder_t const & options) :
            frictionViscous(boost::get<float64_t>(options.at("frictionViscous"))),
            frictionDry(boost::get<float64_t>(options.at("frictionDry"))),
            dryFrictionVelEps(boost::get<float64_t>(options.at("dryFrictionVelEps"))),
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
            float64_t const boundTransitionEps;

            jointOptions_t(configHolder_t const & options) :
            boundStiffness(boost::get<float64_t>(options.at("boundStiffness"))),
            boundDamping(boost::get<float64_t>(options.at("boundDamping"))),
            boundTransitionEps(boost::get<float64_t>(options.at("boundTransitionEps")))
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
            int32_t     const iterMax;
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
            iterMax(boost::get<int32_t>(options.at("iterMax"))),
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
            bool_t const enableTorque;
            bool_t const enableEnergy;

            telemetryOptions_t(configHolder_t const & options) :
            enableConfiguration(boost::get<bool_t>(options.at("enableConfiguration"))),
            enableVelocity(boost::get<bool_t>(options.at("enableVelocity"))),
            enableAcceleration(boost::get<bool_t>(options.at("enableAcceleration"))),
            enableTorque(boost::get<bool_t>(options.at("enableTorque"))),
            enableEnergy(boost::get<bool_t>(options.at("enableEnergy")))
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

    protected:
        class eulerExplicitStepper_t
        {
        public:
            using state_type = vectorN_t;
            using deriv_type = vectorN_t;
            using value_type = float64_t;
            using time_type = float64_t;

            using stepper_category = controlled_stepper_tag;

            static unsigned short order(void)
            {
                return 1;
            }

            template<class System>
            controlled_step_result try_step(System       system,
                                            state_type & x,
                                            deriv_type & dxdt,
                                            time_type  & t,
                                            time_type  & dt) const
            {
                t += dt;
                system(x, dxdt, t);
                x += dt * dxdt;
                return controlled_step_result::success;
            }
        };

        using bulirschStoerStepper_t = bulirsch_stoer<vectorN_t /* x */,
                                                      float64_t /* t */,
                                                      vectorN_t /* dxdt */,
                                                      float64_t /* dt */,
                                                      vector_space_algebra /* Enable Eigen support */>;

        using rungeKuttaBackend_t = runge_kutta_dopri5<
            vectorN_t, float64_t, vectorN_t, float64_t, vector_space_algebra>;

        using rungeKuttaErrorChecker_t = default_error_checker<
            typename rungeKuttaBackend_t::value_type,
            typename rungeKuttaBackend_t::algebra_type,
            typename rungeKuttaBackend_t::operations_type
        >;

        class rungeKuttaStepAdjuster_t
        {
        public:
            using time_type = typename rungeKuttaBackend_t::time_type;
            using value_type = typename rungeKuttaBackend_t::value_type;

            rungeKuttaStepAdjuster_t(void) = default;

            time_type decrease_step(time_type          dt,
                                    value_type const & error,
                                    int        const & error_order) const
            {
                // Returns the decreased time step
                dt *= std::max(
                    static_cast<value_type>(static_cast<value_type>(9) / static_cast<value_type>(10) *
                                            std::pow(error, static_cast<value_type>(-1) / (error_order - 1))),
                    static_cast<value_type>(static_cast<value_type>(1) / static_cast<value_type> (5))
                );
                return dt;
            }

            time_type increase_step(time_type          dt,
                                    value_type         error,
                                    int        const & stepper_order) const
            {
                if(error < 0.5)
                {
                    // Error should be > 0
                    error = std::max(
                        error,
                        static_cast<value_type>(std::pow(static_cast<value_type>(5.0),
                                                -static_cast<value_type>(stepper_order)))
                    );

                    // Error too small - increase dt and keep the evolution and limit scaling factor to 5.0
                    dt *= static_cast<value_type>(9)/static_cast<value_type>(10)
                       * pow(error, static_cast<value_type>(-1) / stepper_order);
                }
                return dt;
            }

            bool check_step_size_limit(time_type const & dt) { return true; }
            time_type get_max_dt() { return static_cast<time_type>(0.0); }
        };

        using rungeKuttaStepper_t = controlled_runge_kutta<rungeKuttaBackend_t,
                                                           rungeKuttaErrorChecker_t,
                                                           rungeKuttaStepAdjuster_t>;

        using stepper_t = boost::variant<
            bulirschStoerStepper_t  /* Bulirsch-Stoer stepper */,
            rungeKuttaStepper_t     /* Runge-Kutta Dopri stepper */,
            eulerExplicitStepper_t  /* Explicit Euler stepper */
        >;

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
        hresult_t removeSystem(std::string const & systemName);

        /// \brief Add a force linking both systems together
        ///
        /// \details This function registers a callback function forceFct that links
        ///          both systems by a given force. This function must return the
        ///          force that the second systems applies to the first system,
        ///          in the world frame.
        ///
        /// \param[in] systemName1 Name of the first system (the one receiving the force)
        /// \param[in] systemName2 Name of the second system (the one applying the force)
        /// \param[in] frameName1 Frame on the first system where the force is applied.
        /// \param[in] frameName2 Frame on the second system where
        ///                       (the opposite of) the force is applied.
        /// \param[in] forceFct Callback function returning the force that systemName2
        ///                     applies on systemName1, in the world frame.
        hresult_t addCouplingForce(std::string            const & systemName1,
                                   std::string            const & systemName2,
                                   std::string            const & frameName1,
                                   std::string            const & frameName2,
                                   forceCouplingFunctor_t         forceFct);
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
        /// \param[in] xInit Initial state.
        /// \param[in] resetRandomNumbers Whether or not to reset the random number generator.
        /// \param[in] resetDynamicForceRegister Whether or not to register the external force profiles applied
        ///                                      during the simulation.
        hresult_t start(std::vector<vectorN_t> const & xInit,
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
        /// \param[in] xInit Initial state, i.e. state at t=0.
        hresult_t simulate(float64_t              const & tEnd,
                           std::vector<vectorN_t> const & xInit);

        hresult_t registerForceImpulse(std::string      const & systemName,
                                       std::string      const & frameName,
                                       float64_t        const & t,
                                       float64_t        const & dt,
                                       pinocchio::Force const & F);
        hresult_t registerForceProfile(std::string           const & systemName,
                                       std::string           const & frameName,
                                       forceProfileFunctor_t         forceFct);

        configHolder_t getOptions(void) const;
        hresult_t setOptions(configHolder_t const & engineOptions);
        bool_t getIsTelemetryConfigured(void) const;
        hresult_t getSystem(std::string        const   & systemName,
                            systemDataHolder_t const * & system) const;
        hresult_t getSystem(std::string        const   & systemName,
                            systemDataHolder_t       * & system);
        stepperState_t const & getStepperState(void) const;
        systemState_t const & getSystemState(std::string const & systemName) const;

        void getLogDataRaw(std::vector<std::string>             & header,
                           std::vector<float64_t>               & timestamps,
                           std::vector<std::vector<int32_t> >   & intData,
                           std::vector<std::vector<float32_t> > & floatData);

        /// \brief Get the full logged content.
        ///
        /// \param[out] header      Header, vector of field names.
        /// \param[out] logData     Corresponding data in the log file.
        void getLogData(std::vector<std::string> & header,
                        matrixN_t                & logData);

        hresult_t writeLogTxt(std::string const & filename);
        hresult_t writeLogBinary(std::string const & filename);

        static hresult_t parseLogBinaryRaw(std::string                          const & filename,
                                           std::vector<std::string>                   & header,
                                           std::vector<float64_t>                     & timestamps,
                                           std::vector<std::vector<int32_t> >         & intData,
                                           std::vector<std::vector<float32_t> >       & floatData);
        static hresult_t parseLogBinary(std::string              const & filename,
                                        std::vector<std::string>       & header,
                                        matrixN_t                      & logData);

    protected:
        hresult_t configureTelemetry(void);
        void updateTelemetry(void);

        stateSplitRef_t<std::add_const> splitState(vectorN_t const & val) const;
        stateSplitRef_t<> splitState(vectorN_t & val) const;

        void syncStepperStateWithSystems(void);
        void syncSystemsStateWithStepper(void);

        static void computeForwardKinematics(systemDataHolder_t       & system,
                                             vectorN_t          const & q,
                                             vectorN_t          const & v,
                                             vectorN_t          const & a);

        pinocchio::Force computeContactDynamics(systemDataHolder_t const & system,
                                                int32_t            const & frameId) const;

        void computeCommand(systemDataHolder_t                & system,
                            float64_t                   const & t,
                            Eigen::Ref<vectorN_t const> const & q,
                            Eigen::Ref<vectorN_t const> const & v,
                            vectorN_t                         & u);
        void computeInternalDynamics(systemDataHolder_t                & system,
                                     float64_t                   const & t,
                                     Eigen::Ref<vectorN_t const> const & q,
                                     Eigen::Ref<vectorN_t const> const & v,
                                     vectorN_t                         & u) const;
        void computeExternalForces(systemDataHolder_t                & system,
                                   float64_t                   const & t,
                                   Eigen::Ref<vectorN_t const> const & q,
                                   Eigen::Ref<vectorN_t const> const & v,
                                   forceVector_t                     & fext);
        void computeInternalForces(float64_t                       const & t,
                                   stateSplitRef_t<std::add_const> const & xSplit);
        void computeAllForces(float64_t                       const & t,
                              stateSplitRef_t<std::add_const> const & xSplit);
        void computeSystemDynamics(float64_t const & t,
                                   vectorN_t const & xCat,
                                   vectorN_t       & dxdtCat);

        void reset(bool_t const & resetRandomNumbers,
                   bool_t const & resetDynamicForceRegister);

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

    protected:
        bool_t isTelemetryConfigured_;
        bool_t isSimulationRunning_;
        configHolder_t engineOptionsHolder_;
        std::vector<systemDataHolder_t> systemsDataHolder_;

    private:
        TelemetrySender telemetrySender_;
        std::shared_ptr<TelemetryData> telemetryData_;
        std::unique_ptr<TelemetryRecorder> telemetryRecorder_;
        stepper_t stepper_;
        float64_t stepperUpdatePeriod_;
        stepperState_t stepperState_;
        forceCouplingRegister_t forcesCoupling_;
    };
}

#endif //end of JIMINY_ENGINE_MULTIROBOT_H
