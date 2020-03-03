#ifndef SIMU_ENGINE_H
#define SIMU_ENGINE_H

#include <tuple>
#include <string>
#include <functional>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/algorithm/energy.hpp"

#include "jiminy/core/Utilities.h"
#include "jiminy/core/Model.h"
#include "jiminy/core/TelemetrySender.h"
#include "jiminy/core/Types.h"

#include <boost/numeric/odeint.hpp>
#include <boost/numeric/odeint/external/eigen/eigen_algebra.hpp>


namespace jiminy
{
    std::string const ENGINE_OBJECT_NAME("HighLevelController");

    extern float64_t const MIN_TIME_STEP;
    extern float64_t const MAX_TIME_STEP;

    using namespace boost::numeric::odeint;

    class AbstractController;
    class TelemetryData;
    class TelemetryRecorder;

    class explicit_euler
    {
    public:
        typedef vectorN_t state_type;
        typedef vectorN_t deriv_type;
        typedef float64_t value_type;
        typedef float64_t time_type;
        typedef unsigned short order_type;

        using stepper_category = controlled_stepper_tag;

        static order_type order(void)
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

    struct stepperState_t
    {
    public:
        typedef pinocchio::container::aligned_vector<pinocchio::Force> forceVector_t;

    public:
        stepperState_t(void) :
        iter(0),
        t(0.0),
        dt(0.0),
        x(),
        dxdt(),
        u(),
        uCommand(),
        uInternal(),
        fExternal(),
        energy(0.0),
        nx_(0),
        nq_(0),
        nv_(0),
        isInitialized_(false)
        {
            // Empty.
        }

        void initialize(Model & model)
        {
            initialize(model, vectorN_t::Zero(model.nx()), MIN_TIME_STEP);
        }

        void initialize(Model           & model,
                        vectorN_t const & x_init,
                        float64_t const & dt_init)
        {
            // Extract some information from the model
            nx_ = model.nx();
            nq_ = model.nq();
            nv_ = model.nv();

            // Initialize the ode stepper state buffers
            iter = 0;
            t = 0.0;
            dt = dt_init;
            x = x_init;

            dxdt = vectorN_t::Zero(nx_);
            computePositionDerivative(model.pncModel_, q(), v(), qDot());

            fExternal = stepperState_t::forceVector_t(model.pncModel_.joints.size(),
                                                      pinocchio::Force::Zero());
            uInternal = vectorN_t::Zero(nv_);
            uCommand = vectorN_t::Zero(model.getMotorsNames().size());
            u = vectorN_t::Zero(nv_);
            energy = 0.0;

            // Set the initialization flag
            isInitialized_ = true;
        }

        bool const & getIsInitialized(void) const
        {
            return isInitialized_;
        }

        Eigen::Ref<vectorN_t> q(void)
        {
            return x.head(nq_);
        }

        Eigen::Ref<vectorN_t> v(void)
        {
            return x.tail(nv_);
        }

        Eigen::Ref<vectorN_t> qDot(void)
        {
            return dxdt.head(nq_);
        }

        Eigen::Ref<vectorN_t> a(void)
        {
            return dxdt.tail(nv_);
        }

    public:
        uint32_t iter;
        float64_t t;
        float64_t dt;
        vectorN_t x;
        vectorN_t dxdt;
        vectorN_t u;
        vectorN_t uCommand;
        vectorN_t uInternal;
        forceVector_t fExternal;
        float64_t energy; ///< Energy of the system (kinetic + potential)

        uint32_t nx_;
        uint32_t nq_;
        uint32_t nv_;

        bool isInitialized_;
    };

    class Engine
    {
    public:
        typedef std::function<vector3_t(float64_t const & /*t*/,
                                        vectorN_t const & /*x*/)> forceFunctor_t; // Impossible to use function pointer since it does not support functors

        typedef std::function<bool(float64_t const & /*t*/,
                                   vectorN_t const & /*x*/)> callbackFunctor_t; // Impossible to use function pointer since it does not support functors

    protected:
        typedef runge_kutta_dopri5<vectorN_t, float64_t, vectorN_t, float64_t, vector_space_algebra> rungeKuttaStepper_t;

        typedef boost::variant<result_of::make_controlled<rungeKuttaStepper_t>::type, explicit_euler> stepper_t;

    public:
        // Disable the copy of the class
        Engine(Engine const & engine) = delete;
        Engine & operator = (Engine const & other) = delete;

    public:
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

        configHolder_t getDefaultJointOptions()
        {
            configHolder_t config;
            config["boundStiffness"] = 1.0e5;
            config["boundDamping"] = 1.0e4;
            config["boundTransitionEps"] = 1.0e-2; // about 0.55 degrees

            return config;
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

        configHolder_t getDefaultStepperOptions()
        {
            configHolder_t config;
            config["verbose"] = false;
            config["randomSeed"] = 0U;
            config["odeSolver"] = std::string("runge_kutta_dopri5"); // ["runge_kutta_dopri5", "explicit_euler"]
            config["tolAbs"] = 1.0e-5;
            config["tolRel"] = 1.0e-4;
            config["dtMax"] = 1.0e-3;
            config["iterMax"] = 100000; // -1: infinity
            config["sensorsUpdatePeriod"] = 0.0;
            config["controllerUpdatePeriod"] = 0.0;
            config["logInternalStepperSteps"] = false;

            return config;
        };

        struct stepperOptions_t
        {
            bool        const verbose;
            uint32_t    const randomSeed;
            std::string const odeSolver;
            float64_t   const tolAbs;
            float64_t   const tolRel;
            float64_t   const dtMax;
            int32_t     const iterMax;
            float64_t   const sensorsUpdatePeriod;
            float64_t   const controllerUpdatePeriod;
            bool        const logInternalStepperSteps;

            stepperOptions_t(configHolder_t const & options) :
            verbose(boost::get<bool>(options.at("verbose"))),
            randomSeed(boost::get<uint32_t>(options.at("randomSeed"))),
            odeSolver(boost::get<std::string>(options.at("odeSolver"))),
            tolAbs(boost::get<float64_t>(options.at("tolAbs"))),
            tolRel(boost::get<float64_t>(options.at("tolRel"))),
            dtMax(boost::get<float64_t>(options.at("dtMax"))),
            iterMax(boost::get<int32_t>(options.at("iterMax"))),
            sensorsUpdatePeriod(boost::get<float64_t>(options.at("sensorsUpdatePeriod"))),
            controllerUpdatePeriod(boost::get<float64_t>(options.at("controllerUpdatePeriod"))),
            logInternalStepperSteps(boost::get<bool>(options.at("logInternalStepperSteps")))
            {
                // Empty.
            }
        };

        configHolder_t getDefaultTelemetryOptions()
        {
            configHolder_t config;
            config["enableConfiguration"] = true;
            config["enableVelocity"] = true;
            config["enableAcceleration"] = true;
            config["enableCommand"] = true;
            config["enableEnergy"] = true;
            return config;
        };

        struct telemetryOptions_t
        {
            bool const enableConfiguration;
            bool const enableVelocity;
            bool const enableAcceleration;
            bool const enableCommand;
            bool const enableEnergy;

            telemetryOptions_t(configHolder_t const & options) :
            enableConfiguration(boost::get<bool>(options.at("enableConfiguration"))),
            enableVelocity(boost::get<bool>(options.at("enableVelocity"))),
            enableAcceleration(boost::get<bool>(options.at("enableAcceleration"))),
            enableCommand(boost::get<bool>(options.at("enableCommand"))),
            enableEnergy(boost::get<bool>(options.at("enableEnergy")))
            {
                // Empty.
            }
        };

        configHolder_t getDefaultOptions()
        {
            configHolder_t config;
            config["telemetry"] = getDefaultTelemetryOptions();
            config["stepper"] = getDefaultStepperOptions();
            config["world"] = getDefaultWorldOptions();
            config["joints"] = getDefaultJointOptions();
            config["contacts"] = getDefaultContactOptions();

            return config;
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
        Engine(void);
        ~Engine(void);

        result_t initialize(std::shared_ptr<Model>              const & model,
                            std::shared_ptr<AbstractController> const & controller,
                            callbackFunctor_t    callbackFct);

        /// \brief Reset engine.
        ///
        /// \details This function resets the engine, the model and the controller.
        ///          This method is made to be called in between simulations, to allow
        ///          registering of new variables to log.
        /// \param[in] resetDynamicForceRegister Whether or not to register the external force profiles applied
        ///                                      during the simulation.
        void reset(bool const & resetDynamicForceRegister = false);

        /// \brief Reset the engine and compute initial state.
        ///
        /// \details This function reset the engine, the model and the controller, and update internal data
        ///          to match the given initial state.
        /// \param[in] x_init Initial state.
        /// \param[in] resetRandomNumbers Whether or not to reset the random number generator.
        /// \param[in] resetDynamicForceRegister Whether or not to register the external force profiles applied
        ///                                      during the simulation.
        result_t setState(vectorN_t const & x_init,
                          bool const & resetRandomNumbers = false,
                          bool const & resetDynamicForceRegister = false);

        /// \brief Run a simulation of duration end_time, starting at x_init.
        ///
        /// \param[in] x_init Initial state, i.e. state at t=0.
        /// \param[in] end_time End time, i.e. amount of time to simulate.
        result_t simulate(vectorN_t const & x_init,
                          float64_t const & end_time);

        /// \brief Integrate system from current state for a duration equal to stepSize
        ///
        /// \details This function performs a single 'integration step', in the sense that only
        ///          the endpoint is added to the log. The integrator object is allowed to perform
        ///          multiple steps inside of this interval.
        ///          One may specify a negative timestep to use the default update value.
        /// \param[in] stepSize Duration for which to integrate ; set to negative value to use default update value.
        result_t step(float64_t const & stepSize = -1);

        void registerForceImpulse(std::string const & frameName,
                                  float64_t   const & t,
                                  float64_t   const & dt,
                                  vector3_t   const & F);
        void registerForceProfile(std::string      const & frameName,
                                  forceFunctor_t           forceFct);

        configHolder_t getOptions(void) const;
        result_t setOptions(configHolder_t const & engineOptions);
        bool getIsInitialized(void) const;
        bool getIsTelemetryConfigured(void) const;
        Model & getModel(void) const;
        AbstractController & getController(void) const;
        stepperState_t const & getStepperState(void) const;
        std::vector<vectorN_t> const & getContactForces(void) const;

        result_t getLogDataRaw(std::vector<std::string>             & header,
                               std::vector<float32_t>               & timestamps,
                               std::vector<std::vector<int32_t> >   & intData,
                               std::vector<std::vector<float32_t> > & floatData);

        /// \brief Get the full logged content.
        ///
        /// \param[out] header      Header, vector of field names.
        /// \param[out] logData     Corresponding data in the log file.
        ///
        /// \return ERROR_INIT_FAILED if telemetry was not initialized, SUCCESS on success.
        result_t getLogData(std::vector<std::string> & header,
                            matrixN_t                & logData);

        /// \brief Get the value of a single logged variable.
        ///
        /// \param[in] fieldName    Full name of the variable to get
        /// \param[in] header       Header, vector of field names.
        /// \param[in] logData      Corresponding data in the log file.
        ///
        /// \return Vector of values for fieldName. If fieldName is not in the header list, this vector will be empty.
        static vectorN_t getLogFieldValue(std::string              const & fieldName,
                                          std::vector<std::string>       & header,
                                          matrixN_t                      & logData);

        result_t writeLogTxt(std::string const & filename);
        result_t writeLogBinary(std::string const & filename);

        static result_t parseLogBinaryRaw(std::string                          const & filename,
                                          std::vector<std::string>                   & header,
                                          std::vector<float32_t>                     & timestamps,
                                          std::vector<std::vector<int32_t> >         & intData,
                                          std::vector<std::vector<float32_t> >       & floatData);
        static result_t parseLogBinary(std::string              const & filename,
                                       std::vector<std::string>       & header,
                                       matrixN_t                      & logData);

    protected:
        result_t configureTelemetry(void);
        void updateTelemetry(void);

        vector6_t computeFrameForceOnParentJoint(int32_t   const & frameId,
                                                 vector3_t const & fExtInWorld) const;
        vector3_t computeContactDynamics(int32_t const & frameId) const;
        void computeForwardKinematics(Eigen::Ref<vectorN_t const> q,
                                      Eigen::Ref<vectorN_t const> v,
                                      Eigen::Ref<vectorN_t const> a);
        void computeCommand(float64_t                   const & t,
                            Eigen::Ref<vectorN_t const>         q,
                            Eigen::Ref<vectorN_t const>         v,
                            vectorN_t                         & u);
        void computeExternalForces(float64_t const & t,
                                   vectorN_t const & x,
                                   pinocchio::container::aligned_vector<pinocchio::Force> & fext);
        void computeInternalDynamics(float64_t                   const & t,
                                     Eigen::Ref<vectorN_t const>         q,
                                     Eigen::Ref<vectorN_t const>         v,
                                     vectorN_t                         & u);
        void computeSystemDynamics(float64_t const & tIn,
                                   vectorN_t const & xIn,
                                   vectorN_t       & dxdtIn);

    private:
        template<typename Scalar, int Options, template<typename, int> class JointCollectionTpl,
                 typename ConfigVectorType, typename TangentVectorType>
        inline Scalar
        kineticEnergy(pinocchio::ModelTpl<Scalar,Options,JointCollectionTpl> const & model,
                      pinocchio::DataTpl<Scalar,Options,JointCollectionTpl>        & data,
                      Eigen::MatrixBase<ConfigVectorType>                    const & q,
                      Eigen::MatrixBase<TangentVectorType>                   const & v,
                      bool                                                   const & update_kinematics);
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
        bool isInitialized_;
        bool isTelemetryConfigured_;
        std::shared_ptr<Model> model_;
        std::shared_ptr<AbstractController> controller_;
        configHolder_t engineOptionsHolder_;
        callbackFunctor_t callbackFct_;

    private:
        TelemetrySender telemetrySender_;
        std::shared_ptr<TelemetryData> telemetryData_;
        std::unique_ptr<TelemetryRecorder> telemetryRecorder_;
        stepper_t stepper_;
        float64_t stepperUpdatePeriod_;
        stepperState_t stepperState_;       ///< Internal buffer with the state for the integration loop
        stepperState_t stepperStateLast_;   ///< Internal state for the integration loop at the end of the previous iteration
        std::map<float64_t, std::tuple<std::string, float64_t, vector3_t> > forcesImpulse_; // Note that one MUST use an ordered map wrt. the application time
        std::map<float64_t, std::tuple<std::string, float64_t, vector3_t> >::const_iterator forceImpulseNextIt_;
        std::vector<std::pair<std::string, std::tuple<int32_t, forceFunctor_t> > > forcesProfile_;
    };
}

#endif //end of SIMU_ENGINE_H
