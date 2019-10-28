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
    // Internal state for the integration loop

    public:
        stepperState_t(void) :
        iterLast(),
        tLast(),
        qLast(),
        vLast(),
        aLast(),
        uLast(),
        uCommandLast(),
        energyLast(0.0),
        t(0.0),
        dt(0.0),
        x(),
        dxdt(),
        uControl(),
        fext(),
        uInternal(),
        isInitialized()
        {
            // Empty.
        }

        bool const & getIsInitialized(void) const
        {
            return isInitialized;
        }

        void initialize(Model & model)
        {
            initialize(model, vectorN_t::Zero(model.nx()), MIN_TIME_STEP);
        }

        void initialize(Model           & model,
                        vectorN_t const & x_init,
                        float64_t const & dt_init)
        {
            // Initialize the ode stepper state buffers
            iterLast = -1;
            tLast = 0.0;
            qLast = x_init.head(model.nq());
            vLast = x_init.tail(model.nv());
            aLast = vectorN_t::Zero(model.nv());
            uCommandLast = vectorN_t::Zero(model.getMotorsNames().size());
            uLast = pinocchio::rnea(model.pncModel_, model.pncData_, qLast, vLast, aLast);
            energyLast = pinocchio::kineticEnergy(model.pncModel_, model.pncData_, qLast, vLast, false)
                + pinocchio::potentialEnergy(model.pncModel_, model.pncData_, qLast, false);

            // Initialize the internal systemDynamics buffers
            t = 0.0;
            dt = dt_init;
            x = x_init;
            dxdt = vectorN_t::Zero(model.nx());
            computePositionDerivative(model.pncModel_, qLast, vLast, dxdt.head(model.nq()));
            uControl = vectorN_t::Zero(model.nv());

            fext = pinocchio::container::aligned_vector<pinocchio::Force>(
                model.pncModel_.joints.size(),
                pinocchio::Force::Zero());
            uInternal = vectorN_t::Zero(model.nv());

            // Set the initialization flag
            isInitialized = true;
        }

        void updateLast(float64_t const & time,
                        vectorN_t const & q,
                        vectorN_t const & v,
                        vectorN_t const & a,
                        vectorN_t const & u,
                        vectorN_t const & uCommand,
                        float64_t const & energy)
        {
            tLast = time;
            qLast = q;
            vLast = v;
            aLast = a;
            uLast = u;
            uCommandLast = uCommand;
            energyLast = energy;
            ++iterLast;
        }

    public:
        // State information about the last iteration
        uint32_t iterLast;
        float64_t tLast;
        vectorN_t qLast;
        vectorN_t vLast;
        vectorN_t aLast;
        vectorN_t uLast;
        vectorN_t uCommandLast;
        float64_t energyLast; ///< Energy (kinetic + potential) of the system at the last state.

        // Internal buffers required for the adaptive step computation and system dynamics
        float64_t t;
        float64_t dt;
        vectorN_t x;
        vectorN_t dxdt;
        vectorN_t uControl;

        // Internal buffers to speed up the evaluation of the system dynamics
        pinocchio::container::aligned_vector<pinocchio::Force> fext;
        vectorN_t uInternal;

    private:
        bool isInitialized;
    };

    class Engine
    {
    protected:
        typedef std::function<vector3_t(float64_t const & /*t*/,
                                        vectorN_t const & /*x*/)> external_force_t;

        typedef std::function<bool(float64_t const & /*t*/,
                                   vectorN_t const & /*x*/)> callbackFct_t;

        typedef runge_kutta_dopri5<vectorN_t, float64_t, vectorN_t, float64_t, vector_space_algebra> runge_kutta_stepper_t;

        typedef boost::variant<result_of::make_controlled<runge_kutta_stepper_t>::type, explicit_euler> stepper_t;

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
            config["zGround"] = 0.0;

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
            float64_t const zGround;

            contactOptions_t(configHolder_t const & options) :
            frictionViscous(boost::get<float64_t>(options.at("frictionViscous"))),
            frictionDry(boost::get<float64_t>(options.at("frictionDry"))),
            dryFrictionVelEps(boost::get<float64_t>(options.at("dryFrictionVelEps"))),
            stiffness(boost::get<float64_t>(options.at("stiffness"))),
            damping(boost::get<float64_t>(options.at("damping"))),
            transitionEps(boost::get<float64_t>(options.at("transitionEps"))),
            zGround(boost::get<float64_t>(options.at("zGround")))
            {
                // Empty.
            }
        };

        configHolder_t getDefaultJointOptions()
        {
            configHolder_t config;
            config["boundStiffness"] = 1.0e5;
            config["boundDamping"] = 1.0e2;
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

            return config;
        };

        struct worldOptions_t
        {
            vectorN_t const gravity;

            worldOptions_t(configHolder_t const & options) :
            gravity(boost::get<vectorN_t>(options.at("gravity")))
            {
                // Empty.
            }
        };

        configHolder_t getDefaultStepperOptions()
        {
            configHolder_t config;
            config["randomSeed"] = 0U;
            config["odeSolver"] = std::string("runge_kutta_dopri5"); // ["runge_kutta_dopri5", "explicit_euler"]
            config["tolAbs"] = 1.0e-5;
            config["tolRel"] = 1.0e-4;
            config["dtMax"] = 1.0e-3;
            config["iterMax"] = 100000; // -1: infinity
            config["sensorsUpdatePeriod"] = 0.0;
            config["controllerUpdatePeriod"] = 0.0;

            return config;
        };

        struct stepperOptions_t
        {
            uint32_t    const randomSeed;
            std::string const odeSolver;
            float64_t   const tolAbs;
            float64_t   const tolRel;
            float64_t   const dtMax;
            int32_t     const iterMax;
            float64_t   const sensorsUpdatePeriod;
            float64_t   const controllerUpdatePeriod;

            stepperOptions_t(configHolder_t const & options) :
            randomSeed(boost::get<uint32_t>(options.at("randomSeed"))),
            odeSolver(boost::get<std::string>(options.at("odeSolver"))),
            tolAbs(boost::get<float64_t>(options.at("tolAbs"))),
            tolRel(boost::get<float64_t>(options.at("tolRel"))),
            dtMax(boost::get<float64_t>(options.at("dtMax"))),
            iterMax(boost::get<int32_t>(options.at("iterMax"))),
            sensorsUpdatePeriod(boost::get<float64_t>(options.at("sensorsUpdatePeriod"))),
            controllerUpdatePeriod(boost::get<float64_t>(options.at("controllerUpdatePeriod")))
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

        result_t initialize(Model              & model,
                            AbstractController & controller,
                            callbackFct_t        callbackFct);
        void reset(bool const & resetDynamicForceRegister = false);

        result_t configureTelemetry(void);
        void updateTelemetry(void);

        result_t reset(vectorN_t const & x_init,
                       bool      const & resetRandomNumbers = false,
                       bool      const & resetDynamicForceRegister = false);
        result_t simulate(vectorN_t const & x_init,
                          float64_t const & end_time);
        result_t step(float64_t const & dtDesired = -1,
                      float64_t         t_end = -1);

        void registerForceImpulse(std::string const & frameName,
                                  float64_t   const & t,
                                  float64_t   const & dt,
                                  vector3_t   const & F);
        void registerForceProfile(std::string      const & frameName,
                                  external_force_t         forceFct);

        configHolder_t getOptions(void) const;
        result_t setOptions(configHolder_t const & engineOptions);
        bool getIsInitialized(void) const;
        bool getIsTelemetryConfigured(void) const;
        Model const & getModel(void) const;
        stepperState_t const & getStepperState(void) const;
        std::vector<vectorN_t> const & getContactForces(void) const;
        void getLogData(std::vector<std::string> & header,
                        matrixN_t                & logData);
        void writeLogTxt(std::string const & filename);
        void writeLogBinary(std::string const & filename);

    protected:
        void systemDynamics(float64_t const & t,
                            vectorN_t const & x,
                            vectorN_t       & dxdt);
        void internalDynamics(vectorN_t const & q,
                              vectorN_t const & v,
                              vectorN_t       & u);
        vector6_t computeFrameForceOnParentJoint(int32_t   const & frameId,
                                                 vector3_t const & fextInWorld) const;
        vector3_t contactDynamics(int32_t const & frameId) const;

    public:
        std::unique_ptr<engineOptions_t const> engineOptions_;

    protected:
        bool isInitialized_;
        bool isTelemetryConfigured_;
        Model * model_;
        AbstractController * controller_;
        configHolder_t engineOptionsHolder_;
        callbackFct_t callbackFct_;

    private:
        TelemetrySender telemetrySender_;
        std::shared_ptr<TelemetryData> telemetryData_;
        std::unique_ptr<TelemetryRecorder> telemetryRecorder_;
        stepper_t stepper_;
        float64_t stepperUpdatePeriod_;
        stepperState_t stepperState_; // Internal state for the integration loop
        std::map<float64_t, std::tuple<std::string, float64_t, vector3_t> > forcesImpulse_; // MUST use ordered map (wrt. the application time)
        std::map<float64_t, std::tuple<std::string, float64_t, vector3_t> >::const_iterator forceImpulseNextIt_;
        std::vector<std::pair<std::string, std::tuple<int32_t, external_force_t> > > forcesProfile_;
    };
}

#endif //end of SIMU_ENGINE_H
