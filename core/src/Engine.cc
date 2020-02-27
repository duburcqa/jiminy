#include <iostream>
#include <cmath>
#include <algorithm>

#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/multibody/model.hpp"
#include "pinocchio/multibody/visitor.hpp"
#include "pinocchio/algorithm/aba.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"

#include "jiminy/core/Utilities.h"
#include "jiminy/core/FileDevice.h"
#include "jiminy/core/TelemetryData.h"
#include "jiminy/core/TelemetryRecorder.h"
#include "jiminy/core/AbstractController.h"
#include "jiminy/core/Model.h"
#include "jiminy/core/Engine.h"

#include <boost/numeric/odeint/iterator/n_step_iterator.hpp>


namespace jiminy
{
    float64_t const MIN_TIME_STEP = 1e-6;
    float64_t const MAX_TIME_STEP = 5e-3;


    Engine::Engine(void):
    engineOptions_(nullptr),
    isInitialized_(false),
    isTelemetryConfigured_(false),
    model_(nullptr),
    controller_(nullptr),
    engineOptionsHolder_(),
    callbackFct_([](float64_t const & t,
                    vectorN_t const & x) -> bool
                 {
                     return true;
                 }),
    telemetrySender_(),
    telemetryData_(nullptr),
    telemetryRecorder_(nullptr),
    stepper_(),
    stepperUpdatePeriod_(),
    stepperState_(),
    stepperStateLast_(),
    forcesImpulse_(),
    forceImpulseNextIt_(forcesImpulse_.begin()),
    forcesProfile_()
    {
        // Initialize the configuration options to the default.
        setOptions(getDefaultOptions());

        // Initialize the global telemetry data holder
        telemetryData_ = std::make_shared<TelemetryData>();
        telemetryData_->reset();

        // Initialize the global telemetry recorder
        telemetryRecorder_ = std::make_unique<TelemetryRecorder>(
            std::const_pointer_cast<TelemetryData const>(telemetryData_));

        // Initialize the engine-specific telemetry sender
        telemetrySender_.configureObject(telemetryData_, ENGINE_OBJECT_NAME);

        // Initialize the random number generators
        resetRandGenerators(engineOptions_->stepper.randomSeed);
    }

    Engine::~Engine(void)
    {
        // Empty
    }

    result_t Engine::initialize(std::shared_ptr<Model>              const & model,
                                std::shared_ptr<AbstractController> const & controller,
                                callbackFunctor_t                           callbackFct)
    {
        result_t returnCode = result_t::SUCCESS;

        if (!model->getIsInitialized())
        {
            std::cout << "Error - Engine::initialize - Model not initialized." << std::endl;
            returnCode = result_t::ERROR_INIT_FAILED;
        }
        model_ = model;

        stepperState_.initialize(*model_);

        if (!controller->getIsInitialized())
        {
            std::cout << "Error - Engine::initialize - Controller not initialized." << std::endl;
            returnCode = result_t::ERROR_INIT_FAILED;
        }
        if (returnCode == result_t::SUCCESS)
        {
            controller_ = controller;
        }

        // TODO: Check that the callback function is working as expected
        if (returnCode == result_t::SUCCESS)
        {
            callbackFct_ = callbackFct;
        }

        // Make sure the gravity is properly set at model level
        if (returnCode == result_t::SUCCESS)
        {
            setOptions(engineOptionsHolder_);
            isInitialized_ = true;
        }

        if (returnCode != result_t::SUCCESS)
        {
            isInitialized_ = false;
        }

        return returnCode;
    }

    result_t Engine::configureTelemetry(void)
    {
        result_t returnCode = result_t::SUCCESS;

        if (!getIsInitialized())
        {
            std::cout << "Error - Engine::configureTelemetry - The engine is not initialized." << std::endl;
            returnCode = result_t::ERROR_INIT_FAILED;
        }

        if (returnCode == result_t::SUCCESS)
        {
            if (!isTelemetryConfigured_)
            {
                // Register variables to the telemetry senders
                if (engineOptions_->telemetry.enableConfiguration)
                {
                    (void) registerNewVectorEntry(telemetrySender_,
                                                  model_->getPositionFieldNames(),
                                                  vectorN_t::Zero(model_->nq()));
                }
                if (engineOptions_->telemetry.enableVelocity)
                {
                    (void) registerNewVectorEntry(telemetrySender_,
                                                  model_->getVelocityFieldNames(),
                                                  vectorN_t::Zero(model_->nv()));
                }
                if (engineOptions_->telemetry.enableAcceleration)
                {
                    (void) registerNewVectorEntry(telemetrySender_,
                                                  model_->getAccelerationFieldNames(),
                                                  vectorN_t::Zero(model_->nv()));
                }
                if (engineOptions_->telemetry.enableCommand)
                {
                    (void) registerNewVectorEntry(telemetrySender_,
                                                  model_->getMotorTorqueFieldNames(),
                                                  vectorN_t::Zero(model_->getMotorsNames().size()));
                }
                if (engineOptions_->telemetry.enableEnergy)
                {
                    telemetrySender_.registerNewEntry<float64_t>("energy", 0.0);
                }
                isTelemetryConfigured_ = true;
            }
        }

        returnCode = controller_->configureTelemetry(telemetryData_);
        if (returnCode == result_t::SUCCESS)
        {
            returnCode = model_->configureTelemetry(telemetryData_);
        }

        if (returnCode != result_t::SUCCESS)
        {
            isTelemetryConfigured_ = false;
        }

        return returnCode;
    }

    void Engine::updateTelemetry(void)
    {
        /* Update internal state of the stepper.
           Note that explicit kinematic computation is not needed to get
           the system energy since they were already done in RNEA. */
        Eigen::Ref<vectorN_t const> q = stepperState_.q();
        Eigen::Ref<vectorN_t const> v = stepperState_.v();
        Eigen::Ref<vectorN_t const> a = stepperState_.a();
        stepperState_t::forceVector_t const & fext = stepperState_.fExternal;
        stepperState_.u = Engine::rnea(model_->pncModel_, model_->pncData_, q, v, a, fext);
        stepperState_.energy = Engine::kineticEnergy(model_->pncModel_, model_->pncData_, q, v, false)
            + pinocchio::potentialEnergy(model_->pncModel_, model_->pncData_, q, false);

        // Backup the state of the stepper
        stepperStateLast_ = stepperState_;

        // Update the telemetry internal state
        if (engineOptions_->telemetry.enableConfiguration)
        {
            updateVectorValue(telemetrySender_,
                              model_->getPositionFieldNames(),
                              stepperStateLast_.q());
        }
        if (engineOptions_->telemetry.enableVelocity)
        {
            updateVectorValue(telemetrySender_,
                              model_->getVelocityFieldNames(),
                              stepperStateLast_.v());
        }
        if (engineOptions_->telemetry.enableAcceleration)
        {
            updateVectorValue(telemetrySender_,
                              model_->getAccelerationFieldNames(),
                              stepperStateLast_.a());
        }
        if (engineOptions_->telemetry.enableCommand)
        {
            updateVectorValue(telemetrySender_,
                              model_->getMotorTorqueFieldNames(),
                              stepperStateLast_.uCommand);
        }
        if (engineOptions_->telemetry.enableEnergy)
        {
            telemetrySender_.updateValue<float64_t>("energy", stepperStateLast_.energy);
        }
        controller_->updateTelemetry(stepperState_.t, q, v);
        model_->updateTelemetry();

        // Flush the telemetry internal state
        telemetryRecorder_->flushDataSnapshot(stepperStateLast_.t);
    }

    void Engine::reset(bool const & resetDynamicForceRegister)
    {
        // Reset the dynamic force register if requested
        if (resetDynamicForceRegister)
        {
            forcesImpulse_.clear();
            forceImpulseNextIt_ = forcesImpulse_.begin();
            forcesProfile_.clear();
        }

        // Reset the internal state of the model and controller
        model_->reset();
        controller_->reset();

        // Reset the telemetry
        telemetryRecorder_->reset();
        telemetryData_->reset();
        isTelemetryConfigured_ = false;
    }

    result_t Engine::setState(vectorN_t const & x_init,
                              bool      const & resetRandomNumbers,
                              bool      const & resetDynamicForceRegister)
    {
        if (!getIsInitialized())
        {
            std::cout << "Error - Engine::reset - The engine is not initialized." << std::endl;
            return result_t::ERROR_INIT_FAILED;
        }

        // Check the dimension of the initial rigid state
        if (x_init.rows() != model_->pncModelRigidOrig_.nq + model_->pncModelRigidOrig_.nv)
        {
            std::cout << "Error - Engine::reset - Size of x_init inconsistent with model size." << std::endl;
            return result_t::ERROR_BAD_INPUT;
        }

        // Reset the random number generators
        if (resetRandomNumbers)
        {
            resetRandGenerators(engineOptions_->stepper.randomSeed);
        }

        // Reset the model, controller, engine, and registered impulse forces if requested
        reset(resetDynamicForceRegister);

        // Propagate the gravity value at Pinocchio model level
        model_->pncModel_.gravity = engineOptions_->world.gravity;

        // Compute the initial flexible state based on the initial rigid state
        std::vector<int32_t> const & rigidJointsPositionIdx = model_->getRigidJointsPositionIdx();
        std::vector<int32_t> const & rigidJointsVelocityIdx = model_->getRigidJointsVelocityIdx();
        vectorN_t x0 = vectorN_t::Zero(model_->nx());
        if (model_->getHasFreeflyer())
        {
            x0.head<7>() = x_init.head<7>();
            for (uint32_t i=0; i<rigidJointsPositionIdx.size(); ++i)
            {
                x0[rigidJointsPositionIdx[i]] = x_init[i + 7];
            }
            x0.segment<6>(model_->nq()) = x_init.segment<6>(7 + rigidJointsPositionIdx.size());
            for (uint32_t i=0; i<rigidJointsVelocityIdx.size(); ++i)
            {
                x0[rigidJointsVelocityIdx[i] + model_->nq()] = x_init[i + 7 + rigidJointsPositionIdx.size() + 6];
            }
        }
        else
        {
            for (uint32_t i=0; i<rigidJointsPositionIdx.size(); ++i)
            {
                x0[rigidJointsPositionIdx[i]] = x_init[i];
            }
            for (uint32_t i=0; i<rigidJointsVelocityIdx.size(); ++i)
            {
                x0[rigidJointsVelocityIdx[i] + model_->nq()] = x_init[i + rigidJointsPositionIdx.size()];
            }
        }
        for (int32_t const & jointIdx : model_->getFlexibleJointsModelIdx())
        {
            x0[model_->pncModel_.joints[jointIdx].idx_q() + 3] = 1.0;
        }

        // Reset the impulse for iterator counter
        forceImpulseNextIt_ = forcesImpulse_.begin();
        for (auto & forceProfile : forcesProfile_)
        {
            getFrameIdx(model_->pncModel_, forceProfile.first, std::get<0>(forceProfile.second));
        }

        // Initialize the ode solver
        if (engineOptions_->stepper.odeSolver == "runge_kutta_dopri5")
        {
            stepper_ = make_controlled(engineOptions_->stepper.tolAbs,
                                       engineOptions_->stepper.tolRel,
                                       rungeKuttaStepper_t());
        }
        else if (engineOptions_->stepper.odeSolver == "explicit_euler")
        {
            stepper_ = explicit_euler();
        }

        // Compute the initial time step
        float64_t dt;
        if (stepperUpdatePeriod_ > MIN_TIME_STEP)
        {
            dt = stepperUpdatePeriod_; // The initial time step is the update period
        }
        else
        {
            dt = engineOptions_->stepper.dtMax; // Use the maximum allowed time step as default
        }

        // Initialize the stepper internal state
        stepperState_.initialize(*model_, x0, dt);

        float64_t & t = stepperState_.t;
        Eigen::Ref<vectorN_t> q = stepperState_.q();
        Eigen::Ref<vectorN_t> v = stepperState_.v();
        vectorN_t & x = stepperState_.x;
        Eigen::Ref<vectorN_t> a = stepperState_.a();
        vectorN_t & u = stepperState_.u;
        vectorN_t & uCommand = stepperState_.uCommand;
        vectorN_t & uInternal = stepperState_.uInternal;
        stepperState_t::forceVector_t & fext = stepperState_.fExternal;

        // Compute the forward kinematics
        computeForwardKinematics(q, v, a);

        // Initialize the external contact forces
        computeExternalForces(t, x, fext);

        // Initialize the sensor data
        model_->setSensorsData(t, q, v, a, u);

        // Initialize the controller's dynamically registered variables
        computeCommand(t, q, v, uCommand);

        // Compute the internal dynamics
        computeInternalDynamics(t, q, v, uInternal);

        // Compute the total torque vector
        u = stepperState_.uInternal;
        std::vector<int32_t> const & motorsVelocityIdx = model_->getMotorsVelocityIdx();
        for (uint32_t i=0; i < motorsVelocityIdx.size(); i++)
        {
            uint32_t const & jointIdx = motorsVelocityIdx[i];
            u[jointIdx] += stepperState_.uCommand[i];
        }

        // Compute dynamics
        a = Engine::aba(model_->pncModel_, model_->pncData_, q, v, u, fext);

        return result_t::SUCCESS;
    }

    result_t Engine::simulate(vectorN_t const & x_init,
                              float64_t const & tEnd)
    {
        result_t returnCode = result_t::SUCCESS;

        if(!isInitialized_)
        {
            std::cout << "Error - Engine::simulate - Engine not initialized. Impossible to run the simulation." << std::endl;
            returnCode = result_t::ERROR_INIT_FAILED;
        }

        if(tEnd < 5e-3)
        {
            std::cout << "Error - Engine::simulate - The duration of the simulation cannot be shorter than 5ms." << std::endl;
            returnCode = result_t::ERROR_BAD_INPUT;
        }

        // Reset the model, controller, and engine
        if (returnCode == result_t::SUCCESS)
        {
            returnCode = setState(x_init, true, false);
        }

        // Integration loop based on boost::numeric::odeint::detail::integrate_times
        while (returnCode == result_t::SUCCESS)
        {
            /* Stop the simulation if the end time has been reached, if
               the callback returns false, or if the max number of
               integration steps is exceeded. */
            if (tEnd - stepperState_.t < MIN_TIME_STEP)
            {
                if (engineOptions_->stepper.verbose)
                {
                    std::cout << "Simulation done: desired final time reached." << std::endl;
                }
                break;
            }
            else if (!callbackFct_(stepperState_.t, stepperState_.x))
            {
                if (engineOptions_->stepper.verbose)
                {
                    std::cout << "Simulation done: callback returned false." << std::endl;
                }
                break;
            }
            else if (engineOptions_->stepper.iterMax > 0
            && stepperStateLast_.iter >= (uint32_t) engineOptions_->stepper.iterMax)
            {
                if (engineOptions_->stepper.verbose)
                {
                    std::cout << "Simulation done: maximum number of integration steps exceeded." << std::endl;
                }
                break;
            }
            // Perform a single integration step up to tEnd, stopping at stepperUpdatePeriod_ to log.
            float64_t stepSize = min(stepperUpdatePeriod_ , tEnd - stepperState_.t);
            returnCode = step(stepSize); // Automatic dt adjustment
        }

        return returnCode;
    }

    result_t Engine::step(float64_t const& stepSize)
    {
        // Check if the engine is initialized
        if(!isInitialized_)
        {
            std::cout << "Error - Engine::step - Engine not initialized. Impossible to perform a simulation step." << std::endl;
            return result_t::ERROR_INIT_FAILED;
        }

        // Configure telemetry if needed.
        if (!isTelemetryConfigured_)
        {
            configureTelemetry();
            // Write the header: this locks the registration of new variables
            telemetryRecorder_->initialize();
            // Log current buffer content as first point of the log data.
            updateTelemetry();
        }

        // Check if there is something wrong with the integration
        if ((stepperState_.x.array() != stepperState_.x.array()).any()) // isnan if NOT equal to itself
        {
            std::cout << "Error - Engine::step - The low-level ode solver failed. Consider increasing accuracy." << std::endl;
            return result_t::ERROR_GENERIC;
        }

        // Check if the desired step size is suitable
        if (stepSize > EPS && stepSize < MIN_TIME_STEP)
        {
            std::cout << "Error - Engine::step - The desired step size 'stepSize' is out of bounds." << std::endl;
            return result_t::ERROR_BAD_INPUT;
        }

        // Set end time: apply default step size (stepperUpdatePeriod_) if negative value given as input.
        float64_t tEnd;
        if (stepSize > EPS)
        {
            tEnd = stepperState_.t + stepSize;
        }
        else
        {
            tEnd = stepperState_.t + stepperUpdatePeriod_;
        }

        // Get references/copies of some internal stepper buffers
        float64_t t = stepperState_.t;
        float64_t & dt = stepperState_.dt;
        vectorN_t & x = stepperState_.x;
        Eigen::Ref<vectorN_t> q = stepperState_.q();
        Eigen::Ref<vectorN_t> v = stepperState_.v();
        vectorN_t & dxdt = stepperState_.dxdt;
        Eigen::Ref<vectorN_t> a = stepperState_.a();
        vectorN_t & u = stepperState_.u;

        // Define the stepper iterators.
        auto system =
            [this](vectorN_t const & xIn,
                   vectorN_t       & dxdtIn,
                   float64_t const & tIn)
            {
                this->computeSystemDynamics(tIn, xIn, dxdtIn);
            };

        // Define a failure checker for the stepper
        failed_step_checker fail_checker;

        // Perform the integration
        while(tEnd - t > EPS)
        {
            float64_t tNext = t;
            // Solver cannot simulate a timestep smaller than MIN_TIME_STEP
            if (stepperUpdatePeriod_ > MIN_TIME_STEP)
            {
                // Update the sensor data if necessary (only for finite update frequency)
                if (engineOptions_->stepper.sensorsUpdatePeriod > EPS)
                {
                    float64_t const & sensorsUpdatePeriod = engineOptions_->stepper.sensorsUpdatePeriod;
                    float64_t dtNextSensorsUpdatePeriod = sensorsUpdatePeriod - std::fmod(t, sensorsUpdatePeriod);
                    if (dtNextSensorsUpdatePeriod < MIN_TIME_STEP
                    || sensorsUpdatePeriod - dtNextSensorsUpdatePeriod < MIN_TIME_STEP)
                    {
                        model_->setSensorsData(t, q, v, a, u);
                    }
                }

                // Update the controller command if necessary (only for finite update frequency)
                if (engineOptions_->stepper.controllerUpdatePeriod > EPS)
                {
                    float64_t const & controllerUpdatePeriod = engineOptions_->stepper.controllerUpdatePeriod;
                    float64_t dtNextControllerUpdatePeriod = controllerUpdatePeriod - std::fmod(t, controllerUpdatePeriod);
                    if (dtNextControllerUpdatePeriod < MIN_TIME_STEP
                    || controllerUpdatePeriod - dtNextControllerUpdatePeriod < MIN_TIME_STEP)
                    {
                        computeCommand(t, q, v, stepperState_.uCommand);

                        /* Update the internal stepper state dxdt since the dynamics has changed.
                            Make sure the next impulse force iterator has NOT been updated at this point ! */
                        if (engineOptions_->stepper.odeSolver != "explicit_euler")
                        {
                            computeSystemDynamics(t, x, dxdt);
                        }
                    }
                }
            }

            // Get the next impulse force application time and update the iterator if necessary
            float64_t tForceImpulseNext = tEnd;
            if (forceImpulseNextIt_ != forcesImpulse_.end())
            {
                float64_t tForceImpulseNextTmp = forceImpulseNextIt_->first;
                float64_t dtForceImpulseNext = std::get<1>(forceImpulseNextIt_->second);
                if (t > tForceImpulseNextTmp + dtForceImpulseNext)
                {
                    ++forceImpulseNextIt_;
                    tForceImpulseNextTmp = forceImpulseNextIt_->first;
                }

                if (forceImpulseNextIt_ != forcesImpulse_.end())
                {
                    if (tForceImpulseNextTmp > t)
                    {
                        tForceImpulseNext = tForceImpulseNextTmp;
                    }
                    else
                    {
                        if (forceImpulseNextIt_ != std::prev(forcesImpulse_.end()))
                        {
                            tForceImpulseNext = std::next(forceImpulseNextIt_)->first;
                        }
                    }
                }
            }

            if (stepperUpdatePeriod_ > EPS)
            {
                // Get the time of the next breakpoint for the ODE solver:
                // a breakpoint occurs if we reached tEnd, if an external force is applied, or if we
                // need to update the sensors / controller.
                float64_t dtNextUpdatePeriod = stepperUpdatePeriod_ - std::fmod(t, stepperUpdatePeriod_);
                if (dtNextUpdatePeriod < MIN_TIME_STEP)
                {
                    // Step to reach next sensors / controller update is too short: skip one
                    // controller update and jump to the next one.
                    // Note that in this case, the sensors have already been updated in
                    // anticipation in previous loop.
                    tNext += min(dtNextUpdatePeriod + stepperUpdatePeriod_,
                                 tEnd - t,
                                 tForceImpulseNext - t);
                }
                else
                {
                    tNext += min(dtNextUpdatePeriod,
                                 tEnd - t,
                                 tForceImpulseNext - t);
                }

                // Compute the next step using adaptive step method
                while (t < tNext)
                {
                    // Adjust stepsize to end up exactly at the next breakpoint
                    // and prevent steps larger than dtMax
                    dt = min(dt, tNext - t, engineOptions_->stepper.dtMax);
                    if (success == boost::apply_visitor(
                        [&](auto && one)
                        {
                            return one.try_step(system, x, dxdt, t, dt);
                        }, stepper_))
                    {
                        fail_checker.reset(); // reset the fail counter
                        if (engineOptions_->stepper.logInternalStepperSteps)
                        {
                            updateTelemetry();
                        }
                    }
                    else
                    {
                        fail_checker();  // check for possible overflow of failed steps in step size adjustment
                    }
                }

                // Update the current time
                t = tNext;
            }
            else
            {
                // Make sure it ends exactly at the tEnd, never exceeds dtMax, and stop to apply impulse forces
                dt = min(dt,
                         engineOptions_->stepper.dtMax,
                         tEnd - t,
                         tForceImpulseNext - t);

                // Compute the next step using adaptive step method
                controlled_step_result res = fail;
                while (res == fail)
                {
                    res = boost::apply_visitor(
                        [&](auto && one)
                        {
                            return one.try_step(system, x, dxdt, t, dt);
                        }, stepper_);
                    if (res == success)
                    {
                        fail_checker.reset(); // reset the fail counter
                        if (engineOptions_->stepper.logInternalStepperSteps)
                        {
                            updateTelemetry();
                        }
                    }
                    else
                    {
                        fail_checker();  // check for possible overflow of failed steps in step size adjustment
                    }
                }
            }
        }

        // Update the iteration counter
        stepperState_.t = t;
        ++stepperState_.iter;

        /* Monitor current iteration number, and log the current time,
           state, command, and sensors data. */
        if (!engineOptions_->stepper.logInternalStepperSteps)
        {
            updateTelemetry();
        }

        return result_t::SUCCESS;
    }


    void Engine::computeForwardKinematics(Eigen::Ref<vectorN_t const> q,
                                          Eigen::Ref<vectorN_t const> v,
                                          Eigen::Ref<vectorN_t const> a)
    {
        pinocchio::forwardKinematics(model_->pncModel_, model_->pncData_, q, v, a);
        pinocchio::updateFramePlacements(model_->pncModel_, model_->pncData_);
    }

    void Engine::computeExternalForces(float64_t const & t,
                                       vectorN_t const & x,
                                       pinocchio::container::aligned_vector<pinocchio::Force> & fext)
    {
        // Reinitialize the external forces
        for (pinocchio::Force & fext_i : fext)
        {
            fext_i.setZero();
        }

        // Compute the contact forces
        std::vector<int32_t> const & contactFramesIdx = model_->getContactFramesIdx();
        for(uint32_t i=0; i < contactFramesIdx.size(); i++)
        {
            // Compute force in the contact frame.
            int32_t const & contactFrameIdx = contactFramesIdx[i];
            vector3_t const fextInFrame = computeContactDynamics(contactFrameIdx);
            model_->contactForces_[i] = pinocchio::Force(fextInFrame, vector3_t::Zero());

            // Apply the force at the origin of the parent joint frame
            vector6_t const fextLocal = computeFrameForceOnParentJoint(contactFrameIdx, fextInFrame);
            int32_t const & parentIdx = model_->pncModel_.frames[contactFrameIdx].parent;
            fext[parentIdx] += pinocchio::Force(fextLocal);
        }

        // Add the effect of user-defined external forces
        if (forceImpulseNextIt_ != forcesImpulse_.end())
        {
            float64_t const & tForceImpulseNext = forceImpulseNextIt_->first;
            float64_t const & dt = std::get<1>(forceImpulseNextIt_->second);
            if (tForceImpulseNext <= t && t <= tForceImpulseNext + dt)
            {
                int32_t frameIdx;
                std::string const & frameName = std::get<0>(forceImpulseNextIt_->second);
                vector3_t const & F = std::get<2>(forceImpulseNextIt_->second);
                getFrameIdx(model_->pncModel_, frameName, frameIdx);
                int32_t const & parentIdx = model_->pncModel_.frames[frameIdx].parent;
                fext[parentIdx] += pinocchio::Force(computeFrameForceOnParentJoint(frameIdx, F));
            }
        }

        for (auto const & forceProfile : forcesProfile_)
        {
            int32_t const & frameIdx = std::get<0>(forceProfile.second);
            int32_t const & parentIdx = model_->pncModel_.frames[frameIdx].parent;
            forceFunctor_t const & forceFct = std::get<1>(forceProfile.second);
            fext[parentIdx] += pinocchio::Force(computeFrameForceOnParentJoint(frameIdx, forceFct(t, x)));
        }
    }

    void Engine::computeCommand(float64_t                   const & t,
                                Eigen::Ref<vectorN_t const>         q,
                                Eigen::Ref<vectorN_t const>         v,
                                vectorN_t                         & u)
    {
        // Reinitialize the external forces
        u.setZero();

        // Command the command
        controller_->computeCommand(t, q, v, u);

        // Enforce the torque limits
        if (model_->mdlOptions_->joints.enableTorqueLimit)
        {
            // The torque is in motor space at this point
            vectorN_t const & torqueLimitMax = model_->getTorqueLimit();
            std::vector<int32_t> const & motorsVelocityIdx = model_->getMotorsVelocityIdx();
            for (uint32_t i=0; i < motorsVelocityIdx.size(); i++)
            {
                float64_t const & torque_max = torqueLimitMax[i];
                u[i] = clamp(u[i], -torque_max, torque_max);
            }
        }
    }

    void Engine::registerForceImpulse(std::string const & frameName,
                                      float64_t   const & t,
                                      float64_t   const & dt,
                                      vector3_t   const & F)
    {
        // Make sure that the forces do NOT overlap while taking into account dt.

        forcesImpulse_[t] = std::make_tuple(frameName, dt, F);
    }

    void Engine::registerForceProfile(std::string      const & frameName,
                                      forceFunctor_t           forceFct)
    {
        forcesProfile_.emplace_back(frameName, std::make_tuple(0, forceFct));
    }

    configHolder_t Engine::getOptions(void) const
    {
        return engineOptionsHolder_;
    }

    result_t Engine::setOptions(configHolder_t const & engineOptions)
    {
        result_t returnCode = result_t::SUCCESS;

        // Make sure the dtMax is not out of bounds
        configHolder_t stepperOptions = boost::get<configHolder_t>(engineOptions.at("stepper"));
        float64_t const & dtMax = boost::get<float64_t>(stepperOptions.at("dtMax"));
        if (MAX_TIME_STEP < dtMax || dtMax < MIN_TIME_STEP)
        {
            std::cout << "Error - Engine::setOptions - 'dtMax' option is out of bounds." << std::endl;
            returnCode = result_t::ERROR_BAD_INPUT;
        }

        // Make sure the selected ode solver is available and instantiate it
        if (returnCode == result_t::SUCCESS)
        {
            std::string const & odeSolver = boost::get<std::string>(stepperOptions.at("odeSolver"));
            if (odeSolver != "runge_kutta_dopri5" && odeSolver != "explicit_euler")
            {
                std::cout << "Error - Engine::setOptions - The requested 'odeSolver' is not available." << std::endl;
                returnCode = result_t::ERROR_BAD_INPUT;
            }
        }

        // Make sure the controller and sensor update periods are multiple of each other
        float64_t const & sensorsUpdatePeriod =
            boost::get<float64_t>(stepperOptions.at("sensorsUpdatePeriod"));
        float64_t const & controllerUpdatePeriod =
            boost::get<float64_t>(stepperOptions.at("controllerUpdatePeriod"));
        if (returnCode == result_t::SUCCESS)
        {
            if ((EPS < sensorsUpdatePeriod && sensorsUpdatePeriod < MIN_TIME_STEP)
            || (EPS < controllerUpdatePeriod && controllerUpdatePeriod < MIN_TIME_STEP))
            {
                std::cout << "Error - Engine::setOptions - Cannot simulate a discrete system with period smaller than" << \
                    MIN_TIME_STEP << "s. Increase period or switch to continuous mode by setting period to zero." << std::endl;
                returnCode = result_t::ERROR_BAD_INPUT;
            }
            // Verify that, if both values are set above sensorsUpdatePeriod, they are multiple of each other:
            // to verify that b devides a with a tolerance EPS, we need to verify that a % b \in [-EPS, EPS] -
            // however since std::fmod yields values in [0, b[, this interval maps to [O, EPS] \union [b - EPS, b[.
            else if (sensorsUpdatePeriod > EPS && controllerUpdatePeriod > EPS
            && (std::min(std::fmod(controllerUpdatePeriod, sensorsUpdatePeriod),
                         sensorsUpdatePeriod - std::fmod(controllerUpdatePeriod, sensorsUpdatePeriod)) > EPS
             && std::min(std::fmod(sensorsUpdatePeriod, controllerUpdatePeriod),
                         controllerUpdatePeriod - std::fmod(sensorsUpdatePeriod, controllerUpdatePeriod)) > EPS))
            {
                std::cout << "Error - Engine::setOptions - In discrete mode, the controller and sensor update periods "\
                             "must be multiple of each other." << std::endl;
                returnCode = result_t::ERROR_BAD_INPUT;
            }
        }

        // Make sure the contacts options are fine
        if (returnCode == result_t::SUCCESS)
        {
            configHolder_t contactsOptions = boost::get<configHolder_t>(engineOptions.at("contacts"));
            float64_t const & dryFrictionVelEps =
                boost::get<float64_t>(contactsOptions.at("dryFrictionVelEps"));
            float64_t const & transitionEps =
                boost::get<float64_t>(contactsOptions.at("transitionEps"));
            if (dryFrictionVelEps < 0.0)
            {
                std::cout << "Error - Engine::setOptions - The contacts option 'dryFrictionVelEps' must be positive." << std::endl;
                returnCode = result_t::ERROR_BAD_INPUT;
            }
            else if (transitionEps < 0.0)
            {
                std::cout << "Error - Engine::setOptions - The contacts option 'transitionEps' must be positive." << std::endl;
                returnCode = result_t::ERROR_BAD_INPUT;
            }
        }

        // Make sure the joints options are fine
        if (returnCode == result_t::SUCCESS)
        {
            configHolder_t jointsOptions = boost::get<configHolder_t>(engineOptions.at("joints"));
            float64_t const & boundTransitionEps =
                boost::get<float64_t>(jointsOptions.at("boundTransitionEps"));
            if (boundTransitionEps < 0.0)
            {
                std::cout << "Error - Engine::setOptions - The joints option 'boundTransitionEps' must be positive." << std::endl;
                returnCode = result_t::ERROR_BAD_INPUT;
            }
        }

        // Compute the breakpoints' period (for command or observation) during the integration loop
        if (returnCode == result_t::SUCCESS)
        {
            if (sensorsUpdatePeriod < MIN_TIME_STEP)
            {
                stepperUpdatePeriod_ = controllerUpdatePeriod;
            }
            else if (controllerUpdatePeriod < MIN_TIME_STEP)
            {
                stepperUpdatePeriod_ = sensorsUpdatePeriod;
            }
            else
            {
                stepperUpdatePeriod_ = std::min(sensorsUpdatePeriod, controllerUpdatePeriod);
            }
        }

        // Make sure the user-defined gravity force has the right dimension
        if (returnCode == result_t::SUCCESS)
        {
            configHolder_t worldOptions = boost::get<configHolder_t>(engineOptions.at("world"));
            vectorN_t gravity = boost::get<vectorN_t>(worldOptions.at("gravity"));
            if (gravity.size() != 6)
            {
                std::cout << "Error - Engine::setOptions - The size of the gravity force vector must be 6." << std::endl;
                returnCode = result_t::ERROR_BAD_INPUT;
            }
        }

        if (returnCode == result_t::SUCCESS)
        {
            // Update the internal options
            engineOptionsHolder_ = engineOptions;

            // Create a fast struct accessor
            engineOptions_ = std::make_unique<engineOptions_t const>(engineOptionsHolder_);
        }

        return returnCode;
    }

    bool Engine::getIsInitialized(void) const
    {
        return isInitialized_;
    }

    Model & Engine::getModel(void) const
    {
        return *model_;
    }

    AbstractController & Engine::getController(void) const
    {
        return *controller_;
    }

    stepperState_t const & Engine::getStepperState(void) const
    {
        return stepperState_;
    }

    void logDataRawToEigenMatrix(std::vector<float32_t>               const & timestamps,
                                 std::vector<std::vector<int32_t> >   const & intData,
                                 std::vector<std::vector<float32_t> > const & floatData,
                                 matrixN_t                                  & logData)
    {
        // Never empty since it contains at least the initial state
        logData.resize(timestamps.size(), 1 + intData[0].size() + floatData[0].size());
        logData.col(0) = Eigen::Matrix<float32_t, 1, Eigen::Dynamic>::Map(
            timestamps.data(), timestamps.size()).cast<float64_t>();
        for (uint32_t i=0; i<intData.size(); i++)
        {
            logData.block(i, 1, 1, intData[i].size()) =
                Eigen::Matrix<int32_t, 1, Eigen::Dynamic>::Map(
                    intData[i].data(), intData[i].size()).cast<float64_t>();
        }
        for (uint32_t i=0; i<floatData.size(); i++)
        {
            logData.block(i, 1 + intData[0].size(), 1, floatData[i].size()) =
                Eigen::Matrix<float32_t, 1, Eigen::Dynamic>::Map(
                    floatData[i].data(), floatData[i].size()).cast<float64_t>();
        }
    }

    result_t Engine::getLogDataRaw(std::vector<std::string>             & header,
                                   std::vector<float32_t>               & timestamps,
                                   std::vector<std::vector<int32_t> >   & intData,
                                   std::vector<std::vector<float32_t> > & floatData)
    {
        if(!isInitialized_ || !telemetryRecorder_->getIsInitialized())
        {
            std::cout << "Error - Engine::getLogDataRaw - Telemetry not initialized. Impossible to get log data." << std::endl;
            return result_t::ERROR_INIT_FAILED;
        }

        telemetryRecorder_->getData(header, timestamps, intData, floatData);

        return result_t::SUCCESS;
    }

    result_t Engine::getLogData(std::vector<std::string> & header,
                                matrixN_t                & logData)
    {
        std::vector<float32_t> timestamps;
        std::vector<std::vector<int32_t> > intData;
        std::vector<std::vector<float32_t> > floatData;
        result_t returnCode = getLogDataRaw(header, timestamps, intData, floatData);

        if (returnCode == result_t::SUCCESS)
        {
            logDataRawToEigenMatrix(timestamps, intData, floatData, logData);
        }

        return returnCode;
    }

    vectorN_t Engine::getLogFieldValue(std::string              const & fieldName,
                                       std::vector<std::string>       & header,
                                       matrixN_t                      & logData)
    {
        vectorN_t fieldData = vectorN_t::Zero(0);

        std::vector<std::string>::iterator iterator = std::find (header.begin(), header.end(), fieldName);
        std::vector<std::string>::iterator start = std::find (header.begin(), header.end(), "StartColumns");
        if (iterator != header.end())
        {
            fieldData = logData.col(std::distance(start, iterator) - 1);
        }

        return fieldData;
    }

    result_t Engine::writeLogTxt(std::string const & filename)
    {
        if(!isInitialized_ || !telemetryRecorder_->getIsInitialized())
        {
            std::cout << "Error - Engine::writeLogTxt - Telemetry not initialized. Impossible to write log txt." << std::endl;
            return result_t::ERROR_INIT_FAILED;
        }

        std::vector<std::string> header;
        matrixN_t log;
        getLogData(header, log);

        std::ofstream myfile = std::ofstream(filename,
                                             std::ios::out |
                                             std::ofstream::trunc);

        if (myfile.is_open())
        {
            auto indexConstantEnd = std::find(header.begin(), header.end(), START_COLUMNS);
            std::copy(header.begin() + 1,
                      indexConstantEnd - 1,
                      std::ostream_iterator<std::string>(myfile, ", ")); // Discard the first one (start constant flag)
            std::copy(indexConstantEnd - 1,
                      indexConstantEnd,
                      std::ostream_iterator<std::string>(myfile, "\n"));
            std::copy(indexConstantEnd + 1,
                      header.end() - 2,
                      std::ostream_iterator<std::string>(myfile, ", "));
            std::copy(header.end() - 2,
                      header.end() - 1,
                      std::ostream_iterator<std::string>(myfile, "\n")); // Discard the last one (start data flag)

            Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");
            myfile << log.format(CSVFormat);

            myfile.close();
        }
        else
        {
            std::cout << "Error - Engine::writeLogTxt - Impossible to create the log file. Check if root folder exists and if you have writing permissions." << std::endl;
            return result_t::ERROR_BAD_INPUT;
        }

        return result_t::SUCCESS;
    }

    result_t Engine::writeLogBinary(std::string const & filename)
    {
        if(!isInitialized_ || !telemetryRecorder_->getIsInitialized())
        {
            std::cout << "Error - Engine::writeLogBinary - Telemetry not initialized. Impossible to write log data." << std::endl;
            return result_t::ERROR_INIT_FAILED;
        }

        telemetryRecorder_->writeDataBinary(filename);

        return result_t::SUCCESS;
    }

    result_t Engine::parseLogBinaryRaw(std::string                          const & filename,
                                       std::vector<std::string>                   & header,
                                       std::vector<float32_t>                     & timestamps,
                                       std::vector<std::vector<int32_t> >         & intData,
                                       std::vector<std::vector<float32_t> >       & floatData)
    {
        int64_t integerSectionSize;
        int64_t floatSectionSize;
        int64_t headerSize;

        std::ifstream myfile = std::ifstream(filename,
                                             std::ios::in |
                                             std::ifstream::binary);

        if (myfile.is_open())
        {
            // Skip the version flag
            int64_t header_version_length = sizeof(int32_t);
            myfile.seekg(header_version_length);

            std::vector<std::string> headerBuffer;
            std::string subHeaderBuffer;

            // Get all the logged constants
            while (std::getline(myfile, subHeaderBuffer, '\0').good() &&
                   subHeaderBuffer != START_COLUMNS)
            {
                headerBuffer.push_back(subHeaderBuffer);
            }

            // Get the names of the logged variables
            while (std::getline(myfile, subHeaderBuffer, '\0').good() &&
                   subHeaderBuffer != (START_DATA + START_LINE_TOKEN))
            {
                // Do nothing
            }

            // Make sure the log file is not corrupted
            if (!myfile.good())
            {
                std::cout << "Error - Engine::parseLogBinary - Corrupted log file." << std::endl;
                return result_t::ERROR_BAD_INPUT;
            }

            // Extract the number of intergers and floats from the list of logged constants
            std::string const & headerNumIntEntries = headerBuffer[headerBuffer.size() - 2];
            int32_t delimiter = headerNumIntEntries.find("=");
            int32_t NumIntEntries = std::stoi(headerNumIntEntries.substr(delimiter + 1));
            std::string const & headerNumFloatEntries = headerBuffer[headerBuffer.size() - 1];
            delimiter = headerNumFloatEntries.find("=");
            int32_t NumFloatEntries = std::stoi(headerNumFloatEntries.substr(delimiter + 1));

            // Deduce the parameters required to parse the whole binary log file
            integerSectionSize = (NumIntEntries - 1) * sizeof(int32_t); // Remove Global.Time
            floatSectionSize = NumFloatEntries * sizeof(float32_t);
            headerSize = ((int32_t) myfile.tellg()) - START_LINE_TOKEN.size() - 1;

            // Close the file
            myfile.close();
        }
        else
        {
            std::cout << "Error - Engine::parseLogBinary - Impossible to open the log file. Check that the file exists and that you have reading permissions." << std::endl;
            return result_t::ERROR_BAD_INPUT;
        }

        FileDevice device(filename);
        device.open(OpenMode::READ_ONLY);
        std::vector<AbstractIODevice *> flows;
        flows.push_back(&device);

        TelemetryRecorder::getData(header,
                                   timestamps,
                                   intData,
                                   floatData,
                                   flows,
                                   integerSectionSize,
                                   floatSectionSize,
                                   headerSize);

        return result_t::SUCCESS;
    }

    result_t Engine::parseLogBinary(std::string              const & filename,
                                    std::vector<std::string>       & header,
                                    matrixN_t                      & logData)
    {
        std::vector<float32_t> timestamps;
        std::vector<std::vector<int32_t> > intData;
        std::vector<std::vector<float32_t> > floatData;
        result_t returnCode = parseLogBinaryRaw(filename,
                                                header,
                                                timestamps,
                                                intData,
                                                floatData);

        if (returnCode == result_t::SUCCESS)
        {
            logDataRawToEigenMatrix(timestamps, intData, floatData, logData);
        }

        return result_t::SUCCESS;
    }

    void Engine::computeSystemDynamics(float64_t const & t,
                                       vectorN_t const & x,
                                       vectorN_t       & dxdt)
    {
        /* Note that the position of the free flyer is in world frame, whereas the
           velocities and accelerations are relative to the parent body frame. */

        // Get references to the internal stepper buffers
        Eigen::Ref<vectorN_t const> q = x.head(model_->nq());
        Eigen::Ref<vectorN_t const> v = x.tail(model_->nv());
        vectorN_t & u = stepperState_.u;
        vectorN_t & uCommand = stepperState_.uCommand;
        vectorN_t & uInternal = stepperState_.uInternal;
        stepperState_t::forceVector_t & fext = stepperState_.fExternal;

        // Compute kinematics information
        computeForwardKinematics(q, v, stepperState_.a());

        /* Compute the external contact forces.
           Note that one must call this method BEFORE updating the sensors since
           the force sensor measurements rely on model_->contactForces_ */
        computeExternalForces(t, x, fext);

        /* Update the sensor data if necessary (only for infinite update frequency).
           Note that it is impossible to have access to the torques
           since they depend on the sensor values themselves. */
        if (engineOptions_->stepper.sensorsUpdatePeriod < MIN_TIME_STEP)
        {
            model_->setSensorsData(t, q, v, stepperStateLast_.a(), stepperStateLast_.u);
        }

        /* Update the controller command if necessary (only for infinite update frequency).
           Make sure that the sensor state has been updated beforehand. */
        if (engineOptions_->stepper.controllerUpdatePeriod < MIN_TIME_STEP)
        {
            computeCommand(t, q, v, uCommand);
        }

        /* Compute the internal dynamics.
           Make sure that the sensor state has been updated beforehand since
           the user-defined internal dynamics may rely on it. */
        computeInternalDynamics(t, q, v, uInternal);

        // Compute the total torque vector
        u = uInternal;
        std::vector<int32_t> const & motorsVelocityIdx = model_->getMotorsVelocityIdx();
        for (uint32_t i=0; i < motorsVelocityIdx.size(); i++)
        {
            uint32_t const & jointIdx = motorsVelocityIdx[i];
            u[jointIdx] += uCommand[i];
        }

        // Compute the dynamics
        vectorN_t a = Engine::aba(model_->pncModel_, model_->pncData_, q, v, u, fext);

        float64_t dt = t - stepperStateLast_.t;
        vectorN_t qDot(model_->nq());
        computePositionDerivative(model_->pncModel_, q, v, qDot, dt);

        // Fill up dxdt
        dxdt.resize(model_->nx());
        dxdt.head(model_->nq()) = qDot;
        dxdt.tail(model_->nv()) = a;
    }

    vector6_t Engine::computeFrameForceOnParentJoint(int32_t   const & frameId,
                                                     vector3_t const & fextInWorld) const
    {
        // Get various transformations
        matrix3_t const & tformFrameRot = model_->pncData_.oMf[frameId].rotation();
        matrix3_t const & tformFrameJointRot = model_->pncModel_.frames[frameId].placement.rotation();
        vector3_t const & posFrameJoint = model_->pncModel_.frames[frameId].placement.translation();

        // Compute the forces at the origin of the parent joint frame
        vector6_t fextLocal;
        fextLocal.head<3>() = tformFrameJointRot * tformFrameRot.transpose() * fextInWorld;
        fextLocal.tail<3>() = posFrameJoint.cross(fextLocal.head<3>());

        return fextLocal;
    }

    vector3_t Engine::computeContactDynamics(int32_t const & frameId) const
    {
        // Returns the external force in the contact frame.
        // It must then be converted into a force onto the parent joint.
        // /* /!\ Note that the contact dynamics depends only on kinematics data. /!\ */

        contactOptions_t const * const contactOptions_ = &engineOptions_->contacts;

        matrix3_t const & tformFrameRot = model_->pncData_.oMf[frameId].rotation();
        vector3_t const & posFrame = model_->pncData_.oMf[frameId].translation();

        // Initialize the contact force
        vector3_t fextInWorld;
        std::pair<float64_t, vector3_t> ground = engineOptions_->world.groundProfile(posFrame);
        float64_t const & zGround = std::get<0>(ground);
        vector3_t nGround = std::get<1>(ground);
        nGround.normalize();
        float64_t depth = (posFrame(2) - zGround) * nGround(2); // First-order projection (exact assuming flat surface)

        if(depth < 0.0)
        {
            // Get frame motion in the motion frame.
            vector3_t motionFrame = pinocchio::getFrameVelocity(model_->pncModel_,
                                                                model_->pncData_,
                                                                frameId).linear();
            vector3_t vFrameInWorld = tformFrameRot * motionFrame;
            float64_t vDepth = vFrameInWorld.dot(nGround);

            // Compute normal force
            float64_t fextNormal = 0.0;
            if(vDepth < 0.0)
            {
                fextNormal -= contactOptions_->damping * vDepth;
            }
            fextNormal -= contactOptions_->stiffness * depth;
            fextInWorld = fextNormal * nGround;

            // Compute friction forces
            vector3_t vTangential = vFrameInWorld - vDepth * nGround;
            float64_t vNorm = vTangential.norm();

            float64_t frictionCoeff = 0.0;
            if(vNorm >= contactOptions_->dryFrictionVelEps)
            {
                if(vNorm < 1.5 * contactOptions_->dryFrictionVelEps)
                {
                    frictionCoeff = -2.0 * (contactOptions_->frictionDry -
                        contactOptions_->frictionViscous) * (vNorm / contactOptions_->dryFrictionVelEps)
                        + 3.0 * contactOptions_->frictionDry - 2.0*contactOptions_->frictionViscous;
                }
                else
                {
                    frictionCoeff = contactOptions_->frictionViscous;
                }
            }
            else
            {
                frictionCoeff = contactOptions_->frictionDry *
                    (vNorm / contactOptions_->dryFrictionVelEps);
            }
            float64_t fextTangential = frictionCoeff * fextNormal;
            fextInWorld += -fextTangential * vTangential;

            // Make sure that the force never exceeds 1e5 N for the sake of numerical stability
            fextInWorld = clamp(fextInWorld, -1e5, 1e5);

            // Add blending factor
            if (contactOptions_->transitionEps > EPS)
            {
                float64_t blendingFactor = -depth / contactOptions_->transitionEps;
                float64_t blendingLaw = std::tanh(2 * blendingFactor);
                fextInWorld *= blendingLaw;
            }
        }
        else
        {
            fextInWorld.setZero();
        }

        return fextInWorld;
    }

    void Engine::computeInternalDynamics(float64_t                   const & t,
                                         Eigen::Ref<vectorN_t const>         q,
                                         Eigen::Ref<vectorN_t const>         v,
                                         vectorN_t                         & u)
    {
        // Reinitialize the internal torque vector
        u.setZero();

        // Compute the user-defined internal dynamics
        controller_->internalDynamics(t, q, v, u);

        // Enforce the position limit (do not support spherical joints)
        if (model_->mdlOptions_->joints.enablePositionLimit)
        {
            Engine::jointOptions_t const & engineJointOptions = engineOptions_->joints;

            std::vector<int32_t> const & jointsModelIdx = model_->getRigidJointsModelIdx();
            vectorN_t const & positionLimitMin = model_->getPositionLimitMin();
            vectorN_t const & positionLimitMax = model_->getPositionLimitMax();
            uint32_t jointIdxOffset = 0;
            for (uint32_t i = 0; i < jointsModelIdx.size(); i++)
            {
                uint32_t const & jointPositionIdx = model_->pncModel_.joints[jointsModelIdx[i]].idx_q();
                uint32_t const & jointVelocityIdx = model_->pncModel_.joints[jointsModelIdx[i]].idx_v();
                uint32_t const & jointDof = model_->pncModel_.joints[jointsModelIdx[i]].nq();

                for (uint32_t j = 0; j < jointDof; j++)
                {
                    float64_t const & qJoint = q(jointPositionIdx + j);
                    float64_t const & vJoint = v(jointVelocityIdx + j);
                    float64_t const & qJointMin = positionLimitMin[jointIdxOffset];
                    float64_t const & qJointMax = positionLimitMax[jointIdxOffset];

                    float64_t forceJoint = 0;
                    float64_t qJointError = 0;
                    if (qJoint > qJointMax)
                    {
                        qJointError = qJoint - qJointMax;
                        float64_t damping = -engineJointOptions.boundDamping * std::max(vJoint, 0.0);
                        forceJoint = -engineJointOptions.boundStiffness * qJointError + damping;
                    }
                    else if (qJoint < qJointMin)
                    {
                        qJointError = qJointMin - qJoint;
                        float64_t damping = -engineJointOptions.boundDamping * std::min(vJoint, 0.0);
                        forceJoint = engineJointOptions.boundStiffness * qJointError + damping;
                    }

                    if (engineJointOptions.boundTransitionEps > EPS)
                    {
                        float64_t blendingFactor = qJointError / engineJointOptions.boundTransitionEps;
                        float64_t blendingLaw = std::tanh(2 * blendingFactor);
                        forceJoint *= blendingLaw;
                    }

                    u[jointVelocityIdx + j] += clamp(forceJoint, -1e5, 1e5);

                    jointIdxOffset += 1;
                }
            }
        }

        // Enforce the velocity limit (do not support spherical joints)
        if (model_->mdlOptions_->joints.enableVelocityLimit)
        {
            Engine::jointOptions_t const & engineJointOptions = engineOptions_->joints;

            std::vector<int32_t> const & jointsModelIdx = model_->getRigidJointsModelIdx();
            vectorN_t const & velocityLimitMax = model_->getVelocityLimit();

            uint32_t jointIdxOffset = 0;
            for (uint32_t i = 0; i < jointsModelIdx.size(); i++)
            {
                uint32_t const & jointVelocityIdx = model_->pncModel_.joints[jointsModelIdx[i]].idx_v();
                uint32_t const & jointDof = model_->pncModel_.joints[jointsModelIdx[i]].nq();

                for (uint32_t j = 0; j < jointDof; j++)
                {
                    float64_t const & vJoint = v(jointVelocityIdx + j);
                    float64_t const & vJointMin = -velocityLimitMax[jointIdxOffset];
                    float64_t const & vJointMax = velocityLimitMax[jointIdxOffset];

                    float64_t forceJoint = 0.0;
                    float64_t vJointError = 0.0;
                    if (vJoint > vJointMax)
                    {
                        vJointError = vJoint - vJointMax;
                        forceJoint = -engineJointOptions.boundDamping * vJointError;
                    }
                    else if (vJoint < vJointMin)
                    {
                        vJointError = vJointMin - vJoint;
                        forceJoint = engineJointOptions.boundDamping * vJointError;
                    }

                    if (engineJointOptions.boundTransitionEps > EPS)
                    {
                        float64_t blendingFactor = vJointError / engineJointOptions.boundTransitionEps;
                        float64_t blendingLaw = std::tanh(2 * blendingFactor);
                        forceJoint *= blendingLaw;
                    }

                    u[jointVelocityIdx + j] += clamp(forceJoint, -1e5, 1e5);

                    jointIdxOffset += 1;
                }
            }
        }

        // Compute the flexibilities (only support joint_t::SPHERICAL so far)
        Model::dynamicsOptions_t const & mdlDynOptions = model_->mdlOptions_->dynamics;
        std::vector<int32_t> const & jointsModelIdx = model_->getFlexibleJointsModelIdx();
        for (uint32_t i=0; i<jointsModelIdx.size(); ++i)
        {
            uint32_t const & jointPositionIdx = model_->pncModel_.joints[jointsModelIdx[i]].idx_q();
            uint32_t const & jointVelocityIdx = model_->pncModel_.joints[jointsModelIdx[i]].idx_v();
            vectorN_t const & jointStiffness = mdlDynOptions.flexibilityConfig[i].stiffness;
            vectorN_t const & jointDamping = mdlDynOptions.flexibilityConfig[i].damping;

            float64_t theta;
            quaternion_t quat(q.segment<4>(jointPositionIdx).data()); // Only way to initialize with [x,y,z,w] order
            vectorN_t axis = pinocchio::quaternion::log3(quat, theta);
            u.segment<3>(jointVelocityIdx).array() += - jointStiffness.array() * axis.array()
                - jointDamping.array() * v.segment<3>(jointVelocityIdx).array();
        }
    }

    // =====================================================================================================
    // ================ Custom implementation of Pinocchio methods to support motor inertia ================
    // =====================================================================================================

    template<typename Scalar, int Options, template<typename, int> class JointCollectionTpl,
             typename ConfigVectorType, typename TangentVectorType>
    inline Scalar
    Engine::kineticEnergy(pinocchio::ModelTpl<Scalar,Options,JointCollectionTpl> const & model,
                          pinocchio::DataTpl<Scalar,Options,JointCollectionTpl>        & data,
                          Eigen::MatrixBase<ConfigVectorType>                    const & q,
                          Eigen::MatrixBase<TangentVectorType>                   const & v,
                          bool                                                   const & update_kinematics)
    {
        pinocchio::kineticEnergy(model, data, q, v, update_kinematics);
        std::vector<int32_t> const & motorsVelocityIdx = model_->getMotorsVelocityIdx();
        for (uint32_t const & motorIdx : motorsVelocityIdx)
        {
            data.kinetic_energy += 0.5 * model.rotorInertia[motorIdx] * std::pow(v[motorIdx], 2);
        }
        return data.kinetic_energy;
    }

    template<typename Scalar, int Options, template<typename, int> class JointCollectionTpl,
             typename ConfigVectorType, typename TangentVectorType1, typename TangentVectorType2,
             typename ForceDerived>
    inline const typename pinocchio::DataTpl<Scalar,Options,JointCollectionTpl>::TangentVectorType &
    Engine::rnea(pinocchio::ModelTpl<Scalar,Options,JointCollectionTpl> const & model,
                 pinocchio::DataTpl<Scalar,Options,JointCollectionTpl>        & data,
                 Eigen::MatrixBase<ConfigVectorType>                    const & q,
                 Eigen::MatrixBase<TangentVectorType1>                  const & v,
                 Eigen::MatrixBase<TangentVectorType2>                  const & a,
                 pinocchio::container::aligned_vector<ForceDerived>     const & fext)
    {
        pinocchio::rnea(model, data, q, v, a, fext);
        data.tau += model_->pncModel_.rotorInertia.asDiagonal() * a;
        return data.tau;
    }

    template<typename Scalar, int Options, template<typename, int> class JointCollectionTpl>
    struct AbaBackwardStep
    : public pinocchio::fusion::JointVisitorBase< AbaBackwardStep<Scalar,Options,JointCollectionTpl> >
    {
        typedef pinocchio::ModelTpl<Scalar,Options,JointCollectionTpl> Model;
        typedef pinocchio::DataTpl<Scalar,Options,JointCollectionTpl> Data;

        typedef boost::fusion::vector<const Model &, Data &> ArgsType;

        template<typename JointModel>
        static typename std::enable_if<!std::is_same<JointModel, pinocchio::JointModelRevoluteTpl<Scalar, 0, 0> >::value
                                    && !std::is_same<JointModel, pinocchio::JointModelRevoluteTpl<Scalar, 0, 1> >::value
                                    && !std::is_same<JointModel, pinocchio::JointModelRevoluteTpl<Scalar, 0, 2> >::value, void>::type
        algo(pinocchio::JointModelBase<JointModel>                const & jmodel,
                         pinocchio::JointDataBase<typename
                                    JointModel::JointDataDerived>       & jdata,
                         pinocchio::Model                         const & model,
                         pinocchio::Data                                & data)
        {
            typedef typename Model::JointIndex JointIndex;
            typedef typename Data::Inertia Inertia;
            typedef typename Data::Force Force;

            const JointIndex & i = jmodel.id();
            const JointIndex & parent  = model.parents[i];
            typename Inertia::Matrix6 & Ia = data.Yaba[i];

            jmodel.jointVelocitySelector(data.u) -= jdata.S().transpose()*data.f[i];
            jmodel.calc_aba(jdata.derived(), Ia, parent > 0);

            if (parent > 0)
            {
                Force & pa = data.f[i];
                pa.toVector() += Ia * data.a[i].toVector() + jdata.UDinv() * jmodel.jointVelocitySelector(data.u);
                data.Yaba[parent] += pinocchio::internal::SE3actOn<Scalar>::run(data.liMi[i], Ia);
                data.f[parent] += data.liMi[i].act(pa);
            }
        }

        template<typename JointModel>
        static typename std::enable_if<std::is_same<JointModel, typename pinocchio::JointModelRevoluteTpl<Scalar, 0, 0> >::value
                                    || std::is_same<JointModel, typename pinocchio::JointModelRevoluteTpl<Scalar, 0, 1> >::value
                                    || std::is_same<JointModel, typename pinocchio::JointModelRevoluteTpl<Scalar, 0, 2> >::value, void>::type
        algo(pinocchio::JointModelBase<JointModel>                const & jmodel,
                         pinocchio::JointDataBase<typename
                                    JointModel::JointDataDerived>       & jdata,
                         pinocchio::Model                         const & model,
                         pinocchio::Data                                & data)
        {
            /// @brief  See equation 9.28 of Roy Featherstone Rigid Body Dynamics

            typedef typename Model::JointIndex JointIndex;
            typedef typename Data::Inertia Inertia;
            typedef typename Data::Force Force;

            const JointIndex & i = jmodel.id();
            const JointIndex & parent  = model.parents[i];
            typename Inertia::Matrix6 & Ia = data.Yaba[i];

            jmodel.jointVelocitySelector(data.u) -= jdata.S().transpose()*data.f[i];

            // jmodel.calc_aba(jdata.derived(), Ia, parent > 0);
            Scalar const & Im = model.rotorInertia[jmodel.idx_v()];
            jdata.derived().U = Ia.col(Inertia::ANGULAR + getAxis(jmodel));
            jdata.derived().Dinv[0] = (Scalar)(1) / (jdata.derived().U[Inertia::ANGULAR + getAxis(jmodel)] + Im); // Here is the modification !
            jdata.derived().UDinv.noalias() = jdata.derived().U * jdata.derived().Dinv[0];
            Ia -= jdata.derived().UDinv * jdata.derived().U.transpose();

            if (parent > 0)
            {
                Force & pa = data.f[i];
                pa.toVector() += Ia * data.a[i].toVector() + jdata.UDinv() * jmodel.jointVelocitySelector(data.u);
                data.Yaba[parent] += pinocchio::internal::SE3actOn<Scalar>::run(data.liMi[i], Ia);
                data.f[parent] += data.liMi[i].act(pa);
            }
        }

        template<int axis>
        static int getAxis(pinocchio::JointModelBase<pinocchio::JointModelRevoluteTpl<Scalar, Options, axis> > const & joint)
        {
            return axis;
        }
    };

    template<typename Scalar, int Options, template<typename, int> class JointCollectionTpl,
             typename ConfigVectorType, typename TangentVectorType1,
             typename TangentVectorType2, typename ForceDerived>
    inline const typename pinocchio::DataTpl<Scalar,Options,JointCollectionTpl>::TangentVectorType &
    Engine::aba(pinocchio::ModelTpl<Scalar,Options,JointCollectionTpl> const & model,
                pinocchio::DataTpl<Scalar,Options,JointCollectionTpl>        & data,
                Eigen::MatrixBase<ConfigVectorType>                    const & q,
                Eigen::MatrixBase<TangentVectorType1>                  const & v,
                Eigen::MatrixBase<TangentVectorType2>                  const & tau,
                pinocchio::container::aligned_vector<ForceDerived>     const & fext)

    {
        assert(model.check(data) && "data is not consistent with model.");
        assert(q.size() == model.nq && "The joint configuration vector is not of right size");
        assert(v.size() == model.nv && "The joint velocity vector is not of right size");
        assert(tau.size() == model.nv && "The joint acceleration vector is not of right size");

        typedef typename pinocchio::ModelTpl<Scalar,Options,JointCollectionTpl>::JointIndex JointIndex;

        data.v[0].setZero();
        data.a[0] = -model.gravity;
        data.u = tau;

        typedef pinocchio::AbaForwardStep1<Scalar, Options, JointCollectionTpl,
                                           ConfigVectorType, TangentVectorType1> Pass1;
        for(JointIndex i=1; i<(JointIndex)model.njoints; ++i)
        {
            Pass1::run(model.joints[i],data.joints[i],
                       typename Pass1::ArgsType(model,data,q.derived(),v.derived()));
            data.f[i] -= fext[i];
        }

        typedef AbaBackwardStep<Scalar,Options,JointCollectionTpl> Pass2;
        for(JointIndex i=(JointIndex)model.njoints-1; i>0; --i)
        {
            Pass2::run(model.joints[i],data.joints[i],
                       typename Pass2::ArgsType(model,data));
        }

        typedef pinocchio::AbaForwardStep2<Scalar,Options,JointCollectionTpl> Pass3;
        for(JointIndex i=1; i<(JointIndex)model.njoints; ++i)
        {
            Pass3::run(model.joints[i],data.joints[i],
                       typename Pass3::ArgsType(model,data));
        }

        return data.ddq;
    }
}
