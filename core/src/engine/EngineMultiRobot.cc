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

#include "jiminy/core/io/FileDevice.h"
#include "jiminy/core/telemetry/TelemetryData.h"
#include "jiminy/core/telemetry/TelemetryRecorder.h"
#include "jiminy/core/robot/AbstractMotor.h"
#include "jiminy/core/robot/AbstractSensor.h"
#include "jiminy/core/robot/Robot.h"
#include "jiminy/core/control/AbstractController.h"
#include "jiminy/core/Utilities.h"
#include "jiminy/core/Constants.h"

#include "jiminy/core/engine/EngineMultiRobot.h"

#include <boost/numeric/odeint/iterator/n_step_iterator.hpp>


namespace jiminy
{
    // ====================================================
    // ================ systemDataHolder_t ================
    // ====================================================

    systemDataHolder_t::systemDataHolder_t(std::string const & systemNameIn,
                                           std::shared_ptr<Robot> robotIn,
                                           std::shared_ptr<AbstractController> controllerIn,
                                           callbackFunctor_t callbackFctIn) :
    name(systemNameIn),
    robot(std::move(robotIn)),
    controller(std::move(controllerIn)),
    callbackFct(std::move(callbackFctIn)),
    positionFieldnames(),
    velocityFieldnames(),
    accelerationFieldnames(),
    motorTorqueFieldnames(),
    energyFieldname(),
    robotLock(nullptr),
    state(),
    stateLast(),
    forcesImpulse(),
    forceImpulseNextIt(),
    forcesProfile()
    {
        // Empty on purpose.
    }

    systemDataHolder_t::systemDataHolder_t(void) :
    systemDataHolder_t("", nullptr, nullptr,
    [](float64_t const & t,
       vectorN_t const & x) -> bool_t
    {
        return true;
    })
    {
        // Empty on purpose.
    }

    // ================================================
    // ================ systemState_t ================
    // ================================================

    void systemState_t::initialize(Robot           * robot,
                                   vectorN_t const & xInit)
    {
        // Extract some information from the robot
        nx_ = robot->nx();
        nq_ = robot->nq();
        nv_ = robot->nv();

        // Initialize the ode stepper state buffers
        x = xInit;
        dxdt = vectorN_t::Zero(nx_);
        fExternal = forceVector_t(robot->pncModel_.joints.size(),
                                  pinocchio::Force::Zero());
        uInternal = vectorN_t::Zero(nv_);
        uCommand = vectorN_t::Zero(robot->getMotorsNames().size());
        uMotor = vectorN_t::Zero(robot->getMotorsNames().size());
        u = vectorN_t::Zero(nv_);

        // Set the initialization flag
        isInitialized_ = true;
    }

    // ==================================================
    // ================ EngineMultiRobot ================
    // ==================================================

    EngineMultiRobot::EngineMultiRobot(void):
    engineOptions_(nullptr),
    isTelemetryConfigured_(false),
    isSimulationRunning_(false),
    engineOptionsHolder_(),
    systemsDataHolder_(),
    telemetrySender_(),
    telemetryData_(nullptr),
    telemetryRecorder_(nullptr),
    stepper_(),
    stepperUpdatePeriod_(-1),
    stepperState_(),
    forcesCoupling_()
    {
        // Initialize the configuration options to the default.
        setOptions(getDefaultEngineOptions());

        // Initialize the global telemetry data holder
        telemetryData_ = std::make_shared<TelemetryData>();
        telemetryData_->reset();

        // Initialize the global telemetry recorder
        telemetryRecorder_ = std::make_unique<TelemetryRecorder>();

        // Initialize the engine-specific telemetry sender
        telemetrySender_.configureObject(telemetryData_, ENGINE_OBJECT_NAME);
    }

    EngineMultiRobot::~EngineMultiRobot(void) = default; // Cannot be default in the header since some types are incomplete at this point

    hresult_t EngineMultiRobot::addSystem(std::string const & systemName,
                                          std::shared_ptr<Robot> robot,
                                          std::shared_ptr<AbstractController> controller,
                                          callbackFunctor_t callbackFct)
    {
        if (!robot->getIsInitialized())
        {
            std::cout << "Error - EngineMultiRobot::initialize - Robot not initialized." << std::endl;
            return hresult_t::ERROR_INIT_FAILED;
        }

        if (!controller->getIsInitialized())
        {
            std::cout << "Error - EngineMultiRobot::initialize - Controller not initialized." << std::endl;
            return hresult_t::ERROR_INIT_FAILED;
        }

        // TODO: Check that the callback function is working as expected

        systemsDataHolder_.emplace_back(systemName,
                                        std::move(robot),
                                        std::move(controller),
                                        std::move(callbackFct));

        return hresult_t::SUCCESS;
    }

    hresult_t EngineMultiRobot::removeSystem(std::string const & systemName)
    {
        auto systemIt = std::find_if(systemsDataHolder_.begin(), systemsDataHolder_.end(),
                                     [&systemName](auto const & sys)
                                     {
                                         return (sys.name == systemName);
                                     });
        if (systemIt == systemsDataHolder_.end())
        {
            std::cout << "Error - EngineMultiRobot::removeSystem - No system with this name has been added to the engine." << std::endl;
            return hresult_t::ERROR_BAD_INPUT;
        }

        // Get the current system index
        int32_t const systemIdx = std::distance(systemsDataHolder_.begin(), systemIt);

        // Remove the system from the list
        systemsDataHolder_.erase(systemIt);

        // Remove every coupling forces involving the system
        removeCouplingForces(systemName);

        // Update the systems' indices for the remaining coupling forces
        for (auto & force : forcesCoupling_)
        {
            if (force.systemIdx1 > systemIdx)
            {
                force.systemIdx1--;
            }
            if (force.systemIdx2 > systemIdx)
            {
                force.systemIdx2--;
            }
        }

        return hresult_t::SUCCESS;
    }

    hresult_t EngineMultiRobot::addCouplingForce(std::string    const & systemName1,
                                                 std::string    const & systemName2,
                                                 std::string    const & frameName1,
                                                 std::string    const & frameName2,
                                                 forceFunctor_t         forceFct)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        auto systemIt1 = std::find_if(systemsDataHolder_.begin(), systemsDataHolder_.end(),
                                      [&systemName1](auto const & sys)
                                      {
                                          return (sys.name == systemName1);
                                      });
        auto systemIt2 = std::find_if(systemsDataHolder_.begin(), systemsDataHolder_.end(),
                                      [&systemName2](auto const & sys)
                                      {
                                          return (sys.name == systemName2);
                                      });
        if (systemIt1 == systemsDataHolder_.end()
         || systemIt2 == systemsDataHolder_.end())
        {
            std::cout << "Error - EngineMultiRobot::removeSystem - At least one of the names does not correspond to any system added to the engine." << std::endl;
            returnCode = hresult_t::ERROR_BAD_INPUT;
        }

        int32_t systemIdx1;
        int32_t frameIdx1;
        int32_t systemIdx2;
        int32_t frameIdx2;
        if (returnCode == hresult_t::ERROR_BAD_INPUT)
        {
            systemIdx1 = std::distance(systemsDataHolder_.begin(), systemIt1);
            returnCode = getFrameIdx(systemIt1->robot->pncModel_, frameName1, frameIdx1);

        }
        if (returnCode == hresult_t::ERROR_BAD_INPUT)
        {
            systemIdx2 = std::distance(systemsDataHolder_.begin(), systemIt2);
            returnCode = getFrameIdx(systemIt1->robot->pncModel_, frameName2, frameIdx2);

        }

        forcesCoupling_.emplace_back(systemName1,
                                     std::move(systemIdx1),
                                     systemName2,
                                     std::move(systemIdx2),
                                     frameName1,
                                     std::move(frameIdx1),
                                     frameName2,
                                     std::move(frameIdx2),
                                     std::move(forceFct));

        return returnCode;
    }

    hresult_t EngineMultiRobot::removeCouplingForces(std::string const & systemName1,
                                                     std::string const & systemName2)
    {
        auto systemIt1 = std::find_if(systemsDataHolder_.begin(), systemsDataHolder_.end(),
                                      [&systemName1](auto const & sys)
                                      {
                                          return (sys.name == systemName1);
                                      });
        auto systemIt2 = std::find_if(systemsDataHolder_.begin(), systemsDataHolder_.end(),
                                      [&systemName2](auto const & sys)
                                      {
                                          return (sys.name == systemName2);
                                      });
        if (systemIt1 == systemsDataHolder_.end()
         || systemIt2 == systemsDataHolder_.end())
        {
            std::cout << "Error - EngineMultiRobot::removeCouplingForces - At least one of the names does not correspond to any system added to the engine." << std::endl;
            return hresult_t::ERROR_BAD_INPUT;
        }

        forcesCoupling_.erase(
            std::remove_if(forcesCoupling_.begin(), forcesCoupling_.end(),
            [&systemName1, &systemName2](auto const & force)
            {
                return (force.systemName1 == systemName1 &&
                        force.systemName2 == systemName2);
            }),
            forcesCoupling_.end()
        );

        return hresult_t::SUCCESS;
    }

    hresult_t EngineMultiRobot::removeCouplingForces(std::string const & systemName)
    {
        auto systemIt = std::find_if(systemsDataHolder_.begin(), systemsDataHolder_.end(),
                                     [&systemName](auto const & sys)
                                     {
                                         return (sys.name == systemName);
                                     });
        if (systemIt == systemsDataHolder_.end())
        {
            std::cout << "Error - EngineMultiRobot::removeCouplingForces - No system with this name has been added to the engine." << std::endl;
            return hresult_t::ERROR_BAD_INPUT;
        }

        forcesCoupling_.erase(
            std::remove_if(forcesCoupling_.begin(), forcesCoupling_.end(),
            [&systemName](auto const & force)
            {
                return (force.systemName1 == systemName ||
                        force.systemName2 == systemName);
            }),
            forcesCoupling_.end()
        );

        return hresult_t::SUCCESS;
    }

    hresult_t EngineMultiRobot::configureTelemetry(void)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (systemsDataHolder_.empty())
        {
            std::cout << "Error - EngineMultiRobot::configureTelemetry - No system added to the engine." << std::endl;
            returnCode = hresult_t::ERROR_INIT_FAILED;
        }

        if (!isTelemetryConfigured_)
        {
            for (auto & system : systemsDataHolder_)
            {
                // Generate the log fieldnames
                system.positionFieldnames =
                    addCircumfix(system.robot->getPositionFieldnames(), system.name);
                system.velocityFieldnames =
                    addCircumfix(system.robot->getVelocityFieldnames(), system.name);
                system.accelerationFieldnames =
                    addCircumfix(system.robot->getAccelerationFieldnames(), system.name);
                system.motorTorqueFieldnames =
                    addCircumfix(system.robot->getMotorTorqueFieldnames(), system.name);
                system.energyFieldname = addCircumfix("energy", system.name);

                // Register variables to the telemetry senders
                if (returnCode == hresult_t::SUCCESS)
                {
                    if (engineOptions_->telemetry.enableConfiguration)
                    {
                        returnCode = telemetrySender_.registerVariable(
                            system.positionFieldnames,
                            system.state.q());
                    }
                }
                if (returnCode == hresult_t::SUCCESS)
                {
                    if (engineOptions_->telemetry.enableVelocity)
                    {
                        returnCode = telemetrySender_.registerVariable(
                            system.velocityFieldnames,
                            system.state.v());
                    }
                }
                if (returnCode == hresult_t::SUCCESS)
                {
                    if (engineOptions_->telemetry.enableAcceleration)
                    {
                        returnCode = telemetrySender_.registerVariable(
                            system.accelerationFieldnames,
                            system.state.a());
                    }
                }
                if (returnCode == hresult_t::SUCCESS)
                {
                    if (engineOptions_->telemetry.enableTorque)
                    {
                        returnCode = telemetrySender_.registerVariable(
                            system.motorTorqueFieldnames,
                            system.state.uMotor);
                    }
                }
                if (returnCode == hresult_t::SUCCESS)
                {
                    if (engineOptions_->telemetry.enableEnergy)
                    {
                        returnCode = telemetrySender_.registerVariable(
                            system.energyFieldname, 0.0);
                    }
                }

                if (returnCode == hresult_t::SUCCESS)
                {
                    returnCode = system.controller->configureTelemetry(
                        telemetryData_, system.name);
                }
                if (returnCode == hresult_t::SUCCESS)
                {
                    returnCode = system.robot->configureTelemetry(
                        telemetryData_, system.name);
                }
            }
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            isTelemetryConfigured_ = true;
        }

        return returnCode;
    }

    void EngineMultiRobot::updateTelemetry(void)
    {
        for (auto & system : systemsDataHolder_)
        {
            // Compute the total energy of the system
            Eigen::Ref<vectorN_t const> q = system.state.q();
            Eigen::Ref<vectorN_t const> v = system.state.v();
            float64_t energy = EngineMultiRobot::kineticEnergy(
                system.robot->pncModel_, system.robot->pncData_, q, v, true);
            energy += pinocchio::potentialEnergy(
                system.robot->pncModel_, system.robot->pncData_, q, false);

            // Update the telemetry internal state
            if (engineOptions_->telemetry.enableConfiguration)
            {
                telemetrySender_.updateValue(system.positionFieldnames,
                                             system.state.q());
            }
            if (engineOptions_->telemetry.enableVelocity)
            {
                telemetrySender_.updateValue(system.velocityFieldnames,
                                             system.state.v());
            }
            if (engineOptions_->telemetry.enableAcceleration)
            {
                telemetrySender_.updateValue(system.accelerationFieldnames,
                                             system.state.a());
            }
            if (engineOptions_->telemetry.enableTorque)
            {
                telemetrySender_.updateValue(system.motorTorqueFieldnames,
                                             system.state.uMotor);
            }
            if (engineOptions_->telemetry.enableEnergy)
            {
                telemetrySender_.updateValue(system.energyFieldname, energy);
            }

            system.controller->updateTelemetry();
            system.robot->updateTelemetry();
        }

        // Flush the telemetry internal state
        telemetryRecorder_->flushDataSnapshot(stepperState_.t);
    }

    void EngineMultiRobot::reset(bool_t const & resetRandomNumbers,
                                 bool_t const & resetDynamicForceRegister)
    {
        // Reset the dynamic force register if requested
        if (resetDynamicForceRegister)
        {
            for (auto & system : systemsDataHolder_)
            {
                system.forcesImpulse.clear();
                system.forceImpulseNextIt = system.forcesImpulse.begin();
                system.forcesProfile.clear();
            }
        }

        // Reset the random number generators
        if (resetRandomNumbers)
        {
            resetRandGenerators(engineOptions_->stepper.randomSeed);
        }

        // Reset the internal state of the robot and controller
        for (auto & system : systemsDataHolder_)
        {
            system.robot->reset();
            system.controller->reset();
        }

        // Make sure the simulation is properly stopped
        stop();
    }

    void EngineMultiRobot::reset(bool_t const & resetDynamicForceRegister)
    {
        reset(true, resetDynamicForceRegister);
    }

    hresult_t EngineMultiRobot::start(std::vector<vectorN_t> const & xInit,
                                      bool_t const & resetRandomNumbers,
                                      bool_t const & resetDynamicForceRegister)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (systemsDataHolder_.empty())
        {
            std::cout << "Error - EngineMultiRobot::start - No system to simulate. Please add one before starting a simulation." << std::endl;
            returnCode = hresult_t::ERROR_INIT_FAILED;
        }

        if (xInit.size() != systemsDataHolder_.size())
        {
            std::cout << "Error - EngineMultiRobot::start - The number of initial state must match the number of systems." << std::endl;
            returnCode = hresult_t::ERROR_BAD_INPUT;
        }

        // Check the dimension of the state
        auto xInitIt = xInit.begin();
        auto systemIt = systemsDataHolder_.begin();
        for ( ; systemIt != systemsDataHolder_.end() ; xInitIt++, systemIt++)
        {

            if (xInitIt->rows() != systemIt->robot->nx())
            {
                std::cout << "Error - EngineMultiRobot::start - Size of xInit inconsistent with model size." << std::endl;
                returnCode = hresult_t::ERROR_BAD_INPUT;
            }
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            // Reset the robot, controller, engine, and registered impulse forces if requested
            reset(resetRandomNumbers, resetDynamicForceRegister);
        }

        for (auto & system : systemsDataHolder_)
        {
            if (returnCode == hresult_t::SUCCESS)
            {
                // Propagate the user-defined gravity at Pinocchio model level
                system.robot->pncModel_.gravity = engineOptions_->world.gravity;

                // Propage the user-defined motor inertia at Pinocchio model level
                system.robot->pncModel_.rotorInertia = system.robot->getMotorInertia();

                // Lock the robot. At this point it is no longer possible to change the robot anymore.
                returnCode = system.robot->getLock(system.robotLock);
            }
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            // Consider that the simulation is running
            isSimulationRunning_ = true;

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
            if (stepperUpdatePeriod_ > SIMULATION_MIN_TIMESTEP)
            {
                // The initial time step is the global update period (breakpoint frequency)
                dt = stepperUpdatePeriod_;
            }
            else
            {
                // Use the maximum allowed time step as default
                dt = engineOptions_->stepper.dtMax;
            }

            // Initialize the stepper state
            stepperState_.reset(dt, cat(xInit));

            // Initialize the stepper internal state
            xInitIt = xInit.begin();
            systemIt = systemsDataHolder_.begin();
            for ( ; systemIt != systemsDataHolder_.end() ; xInitIt++, systemIt++)
            {
                systemIt->state.initialize(systemIt->robot.get(), *xInitIt);
            }

            // Update the frame indices associated with the coupling forces
            for (auto & force : forcesCoupling_)
            {
                getFrameIdx(systemsDataHolder_[force.systemIdx1].robot->pncModel_,
                            force.frameName1,
                            force.frameIdx1);
                getFrameIdx(systemsDataHolder_[force.systemIdx2].robot->pncModel_,
                            force.frameName2,
                            force.frameIdx2);
            }

            for (auto & system : systemsDataHolder_)
            {
                // Reset the impulse for iterator counter
                system.forceImpulseNextIt = system.forcesImpulse.begin();

                // Update the frame indices associated with the force profiles
                for (auto & force : system.forcesProfile)
                {
                    getFrameIdx(system.robot->pncModel_,
                                force.frameName,
                                force.frameIdx);
                }

                // Get some system state proxies
                float64_t & t = stepperState_.t;
                Eigen::Ref<vectorN_t> q = system.state.q();
                Eigen::Ref<vectorN_t> v = system.state.v();
                vectorN_t & x = system.state.x;
                Eigen::Ref<vectorN_t> qDot = system.state.qDot();
                Eigen::Ref<vectorN_t> a = system.state.a();
                vectorN_t & u = system.state.u;
                vectorN_t & uCommand = system.state.uCommand;
                vectorN_t & uMotor = system.state.uMotor;
                vectorN_t & uInternal = system.state.uInternal;
                forceVector_t & fext = system.state.fExternal;

                // Compute the forward kinematics
                computeForwardKinematics(system, q, v, a);

                // Initialize the external contact forces
                computeExternalForces(system, t, x, fext);

                // Initialize the sensor data
                system.robot->setSensorsData(t, q, v, a, uMotor);

                // Compute the actual motor torque
                computeCommand(system, t, q, v, uCommand);

                // Compute the actual motor torque
                system.robot->computeMotorsTorques(t, q, v, a, uCommand);
                uMotor = system.robot->getMotorsTorques();

                // Compute the internal dynamics
                computeInternalDynamics(system, t, q, v, uInternal);

                // Compute the total torque vector
                u = uInternal;
                for (auto const & motor : system.robot->getMotors())
                {
                    int32_t const & motorId = motor->getIdx();
                    int32_t const & motorVelocityIdx = motor->getJointVelocityIdx();
                    u[motorVelocityIdx] += uMotor[motorId];
                }

                // Compute dynamics
                a = EngineMultiRobot::aba(system.robot->pncModel_,
                                          system.robot->pncData_,
                                          q, v, u, fext);

                // Project the derivative in state space
                computePositionDerivative(system.robot->pncModel_, q, v, qDot, dt);

                // Update the sensor data with the updated torque and acceleration
                system.robot->setSensorsData(t, q, v, a, uMotor);
            }
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            // Lock the telemetry. At this point it is no longer possible to register new variables.
            configureTelemetry();

            // Write the header: this locks the registration of new variables
            telemetryRecorder_->initialize(telemetryData_.get());

            // Log current buffer content as first point of the log data.
            updateTelemetry();

            // Initialize the last system states
            for (auto & system : systemsDataHolder_)
            {
                system.stateLast = system.state;
            }
        }

        if (returnCode != hresult_t::SUCCESS)
        {
            stop();
        }

        return returnCode;
    }

    hresult_t EngineMultiRobot::simulate(float64_t              const & tEnd,
                                         std::vector<vectorN_t> const & xInit)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (systemsDataHolder_.empty())
        {
            std::cout << "Error - EngineMultiRobot::simulate - No system to simulate. Please add one before starting a simulation." << std::endl;
            returnCode = hresult_t::ERROR_INIT_FAILED;
        }

        if (tEnd < 5e-3)
        {
            std::cout << "Error - EngineMultiRobot::simulate - The duration of the simulation cannot be shorter than 5ms." << std::endl;
            returnCode = hresult_t::ERROR_BAD_INPUT;
        }

        // Reset the robot, controller, and engine
        if (returnCode == hresult_t::SUCCESS)
        {
            returnCode = start(xInit, true, false);
        }

        // Integration loop based on boost::numeric::odeint::detail::integrate_times
        while (returnCode == hresult_t::SUCCESS)
        {
            // Stop the simulation if the end time has been reached
            if (tEnd - stepperState_.t < SIMULATION_MIN_TIMESTEP)
            {
                if (engineOptions_->stepper.verbose)
                {
                    std::cout << "Simulation done: desired final time reached." << std::endl;
                }
                break;
            }

            // Stop the simulation if any of the callbacks return false
            for (auto & system : systemsDataHolder_)
            {
                if (!system.callbackFct(stepperState_.t, system.state.x))
                {
                    if (engineOptions_->stepper.verbose)
                    {
                        std::cout << "Simulation done: callback returned false." << std::endl;
                    }
                    break;
                }
            }

            // Stop the simulation if the max number of integration steps is reached
            if (engineOptions_->stepper.iterMax > 0
            && stepperState_.iter >= (uint32_t) engineOptions_->stepper.iterMax)
            {
                if (engineOptions_->stepper.verbose)
                {
                    std::cout << "Simulation done: maximum number of integration steps exceeded." << std::endl;
                }
                break;
            }

            // Perform a single integration step up to tEnd, stopping at stepperUpdatePeriod_ to log.
            float64_t stepSize;
            if (stepperUpdatePeriod_ > 0)
            {
                stepSize = min(stepperUpdatePeriod_ , tEnd - stepperState_.t);
            }
            else
            {
                stepSize = min(engineOptions_->stepper.dtMax, tEnd - stepperState_.t);
            }
            returnCode = step(stepSize); // Automatic dt adjustment
        }

        /* Stop the simulation. New variables can be registered again,
           and the lock on the robot is released. */
        stop();

        return returnCode;
    }

    hresult_t EngineMultiRobot::step(float64_t stepSize)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        // Check if the simulation has started
        if (!isSimulationRunning_)
        {
            std::cout << "Error - EngineMultiRobot::step - No simulation running. Please start it before using step method." << std::endl;
            returnCode = hresult_t::ERROR_GENERIC;
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            // Check if there is something wrong with the integration
            for (auto & system : systemsDataHolder_)
            {
                if ((system.state.x.array() != system.state.x.array()).any()) // isnan if NOT equal to itself
                {
                    std::cout << "Error - EngineMultiRobot::step - The low-level ode solver failed."\
                                 "Consider increasing the stepper accuracy." << std::endl;
                    return hresult_t::ERROR_GENERIC;
                }
            }

            // Check if the desired step size is suitable
            if (stepSize > EPS && stepSize < SIMULATION_MIN_TIMESTEP)
            {
                std::cout << "Error - EngineMultiRobot::step - The requested step size is out of bounds." << std::endl;
                return hresult_t::ERROR_BAD_INPUT;
            }

            /* Set end time: The default step size is equal to the controller update period if
            discrete-time, otherwise it uses the sensor update period if discrete-time,
            otherwise it uses the user-defined parameter dtMax. */
            float64_t tEnd;
            if (stepSize < EPS)
            {
                float64_t const & controllerUpdatePeriod = engineOptions_->stepper.controllerUpdatePeriod;
                if (controllerUpdatePeriod > EPS)
                {
                    stepSize = controllerUpdatePeriod;
                }
                else
                {
                    float64_t const & sensorsUpdatePeriod = engineOptions_->stepper.sensorsUpdatePeriod;
                    if (sensorsUpdatePeriod > EPS)
                    {
                        stepSize = sensorsUpdatePeriod;
                    }
                    else
                    {
                        stepSize = engineOptions_->stepper.dtMax;
                    }
                }
            }

            // Define the stepper iterators.
            auto systemOde =
                [this](vectorN_t const & xIn,
                       vectorN_t       & dxdtIn,
                       float64_t const & tIn)
                {
                    this->computeSystemDynamics(tIn, xIn, dxdtIn);
                };

            /* Avoid compounding of error using Kahan algorithm. It
               consists in keeping track of the cumulative rounding error
               to add it back to the sum when it gets larger than the
               numerical precision, thus avoiding it to grows unbounded. */
            float64_t stepSize_true = stepSize - stepperState_.tError;
            tEnd = stepperState_.t + stepSize_true;
            stepperState_.tError = (tEnd - stepperState_.t) - stepSize_true;

            // Get references to some internal stepper buffers
            float64_t & t = stepperState_.t;
            float64_t & dt = stepperState_.dt;
            vectorN_t & x = stepperState_.x;
            vectorN_t & dxdt = stepperState_.dxdt;

            // Define a failure checker for the stepper
            failed_step_checker fail_checker;

            /* Perform the integration.
               Do not simulate a timestep smaller than STEPPER_MIN_TIMESTEP. */
            while (tEnd - t > STEPPER_MIN_TIMESTEP)
            {
                float64_t tNext = t;

                if (stepperUpdatePeriod_ > EPS)
                {
                    // Update the sensor data if necessary (only for finite update frequency)
                    if (engineOptions_->stepper.sensorsUpdatePeriod > EPS)
                    {
                        float64_t const & sensorsUpdatePeriod = engineOptions_->stepper.sensorsUpdatePeriod;
                        float64_t dtNextSensorsUpdatePeriod = sensorsUpdatePeriod - std::fmod(t, sensorsUpdatePeriod);
                        if (dtNextSensorsUpdatePeriod < SIMULATION_MIN_TIMESTEP
                        || sensorsUpdatePeriod - dtNextSensorsUpdatePeriod < SIMULATION_MIN_TIMESTEP)
                        {
                            for (auto & system : systemsDataHolder_)
                            {
                                Eigen::Ref<vectorN_t const> q = system.state.q();
                                Eigen::Ref<vectorN_t const> v = system.state.v();
                                Eigen::Ref<vectorN_t const> a = system.state.a();
                                vectorN_t const & uMotor = system.state.uMotor;
                                system.robot->setSensorsData(t, q, v, a, uMotor);
                            }
                        }
                    }

                    // Update the controller command if necessary (only for finite update frequency)
                    if (engineOptions_->stepper.controllerUpdatePeriod > EPS)
                    {
                        float64_t const & controllerUpdatePeriod = engineOptions_->stepper.controllerUpdatePeriod;
                        float64_t dtNextControllerUpdatePeriod = controllerUpdatePeriod - std::fmod(t, controllerUpdatePeriod);
                        if (dtNextControllerUpdatePeriod < SIMULATION_MIN_TIMESTEP
                        || controllerUpdatePeriod - dtNextControllerUpdatePeriod < SIMULATION_MIN_TIMESTEP)
                        {
                            for (auto & system : systemsDataHolder_)
                            {
                                Eigen::Ref<vectorN_t const> q = system.state.q();
                                Eigen::Ref<vectorN_t const> v = system.state.v();
                                vectorN_t & uCommand = system.state.uCommand;
                                computeCommand(system, t, q, v, uCommand);
                            }

                            /* Update the internal stepper state dxdt since the dynamics has changed.
                                -> Make sure the next impulse force iterator has NOT been updated yet !
                                Note: This point is still subject to debate: it's more a choice than
                                mathematical condition. Anyway, numerically, the results are similar. */
                            if (engineOptions_->stepper.odeSolver != "explicit_euler")
                            {
                                computeSystemDynamics(t, x, dxdt);
                            }
                        }
                    }
                }

                // Get the next impulse force application time and update the iterators if necessary
                float64_t tForceImpulseNext = tEnd;
                for (auto & system : systemsDataHolder_)
                {
                    if (system.forceImpulseNextIt != system.forcesImpulse.end())
                    {
                        auto & forceImpulseNextIt = system.forceImpulseNextIt;

                        float64_t tForceImpulse = forceImpulseNextIt->t;
                        float64_t dtForceImpulse = forceImpulseNextIt->dt;
                        if (t > tForceImpulse + dtForceImpulse)
                        {
                            // The current force is over. Switch to the next one.
                            forceImpulseNextIt++;
                        }

                        if (forceImpulseNextIt != system.forcesImpulse.end())
                        {
                            tForceImpulse = forceImpulseNextIt->t;
                            if (tForceImpulse > t)
                            {
                                /* The application time of the current force is
                                   ahead of time. So waiting for it to begin... */
                                tForceImpulseNext = min(tForceImpulseNext, tForceImpulse);
                            }
                            else
                            {
                                /* The application time of the current force is past BUT
                                   the application duration may not be over. In such a
                                   case, one must NOT increment the force iterator, yet
                                   the next application time does not correspond to the
                                   current force but the next one. */
                                if (forceImpulseNextIt != std::prev(system.forcesImpulse.end()))
                                {
                                    tForceImpulse = std::next(forceImpulseNextIt)->t;
                                    tForceImpulseNext = min(tForceImpulseNext, tForceImpulse);
                                }
                            }
                        }
                    }
                }

                /* Increase back the timestep dt if it has been decreased
                   to a ridiculously small value because of a breakpoint. */
                dt = std::max(dt, SIMULATION_DEFAULT_TIMESTEP);

                if (stepperUpdatePeriod_ > EPS)
                {
                    // Get the time of the next breakpoint for the ODE solver:
                    // a breakpoint occurs if we reached tEnd, if an external force is applied, or if we
                    // need to update the sensors / controller.
                    float64_t dtNextGlobal; // dt to apply for the next stepper step because of the various breakpoints
                    float64_t dtNextUpdatePeriod = stepperUpdatePeriod_ - std::fmod(t, stepperUpdatePeriod_);
                    if (dtNextUpdatePeriod < SIMULATION_MIN_TIMESTEP)
                    {
                        // Step to reach next sensors / controller update is too short: skip one
                        // controller update and jump to the next one.
                        // Note that in this case, the sensors have already been updated in
                        // anticipation in previous loop.
                        dtNextGlobal = min(dtNextUpdatePeriod + stepperUpdatePeriod_,
                                           tForceImpulseNext - t);
                    }
                    else
                    {
                        dtNextGlobal = min(dtNextUpdatePeriod, tForceImpulseNext - t);
                    }

                    /* Check if the next dt to about equal to the time difference
                       between the current time (it can only be smaller) and
                       enforce next dt to exactly match this value in such a case. */
                    if (tEnd - t - STEPPER_MIN_TIMESTEP < dtNextGlobal)
                    {
                        dtNextGlobal = tEnd - t;
                    }
                    tNext += dtNextGlobal;

                    // Compute the next step using adaptive step method
                    while (tNext - t > EPS)
                    {
                        /* Adjust stepsize to end up exactly at the next breakpoint,
                           prevent steps larger than dtMax, and make sure that dt is
                           multiple of TELEMETRY_TIME_DISCRETIZATION_FACTOR whenever
                           it is possible, to reduce rounding errors of logged data. */
                        dt = min(dt, tNext - t, engineOptions_->stepper.dtMax);
                        if (tNext - (t + dt) < STEPPER_MIN_TIMESTEP)
                        {
                            dt = tNext - t;
                        }
                        if (dt > SIMULATION_MIN_TIMESTEP)
                        {
                            float64_t const dtResidual = std::fmod(dt, SIMULATION_MIN_TIMESTEP);
                            if (dtResidual > STEPPER_MIN_TIMESTEP
                             && dtResidual < SIMULATION_MIN_TIMESTEP - STEPPER_MIN_TIMESTEP
                             && dt - dtResidual > STEPPER_MIN_TIMESTEP)
                            {
                                dt -= dtResidual;
                            }
                        }

                        if (success == boost::apply_visitor(
                            [&](auto && one)
                            {
                                return one.try_step(systemOde, x, dxdt, t, dt);
                            }, stepper_))
                        {
                            // reset the fail counter
                            fail_checker.reset();

                            // Increment the iteration counter only for successful steps
                            stepperState_.iter++;

                            // Log every stepper state only if the user asked for
                            if (engineOptions_->stepper.logInternalStepperSteps)
                            {
                                updateTelemetry();
                            }

                            /* Backup the stepper and systems' state on success only:
                               - t at last successful iteration is used to compute dt,
                                 which is project the accelation in the state space
                                 instead of SO3^2.
                               - the acceleration and torque at the last successful
                                 iteration is used to update the sensors' data in
                                 case of continuous sensing. */
                            stepperState_.tLast = t;
                            for (auto & system : systemsDataHolder_)
                            {
                                system.stateLast = system.state;
                            }
                        }
                        else
                        {
                            // check for possible overflow of failed steps in step size adjustment
                            fail_checker();
                        }
                    }
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
                                return one.try_step(systemOde, x, dxdt, t, dt);
                            }, stepper_);
                        if (res == success)
                        {
                            // reset the fail counter
                            fail_checker.reset();

                            // Increment the iteration counter
                            stepperState_.iter++;

                            // Log every stepper state only if the user asked for
                            if (engineOptions_->stepper.logInternalStepperSteps)
                            {
                                updateTelemetry();
                            }

                            // Backup the stepper and systems' state
                            stepperState_.tLast = t;
                            for (auto & system : systemsDataHolder_)
                            {
                                system.stateLast = system.state;
                            }
                        }
                        else
                        {
                            // check for possible overflow of failed steps in step size adjustment
                            fail_checker();
                        }
                    }
                }
            }

            /* Update the final time to make sure it corresponds
            to the desired tEnd and avoid compounding of error.
            Anyway the user asked for a step of exactly stepSize,
            so he is expecting this value to be reached. */
            stepperState_.t = tEnd;

            /* Monitor current iteration number, and log the current time,
            state, command, and sensors data. */
            if (!engineOptions_->stepper.logInternalStepperSteps)
            {
                updateTelemetry();
            }
        }

        return returnCode;
    }

    void EngineMultiRobot::stop(void)
    {
        // Make sure that a simulation running
        if (isSimulationRunning_)
        {
            // Release the lock on the robots
            for (auto & system : systemsDataHolder_)
            {
                system.robotLock.reset(nullptr);
            }

            /* Reset the telemetry. Note that calling `reset` does NOT clear the
            internal data buffer of telemetryRecorder_. Clearing is done at init
            time, so that it remains accessible until the next initialization. */
            telemetryRecorder_->reset();
            telemetryData_->reset();
            isTelemetryConfigured_ = false;

            isSimulationRunning_ = false;
        }
    }

    hresult_t EngineMultiRobot::registerForceImpulse(std::string const & systemName,
                                                     std::string const & frameName,
                                                     float64_t   const & t,
                                                     float64_t   const & dt,
                                                     vector3_t   const & F)
    {
        // Make sure that the forces do NOT overlap while taking into account dt.

        hresult_t returnCode = hresult_t::SUCCESS;

        if (isSimulationRunning_)
        {
            std::cout << "Error - EngineMultiRobot::registerForceImpulse - A simulation is running. "\
                         "Please stop it before registering new forces." << std::endl;
            returnCode = hresult_t::ERROR_GENERIC;
        }

        systemDataHolder_t * system;
        returnCode = getSystem(systemName, system);
        if (returnCode == hresult_t::SUCCESS)
        {
            system->forcesImpulse.emplace(frameName, t, dt, F);
        }

        return hresult_t::SUCCESS;
    }

    hresult_t EngineMultiRobot::registerForceProfile(std::string    const & systemName,
                                                     std::string    const & frameName,
                                                     forceFunctor_t         forceFct)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (isSimulationRunning_)
        {
            std::cout << "Error - EngineMultiRobot::registerForceProfile - A simulation is running. "\
                         "Please stop it before registering new forces." << std::endl;
            returnCode = hresult_t::ERROR_GENERIC;
        }

        systemDataHolder_t * system;
        returnCode = getSystem(systemName, system);

        int32_t frameIdx;
        if (returnCode == hresult_t::SUCCESS)
        {
            returnCode =  getFrameIdx(system->robot->pncModel_, frameName, frameIdx);
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            system->forcesProfile.emplace_back(
                frameName, std::move(frameIdx), std::move(forceFct));
        }

        return returnCode;
    }

    configHolder_t EngineMultiRobot::getOptions(void) const
    {
        return engineOptionsHolder_;
    }

    hresult_t EngineMultiRobot::setOptions(configHolder_t const & engineOptions)
    {
        if (isSimulationRunning_)
        {
            std::cout << "Error - EngineMultiRobot::setOptions - A simulation is running. "\
                         "Please stop it before updating the options." << std::endl;
            return hresult_t::ERROR_GENERIC;
        }

        // Make sure the dtMax is not out of bounds
        configHolder_t stepperOptions = boost::get<configHolder_t>(engineOptions.at("stepper"));
        float64_t const & dtMax = boost::get<float64_t>(stepperOptions.at("dtMax"));
        if (SIMULATION_MAX_TIMESTEP < dtMax || dtMax < SIMULATION_MIN_TIMESTEP)
        {
            std::cout << "Error - EngineMultiRobot::setOptions - 'dtMax' option is out of bounds." << std::endl;
            return hresult_t::ERROR_BAD_INPUT;
        }

        // Make sure the selected ode solver is available and instantiate it
        std::string const & odeSolver = boost::get<std::string>(stepperOptions.at("odeSolver"));
        if (odeSolver != "runge_kutta_dopri5" && odeSolver != "explicit_euler")
        {
            std::cout << "Error - EngineMultiRobot::setOptions - The requested 'odeSolver' is not available." << std::endl;
            return hresult_t::ERROR_BAD_INPUT;
        }

        // Make sure the controller and sensor update periods are multiple of each other
        float64_t const & sensorsUpdatePeriod =
            boost::get<float64_t>(stepperOptions.at("sensorsUpdatePeriod"));
        float64_t const & controllerUpdatePeriod =
            boost::get<float64_t>(stepperOptions.at("controllerUpdatePeriod"));
        if ((EPS < sensorsUpdatePeriod && sensorsUpdatePeriod < SIMULATION_MIN_TIMESTEP)
        || (EPS < controllerUpdatePeriod && controllerUpdatePeriod < SIMULATION_MIN_TIMESTEP))
        {
            std::cout << "Error - EngineMultiRobot::setOptions - Cannot simulate a discrete system with period smaller than";
            std::cout << SIMULATION_MIN_TIMESTEP << "s. Increase period or switch to continuous mode by setting period to zero." << std::endl;
            return hresult_t::ERROR_BAD_INPUT;
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
            std::cout << "Error - EngineMultiRobot::setOptions - In discrete mode, the controller and sensor update periods "\
                         "must be multiple of each other." << std::endl;
            return hresult_t::ERROR_BAD_INPUT;
        }

        // Make sure the contacts options are fine
        configHolder_t contactsOptions = boost::get<configHolder_t>(engineOptions.at("contacts"));
        float64_t const & dryFrictionVelEps =
            boost::get<float64_t>(contactsOptions.at("dryFrictionVelEps"));
        float64_t const & transitionEps =
            boost::get<float64_t>(contactsOptions.at("transitionEps"));
        if (dryFrictionVelEps < 0.0)
        {
            std::cout << "Error - EngineMultiRobot::setOptions - The contacts option 'dryFrictionVelEps' must be positive." << std::endl;
            return hresult_t::ERROR_BAD_INPUT;
        }
        else if (transitionEps < 0.0)
        {
            std::cout << "Error - EngineMultiRobot::setOptions - The contacts option 'transitionEps' must be positive." << std::endl;
            return hresult_t::ERROR_BAD_INPUT;
        }

        // Make sure the joints options are fine
        configHolder_t jointsOptions = boost::get<configHolder_t>(engineOptions.at("joints"));
        float64_t const & boundTransitionEps =
            boost::get<float64_t>(jointsOptions.at("boundTransitionEps"));
        if (boundTransitionEps < 0.0)
        {
            std::cout << "Error - EngineMultiRobot::setOptions - The joints option 'boundTransitionEps' must be positive." << std::endl;
            return hresult_t::ERROR_BAD_INPUT;
        }

        // Compute the breakpoints' period (for command or observation) during the integration loop
        if (sensorsUpdatePeriod < SIMULATION_MIN_TIMESTEP)
        {
            stepperUpdatePeriod_ = controllerUpdatePeriod;
        }
        else if (controllerUpdatePeriod < SIMULATION_MIN_TIMESTEP)
        {
            stepperUpdatePeriod_ = sensorsUpdatePeriod;
        }
        else
        {
            stepperUpdatePeriod_ = std::min(sensorsUpdatePeriod, controllerUpdatePeriod);
        }

        // Make sure the user-defined gravity force has the right dimension
        configHolder_t worldOptions = boost::get<configHolder_t>(engineOptions.at("world"));
        vectorN_t gravity = boost::get<vectorN_t>(worldOptions.at("gravity"));
        if (gravity.size() != 6)
        {
            std::cout << "Error - EngineMultiRobot::setOptions - The size of the gravity force vector must be 6." << std::endl;
            return hresult_t::ERROR_BAD_INPUT;
        }

        // Update the internal options
        engineOptionsHolder_ = engineOptions;

        // Create a fast struct accessor
        engineOptions_ = std::make_unique<engineOptions_t const>(engineOptionsHolder_);

        return hresult_t::SUCCESS;
    }

    hresult_t EngineMultiRobot::getSystem(std::string        const   & systemName,
                                          systemDataHolder_t const * & system) const
    {
        auto systemIt = std::find_if(systemsDataHolder_.begin(), systemsDataHolder_.end(),
                                     [&systemName](auto const & sys)
                                     {
                                         return (sys.name == systemName);
                                     });
        if (systemIt == systemsDataHolder_.end())
        {
            std::cout << "Error - EngineMultiRobot::getSystem - No system with this name has been added to the engine." << std::endl;
            return hresult_t::ERROR_BAD_INPUT;
        }

        system = &(*systemIt);

        return hresult_t::SUCCESS;
    }

    hresult_t EngineMultiRobot::getSystem(std::string        const   & systemName,
                                          systemDataHolder_t       * & system)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        systemDataHolder_t const * systemConst;
        returnCode = const_cast<EngineMultiRobot const *>(this)->getSystem(systemName, systemConst);
        if (returnCode == hresult_t::SUCCESS)
        {
            system = const_cast<systemDataHolder_t *>(systemConst);
        }

        return returnCode;
    }

    systemState_t const & EngineMultiRobot::getSystemState(std::string const & systemName) const
    {
        static systemState_t const systemStateEmpty;

        systemDataHolder_t const * system;
        hresult_t returnCode = getSystem(systemName, system);
        if (returnCode == hresult_t::SUCCESS)
        {
            return system->state;
        }

        return systemStateEmpty;
    }

    stepperState_t const & EngineMultiRobot::getStepperState(void) const
    {
        return stepperState_;
    }

    // ========================================================
    // ================ Core physics utilities ================
    // ========================================================

    void EngineMultiRobot::computeForwardKinematics(systemDataHolder_t          & system,
                                                    Eigen::Ref<vectorN_t const>   q,
                                                    Eigen::Ref<vectorN_t const>   v,
                                                    Eigen::Ref<vectorN_t const>   a)
    {
        pinocchio::forwardKinematics(system.robot->pncModel_, system.robot->pncData_, q, v, a);
        pinocchio::updateFramePlacements(system.robot->pncModel_, system.robot->pncData_);
    }

    pinocchio::Force EngineMultiRobot::computeContactDynamics(systemDataHolder_t const & system,
                                                              int32_t            const & frameId) const
    {
        // Returns the external force in the contact frame.
        // It must then be converted into a force onto the parent joint.
        // /* /!\ Note that the contact dynamics depends only on kinematics data. /!\ */

        contactOptions_t const & contactOptions_ = engineOptions_->contacts;

        matrix3_t const & tformFrameRot = system.robot->pncData_.oMf[frameId].rotation();
        vector3_t const & posFrame = system.robot->pncData_.oMf[frameId].translation();

        // Initialize the contact force
        vector3_t fextInWorld;
        std::pair<float64_t, vector3_t> ground = engineOptions_->world.groundProfile(posFrame);
        float64_t const & zGround = std::get<float64_t>(ground);
        vector3_t nGround = std::get<vector3_t>(ground);
        nGround.normalize();
        float64_t depth = (posFrame(2) - zGround) * nGround(2); // First-order projection (exact assuming flat surface)

        if (depth < 0.0)
        {
            // Get frame motion in the motion frame.
            vector3_t motionFrame = pinocchio::getFrameVelocity(
                system.robot->pncModel_, system.robot->pncData_, frameId).linear();
            vector3_t vFrameInWorld = tformFrameRot * motionFrame;
            float64_t vDepth = vFrameInWorld.dot(nGround);

            // Compute normal force
            float64_t fextNormal = 0.0;
            if (vDepth < 0.0)
            {
                fextNormal -= contactOptions_.damping * vDepth;
            }
            fextNormal -= contactOptions_.stiffness * depth;
            fextInWorld = fextNormal * nGround;

            // Compute friction forces
            vector3_t vTangential = vFrameInWorld - vDepth * nGround;
            float64_t vNorm = vTangential.norm();

            float64_t frictionCoeff = 0.0;
            if (vNorm >= contactOptions_.dryFrictionVelEps)
            {
                if (vNorm < 1.5 * contactOptions_.dryFrictionVelEps)
                {
                    frictionCoeff = -2.0 * (contactOptions_.frictionDry -
                        contactOptions_.frictionViscous) * (vNorm / contactOptions_.dryFrictionVelEps)
                        + 3.0 * contactOptions_.frictionDry - 2.0*contactOptions_.frictionViscous;
                }
                else
                {
                    frictionCoeff = contactOptions_.frictionViscous;
                }
            }
            else
            {
                frictionCoeff = contactOptions_.frictionDry *
                    (vNorm / contactOptions_.dryFrictionVelEps);
            }
            float64_t fextTangential = frictionCoeff * fextNormal;
            fextInWorld += -fextTangential * vTangential;

            // Add blending factor
            if (contactOptions_.transitionEps > EPS)
            {
                float64_t blendingFactor = -depth / contactOptions_.transitionEps;
                float64_t blendingLaw = std::tanh(2 * blendingFactor);
                fextInWorld *= blendingLaw;
            }
        }
        else
        {
            fextInWorld.setZero();
        }

        return {fextInWorld, vector3_t::Zero()};
    }

    void EngineMultiRobot::computeExternalForces(systemDataHolder_t       & system,
                                                 float64_t          const & t,
                                                 vectorN_t          const & x,
                                                 forceVector_t            & fext)
    {
        // Reinitialize the external forces
        for (pinocchio::Force & fext_i : fext)
        {
            fext_i.setZero();
        }

        // Compute the contact forces
        std::vector<int32_t> const & contactFramesIdx = system.robot->getContactFramesIdx();
        for (uint32_t i=0; i < contactFramesIdx.size(); i++)
        {
            // Compute force in the contact frame.
            int32_t const & frameIdx = contactFramesIdx[i];
            pinocchio::Force & fextInFrame = system.robot->contactForces_[i];
            fextInFrame = computeContactDynamics(system, frameIdx);

            // Apply the force at the origin of the parent joint frame
            pinocchio::Force const fextLocal = computeFrameForceOnParentJoint(
                system.robot->pncModel_, system.robot->pncData_, frameIdx, fextInFrame);
            int32_t const & parentIdx = system.robot->pncModel_.frames[frameIdx].parent;
            fext[parentIdx] += fextLocal;
        }

        // Add the effect of user-defined external forces
        if (system.forceImpulseNextIt != system.forcesImpulse.end())
        {
            float64_t const & tForceImpulseNext = system.forceImpulseNextIt->t;
            float64_t const & dt = system.forceImpulseNextIt->dt;
            if (tForceImpulseNext <= t && t <= tForceImpulseNext + dt)
            {
                std::string const & frameName = system.forceImpulseNextIt->frameName;
                pinocchio::Force const & F = system.forceImpulseNextIt->F;
                int32_t frameIdx;
                getFrameIdx(system.robot->pncModel_, frameName, frameIdx);
                int32_t const & parentIdx = system.robot->pncModel_.frames[frameIdx].parent;
                fext[parentIdx] += computeFrameForceOnParentJoint(
                    system.robot->pncModel_, system.robot->pncData_, frameIdx, F);
            }
        }

        for (auto const & forceProfile : system.forcesProfile)
        {
            int32_t const & frameIdx = forceProfile.frameIdx;
            int32_t const & parentIdx = system.robot->pncModel_.frames[frameIdx].parent;
            forceFunctor_t const & forceFct = forceProfile.forceFct;
            fext[parentIdx] += computeFrameForceOnParentJoint(
                system.robot->pncModel_, system.robot->pncData_, frameIdx, forceFct(t, x));
        }
    }

    void EngineMultiRobot::computeInternalDynamics(systemDataHolder_t          & system,
                                                   float64_t            const  & t,
                                                   Eigen::Ref<vectorN_t const>   q,
                                                   Eigen::Ref<vectorN_t const>   v,
                                                   vectorN_t                   & u) const
    {
        // Reinitialize the internal torque vector
        u.setZero();

        // Compute the user-defined internal dynamics
        system.controller->internalDynamics(t, q, v, u);

        // Define some proxies
        auto const & jointOptions = engineOptions_->joints;
        pinocchio::Model const & pncModel = system.robot->pncModel_;

        // Enforce the position limit (do not support spherical joints)
        if (system.robot->mdlOptions_->joints.enablePositionLimit)
        {
            std::vector<int32_t> const & rigidIdx = system.robot->getRigidJointsModelIdx();
            vectorN_t const & positionLimitMin = system.robot->getPositionLimitMin();
            vectorN_t const & positionLimitMax = system.robot->getPositionLimitMax();
            uint32_t idxOffset = 0;
            for (uint32_t i = 0; i < rigidIdx.size(); i++)
            {
                uint32_t const & positionIdx = pncModel.joints[rigidIdx[i]].idx_q();
                uint32_t const & velocityIdx = pncModel.joints[rigidIdx[i]].idx_v();

                uint32_t const & jointDof = pncModel.joints[rigidIdx[i]].nq();
                for (uint32_t j = 0; j < jointDof; j++)
                {
                    float64_t const & qJoint = q[positionIdx + j];
                    float64_t const & vJoint = v[velocityIdx + j];
                    float64_t const & qJointMin = positionLimitMin[idxOffset];
                    float64_t const & qJointMax = positionLimitMax[idxOffset];

                    float64_t forceJoint = 0;
                    float64_t qJointError = 0;
                    if (qJoint > qJointMax)
                    {
                        qJointError = qJoint - qJointMax;
                        float64_t const damping = -jointOptions.boundDamping * std::max(vJoint, 0.0);
                        forceJoint = -jointOptions.boundStiffness * qJointError + damping;
                    }
                    else if (qJoint < qJointMin)
                    {
                        qJointError = qJointMin - qJoint;
                        float64_t const damping = -jointOptions.boundDamping * std::min(vJoint, 0.0);
                        forceJoint = jointOptions.boundStiffness * qJointError + damping;
                    }

                    if (jointOptions.boundTransitionEps > EPS)
                    {
                        float64_t const blendingFactor = qJointError / jointOptions.boundTransitionEps;
                        float64_t const blendingLaw = std::tanh(2 * blendingFactor);
                        forceJoint *= blendingLaw;
                    }

                    u[velocityIdx + j] += clamp(forceJoint, -1e5, 1e5);

                    idxOffset++;
                }
            }
        }

        // Enforce the velocity limit (do not support spherical joints)
        if (system.robot->mdlOptions_->joints.enableVelocityLimit)
        {
            std::vector<int32_t> const & rigidIdx = system.robot->getRigidJointsModelIdx();
            vectorN_t const & velocityLimitMax = system.robot->getVelocityLimit();

            uint32_t idxOffset = 0U;
            for (uint32_t i = 0; i < rigidIdx.size(); i++)
            {
                uint32_t const & velocityIdx = pncModel.joints[rigidIdx[i]].idx_v();
                uint32_t const & jointDof = pncModel.joints[rigidIdx[i]].nq();

                for (uint32_t j = 0; j < jointDof; j++)
                {
                    float64_t const & vJoint = v[velocityIdx + j];
                    float64_t const & vJointMin = -velocityLimitMax[idxOffset];
                    float64_t const & vJointMax = velocityLimitMax[idxOffset];

                    float64_t forceJoint = 0.0;
                    float64_t vJointError = 0.0;
                    if (vJoint > vJointMax)
                    {
                        vJointError = vJoint - vJointMax;
                        forceJoint = -jointOptions.boundDamping * vJointError;
                    }
                    else if (vJoint < vJointMin)
                    {
                        vJointError = vJointMin - vJoint;
                        forceJoint = jointOptions.boundDamping * vJointError;
                    }

                    if (jointOptions.boundTransitionEps > EPS)
                    {
                        float64_t const blendingFactor = vJointError / jointOptions.boundTransitionEps;
                        float64_t const blendingLaw = std::tanh(2 * blendingFactor);
                        forceJoint *= blendingLaw;
                    }

                    u[velocityIdx + j] += clamp(forceJoint, -1e5, 1e5);

                    idxOffset++;
                }
            }
        }

        // Compute the flexibilities (only support joint_t::SPHERICAL so far)
        Robot::dynamicsOptions_t const & mdlDynOptions = system.robot->mdlOptions_->dynamics;
        std::vector<int32_t> const & flexibilityIdx = system.robot->getFlexibleJointsModelIdx();
        for (uint32_t i=0; i<flexibilityIdx.size(); ++i)
        {
            uint32_t const & positionIdx = pncModel.joints[flexibilityIdx[i]].idx_q();
            uint32_t const & velocityIdx = pncModel.joints[flexibilityIdx[i]].idx_v();
            vectorN_t const & stiffness = mdlDynOptions.flexibilityConfig[i].stiffness;
            vectorN_t const & damping = mdlDynOptions.flexibilityConfig[i].damping;

            float64_t theta;
            quaternion_t const quat(q.segment<4>(positionIdx).data()); // Only way to initialize with [x,y,z,w] order
            vectorN_t const axis = pinocchio::quaternion::log3(quat, theta);
            u.segment<3>(velocityIdx).array() += - stiffness.array() * axis.array()
                - damping.array() * v.segment<3>(velocityIdx).array();
        }
    }

    void EngineMultiRobot::computeCommand(systemDataHolder_t                & system,
                                          float64_t                   const & t,
                                          Eigen::Ref<vectorN_t const>         q,
                                          Eigen::Ref<vectorN_t const>         v,
                                          vectorN_t                         & u)
    {
        // Reinitialize the external forces
        u.setZero();

        // Command the command
        system.controller->computeCommand(t, q, v, u);
    }

    void EngineMultiRobot::computeSystemDynamics(float64_t const & t,
                                                 vectorN_t const & xCat,
                                                 vectorN_t       & dxdtCat)
    {
        /* Note that the position of the free flyer is in world frame, whereas the
           velocities and accelerations are relative to the parent body frame. */

        uint32_t xIdx = 0U;
        for (auto & system : systemsDataHolder_)
        {
            // Extract the state and derivate associated with the system
            Eigen::Ref<vectorN_t const> x = xCat.segment(xIdx, system.robot->nx());
            Eigen::Ref<vectorN_t> dxdt = dxdtCat.segment(xIdx, system.robot->nx());

            // Get references to the internal stepper buffers
            Eigen::Ref<vectorN_t const> q = x.head(system.robot->nq());
            Eigen::Ref<vectorN_t const> v = x.tail(system.robot->nv());
            vectorN_t & u = system.state.u;
            vectorN_t & uCommand = system.state.uCommand;
            vectorN_t & uMotor = system.state.uMotor;
            vectorN_t & uInternal = system.state.uInternal;
            forceVector_t & fext = system.state.fExternal;

            // Get proxies to the state derivative
            Eigen::Ref<vectorN_t> qDot = dxdt.head(system.robot->nq());
            Eigen::Ref<vectorN_t> a = dxdt.tail(system.robot->nv());

            // Compute kinematics information
            computeForwardKinematics(system, q, v, system.state.a());

            /* Compute the external contact forces.
            Note that one must call this method BEFORE updating the sensors since
            the force sensor measurements rely on robot_->contactForces_ */
            computeExternalForces(system, t, x, fext);

            /* Update the sensor data if necessary (only for infinite update frequency).
            Note that it is impossible to have access to the current accelerations
            and torques since they depend on the sensor values themselves. */
            if (engineOptions_->stepper.sensorsUpdatePeriod < SIMULATION_MIN_TIMESTEP)
            {
                system.robot->setSensorsData(t, q, v, system.stateLast.a(), system.stateLast.uMotor);
            }

            /* Update the controller command if necessary (only for infinite update frequency).
            Make sure that the sensor state has been updated beforehand. */
            if (engineOptions_->stepper.controllerUpdatePeriod < SIMULATION_MIN_TIMESTEP)
            {
                computeCommand(system, t, q, v, uCommand);
            }

            /* Compute the actual motor torque.
            Note that it is impossible to have access to the current accelerations. */
            system.robot->computeMotorsTorques(t, q, v, system.stateLast.a(), uCommand);
            uMotor = system.robot->getMotorsTorques();

            /* Compute the internal dynamics.
            Make sure that the sensor state has been updated beforehand since
            the user-defined internal dynamics may rely on it. */
            computeInternalDynamics(system, t, q, v, uInternal);

            // Compute the total torque vector
            u = uInternal;
            for (auto const & motor : system.robot->getMotors())
            {
                int32_t const & motorId = motor->getIdx();
                int32_t const & motorVelocityIdx = motor->getJointVelocityIdx();
                u[motorVelocityIdx] += uMotor[motorId];
            }

            // Compute the dynamics
            a = EngineMultiRobot::aba(system.robot->pncModel_, system.robot->pncData_, q, v, u, fext);

            // Project the derivative in state space
            float64_t const dt = t - stepperState_.tLast;
            computePositionDerivative(system.robot->pncModel_, q, v, qDot, dt);

            // Update the running first index
            xIdx += system.robot->nx();
        }
    }

    // ===================================================================
    // ================ Log reading and writing utilities ================
    // ===================================================================

    void logDataRawToEigenMatrix(std::vector<float64_t>               const & timestamps,
                                 std::vector<std::vector<int32_t> >   const & intData,
                                 std::vector<std::vector<float32_t> > const & floatData,
                                 matrixN_t                                  & logData)
    {
        // Never empty since it contains at least the initial state
        logData.resize(timestamps.size(), 1 + intData[0].size() + floatData[0].size());
        logData.col(0) = Eigen::Matrix<float64_t, 1, Eigen::Dynamic>::Map(
            timestamps.data(), timestamps.size());
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

    void EngineMultiRobot::getLogDataRaw(std::vector<std::string>             & header,
                                         std::vector<float64_t>               & timestamps,
                                         std::vector<std::vector<int32_t> >   & intData,
                                         std::vector<std::vector<float32_t> > & floatData)
    {
        telemetryRecorder_->getData(header, timestamps, intData, floatData);
    }

    void EngineMultiRobot::getLogData(std::vector<std::string> & header,
                                      matrixN_t                & logData)
    {
        std::vector<float64_t> timestamps;
        std::vector<std::vector<int32_t> > intData;
        std::vector<std::vector<float32_t> > floatData;
        getLogDataRaw(header, timestamps, intData, floatData);
        logDataRawToEigenMatrix(timestamps, intData, floatData, logData);
    }

    hresult_t EngineMultiRobot::writeLogTxt(std::string const & filename)
    {
        std::vector<std::string> header;
        matrixN_t log;
        getLogData(header, log);

        std::ofstream myFile = std::ofstream(filename,
                                             std::ios::out |
                                             std::ofstream::trunc);

        if (myFile.is_open())
        {
            auto indexConstantEnd = std::find(header.begin(), header.end(), START_COLUMNS);
            std::copy(header.begin() + 1,
                      indexConstantEnd - 1,
                      std::ostream_iterator<std::string>(myFile, ", ")); // Discard the first one (start constant flag)
            std::copy(indexConstantEnd - 1,
                      indexConstantEnd,
                      std::ostream_iterator<std::string>(myFile, "\n"));
            std::copy(indexConstantEnd + 1,
                      header.end() - 2,
                      std::ostream_iterator<std::string>(myFile, ", "));
            std::copy(header.end() - 2,
                      header.end() - 1,
                      std::ostream_iterator<std::string>(myFile, "\n")); // Discard the last one (start data flag)

            Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");
            myFile << log.format(CSVFormat);

            myFile.close();
        }
        else
        {
            std::cout << "Error - EngineMultiRobot::writeLogTxt - Impossible to create the log file. "\
                         "Check if root folder exists and if you have writing permissions." << std::endl;
            return hresult_t::ERROR_BAD_INPUT;
        }
        return hresult_t::SUCCESS;
    }

    hresult_t EngineMultiRobot::writeLogBinary(std::string const & filename)
    {
        return telemetryRecorder_->writeDataBinary(filename);
    }

    hresult_t EngineMultiRobot::parseLogBinaryRaw(std::string                          const & filename,
                                                  std::vector<std::string>                   & header,
                                                  std::vector<float64_t>                     & timestamps,
                                                  std::vector<std::vector<int32_t> >         & intData,
                                                  std::vector<std::vector<float32_t> >       & floatData)
    {
        int64_t integerSectionSize;
        int64_t floatSectionSize;
        int64_t headerSize;

        std::ifstream myFile = std::ifstream(filename,
                                             std::ios::in |
                                             std::ifstream::binary);

        if (myFile.is_open())
        {
            // Skip the version flag
            int64_t header_version_length = sizeof(int32_t);
            myFile.seekg(header_version_length);

            std::vector<std::string> headerBuffer;
            std::string subHeaderBuffer;

            // Get all the logged constants
            while (std::getline(myFile, subHeaderBuffer, '\0').good() &&
                   subHeaderBuffer != START_COLUMNS)
            {
                headerBuffer.push_back(subHeaderBuffer);
            }

            // Get the names of the logged variables
            while (std::getline(myFile, subHeaderBuffer, '\0').good() &&
                   subHeaderBuffer != (START_DATA + START_LINE_TOKEN))
            {
                // Do nothing
            }

            // Make sure the log file is not corrupted
            if (!myFile.good())
            {
                std::cout << "Error - EngineMultiRobot::parseLogBinary - Corrupted log file." << std::endl;
                return hresult_t::ERROR_BAD_INPUT;
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
            headerSize = ((int32_t) myFile.tellg()) - START_LINE_TOKEN.size() - 1;

            // Close the file
            myFile.close();
        }
        else
        {
            std::cout << "Error - EngineMultiRobot::parseLogBinary - Impossible to open the log file. "\
                         "Check that the file exists and that you have reading permissions." << std::endl;
            return hresult_t::ERROR_BAD_INPUT;
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

        return hresult_t::SUCCESS;
    }

    hresult_t EngineMultiRobot::parseLogBinary(std::string              const & filename,
                                               std::vector<std::string>       & header,
                                               matrixN_t                      & logData)
    {
        std::vector<float64_t> timestamps;
        std::vector<std::vector<int32_t> > intData;
        std::vector<std::vector<float32_t> > floatData;
        hresult_t returnCode = parseLogBinaryRaw(
            filename, header, timestamps, intData, floatData);
        if (returnCode == hresult_t::SUCCESS)
        {
            logDataRawToEigenMatrix(timestamps, intData, floatData, logData);
        }
        return returnCode;
    }

    // =====================================================================================================
    // ================ Custom implementation of Pinocchio methods to support motor inertia ================
    // =====================================================================================================

    template<typename Scalar, int Options, template<typename, int> class JointCollectionTpl,
             typename ConfigVectorType, typename TangentVectorType>
    inline Scalar
    EngineMultiRobot::kineticEnergy(pinocchio::ModelTpl<Scalar,Options,JointCollectionTpl> const & model,
                                    pinocchio::DataTpl<Scalar,Options,JointCollectionTpl>        & data,
                                    Eigen::MatrixBase<ConfigVectorType>                    const & q,
                                    Eigen::MatrixBase<TangentVectorType>                   const & v,
                                    bool_t                                                 const & update_kinematics)
    {
        pinocchio::kineticEnergy(model, data, q, v, update_kinematics);
        data.kinetic_energy += 0.5 * (model.rotorInertia.array() * Eigen::pow(v.array(), 2)).sum();
        return data.kinetic_energy;
    }

    template<typename Scalar, int Options, template<typename, int> class JointCollectionTpl,
             typename ConfigVectorType, typename TangentVectorType1, typename TangentVectorType2,
             typename ForceDerived>
    inline const typename pinocchio::DataTpl<Scalar,Options,JointCollectionTpl>::TangentVectorType &
    EngineMultiRobot::rnea(pinocchio::ModelTpl<Scalar,Options,JointCollectionTpl> const & model,
                           pinocchio::DataTpl<Scalar,Options,JointCollectionTpl>        & data,
                           Eigen::MatrixBase<ConfigVectorType>                    const & q,
                           Eigen::MatrixBase<TangentVectorType1>                  const & v,
                           Eigen::MatrixBase<TangentVectorType2>                  const & a,
                           pinocchio::container::aligned_vector<ForceDerived>     const & fext)
    {
        pinocchio::rnea(model, data, q, v, a, fext);
        data.tau += model.rotorInertia.asDiagonal() * a;
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
        static enable_if_t<!std::is_same<JointModel, pinocchio::JointModelRevoluteTpl<Scalar, 0, 0> >::value
                        && !std::is_same<JointModel, pinocchio::JointModelRevoluteTpl<Scalar, 0, 1> >::value
                        && !std::is_same<JointModel, pinocchio::JointModelRevoluteTpl<Scalar, 0, 2> >::value, void>
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
        static enable_if_t<std::is_same<JointModel, typename pinocchio::JointModelRevoluteTpl<Scalar, 0, 0> >::value
                        || std::is_same<JointModel, typename pinocchio::JointModelRevoluteTpl<Scalar, 0, 1> >::value
                        || std::is_same<JointModel, typename pinocchio::JointModelRevoluteTpl<Scalar, 0, 2> >::value, void>
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
    EngineMultiRobot::aba(pinocchio::ModelTpl<Scalar,Options,JointCollectionTpl> const & model,
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
        for (JointIndex i=1; i<(JointIndex)model.njoints; ++i)
        {
            Pass1::run(model.joints[i],data.joints[i],
                       typename Pass1::ArgsType(model,data,q.derived(),v.derived()));
            data.f[i] -= fext[i];
        }

        typedef AbaBackwardStep<Scalar,Options,JointCollectionTpl> Pass2;
        for (JointIndex i=(JointIndex)model.njoints-1; i>0; --i)
        {
            Pass2::run(model.joints[i],data.joints[i],
                       typename Pass2::ArgsType(model,data));
        }

        typedef pinocchio::AbaForwardStep2<Scalar,Options,JointCollectionTpl> Pass3;
        for (JointIndex i=1; i<(JointIndex)model.njoints; ++i)
        {
            Pass3::run(model.joints[i],data.joints[i],
                       typename Pass3::ArgsType(model,data));
        }

        return data.ddq;
    }
}
