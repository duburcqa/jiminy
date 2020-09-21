#include <iostream>
#include <cmath>
#include <algorithm>

#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/algorithm/contact-dynamics.hpp"
#include "pinocchio/algorithm/geometry.hpp"

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
#include "jiminy/core/engine/PinocchioOverloadAlgorithms.h"

#include <boost/numeric/odeint/iterator/n_step_iterator.hpp>


namespace jiminy
{
    // ===============================================
    // ================ systemState_t ================
    // ===============================================

    void systemState_t::initialize(Robot const * robot)
    {
        robot_ = robot;
        q = pinocchio::neutral(robot->pncModel_);
        v = vectorN_t::Zero(robot_->nv());
        qDot = vectorN_t::Zero(robot_->nq());
        a = vectorN_t::Zero(robot_->nv());
        uInternal = vectorN_t::Zero(robot_->nv());
        uCommand = vectorN_t::Zero(robot_->getMotorsNames().size());
        uMotor = vectorN_t::Zero(robot_->getMotorsNames().size());
        u = vectorN_t::Zero(robot_->nv());
        fExternal = forceVector_t(robot_->pncModel_.joints.size(),
                                    pinocchio::Force::Zero());

        isInitialized_ = true;
    }

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
    motorEffortFieldnames(),
    energyFieldname(),
    robotLock(nullptr),
    state(),
    statePrev(),
    forcesProfile(),
    forcesImpulse(),
    forcesImpulseBreaks(),
    forcesImpulseBreakNextIt(),
    forcesImpulseActive()
    {
        state.initialize(robot.get());
        statePrev.initialize(robot.get());
    }

    systemDataHolder_t::systemDataHolder_t(void) :
    systemDataHolder_t("", nullptr, nullptr,
    [](float64_t const & t,
       vectorN_t const & q,
       vectorN_t const & v) -> bool_t
    {
        return true;
    })
    {
        // Empty on purpose.
    }

    // ==================================================
    // ================ EngineMultiRobot ================
    // ==================================================

    EngineMultiRobot::EngineMultiRobot(void):
    engineOptions_(nullptr),
    systemsDataHolder_(),
    isTelemetryConfigured_(false),
    isSimulationRunning_(false),
    engineOptionsHolder_(),
    telemetrySender_(),
    telemetryData_(nullptr),
    telemetryRecorder_(nullptr),
    timer_(),
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

    hresult_t EngineMultiRobot::addCouplingForce(std::string            const & systemName1,
                                                 std::string            const & systemName2,
                                                 std::string            const & frameName1,
                                                 std::string            const & frameName2,
                                                 forceCouplingFunctor_t         forceFct)
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
            std::cout << "Error - EngineMultiRobot::addCouplingForce - At least one of the names does not correspond to any system added to the engine." << std::endl;
            returnCode = hresult_t::ERROR_BAD_INPUT;
        }

        int32_t systemIdx1;
        int32_t frameIdx1;
        int32_t systemIdx2;
        int32_t frameIdx2;
        if (returnCode == hresult_t::SUCCESS)
        {
            systemIdx1 = std::distance(systemsDataHolder_.begin(), systemIt1);
            returnCode = getFrameIdx(systemIt1->robot->pncModel_, frameName1, frameIdx1);

        }
        if (returnCode == hresult_t::SUCCESS)
        {
            systemIdx2 = std::distance(systemsDataHolder_.begin(), systemIt2);
            returnCode = getFrameIdx(systemIt1->robot->pncModel_, frameName2, frameIdx2);
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            forcesCoupling_.emplace_back(systemName1,
                                         std::move(systemIdx1),
                                         systemName2,
                                         std::move(systemIdx2),
                                         frameName1,
                                         std::move(frameIdx1),
                                         frameName2,
                                         std::move(frameIdx2),
                                         std::move(forceFct));
        }

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
                    addCircumfix(system.robot->getPositionFieldnames(),
                                 system.name, "", TELEMETRY_DELIMITER);
                system.velocityFieldnames =
                    addCircumfix(system.robot->getVelocityFieldnames(),
                                 system.name, "", TELEMETRY_DELIMITER);
                system.accelerationFieldnames =
                    addCircumfix(system.robot->getAccelerationFieldnames(),
                                 system.name, "", TELEMETRY_DELIMITER);
                system.motorEffortFieldnames =
                    addCircumfix(system.robot->getMotorEffortFieldnames(),
                                 system.name, "", TELEMETRY_DELIMITER);
                system.energyFieldname =
                    addCircumfix("energy",
                                 system.name, "", TELEMETRY_DELIMITER);

                // Register variables to the telemetry senders
                if (returnCode == hresult_t::SUCCESS)
                {
                    if (engineOptions_->telemetry.enableConfiguration)
                    {
                        returnCode = telemetrySender_.registerVariable(
                            system.positionFieldnames,
                            system.state.q);
                    }
                }
                if (returnCode == hresult_t::SUCCESS)
                {
                    if (engineOptions_->telemetry.enableVelocity)
                    {
                        returnCode = telemetrySender_.registerVariable(
                            system.velocityFieldnames,
                            system.state.v);
                    }
                }
                if (returnCode == hresult_t::SUCCESS)
                {
                    if (engineOptions_->telemetry.enableAcceleration)
                    {
                        returnCode = telemetrySender_.registerVariable(
                            system.accelerationFieldnames,
                            system.state.a);
                    }
                }
                if (returnCode == hresult_t::SUCCESS)
                {
                    if (engineOptions_->telemetry.enableEffort)
                    {
                        returnCode = telemetrySender_.registerVariable(
                            system.motorEffortFieldnames,
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
            float64_t energy = pinocchio_overload::kineticEnergy(
                system.robot->pncModel_,
                system.robot->pncData_,
                system.state.q,
                system.state.v,
                true);
            energy += pinocchio::potentialEnergy(
                system.robot->pncModel_,
                system.robot->pncData_,
                system.state.q,
                false);

            // Update the telemetry internal state
            if (engineOptions_->telemetry.enableConfiguration)
            {
                telemetrySender_.updateValue(system.positionFieldnames,
                                             system.state.q);
            }
            if (engineOptions_->telemetry.enableVelocity)
            {
                telemetrySender_.updateValue(system.velocityFieldnames,
                                             system.state.v);
            }
            if (engineOptions_->telemetry.enableAcceleration)
            {
                telemetrySender_.updateValue(system.accelerationFieldnames,
                                             system.state.a);
            }
            if (engineOptions_->telemetry.enableEffort)
            {
                telemetrySender_.updateValue(system.motorEffortFieldnames,
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
        // Make sure the simulation is properly stopped
        if (isSimulationRunning_)
        {
            stop();
        }

        // Reset the dynamic force register if requested
        if (resetDynamicForceRegister)
        {
            for (auto & system : systemsDataHolder_)
            {
                system.forcesImpulse.clear();
                system.forcesImpulseBreaks.clear();
                system.forcesImpulseActive.clear();
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
    }

    void EngineMultiRobot::reset(bool_t const & resetDynamicForceRegister)
    {
        reset(true, resetDynamicForceRegister);
    }

    hresult_t EngineMultiRobot::start(std::map<std::string, vectorN_t> const & xInit,
                                      bool_t const & resetRandomNumbers,
                                      bool_t const & resetDynamicForceRegister)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        /* Make sure that no simulation is running.
           Note that one must return early to avoid configuring multiple times the telemetry
           and stopping the simulation because of the unsuccessful returnCode. */
        if (isSimulationRunning_)
        {
            std::cout << "Error - EngineMultiRobot::start - A simulation is already running. Stop it before starting again." << std::endl;
            returnCode = hresult_t::ERROR_GENERIC;
            return returnCode;
        }

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

        // Check the dimension of the initial state associated with every system and order them
        std::vector<vectorN_t> xInitOrdered;
        xInitOrdered.reserve(systemsDataHolder_.size());
        for (auto & system : systemsDataHolder_)
        {
            auto xInitIt = xInit.find(system.name);
            if (xInitIt == xInit.end())
            {
                    std::cout << "Error - EngineMultiRobot::start - At least one system does not have an initial state." << std::endl;
                    returnCode = hresult_t::ERROR_BAD_INPUT;
            }
            if (returnCode == hresult_t::SUCCESS)
            {
                if (xInitIt->second.rows() != system.robot->nx())
                {
                    std::cout << "Error - EngineMultiRobot::start - The size of the initial state is inconsistent "
                                 "with model size for at least one system." << std::endl;
                    returnCode = hresult_t::ERROR_BAD_INPUT;
                }
            }
            if (returnCode == hresult_t::SUCCESS)
            {
                xInitOrdered.emplace_back(std::move(xInitIt->second));
            }
        }

        for (auto & system : systemsDataHolder_)
        {
            for (auto const & sensorGroup : system.robot->getSensors())
            {
                for (auto const & sensor : sensorGroup.second)
                {
                    if (returnCode == hresult_t::SUCCESS)
                    {
                        if (!sensor->getIsInitialized())
                        {
                            std::cout << "Error - EngineMultiRobot::start - At least a sensor of a robot is not initialized." << std::endl;
                            returnCode = hresult_t::ERROR_INIT_FAILED;
                        }
                    }
                }
            }

            for (auto const & motor : system.robot->getMotors())
            {
                if (returnCode == hresult_t::SUCCESS)
                {
                    if (!motor->getIsInitialized())
                    {
                        std::cout << "Error - EngineMultiRobot::start - At least a motor of a robot is not initialized." << std::endl;
                        returnCode = hresult_t::ERROR_INIT_FAILED;
                    }
                }
            }
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            // Reset the robot, controller, engine, and registered impulse forces if requested
            reset(resetRandomNumbers, resetDynamicForceRegister);
        }

        // At this point, consider that the simulation is running
        isSimulationRunning_ = true;

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

                /* Reinitialize the system state buffers, since the robot kinematic may have changed.
                   For example, it may happens if one activates or deactivates the flexibility between
                   two successive simulations. */
                system.state.initialize(system.robot.get());
                system.statePrev.initialize(system.robot.get());
            }
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            // Initialize the ode solver
            if (engineOptions_->stepper.odeSolver == "runge_kutta_dopri5")
            {
                stepper_ = stepper::RungeKutta(
                    stepper::runge_kutta::ErrorChecker(
                        engineOptions_->stepper.tolAbs,
                        engineOptions_->stepper.tolRel
                    ), stepper::runge_kutta::StepAdjuster());
            }
            else if (engineOptions_->stepper.odeSolver == "bulirsch_stoer")
            {
                stepper_ = stepper::BulirschStoer(
                    engineOptions_->stepper.tolAbs,
                    engineOptions_->stepper.tolRel
                );
            }
            else if (engineOptions_->stepper.odeSolver == "explicit_euler")
            {
                stepper_ = stepper::EulerExplicit();
            }

            // Set the initial time step
            float64_t const dt = SIMULATION_INITIAL_TIMESTEP;

            // Initialize the stepper state
            float64_t const t = 0.0;
            vectorN_t const xCat = cat(xInitOrdered);
            stepperState_.reset(dt, xCat);

            // Synchronize the individual system states with the global stepper state
            syncSystemsStateWithStepper();

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
                // Update the frame indices associated with the impulse forces and force profiles
                for (auto & force : system.forcesProfile)
                {
                    getFrameIdx(system.robot->pncModel_,
                                force.frameName,
                                force.frameIdx);
                }
                for (auto & force : system.forcesImpulse)
                {
                    getFrameIdx(system.robot->pncModel_,
                                force.frameName,
                                force.frameIdx);
                }

                // Initialize the impulse force breakpoint point iterator
                system.forcesImpulseBreakNextIt = system.forcesImpulseBreaks.begin();

                // Reset the active set of impulse forces
                std::fill(system.forcesImpulseActive.begin(),
                          system.forcesImpulseActive.end(),
                          false);

                // Compute the forward kinematics for each system
                vectorN_t const & q = system.state.q;
                vectorN_t const & v = system.state.v;
                vectorN_t const & a = system.state.a;
                computeForwardKinematics(system, q, v, a);

                // Make sure that the contact forces are bounded.
                // TODO: One should rather use something like 10 * m * g instead of a fix threshold
                float64_t forceMax = 0.0;
                auto const & contactFramesIdx = system.robot->getContactFramesIdx();
                for (uint32_t i=0; i < contactFramesIdx.size(); i++)
                {
                    pinocchio::Force fext = computeContactDynamicsAtFrame(system, contactFramesIdx[i]);
                    forceMax = std::max(forceMax, fext.linear().norm());
                }

                auto const & collisionBodiesIdx = system.robot->getCollisionBodiesIdx();
                for (uint32_t i=0; i < collisionBodiesIdx.size(); i++)
                {
                    pinocchio::Force fext = computeContactDynamicsAtBody(system, i);
                    forceMax = std::max(forceMax, fext.linear().norm());
                }

                if (forceMax > 1e5)
                {
                    std::cout << "Error - EngineMultiRobot::start - The initial force exceeds 1e5 for at least one contact point, "\
                                 "which is forbidden for the sake of numerical stability. Please update the initial state." << std::endl;
                    returnCode = hresult_t::ERROR_BAD_INPUT;
                }

                // Activate every force impulse starting at t=0
                auto forcesImpulseActiveIt = system.forcesImpulseActive.begin();
                auto forcesImpulseIt = system.forcesImpulse.begin();
                for ( ; forcesImpulseIt != system.forcesImpulse.end() ;
                    forcesImpulseActiveIt++, forcesImpulseIt++)
                {
                    if (forcesImpulseIt->t < STEPPER_MIN_TIMESTEP)
                    {
                        *forcesImpulseActiveIt = true;
                    }
                }
            }

            // Compute the internal and external forces applied on every systems
            auto const xSplit = splitState(xCat);
            computeAllForces(t, xSplit);

            for (auto & system : systemsDataHolder_)
            {
                // Get some system state proxies
                vectorN_t const & q = system.state.q;
                vectorN_t const & v = system.state.v;
                vectorN_t & qDot = system.state.qDot;
                vectorN_t & a = system.state.a;
                vectorN_t & u = system.state.u;
                vectorN_t & uCommand = system.state.uCommand;
                vectorN_t & uMotor = system.state.uMotor;
                vectorN_t & uInternal = system.state.uInternal;
                forceVector_t & fext = system.state.fExternal;

                // Initialize the sensor data
                system.robot->setSensorsData(t, q, v, a, uMotor);

                // Compute the actual motor effort
                computeCommand(system, t, q, v, uCommand);

                // Compute the actual motor effort
                system.robot->computeMotorsEfforts(t, q, v, a, uCommand);
                uMotor = system.robot->getMotorsEfforts();

                // Compute the internal dynamics
                computeInternalDynamics(system, t, q, v, uInternal);

                // Compute the total effort vector
                u = uInternal;
                for (auto const & motor : system.robot->getMotors())
                {
                    int32_t const & motorIdx = motor->getIdx();
                    int32_t const & motorVelocityIdx = motor->getJointVelocityIdx();
                    u[motorVelocityIdx] += uMotor[motorIdx];
                }

                // Compute dynamics
                a = computeAcceleration(system, q, v, u, fext);

                // Project the derivative in state space
                computePositionDerivative(system.robot->pncModel_, q, v, qDot, dt);

                // Compute the forward kinematics once again, with the updated acceleration
                computeForwardKinematics(system, q, v, a);

                // Update the sensor data once again, with the updated effort and acceleration
                system.robot->setSensorsData(t, q, v, a, uMotor);
            }

            // Synchronize the global stepper state with the individual system states
            syncStepperStateWithSystems();
        }

        // Lock the telemetry. At this point it is no longer possible to register new variables.
        configureTelemetry();

        // Write the header: this locks the registration of new variables
        telemetryRecorder_->initialize(telemetryData_.get(), engineOptions_->telemetry.timeUnit);

        // Log current buffer content as first point of the log data.
        updateTelemetry();

        // Initialize the last system states
        for (auto & system : systemsDataHolder_)
        {
            system.statePrev = system.state;
        }

        if (returnCode != hresult_t::SUCCESS)
        {
            stop();
        }

        return returnCode;
    }

    hresult_t EngineMultiRobot::simulate(float64_t              const & tEnd,
                                         std::map<std::string, vectorN_t> const & xInit)
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

        // Now that telemetry has been initialized, check simulation duration.
        if (tEnd > telemetryRecorder_->getMaximumLogTime())
        {
            std::cout << "Error - EngineMultiRobot::simulate - Time overflow: with the current precision ";
            std::cout << "the maximum value that can be logged is " << telemetryRecorder_->getMaximumLogTime();
            std::cout << "s. Decrease logger precision to simulate for longer than that." << std::endl;
            return hresult_t::ERROR_BAD_INPUT;
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
            bool_t isCallbackFalse = false;
            for (auto & system : systemsDataHolder_)
            {
                if (!system.callbackFct(stepperState_.t, system.state.q, system.state.v))
                {
                    isCallbackFalse = true;
                    break;
                }
            }
            if (isCallbackFalse)
            {
                if (engineOptions_->stepper.verbose)
                {
                    std::cout << "Simulation done: callback returned false." << std::endl;
                }
                break;
            }

            // Stop the simulation if the max number of integration steps is reached
            if (0 < engineOptions_->stepper.iterMax
                && (uint32_t) engineOptions_->stepper.iterMax <= stepperState_.iter)
            {
                if (engineOptions_->stepper.verbose)
                {
                    std::cout << "Simulation done: maximum number of integration steps exceeded." << std::endl;
                }
                break;
            }

            // Perform a single integration step up to tEnd, stopping at stepperUpdatePeriod_ to log.
            float64_t stepSize;
            if (stepperUpdatePeriod_ > EPS)
            {
                stepSize = min(stepperUpdatePeriod_ , tEnd - stepperState_.t);
            }
            else
            {
                stepSize = min(engineOptions_->stepper.dtMax, tEnd - stepperState_.t);
            }
            returnCode = step(stepSize); // Automatic dt adjustment
        }

        // Stop the simulation. New variables can be registered again, and the lock on the robot is released
        stop();

        return returnCode;
    }

    hresult_t EngineMultiRobot::step(float64_t stepSize)
    {
        // Check if the simulation has started
        if (!isSimulationRunning_)
        {
            std::cout << "Error - EngineMultiRobot::step - No simulation running. Please start it before using step method." << std::endl;
            return hresult_t::ERROR_GENERIC;
        }

        // Check if there is something wrong with the integration
        if ((stepperState_.x.array() != stepperState_.x.array()).any()) // isnan if NOT equal to itself
        {
            std::cout << "Error - EngineMultiRobot::step - The low-level ode solver failed. "\
                            "Consider increasing the stepper accuracy." << std::endl;
            return hresult_t::ERROR_GENERIC;
        }

        // Check if the desired step size is suitable
        if (stepSize > EPS && stepSize < SIMULATION_MIN_TIMESTEP)
        {
            std::cout << "Error - EngineMultiRobot::step - The requested step size is out of bounds." << std::endl;
            return hresult_t::ERROR_BAD_INPUT;
        }

        /* Set end time: The default step size is equal to the
           controller update period if discrete-time, otherwise
           it uses the sensor update period if discrete-time,
           otherwise it uses the user-defined parameter dtMax. */
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

        /* Check that end time is not too large for the current
           logging precision, otherwise abort integration. */
        if (stepperState_.t + stepSize > telemetryRecorder_->getMaximumLogTime())
        {
            std::cout << "Error - EngineMultiRobot::step - Time overflow: with the current precision ";
            std::cout << "the maximum value that can be logged is " << telemetryRecorder_->getMaximumLogTime();
            std::cout << "s. Decrease logger precision to simulate for longer than that." << std::endl;
            return hresult_t::ERROR_GENERIC;
        }

        /* Avoid compounding of error using Kahan algorithm. It
           consists in keeping track of the cumulative rounding error
           to add it back to the sum when it gets larger than the
           numerical precision, thus avoiding it to grows unbounded. */
        float64_t stepSize_true = stepSize - stepperState_.tError;
        float64_t tEnd = stepperState_.t + stepSize_true;
        stepperState_.tError = (tEnd - stepperState_.t) - stepSize_true;

        // Get references to some internal stepper buffers
        float64_t & t = stepperState_.t;
        float64_t & dt = stepperState_.dt;
        float64_t & dtLargest = stepperState_.dtLargest;
        vectorN_t & x = stepperState_.x;
        vectorN_t & dxdt = stepperState_.dxdt;

        // Define the stepper iterators.
        auto systemOde =
            [this](vectorN_t const & xIn,
                    vectorN_t       & dxdtIn,
                    float64_t const & tIn)
            {
                this->computeSystemDynamics(tIn, xIn, dxdtIn);
            };

        // Define a failure checker for the stepper
        failed_step_checker fail_checker;

        // Successive iteration failure
        uint32_t sucessiveIterFailed = 0;

        /* Flag monitoring if the current time step depends of a breakpoint
           or the integration tolerance. It will be used by the restoration
           mechanism, if dt gets very small to reach a breakpoint, in order
           to avoid having to perform several steps to stabilize again the
           estimation of the optimal time step. */
        bool_t isBreakpointReached = false;

        /* Flag monitoring if the dynamics has changed because of impulse
           forces or the command (only in the case of discrete control).

           `try_step(rhs, x, dxdt, t, dt)` method of error controlled boost
           steppers leverage the FSAL (first same as last) principle. It is
           implemented by considering at the value of (x, dxdt) in argument
           have been initialized by the user with the system dynamics at
           current time t. Thus, if the system dynamics is discontinuous,
           one has to manually integrate up to t-, then update dxdt to take
           into the acceleration at t+.

           Note that ONLY the acceleration part of dxdt must be updated since
           the  projection of the velocity on the state space is not supposed
           to have changed, and on top of that tPrev is invalid at this point
           because it has been updated just after the last successful step.

           Note that the estimated dt is no longer very meaningful since the
           dynamics has changed. Maybe dt should be reschedule... */
        bool_t hasDynamicsChanged = false;

        // Start the timer used for timeout handling
        timer_.tic();

        // Perform the integration. Do not simulate extremely small time steps.
        while (tEnd - t > STEPPER_MIN_TIMESTEP)
        {
            float64_t tNext = t;

            // Update the active set and get the next breakpoint of impulse forces
            float64_t tForceImpulseNext = INF;
            for (auto & system : systemsDataHolder_)
            {
                /* Update the active set: activate an impulse force as soon as
                   the current time gets close enough of the application time,
                   and deactivate it once the following the same reasoning.

                   Note that breakpoints at the begining and the end of every
                   impulse force at already enforced, so that the forces
                   cannot get activated/desactivate too late. */
                auto forcesImpulseActiveIt = system.forcesImpulseActive.begin();
                auto forcesImpulseIt = system.forcesImpulse.begin();
                for ( ; forcesImpulseIt != system.forcesImpulse.end() ;
                    forcesImpulseActiveIt++, forcesImpulseIt++)
                {
                    float64_t const & tForceImpulse = forcesImpulseIt->t;
                    float64_t const & dtForceImpulse = forcesImpulseIt->dt;

                    if (t > tForceImpulse - STEPPER_MIN_TIMESTEP)
                    {
                        *forcesImpulseActiveIt = true;
                        hasDynamicsChanged = true;
                    }
                    if (t > tForceImpulse + dtForceImpulse - STEPPER_MIN_TIMESTEP)
                    {
                        *forcesImpulseActiveIt = false;
                        hasDynamicsChanged = true;
                    }
                }

                // Update the breakpoint time iterator if necessary
                auto & tBreakNextIt = system.forcesImpulseBreakNextIt;
                if (tBreakNextIt != system.forcesImpulseBreaks.end())
                {
                    if (t > *tBreakNextIt - STEPPER_MIN_TIMESTEP)
                    {
                        // The current breakpoint is behind in time. Switching to the next one.
                        tBreakNextIt++;
                    }
                }

                // Get the next breakpoint time if any
                if (tBreakNextIt != system.forcesImpulseBreaks.end())
                {
                    tForceImpulseNext = min(tForceImpulseNext, *tBreakNextIt);
                }
            }

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
                            vectorN_t const & q = system.state.q;
                            vectorN_t const & v = system.state.v;
                            vectorN_t const & a = system.state.a;
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
                            vectorN_t const & q = system.state.q;
                            vectorN_t const & v = system.state.v;
                            vectorN_t & uCommand = system.state.uCommand;
                            computeCommand(system, t, q, v, uCommand);
                        }
                        hasDynamicsChanged = true;
                    }
                }
            }

            // Fix the FSAL issue if the dynamics has changed
            if (hasDynamicsChanged)
            {
                computeSystemDynamics(t, x, dxdt);
                syncSystemsStateWithStepper();
            }

            if (stepperUpdatePeriod_ > EPS)
            {
                /* Get the time of the next breakpoint for the ODE solver:
                   a breakpoint occurs if we reached tEnd, if an external force
                   is applied, or if we need to update the sensors / controller. */
                float64_t dtNextGlobal; // dt to apply for the next stepper step because of the various breakpoints
                float64_t dtNextUpdatePeriod = stepperUpdatePeriod_ - std::fmod(t, stepperUpdatePeriod_);
                if (dtNextUpdatePeriod < SIMULATION_MIN_TIMESTEP)
                {
                    /* Step to reach next sensors/controller update is too short:
                       skip one controller update and jump to the next one.
                       Note that in this case, the sensors have already been
                       updated in anticipation in previous loop. */
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
                sucessiveIterFailed = 0;
                while (tNext - t > EPS)
                {
                    /* Adjust stepsize to end up exactly at the next breakpoint,
                       prevent steps larger than dtMax, trying to reach multiples of
                       STEPPER_MIN_TIMESTEP whenever possible. The idea here is to
                       reach only multiples of 1us, making logging easier, given that,
                       in robotics, 1us can be consider an 'infinitesimal' time. This
                       arbitrary threshold many not be suited for simulating different,
                       faster dynamics, that require sub-microsecond precision. */
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

                    /* Break the loop if dt is getting too small.
                       Don't worry, an exception will be raised later. */
                    if (dt < STEPPER_MIN_TIMESTEP)
                    {
                        break;
                    }

                    /* Break the loop in case of timeout.
                       Don't worry, an exception will be raised later. */
                    timer_.toc();
                    if (EPS < engineOptions_->stepper.timeout
                        && engineOptions_->stepper.timeout < timer_.dt)
                    {
                        break;
                    }

                    // Break the loop in case of too many successive failed inner iteration
                    if (sucessiveIterFailed > engineOptions_->stepper.successiveIterFailedMax)
                    {
                        break;
                    }

                    /* A breakpoint has been reached dt has been decreased
                       wrt the largest possible dt within integration tol. */
                    isBreakpointReached = (dtLargest > dt);

                    // Set the timestep to be tried by the stepper
                    dtLargest = dt;

                    if (try_step(stepper_, systemOde, x, dxdt, t, dtLargest))
                    {
                        // Reset the fail counter
                        fail_checker.reset();

                        // Project vector onto Lie group, to prevent accumulation of numerical error due to integration.
                        x = normalizeState(x);

                        // Synchronize the individual system states
                        syncSystemsStateWithStepper();

                        // Increment the iteration counter only for successful steps
                        stepperState_.iter++;

                        // Log every stepper state only if the user asked for
                        if (engineOptions_->stepper.logInternalStepperSteps)
                        {
                            updateTelemetry();
                        }

                        /* Restore the step size dt if it has been significantly
                           decreased to because of a breakpoint. It is set
                           equal to the last available largest dt to be known,
                           namely the second to last successfull step. */
                        if (isBreakpointReached)
                        {
                            /* Restore the step size if and only if:
                               - the next estimated largest step size is larger than
                                 the requested one for the current (successful) step.
                               - the next estimated largest step size is significantly
                                 smaller than the estimated largest step size for the
                                 previous step. */
                            float64_t dtRestoreThresholdAbs = stepperState_.dtLargestPrev *
                                engineOptions_->stepper.dtRestoreThresholdRel;
                            if (dt < dtLargest && dtLargest < dtRestoreThresholdAbs)
                            {
                                dtLargest = stepperState_.dtLargestPrev;
                            }
                        }

                        /* Backup the stepper and systems' state on success only:
                           - t at last successful iteration is used to compute dt,
                             which is project the accelation in the state space
                             instead of SO3^2.
                           - dtLargestPrev is used to restore the largest step
                             size in case of a breakpoint requiring lowering it.
                           - the acceleration and effort at the last successful
                             iteration is used to update the sensors' data in
                             case of continuous sensing. */
                        stepperState_.tPrev = t;
                        stepperState_.dtLargestPrev = dtLargest;
                        for (auto & system : systemsDataHolder_)
                        {
                            system.statePrev = system.state;
                        }
                    }
                    else
                    {
                        /* Check for possible overflow of failed steps
                           in step size adjustment. */
                        fail_checker();

                        // Increment the failed iteration counters
                        sucessiveIterFailed++;
                        stepperState_.iterFailed++;
                    }

                    // Initialize the next dt
                    dt = dtLargest;
                }
            }
            else
            {
                /* Make sure it ends exactly at the tEnd, never exceeds
                   dtMax, and stop to apply impulse forces. */
                dt = min(dt,
                         engineOptions_->stepper.dtMax,
                         tEnd - t,
                         tForceImpulseNext - t);

                /* A breakpoint has been reached, because dt has been decreased
                   wrt the largest possible dt within integration tol. */
                isBreakpointReached = (dtLargest > dt);

                // Compute the next step using adaptive step method
                bool_t isStepSuccessful = false;
                while (!isStepSuccessful)
                {
                    // Set the timestep to be tried by the stepper
                    dtLargest = dt;

                    // Break the loop in case of too many successive failed inner iteration
                    if (sucessiveIterFailed > engineOptions_->stepper.successiveIterFailedMax)
                    {
                        break;
                    }

                    // Try to do a step
                    isStepSuccessful = try_step(stepper_, systemOde, x, dxdt, t, dtLargest);

                    if (isStepSuccessful)
                    {
                        // Reset the fail counter
                        fail_checker.reset();

                        // Project vector onto Lie group
                        x = normalizeState(x);

                        // Synchronize the individual system states
                        syncSystemsStateWithStepper();

                        // Increment the iteration counter
                        stepperState_.iter++;

                        // Log every stepper state only if required
                        if (engineOptions_->stepper.logInternalStepperSteps)
                        {
                            updateTelemetry();
                        }

                        // Restore the step size if necessary
                        if (isBreakpointReached)
                        {
                            float64_t dtRestoreThresholdAbs = stepperState_.dtLargestPrev *
                                engineOptions_->stepper.dtRestoreThresholdRel;
                            if (dt < dtLargest && dtLargest < dtRestoreThresholdAbs)
                            {
                                dtLargest = stepperState_.dtLargestPrev;
                            }
                        }

                        // Backup the stepper and systems' state
                        stepperState_.tPrev = t;
                        stepperState_.dtLargestPrev = dtLargest;
                        for (auto & system : systemsDataHolder_)
                        {
                            system.statePrev = system.state;
                        }
                    }
                    else
                    {
                        /* check for possible overflow of failed steps
                           in step size adjustment. */
                        fail_checker();

                        // Increment the failed iteration counter
                        sucessiveIterFailed++;
                        stepperState_.iterFailed++;
                    }

                    // Initialize the next dt
                    dt = dtLargest;
                }
            }

            if (sucessiveIterFailed > engineOptions_->stepper.successiveIterFailedMax)
            {
                std::cout << "Error - EngineMultiRobot::step - Too many successive iteration failures. "\
                             "Probably something is going wrong with the physics. Aborting integration." << std::endl;
                return hresult_t::ERROR_GENERIC;
            }

            if (dt < STEPPER_MIN_TIMESTEP)
            {
                std::cout << "Error - EngineMultiRobot::step - The internal time step is getting too small. "\
                             "Impossible to integrate physics further in time." << std::endl;
                return hresult_t::ERROR_GENERIC;
            }

            timer_.toc();
            if (EPS < engineOptions_->stepper.timeout
                && engineOptions_->stepper.timeout < timer_.dt)
            {
                std::cout << "Error - EngineMultiRobot::step - Step computation timeout." << std::endl;
                return hresult_t::ERROR_GENERIC;
            }
        }

        /* Update the final time and dt to make sure it corresponds
           to the desired values and avoid compounding of error.
           Anyway the user asked for a step of exactly stepSize,
           so he is expecting this value to be reached. */
        stepperState_.t = tEnd;
        stepperState_.dt = stepSize;

        /* Monitor current iteration number, and log the current time,
           state, command, and sensors data. */
        if (!engineOptions_->stepper.logInternalStepperSteps)
        {
            updateTelemetry();
        }

        return hresult_t::SUCCESS;
    }

    void EngineMultiRobot::stop(void)
    {
        // Make sure that a simulation running
        if (!isSimulationRunning_)
        {
            return;
        }

        // Release the lock on the robots
        for (auto & system : systemsDataHolder_)
        {
            system.robotLock.reset(nullptr);
        }

        /* Reset the telemetry.
           Note that calling ``stop` or  `reset` does NOT clear
           the internal data buffer of telemetryRecorder_.
           Clearing is done at init time, so that it remains
           accessible until the next initialization. */
        telemetryRecorder_->reset();
        telemetryData_->reset();

        // Update some internal flags
        isTelemetryConfigured_ = false;
        isSimulationRunning_ = false;
    }

    hresult_t EngineMultiRobot::registerForceImpulse(std::string      const & systemName,
                                                     std::string      const & frameName,
                                                     float64_t        const & t,
                                                     float64_t        const & dt,
                                                     pinocchio::Force const & F)
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

        if (dt < STEPPER_MIN_TIMESTEP)
        {
            std::cout << "Error - EngineMultiRobot::registerForceImpulse - The force duration cannot be smaller than "
                      << STEPPER_MIN_TIMESTEP << "." << std::endl;
            returnCode = hresult_t::ERROR_BAD_INPUT;
        }

        int32_t frameIdx;
        if (returnCode == hresult_t::SUCCESS)
        {
            returnCode = getFrameIdx(system->robot->pncModel_, frameName, frameIdx);
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            system->forcesImpulse.emplace_back(frameName, frameIdx, t, dt, F);
            system->forcesImpulseBreaks.emplace(t);
            system->forcesImpulseBreaks.emplace(t + dt);
            system->forcesImpulseActive.emplace_back(false);
        }

        return hresult_t::SUCCESS;
    }

    hresult_t EngineMultiRobot::registerForceProfile(std::string           const & systemName,
                                                     std::string           const & frameName,
                                                     forceProfileFunctor_t         forceFct)
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
            returnCode = getFrameIdx(
                system->robot->pncModel_, frameName, frameIdx);
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            system->forcesProfile.emplace_back(
                frameName, frameIdx, std::move(forceFct));
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

        // Make sure that the selected time unit for logging makes sense
        configHolder_t telemetryOptions = boost::get<configHolder_t>(engineOptions.at("telemetry"));
        float64_t const & timeUnit = boost::get<float64_t>(telemetryOptions.at("timeUnit"));
        if (1.0 / STEPPER_MIN_TIMESTEP < timeUnit || timeUnit < 1.0 / SIMULATION_MAX_TIMESTEP)
        {
            std::cout << "Error - EngineMultiRobot::setOptions - 'timeUnit' is out of range." << std::endl;
            return hresult_t::ERROR_BAD_INPUT;
        }

        // Make sure the dtMax is not out of range
        configHolder_t stepperOptions = boost::get<configHolder_t>(engineOptions.at("stepper"));
        float64_t const & dtMax = boost::get<float64_t>(stepperOptions.at("dtMax"));
        if (SIMULATION_MAX_TIMESTEP < dtMax || dtMax < SIMULATION_MIN_TIMESTEP)
        {
            std::cout << "Error - EngineMultiRobot::setOptions - 'dtMax' option is out of range." << std::endl;
            return hresult_t::ERROR_BAD_INPUT;
        }

        // Make sure successiveIterFailedMax is strictly positive
        uint32_t const & successiveIterFailedMax = boost::get<uint32_t>(stepperOptions.at("successiveIterFailedMax"));
        if (successiveIterFailedMax < 1)
        {
            std::cout << "Error - EngineMultiRobot::setOptions - 'successiveIterFailedMax' must be strictly positive." << std::endl;
            return hresult_t::ERROR_BAD_INPUT;
        }

        // Make sure the selected ode solver is available and instantiate it
        std::string const & odeSolver = boost::get<std::string>(stepperOptions.at("odeSolver"));
        if (STEPPERS.find(odeSolver) == STEPPERS.end())
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
        || sensorsUpdatePeriod > SIMULATION_MAX_TIMESTEP
        || (EPS < controllerUpdatePeriod && controllerUpdatePeriod < SIMULATION_MIN_TIMESTEP)
        || controllerUpdatePeriod > SIMULATION_MAX_TIMESTEP)
        {
            std::cout << "Error - EngineMultiRobot::setOptions - Cannot simulate a discrete system with update period smaller than "
                      << SIMULATION_MIN_TIMESTEP << "s or larger than " << SIMULATION_MAX_TIMESTEP << "s. "
                      << "Increase period or switch to continuous mode by setting period to zero." << std::endl;
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
        float64_t const & frictionStictionVel =
            boost::get<float64_t>(contactsOptions.at("frictionStictionVel"));
        if (frictionStictionVel < 0.0)
        {
            std::cout << "Error - EngineMultiRobot::setOptions - The contacts option 'frictionStictionVel' must be positive." << std::endl;
            return hresult_t::ERROR_BAD_INPUT;
        }
        float64_t const & frictionStictionRatio =
            boost::get<float64_t>(contactsOptions.at("frictionStictionRatio"));
        if (frictionStictionRatio < 0.0)
        {
            std::cout << "Error - EngineMultiRobot::setOptions - The contacts option 'frictionStictionRatio' must be positive." << std::endl;
            return hresult_t::ERROR_BAD_INPUT;
        }
        float64_t const & contactsTransitionEps =
            boost::get<float64_t>(contactsOptions.at("transitionEps"));
        if (contactsTransitionEps < 0.0)
        {
            std::cout << "Error - EngineMultiRobot::setOptions - The contacts option 'transitionEps' must be positive." << std::endl;
            return hresult_t::ERROR_BAD_INPUT;
        }

        // Make sure the joints options are fine
        configHolder_t jointsOptions = boost::get<configHolder_t>(engineOptions.at("joints"));
        float64_t const & jointsTransitionPositionEps =
            boost::get<float64_t>(jointsOptions.at("transitionPositionEps"));
        if (jointsTransitionPositionEps < EPS)
        {
            std::cout << "Error - EngineMultiRobot::setOptions - The joints option 'transitionPositionEps' must be strictly positive." << std::endl;
            return hresult_t::ERROR_BAD_INPUT;
        }
        float64_t const & jointsTransitionVelocityEps =
            boost::get<float64_t>(jointsOptions.at("transitionVelocityEps"));
        if (jointsTransitionVelocityEps < EPS)
        {
            std::cout << "Error - EngineMultiRobot::setOptions - The joints option 'transitionVelocityEps' must be strictly positive." << std::endl;
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

    std::vector<std::string> EngineMultiRobot::getSystemsNames(void) const
    {
        std::vector<std::string> systemsNames;
        systemsNames.reserve(systemsDataHolder_.size());
        std::transform(systemsDataHolder_.begin(), systemsDataHolder_.end(),
                       std::back_inserter(systemsNames),
                       [](auto const & sys) -> std::string
                       {
                           return sys.name;
                       });
        return systemsNames;
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

    bool_t const & EngineMultiRobot::getIsSimulationRunning(void) const
    {
        return isSimulationRunning_;
    }

    // ========================================================
    // ================ Core physics utilities ================
    // ========================================================

    template<template<typename> class F = type_identity>
    stateSplitRef_t<F> splitStateImpl(std::vector<systemDataHolder_t> const & systemsData,
                                      typename F<vectorN_t>::type & val)
    {
        stateSplitRef_t<F> valSplit;

        uint8_t const nSystems = systemsData.size();
        valSplit.first.reserve(nSystems);
        valSplit.second.reserve(nSystems);

        uint32_t xIdx = 0U;
        for (auto const & system : systemsData)
        {
            int32_t const & nq = system.robot->nq();
            int32_t const & nv = system.robot->nv();
            int32_t const & nx = system.robot->nx();
            valSplit.first.emplace_back(val.segment(xIdx, nq));
            valSplit.second.emplace_back(val.segment(xIdx + nq,  nv));
            xIdx += nx;
        }

        return valSplit;
    }

    stateSplitRef_t<std::add_const> EngineMultiRobot::splitState(vectorN_t const & val) const
    {
        return splitStateImpl<std::add_const>(systemsDataHolder_, val);
    }

    stateSplitRef_t<> EngineMultiRobot::splitState(vectorN_t & val) const
    {
        return splitStateImpl<>(systemsDataHolder_, val);
    }

    vectorN_t EngineMultiRobot::normalizeState(vectorN_t xCat) const
    {
        auto xSplit = splitState(xCat);
        auto systemIt = systemsDataHolder_.begin();
        auto qSplitIt = xSplit.first.begin();
        for ( ; systemIt != systemsDataHolder_.end(); systemIt++, qSplitIt++)
        {
            pinocchio::normalize(systemIt->robot->pncModel_, *qSplitIt);
        }
        return xCat;
    }

    void EngineMultiRobot::syncStepperStateWithSystems(void)
    {
        auto xSplit = splitState(stepperState_.x);
        auto dxdtSplit = splitState(stepperState_.dxdt);

        auto systemIt = systemsDataHolder_.begin();
        auto qSplitIt = xSplit.first.begin();
        auto vSplitIt = xSplit.second.begin();
        auto qDotSplitIt = dxdtSplit.first.begin();
        auto aSplitIt = dxdtSplit.second.begin();
        for ( ; systemIt != systemsDataHolder_.end();
             systemIt++, qSplitIt++, vSplitIt++, qDotSplitIt++, aSplitIt++)
        {
            *qSplitIt = systemIt->state.q;
            *vSplitIt = systemIt->state.v;
            *qDotSplitIt = systemIt->state.qDot;
            *aSplitIt = systemIt->state.a;
        }
    }

    void EngineMultiRobot::syncSystemsStateWithStepper(void)
    {
        auto xSplit = splitState(stepperState_.x);
        auto dxdtSplit = splitState(stepperState_.dxdt);

        auto systemIt = systemsDataHolder_.begin();
        auto qSplitIt = xSplit.first.begin();
        auto vSplitIt = xSplit.second.begin();
        auto qDotSplitIt = dxdtSplit.first.begin();
        auto aSplitIt = dxdtSplit.second.begin();
        for ( ; systemIt != systemsDataHolder_.end();
             systemIt++, qSplitIt++, vSplitIt++, qDotSplitIt++, aSplitIt++)
        {
            systemIt->state.q = *qSplitIt;
            systemIt->state.v = *vSplitIt;
            systemIt->state.qDot = *qDotSplitIt;
            systemIt->state.a = *aSplitIt;
        }
    }

    void EngineMultiRobot::computeForwardKinematics(systemDataHolder_t       & system,
                                                    vectorN_t          const & q,
                                                    vectorN_t          const & v,
                                                    vectorN_t          const & a)
    {
        pinocchio::forwardKinematics(system.robot->pncModel_, system.robot->pncData_, q, v, a);
        pinocchio::updateFramePlacements(system.robot->pncModel_, system.robot->pncData_);
        pinocchio::updateGeometryPlacements(system.robot->pncModel_,
                                            system.robot->pncData_,
                                            system.robot->pncGeometryModel_,
                                            *system.robot->pncGeometryData_);
        pinocchio::computeCollisions(system.robot->pncGeometryModel_,
                                     *system.robot->pncGeometryData_,
                                     false);  // Update collision results
        pinocchio::computeDistances(system.robot->pncGeometryModel_,
                                    *system.robot->pncGeometryData_); // Update distance results.
    }

    pinocchio::Force EngineMultiRobot::computeContactDynamicsAtBody(systemDataHolder_t const & system,
                                                                    int32_t            const & collisionPairIdx) const
    {
        // TODO: It is assumed that the ground is flat. For now ground profile is not supported
        // with body collision. Nevertheless it should not be to hard to generated a collision
        // object simply by sampling points on the profile.

        // Get the frame and joint indices
        uint32_t const & geometryIdx = system.robot->pncGeometryModel_.collisionPairs[collisionPairIdx].first;
        uint32_t const & parentFrameIdx = system.robot->pncGeometryModel_.geometryObjects[geometryIdx].parentFrame;
        uint32_t const & parentJointIdx =  system.robot->pncModel_.frames[parentFrameIdx].parent;

        // Extract collision and distance results
        hpp::fcl::CollisionResult const & collisionResult = system.robot->pncGeometryData_->collisionResults[collisionPairIdx];
        hpp::fcl::DistanceResult const & distanceResult = system.robot->pncGeometryData_->distanceResults[collisionPairIdx];

        if (collisionResult.isCollision())
        {
            // Extract the contact information.
            // Note that there is always a single contact point while computing the collision
            // between two shape objects, for instance convex geometry and box primitive.
            vector3_t const & nGround = - distanceResult.normal;  // Normal of the ground in world (at least in the case of box primitive ground)
            float64_t const & depth = distanceResult.min_distance;
            pinocchio::SE3 posContactInWorld = pinocchio::SE3::Identity();
            posContactInWorld.translation() = distanceResult.nearest_points[1]; //  Point at the surface of the ground (it is hill-defined for the body geometry since it depends on its type)

            /* Make sure the collision computation didn't failed. If it happends the
               norm of the distance normal close to zero. It so, just assume there is
               no collision at all. */
            if (nGround.norm() < EPS)
            {
                return pinocchio::Force::Zero();
            }

            // Compute the linear velocity of the contact point in world frame
            pinocchio::Motion const & motionJointLocal = system.robot->pncData_.v[parentJointIdx];
            pinocchio::SE3 const & transformJointFrameInWorld = system.robot->pncData_.oMi[parentJointIdx];
            pinocchio::SE3 const transformJointFrameInContact = posContactInWorld.actInv(transformJointFrameInWorld);
            vector3_t const vContactInWorld = transformJointFrameInContact.act(motionJointLocal).linear();

            // Compute the ground reaction force at contact point in world frame
            pinocchio::Force const fextAtContactInGlobal = computeContactDynamics(system, nGround, depth, vContactInWorld);

            // Move the force at parent frame location
            pinocchio::Force const fextAtParentJointInLocal = transformJointFrameInContact.actInv(fextAtContactInGlobal);

            return fextAtParentJointInLocal;
        }
        else
        {
            return pinocchio::Force::Zero();
        }
    }

    pinocchio::Force EngineMultiRobot::computeContactDynamicsAtFrame(systemDataHolder_t const & system,
                                                                     int32_t            const & frameIdx) const
    {
        /* Returns the external force in the contact frame.
           It must then be converted into a force onto the parent joint.
           /!\ Note that the contact dynamics depends only on kinematics data. /!\ */

        // Get the pose of the frame wrt the world
        pinocchio::SE3 const & transformFrameInWorld = system.robot->pncData_.oMf[frameIdx];

        // Compute the ground normal and penetration depth at the contact point
        vector3_t const & posFrame = transformFrameInWorld.translation();
        auto ground = engineOptions_->world.groundProfile(posFrame);
        float64_t const & zGround = std::get<float64_t>(ground);
        vector3_t & nGround = std::get<vector3_t>(ground);
        nGround.normalize();  // Make sure the ground normal is normalized
        float64_t const depth = (posFrame(2) - zGround) * nGround(2); // First-order projection (exact assuming flat surface)

        // Only compute the ground reaction force if the penetration depth is positive
        if (depth < 0.0)
        {
            // Compute the linear velocity of the contact point in world frame.
            // Note that for Pinocchio >= v2.4.4, it is possible to specify directly the desired reference frame.
            vector3_t const motionFrameLocal = pinocchio::getFrameVelocity(
                system.robot->pncModel_, system.robot->pncData_, frameIdx).linear();
            matrix3_t const & rotFrame = transformFrameInWorld.rotation();
            vector3_t const vContactInWorld = rotFrame * motionFrameLocal;

            // Compute the ground reaction force in world frame
            pinocchio::Force const fextAtContactInGlobal = computeContactDynamics(system, nGround, depth, vContactInWorld);

            // Apply the force at the origin of the parent joint frame
            pinocchio::Force const fextAtParentJointInLocal = convertForceGlobalFrameToJoint(
                system.robot->pncModel_, system.robot->pncData_, frameIdx, fextAtContactInGlobal);

            return fextAtParentJointInLocal;
        }
        else
        {
            // Not in contact with the ground, thus no force applied
            return pinocchio::Force::Zero();
        }
    }

    pinocchio::Force EngineMultiRobot::computeContactDynamics(systemDataHolder_t const & system,
                                                              vector3_t          const & nGround,
                                                              float64_t          const & depth,
                                                              vector3_t          const & vContactInWorld) const
    {
        // Initialize the contact force
        vector3_t fextInWorld;

        if (depth < 0.0)
        {
            // Extract some proxies
            contactOptions_t const & contactOptions_ = engineOptions_->contacts;

            // Compute the penetration speed
            float64_t const vDepth = vContactInWorld.dot(nGround);

            // Compute normal force
            float64_t fextNormal = 0.0;
            if (vDepth < 0.0)
            {
                fextNormal -= contactOptions_.damping * vDepth;
            }
            fextNormal -= contactOptions_.stiffness * depth;
            fextInWorld = fextNormal * nGround;

            // Compute friction forces
            vector3_t const vTangential = vContactInWorld - vDepth * nGround;
            float64_t const vNorm = vTangential.norm();

            float64_t frictionCoeff = 0.0;
            if (vNorm > contactOptions_.frictionStictionVel)
            {
                if (vNorm < (1.0 + contactOptions_.frictionStictionRatio) * contactOptions_.frictionStictionVel)
                {
                    float64_t const vRatio = vNorm / contactOptions_.frictionStictionVel;
                    frictionCoeff = (contactOptions_.frictionDry * ((1.0 + contactOptions_.frictionStictionRatio) - vRatio)
                                  - contactOptions_.frictionViscous * (1.0 - vRatio)) / contactOptions_.frictionStictionRatio;
                }
                else
                {
                    frictionCoeff = contactOptions_.frictionViscous;
                }
            }
            else
            {
                float64_t const vRatio = vNorm / contactOptions_.frictionStictionVel;
                frictionCoeff = contactOptions_.frictionDry * vRatio;
            }
            float64_t const fextTangential = frictionCoeff * fextNormal;
            fextInWorld += -fextTangential * vTangential;

            // Add blending factor
            if (contactOptions_.transitionEps > EPS)
            {
                float64_t const blendingFactor = -depth / contactOptions_.transitionEps;
                float64_t const blendingLaw = std::tanh(2 * blendingFactor);
                fextInWorld *= blendingLaw;
            }
        }
        else
        {
            fextInWorld.setZero();
        }

        return {fextInWorld, vector3_t::Zero()};
    }

    void EngineMultiRobot::computeCommand(systemDataHolder_t                & system,
                                          float64_t                   const & t,
                                          Eigen::Ref<vectorN_t const> const & q,
                                          Eigen::Ref<vectorN_t const> const & v,
                                          vectorN_t                         & u)
    {
        // Reinitialize the external forces
        u.setZero();

        // Command the command
        system.controller->computeCommand(t, q, v, u);
    }

    void EngineMultiRobot::computeInternalDynamics(systemDataHolder_t                & system,
                                                   float64_t                   const & t,
                                                   Eigen::Ref<vectorN_t const> const & q,
                                                   Eigen::Ref<vectorN_t const> const & v,
                                                   vectorN_t                         & u) const
    {
        // Reinitialize the internal effort vector
        u.setZero();

        // Compute the user-defined internal dynamics
        system.controller->internalDynamics(t, q, v, u);

        // Define some proxies
        auto const & jointOptions = engineOptions_->joints;
        pinocchio::Model const & pncModel = system.robot->pncModel_;

        // Enforce the position limit for the rigid joints only (TODO: Add support of spherical and planar joints)
        if (system.robot->mdlOptions_->joints.enablePositionLimit)
        {
            vectorN_t const & positionLimitMin = system.robot->getPositionLimitMin();
            vectorN_t const & positionLimitMax = system.robot->getPositionLimitMax();
            for (int32_t const & rigidIdx : system.robot->getRigidJointsModelIdx())
            {
                uint32_t const & positionIdx = pncModel.joints[rigidIdx].idx_q();
                uint32_t const & velocityIdx = pncModel.joints[rigidIdx].idx_v();
                int32_t const & jointDof = pncModel.joints[rigidIdx].nq();
                for (int32_t j = 0; j < jointDof; j++)
                {
                    float64_t const & qJoint = q[positionIdx + j];
                    float64_t const & vJoint = v[velocityIdx + j];
                    float64_t const & qJointMin = positionLimitMin[positionIdx + j];
                    float64_t const & qJointMax = positionLimitMax[positionIdx + j];

                    float64_t qJointError = 0.0;
                    float64_t vJointError = 0.0;
                    if (qJoint > qJointMax)
                    {
                        qJointError = qJoint - qJointMax;
                        vJointError = std::max(vJoint, 0.0);
                    }
                    else if (qJoint < qJointMin)
                    {
                        qJointError = qJoint - qJointMin;
                        vJointError = std::min(vJoint, 0.0);
                    }
                    float64_t const blendingFactor = std::abs(qJointError - jointOptions.transitionPositionEps *
                        std::tanh(qJointError / jointOptions.transitionPositionEps));
                    float64_t const forceJoint = - jointOptions.boundStiffness * qJointError
                                                 - jointOptions.boundDamping * blendingFactor * vJointError;

                    u[velocityIdx + j] += forceJoint;
                }
            }
        }

        // Enforce the velocity limit (do not support spherical joints)
        if (system.robot->mdlOptions_->joints.enableVelocityLimit)
        {
            vectorN_t const & velocityLimitMax = system.robot->getVelocityLimit();
            for (int32_t const & rigidIdx : system.robot->getRigidJointsModelIdx())
            {
                uint32_t const & velocityIdx = pncModel.joints[rigidIdx].idx_v();
                uint32_t const & jointDof = pncModel.joints[rigidIdx].nq();
                for (uint32_t j = 0; j < jointDof; j++)
                {
                    float64_t const & vJoint = v[velocityIdx + j];
                    float64_t const & vJointMin = -velocityLimitMax[velocityIdx + j];
                    float64_t const & vJointMax = velocityLimitMax[velocityIdx + j];

                    float64_t vJointError = 0.0;
                    if (vJoint > vJointMax)
                    {
                        vJointError = vJoint - vJointMax;
                    }
                    else if (vJoint < vJointMin)
                    {
                        vJointError = vJoint - vJointMin;
                    }
                    float64_t forceJoint = - jointOptions.boundDamping *
                        std::tanh(vJointError / jointOptions.transitionVelocityEps);

                    u[velocityIdx + j] += forceJoint;
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

    void EngineMultiRobot::computeExternalForces(systemDataHolder_t                & system,
                                                 float64_t                   const & t,
                                                 Eigen::Ref<vectorN_t const> const & q,
                                                 Eigen::Ref<vectorN_t const> const & v,
                                                 forceVector_t                     & fext)
    {
        // Compute the forces at contact points
        std::vector<int32_t> const & contactFramesIdx = system.robot->getContactFramesIdx();
        for (uint32_t i=0; i < contactFramesIdx.size(); i++)
        {
            // Compute force at the given contact frame.
            int32_t const & frameIdx = contactFramesIdx[i];
            pinocchio::Force const fextLocal = computeContactDynamicsAtFrame(system, frameIdx);

            // Apply the force at the origin of the parent joint frame, in local joint frame
            int32_t const & parentJointIdx = system.robot->pncModel_.frames[frameIdx].parent;
            fext[parentJointIdx] += fextLocal;

            // Convert contact force from the global frame to the local frame to store it in contactForces_
            pinocchio::SE3 const & transformContactInJoint = system.robot->pncModel_.frames[frameIdx].placement;
            system.robot->contactForces_[i] = transformContactInJoint.act(fextLocal);
        }

        // Compute the force at collision bodies
        std::vector<int32_t> const & collisionBodiesIdx = system.robot->getCollisionBodiesIdx();
        for (uint32_t i=0; i < collisionBodiesIdx.size(); i++)
        {
            // Compute force at the given collision body.
            // It returns the force applied at the origin of the parent joint frame, in global frame
            int32_t const & frameIdx = collisionBodiesIdx[i];
            pinocchio::Force const fextLocal = computeContactDynamicsAtBody(system, i);

            // Apply the force at the origin of the parent joint frame, in local joint frame
            int32_t const & parentJointIdx = system.robot->pncModel_.frames[frameIdx].parent;
            fext[parentJointIdx] += fextLocal;
        }

        // Add the effect of user-defined external impulse forces
        auto forcesImpulseActiveIt = system.forcesImpulseActive.begin();
        auto forcesImpulseIt = system.forcesImpulse.begin();
        for ( ; forcesImpulseIt != system.forcesImpulse.end() ;
             forcesImpulseActiveIt++, forcesImpulseIt++)
        {
            /* Do not check if the force is active at this point.
               This is managed at stepper level to get around the
               ambiguous t- versus t+. */
            if (*forcesImpulseActiveIt)
            {
                int32_t const & frameIdx = forcesImpulseIt->frameIdx;
                int32_t const & parentJointIdx = system.robot->pncModel_.frames[frameIdx].parent;
                pinocchio::Force const & F = forcesImpulseIt->F;

                fext[parentJointIdx] += convertForceGlobalFrameToJoint(
                    system.robot->pncModel_, system.robot->pncData_, frameIdx, F);
            }
        }

        // Add the effect of user-defined external force profiles
        for (auto const & forceProfile : system.forcesProfile)
        {
            int32_t const & frameIdx = forceProfile.frameIdx;
            int32_t const & parentJointIdx = system.robot->pncModel_.frames[frameIdx].parent;
            forceProfileFunctor_t const & forceFct = forceProfile.forceFct;

            pinocchio::Force const force = forceFct(t, q, v);
            fext[parentJointIdx] += convertForceGlobalFrameToJoint(
                system.robot->pncModel_, system.robot->pncData_, frameIdx, force);
        }
    }

    void EngineMultiRobot::computeInternalForces(float64_t                       const & t,
                                                 stateSplitRef_t<std::add_const> const & xSplit)
    {
        for (auto & forceCoupling : forcesCoupling_)
        {
            int32_t const & systemIdx1 = forceCoupling.systemIdx1;
            int32_t const & systemIdx2 = forceCoupling.systemIdx2;
            int32_t const & frameIdx1 = forceCoupling.frameIdx1;
            int32_t const & frameIdx2 = forceCoupling.frameIdx2;
            forceCouplingFunctor_t const & forceFct = forceCoupling.forceFct;

            systemDataHolder_t & system1 = systemsDataHolder_[systemIdx1];
            systemDataHolder_t & system2 = systemsDataHolder_[systemIdx2];
            Eigen::Ref<vectorN_t const> const & q1 = xSplit.first[systemIdx1];
            Eigen::Ref<vectorN_t const> const & v1 = xSplit.second[systemIdx1];
            Eigen::Ref<vectorN_t const> const & q2 = xSplit.first[systemIdx2];
            Eigen::Ref<vectorN_t const> const & v2 = xSplit.second[systemIdx2];
            forceVector_t & fext1 = system1.state.fExternal;
            forceVector_t & fext2 = system2.state.fExternal;

            pinocchio::Force const force = forceFct(t, q1, v1, q2, v2);
            int32_t const & parentJointIdx1 = system1.robot->pncModel_.frames[frameIdx1].parent;
            fext1[parentJointIdx1] += convertForceGlobalFrameToJoint(
                system1.robot->pncModel_, system1.robot->pncData_, frameIdx1, force);
            int32_t const & parentJointIdx2 = system2.robot->pncModel_.frames[frameIdx2].parent;
            // Move force from frame1 to frame2 to apply it to the second system.
            pinocchio::SE3 offset(
                matrix3_t::Identity(),
                system1.robot->pncData_.oMf[frameIdx2].translation() - system1.robot->pncData_.oMf[frameIdx1].translation());
            fext2[parentJointIdx2] += convertForceGlobalFrameToJoint(
                system2.robot->pncModel_, system2.robot->pncData_, frameIdx2, -offset.act(force));
        }
    }

    void EngineMultiRobot::computeAllForces(float64_t                       const & t,
                                            stateSplitRef_t<std::add_const> const & xSplit)
    {
        // Reinitialize the external forces
        for (auto & system : systemsDataHolder_)
        {
            for (pinocchio::Force & fext_i : system.state.fExternal)
            {
                fext_i.setZero();
            }
        }

        // Compute the internal forces
        computeInternalForces(t, xSplit);

        // Compute each individual system dynamics
        auto systemIt = systemsDataHolder_.begin();
        auto qSplitIt = xSplit.first.begin();
        auto vSplitIt = xSplit.second.begin();
        for ( ; systemIt != systemsDataHolder_.end();
             systemIt++, qSplitIt++, vSplitIt++)
        {
            // Define some proxies
            Eigen::Ref<vectorN_t const> const & q = *qSplitIt;
            Eigen::Ref<vectorN_t const> const & v = *vSplitIt;
            forceVector_t & fext = systemIt->state.fExternal;

            // Compute the external contact forces.
            computeExternalForces(*systemIt, t, q, v, fext);
        }
    }

    void EngineMultiRobot::computeSystemDynamics(float64_t const & t,
                                                 vectorN_t const & xCat,
                                                 vectorN_t       & dxdtCat)
    {
        /* - Note that the position of the free flyer is in world frame,
             whereas the velocities and accelerations are relative to
             the parent body frame.
           - Note that dxdtCat is a different preallocated buffer for
             each midpoint of the stepper, so there is 6 different
             buffers in the case of the Dopri5. The actually stepper
             buffer never directly use by this method. */

        /* Project vector onto Lie group, to prevent numerical error due
           to Runge-Kutta internal steps being based on vector algebra. */
        vectorN_t const xN = normalizeState(xCat);

        // Split the input state and derivative (by reference)
        auto xSplit = splitState(xN);
        auto dxdtSplit = splitState(dxdtCat);

        // Update the kinematics of each system
        auto systemIt = systemsDataHolder_.begin();
        auto qSplitIt = xSplit.first.begin();
        auto vSplitIt = xSplit.second.begin();
        for ( ; systemIt != systemsDataHolder_.end();
             systemIt++, qSplitIt++, vSplitIt++)
        {
            // Define some proxies
            Eigen::Ref<vectorN_t const> const & q = *qSplitIt;
            Eigen::Ref<vectorN_t const> const & v = *vSplitIt;
            vectorN_t const & aPrev = systemIt->statePrev.a;

            computeForwardKinematics(*systemIt, q, v, aPrev);
        }

        /* Compute the internal and external forces applied on every systems.
           Note that one must call this method BEFORE updating the sensors
           since the force sensor measurements rely on robot_->contactForces_. */
        computeAllForces(t, xSplit);

        // Compute each individual system dynamics
        systemIt = systemsDataHolder_.begin();
        qSplitIt = xSplit.first.begin();
        vSplitIt = xSplit.second.begin();
        auto qDotSplitIt = dxdtSplit.first.begin();
        auto aSplitIt = dxdtSplit.second.begin();
        for ( ; systemIt != systemsDataHolder_.end();
             systemIt++, qSplitIt++, vSplitIt++, qDotSplitIt++, aSplitIt++)
        {
            // Define some proxies
            Eigen::Ref<vectorN_t const> const & q = *qSplitIt;
            Eigen::Ref<vectorN_t const> const & v = *vSplitIt;
            Eigen::Ref<vectorN_t> & qDot = *qDotSplitIt;
            Eigen::Ref<vectorN_t> & a = *aSplitIt;
            vectorN_t & u = systemIt->state.u;
            vectorN_t & uCommand = systemIt->state.uCommand;
            vectorN_t & uMotor = systemIt->state.uMotor;
            vectorN_t & uInternal = systemIt->state.uInternal;
            forceVector_t & fext = systemIt->state.fExternal;
            vectorN_t const & aPrev = systemIt->statePrev.a;
            vectorN_t const & uMotorPrev = systemIt->statePrev.uMotor;

            /* Update the sensor data if necessary (only for infinite update frequency).
               Note that it is impossible to have access to the current accelerations
               and efforts since they depend on the sensor values themselves. */
            if (engineOptions_->stepper.sensorsUpdatePeriod < SIMULATION_MIN_TIMESTEP)
            {
                systemIt->robot->setSensorsData(t, q, v, aPrev, uMotorPrev);
            }

            /* Update the controller command if necessary (only for infinite update frequency).
               Make sure that the sensor state has been updated beforehand. */
            if (engineOptions_->stepper.controllerUpdatePeriod < SIMULATION_MIN_TIMESTEP)
            {
                computeCommand(*systemIt, t, q, v, uCommand);
            }

            /* Compute the actual motor effort.
               Note that it is impossible to have access to the current accelerations. */
            systemIt->robot->computeMotorsEfforts(t, q, v, aPrev, uCommand);
            uMotor = systemIt->robot->getMotorsEfforts();

            /* Compute the internal dynamics.
               Make sure that the sensor state has been updated beforehand since
               the user-defined internal dynamics may rely on it. */
            computeInternalDynamics(*systemIt, t, q, v, uInternal);

            // Compute the total effort vector
            u = uInternal;
            for (auto const & motor : systemIt->robot->getMotors())
            {
                int32_t const & motorIdx = motor->getIdx();
                int32_t const & motorVelocityIdx = motor->getJointVelocityIdx();
                u[motorVelocityIdx] += uMotor[motorIdx];
            }

            // Compute the dynamics
            a = computeAcceleration(*systemIt, q, v, u, fext);

            // Project the derivative in state space (only if moving forward in time)
            float64_t const dt = t - stepperState_.tPrev;
            if (dt >= STEPPER_MIN_TIMESTEP)
            {
                computePositionDerivative(systemIt->robot->pncModel_, q, v, qDot, dt);
            }
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

    vectorN_t EngineMultiRobot::computeAcceleration(systemDataHolder_t & system,
                                                    Eigen::Ref<vectorN_t const> const & q,
                                                    Eigen::Ref<vectorN_t const> const & v,
                                                    vectorN_t const & u,
                                                    forceVector_t const & fext)
    {
        if (system.robot->hasConstraint())
        {
            // Compute kinematic constraints.
            system.robot->computeConstraints(q, v);

            // Project external forces from cartesian space to joint space.
            vectorN_t uTotal = u;
            matrixN_t jointJacobian = matrixN_t::Zero(6, system.robot->pncModel_.nv);
            for (int i = 1; i < system.robot->pncModel_.njoints; i++)
            {
                jointJacobian.setZero();
                pinocchio::getJointJacobian(system.robot->pncModel_,
                                            system.robot->pncData_,
                                            i,
                                            pinocchio::LOCAL,
                                            jointJacobian);
                uTotal += jointJacobian.transpose() * fext[i].toVector();
            }
            // Compute non-linear effects.
            pinocchio::nonLinearEffects(system.robot->pncModel_,
                                        system.robot->pncData_,
                                        q,
                                        v);

            // Compute inertia matrix, adding rotor inertia.
            pinocchio::crba(system.robot->pncModel_,
                            system.robot->pncData_,
                            q);
            for (int i = 1; i < system.robot->pncModel_.njoints; i++)
            {
                // Only support inertia for 1DoF joints.
                if (system.robot->pncModel_.joints[i].nv() == 1)
                {
                    int jointIdx = system.robot->pncModel_.joints[i].idx_v();
                    system.robot->pncData_.M(jointIdx, jointIdx) +=
                            system.robot->pncModel_.rotorInertia[jointIdx];
                }
            }

            // Call forward dynamics.
            return pinocchio::forwardDynamics(system.robot->pncModel_,
                                              system.robot->pncData_,
                                              q,
                                              v,
                                              uTotal,
                                              system.robot->getConstraintsJacobian(),
                                              system.robot->getConstraintsDrift(),
                                              CONSTRAINT_INVERSION_DAMPING,
                                              false);
        }
        else
        {
            // No kinematic constraint: run aba algorithm.
            return pinocchio_overload::aba(
                system.robot->pncModel_, system.robot->pncData_, q, v, u, fext);
        }
    }
}
