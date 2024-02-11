#include <cmath>
#include <ctime>
#include <unordered_set>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <fstream>
#include <streambuf>

#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/spatial/inertia.hpp"            // `pinocchio::Inertia`
#include "pinocchio/spatial/force.hpp"              // `pinocchio::Force`
#include "pinocchio/spatial/se3.hpp"                // `pinocchio::SE3`
#include "pinocchio/spatial/explog.hpp"             // `pinocchio::exp3`, `pinocchio::log3`
#include "pinocchio/spatial/explog-quaternion.hpp"  // `pinocchio::quaternion::log3`
#include "pinocchio/multibody/visitor.hpp"          // `pinocchio::fusion::JointUnaryVisitorBase`
#include "pinocchio/multibody/joint/joint-model-base.hpp"  // `pinocchio::JointModelBase`
#include "pinocchio/algorithm/center-of-mass.hpp"          // `pinocchio::getComFromCrba`
#include "pinocchio/algorithm/frames.hpp"                  // `pinocchio::getFrameVelocity`
#include "pinocchio/algorithm/jacobian.hpp"                // `pinocchio::getJointJacobian`
#include "pinocchio/algorithm/energy.hpp"                  // `pinocchio::computePotentialEnergy`
#include "pinocchio/algorithm/joint-configuration.hpp"     // `pinocchio::normalize`
#include "pinocchio/algorithm/geometry.hpp"                // `pinocchio::computeCollisions`

#include "H5Cpp.h"
#include "json/json.h"

#include "jiminy/core/telemetry/fwd.h"
#include "jiminy/core/hardware/fwd.h"
#include "jiminy/core/utilities/pinocchio.h"
#include "jiminy/core/utilities/json.h"
#include "jiminy/core/io/file_device.h"
#include "jiminy/core/io/serialization.h"
#include "jiminy/core/telemetry/telemetry_sender.h"
#include "jiminy/core/telemetry/telemetry_data.h"
#include "jiminy/core/telemetry/telemetry_recorder.h"
#include "jiminy/core/constraints/abstract_constraint.h"
#include "jiminy/core/constraints/joint_constraint.h"
#include "jiminy/core/constraints/frame_constraint.h"
#include "jiminy/core/hardware/abstract_motor.h"
#include "jiminy/core/hardware/abstract_sensor.h"
#include "jiminy/core/robot/robot.h"
#include "jiminy/core/robot/pinocchio_overload_algorithms.h"
#include "jiminy/core/control/abstract_controller.h"
#include "jiminy/core/control/controller_functor.h"
#include "jiminy/core/solver/constraint_solvers.h"
#include "jiminy/core/stepper/abstract_stepper.h"
#include "jiminy/core/stepper/euler_explicit_stepper.h"
#include "jiminy/core/stepper/runge_kutta_dopri_stepper.h"
#include "jiminy/core/stepper/runge_kutta4_stepper.h"

#include "jiminy/core/engine/engine_multi_robot.h"

namespace jiminy
{
    inline constexpr uint32_t INIT_ITERATIONS{4U};
    inline constexpr uint32_t PGS_MAX_ITERATIONS{100U};

    EngineMultiRobot::EngineMultiRobot() noexcept :
    generator_{std::seed_seq{std::random_device{}()}},
    telemetrySender_{std::make_unique<TelemetrySender>()},
    telemetryData_{std::make_shared<TelemetryData>()},
    telemetryRecorder_{std::make_unique<TelemetryRecorder>()}
    {
        // Initialize the configuration options to the default.
        setOptions(getDefaultEngineOptions());
    }

    // Cannot be default in the header since some types are incomplete at this point
    EngineMultiRobot::~EngineMultiRobot() = default;

    void EngineMultiRobot::addSystem(const std::string & systemName,
                                     std::shared_ptr<Robot> robot,
                                     std::shared_ptr<AbstractController> controller,
                                     const AbortSimulationFunction & callback)
    {
        // Make sure that no simulation is running
        if (isSimulationRunning_)
        {
            THROW_ERROR(bad_control_flow,
                        "Simulation already running. Stop it before adding new system.");
        }

        if (!robot)
        {
            THROW_ERROR(std::invalid_argument, "Robot unspecified.");
        }

        if (!robot->getIsInitialized())
        {
            THROW_ERROR(bad_control_flow, "Robot not initialized.");
        }

        if (!controller)
        {
            THROW_ERROR(std::invalid_argument, "Controller unspecified.");
        }

        if (!controller->getIsInitialized())
        {
            THROW_ERROR(bad_control_flow, "Controller not initialized.");
        }

        auto robot_controller = controller->robot_.lock();
        if (!robot_controller)
        {
            THROW_ERROR(std::invalid_argument, "Controller's robot expired or unset.");
        }

        if (robot != robot_controller)
        {
            THROW_ERROR(std::invalid_argument, "Controller not initialized for specified robot.");
        }

        // TODO: Check that the callback function is working as expected
        // FIXME: remove explicit constructor call when moving to C++20
        systems_.emplace_back(System{systemName, robot, controller, callback});
        systemDataVec_.resize(systems_.size());
    }

    void EngineMultiRobot::addSystem(const std::string & systemName,
                                     std::shared_ptr<Robot> robot,
                                     const AbortSimulationFunction & callback)
    {
        // Make sure an actual robot is specified
        if (!robot)
        {
            THROW_ERROR(std::invalid_argument, "Robot unspecified.");
        }

        // Make sure the robot is properly initialized
        if (!robot->getIsInitialized())
        {
            THROW_ERROR(std::invalid_argument, "Robot not initialized.");
        }

        // When using several robots the robots names are specified
        // as a circumfix in the log, for consistency they must all
        // have a name
        if (systems_.size() && systemName == "")
        {
            THROW_ERROR(std::invalid_argument, "All systems but the first one must have a name.");
        }

        // Check if a system with the same name already exists
        auto systemIt = std::find_if(systems_.begin(),
                                     systems_.end(),
                                     [&systemName](const auto & sys)
                                     { return (sys.name == systemName); });
        if (systemIt != systems_.end())
        {
            THROW_ERROR(std::invalid_argument,
                        "System with name '",
                        systemName,
                        "'has already been added to the engine.");
        }

        // Make sure none of the existing system is referring to the same robot
        systemIt = std::find_if(systems_.begin(),
                                systems_.end(),
                                [&robot](const auto & sys) { return (sys.robot == robot); });
        if (systemIt != systems_.end())
        {
            THROW_ERROR(std::invalid_argument,
                        "System '",
                        systemIt->name,
                        "' already referring to this robot.");
        }

        // Create and initialize a controller doing nothing
        auto noop = [](double /* t */,
                       const Eigen::VectorXd & /* q */,
                       const Eigen::VectorXd & /* v */,
                       const SensorMeasurementTree & /* sensorMeasurements */,
                       Eigen::VectorXd & /* out */)
        {
            // Empty on purpose
        };
        auto controller = std::make_shared<FunctionalController<>>(noop, noop);
        controller->initialize(robot);

        return addSystem(systemName, robot, controller, callback);
    }

    void EngineMultiRobot::removeSystem(const std::string & systemName)
    {
        // Make sure that no simulation is running
        if (isSimulationRunning_)
        {
            THROW_ERROR(bad_control_flow,
                        "Simulation already running. Stop it before removing a system.");
        }

        /* Remove every coupling forces involving the system.
           Note that it is already checking that the system exists. */
        removeCouplingForces(systemName);

        // Get system index
        std::ptrdiff_t systemIndex = getSystemIndex(systemName);

        // Update the systems' indices for the remaining coupling forces
        for (auto & force : couplingForces_)
        {
            if (force.systemIndex1 > systemIndex)
            {
                --force.systemIndex1;
            }
            if (force.systemIndex2 > systemIndex)
            {
                --force.systemIndex2;
            }
        }

        // Remove the system from the list
        systems_.erase(systems_.begin() + systemIndex);
        systemDataVec_.erase(systemDataVec_.begin() + systemIndex);
    }

    void EngineMultiRobot::setController(const std::string & systemName,
                                         std::shared_ptr<AbstractController> controller)
    {
        // Make sure that no simulation is running
        if (isSimulationRunning_)
        {
            THROW_ERROR(bad_control_flow,
                        "Simulation already running. Stop it before setting a new "
                        "controller for a system.");
        }

        // Make sure that the controller is initialized
        if (!controller->getIsInitialized())
        {
            THROW_ERROR(bad_control_flow, "Controller not initialized.");
        }

        auto robot_controller = controller->robot_.lock();
        if (!robot_controller)
        {
            THROW_ERROR(bad_control_flow, "Controller's robot expired or unset.");
        }

        // Get the system for which to set the controller
        System & system = getSystem(systemName);

        if (system.robot != robot_controller)
        {
            THROW_ERROR(std::invalid_argument,
                        "Controller not initialized for robot associated with specified system.");
        }

        // Set the controller
        system.controller = controller;
    }

    void EngineMultiRobot::registerCouplingForce(const std::string & systemName1,
                                                 const std::string & systemName2,
                                                 const std::string & frameName1,
                                                 const std::string & frameName2,
                                                 const CouplingForceFunction & forceFunc)
    {
        // Make sure that no simulation is running
        if (isSimulationRunning_)
        {
            THROW_ERROR(bad_control_flow,
                        "Simulation already running. Stop it before adding coupling forces.");
        }

        // Get system and frame indices
        const std::ptrdiff_t systemIndex1 = getSystemIndex(systemName1);
        const std::ptrdiff_t systemIndex2 = getSystemIndex(systemName2);
        const pinocchio::FrameIndex frameIndex1 =
            getFrameIndex(systems_[systemIndex1].robot->pinocchioModel_, frameName1);
        const pinocchio::FrameIndex frameIndex2 =
            getFrameIndex(systems_[systemIndex2].robot->pinocchioModel_, frameName2);

        // Make sure it is not coupling the exact same frame
        if (systemIndex1 == systemIndex2 && frameIndex1 == frameIndex2)
        {
            THROW_ERROR(std::invalid_argument,
                        "A coupling force must involve two different frames.");
        }

        couplingForces_.emplace_back(systemName1,
                                     systemIndex1,
                                     systemName2,
                                     systemIndex2,
                                     frameName1,
                                     frameIndex1,
                                     frameName2,
                                     frameIndex2,
                                     forceFunc);
    }

    void EngineMultiRobot::registerViscoelasticCouplingForce(const std::string & systemName1,
                                                             const std::string & systemName2,
                                                             const std::string & frameName1,
                                                             const std::string & frameName2,
                                                             const Vector6d & stiffness,
                                                             const Vector6d & damping,
                                                             double alpha)
    {
        // Make sure that the input arguments are valid
        if ((stiffness.array() < 0.0).any() || (damping.array() < 0.0).any())
        {
            THROW_ERROR(std::invalid_argument,
                        "Stiffness and damping parameters must be positive.");
        }

        // Get system and frame indices
        System * system1 = &getSystem(systemName1);
        System * system2 = &getSystem(systemName2);
        pinocchio::FrameIndex frameIndex1 =
            getFrameIndex(system1->robot->pinocchioModel_, frameName1);
        pinocchio::FrameIndex frameIndex2 =
            getFrameIndex(system2->robot->pinocchioModel_, frameName2);

        // Allocate memory
        double angle{0.0};
        Eigen::Matrix3d rot12{}, rotJLog12{}, rotJExp12{}, rotRef12{}, omega{};
        Eigen::Vector3d rotLog12{}, pos12{}, posLocal12{}, fLin{}, fAng{};

        auto forceFunc = [=](double /* t */,
                             const Eigen::VectorXd & /* q1 */,
                             const Eigen::VectorXd & /* v1 */,
                             const Eigen::VectorXd & /* q2 */,
                             const Eigen::VectorXd & /* v2 */) mutable -> pinocchio::Force
        {
            /* One must keep track of system pointers and frames indices internally and update
               them at reset since the systems may have changed between simulations. Note that
               `isSimulationRunning_` is false when called for the first time in `start` method
               before locking the robot, so it is the right time to update those proxies. */
            if (!isSimulationRunning_)
            {
                system1 = &getSystem(systemName1);
                system2 = &getSystem(systemName2);
                frameIndex1 = getFrameIndex(system1->robot->pinocchioModel_, frameName1);
                frameIndex2 = getFrameIndex(system2->robot->pinocchioModel_, frameName2);
            }

            // Get the frames positions and velocities in world
            const pinocchio::SE3 & oMf1{system1->robot->pinocchioData_.oMf[frameIndex1]};
            const pinocchio::SE3 & oMf2{system2->robot->pinocchioData_.oMf[frameIndex2]};
            const pinocchio::Motion oVf1{getFrameVelocity(system1->robot->pinocchioModel_,
                                                          system1->robot->pinocchioData_,
                                                          frameIndex1,
                                                          pinocchio::LOCAL_WORLD_ALIGNED)};
            const pinocchio::Motion oVf2{getFrameVelocity(system2->robot->pinocchioModel_,
                                                          system2->robot->pinocchioData_,
                                                          frameIndex2,
                                                          pinocchio::LOCAL_WORLD_ALIGNED)};

            // Compute intermediary quantities
            rot12.noalias() = oMf1.rotation().transpose() * oMf2.rotation();
            rotLog12 = pinocchio::log3(rot12, angle);
            if (angle < 0.95 * M_PI)
            {
                THROW_ERROR(std::runtime_error,
                            "Relative angle between reference frames of viscoelastic "
                            "coupling must be smaller than 0.95 * pi.");
            }
            pinocchio::Jlog3(angle, rotLog12, rotJLog12);
            fAng = stiffness.tail<3>().array() * rotLog12.array();
            rotLog12 *= alpha;
            pinocchio::Jexp3(rotLog12, rotJExp12);
            rotRef12.noalias() = oMf1.rotation() * pinocchio::exp3(rotLog12);
            pos12 = oMf2.translation() - oMf1.translation();
            posLocal12.noalias() = rotRef12.transpose() * pos12;
            fLin = stiffness.head<3>().array() * posLocal12.array();
            omega.noalias() = alpha * rotJExp12 * rotJLog12;

            /* Compute the relative velocity. The application point is the "linear"
               interpolation between the frames placement with alpha ratio. */
            const pinocchio::Motion oVf12{oVf2 - oVf1};
            pinocchio::Motion velLocal12{
                rotRef12.transpose() *
                    (oVf12.linear() + pos12.cross(oVf2.angular() - alpha * oVf12.angular())),
                rotRef12.transpose() * oVf12.angular()};

            // Compute the coupling force acting on frame 2
            pinocchio::Force f{};
            f.linear() = damping.head<3>().array() * velLocal12.linear().array();
            f.angular() = (1.0 - alpha) * f.linear().cross(posLocal12);
            f.angular().array() += damping.tail<3>().array() * velLocal12.angular().array();
            f.linear() += fLin;
            f.linear() = rotRef12 * f.linear();
            f.angular() = rotRef12 * f.angular();
            f.angular() -= oMf2.rotation() * omega.colwise().cross(posLocal12).transpose() * fLin;
            f.angular() += oMf1.rotation() * rotJLog12 * fAng;

            // Deduce the force acting on frame 1 from action-reaction law
            f.angular() += pos12.cross(f.linear());

            return f;
        };

        registerCouplingForce(systemName1, systemName2, frameName1, frameName2, forceFunc);
    }

    void EngineMultiRobot::registerViscoelasticCouplingForce(const std::string & systemName,
                                                             const std::string & frameName1,
                                                             const std::string & frameName2,
                                                             const Vector6d & stiffness,
                                                             const Vector6d & damping,
                                                             double alpha)
    {
        return registerViscoelasticCouplingForce(
            systemName, systemName, frameName1, frameName2, stiffness, damping, alpha);
    }

    void EngineMultiRobot::registerViscoelasticDirectionalCouplingForce(
        const std::string & systemName1,
        const std::string & systemName2,
        const std::string & frameName1,
        const std::string & frameName2,
        double stiffness,
        double damping,
        double restLength)
    {
        // Make sure that the input arguments are valid
        if (stiffness < 0.0 || damping < 0.0)
        {
            THROW_ERROR(std::invalid_argument,
                        "The stiffness and damping parameters must be positive.");
        }

        // Get system and frame indices
        System * system1 = &getSystem(systemName1);
        System * system2 = &getSystem(systemName2);
        pinocchio::FrameIndex frameIndex1 =
            getFrameIndex(system1->robot->pinocchioModel_, frameName1);
        pinocchio::FrameIndex frameIndex2 =
            getFrameIndex(system2->robot->pinocchioModel_, frameName2);

        auto forceFunc = [=](double /* t */,
                             const Eigen::VectorXd & /* q1 */,
                             const Eigen::VectorXd & /* v1 */,
                             const Eigen::VectorXd & /* q2 */,
                             const Eigen::VectorXd & /* v2 */) mutable -> pinocchio::Force
        {
            /* One must keep track of system pointers and frames indices internally and update
               them at reset since the systems may have changed between simulations. Note that
               `isSimulationRunning_` is false when called for the first time in `start` method
               before locking the robot, so it is the right time to update those proxies. */
            if (!isSimulationRunning_)
            {
                system1 = &getSystem(systemName1);
                system2 = &getSystem(systemName2);
                frameIndex1 = getFrameIndex(system1->robot->pinocchioModel_, frameName1);
                frameIndex2 = getFrameIndex(system2->robot->pinocchioModel_, frameName2);
            }

            // Get the frames positions and velocities in world
            const pinocchio::SE3 & oMf1{system1->robot->pinocchioData_.oMf[frameIndex1]};
            const pinocchio::SE3 & oMf2{system2->robot->pinocchioData_.oMf[frameIndex2]};
            const pinocchio::Motion oVf1{getFrameVelocity(system1->robot->pinocchioModel_,
                                                          system1->robot->pinocchioData_,
                                                          frameIndex1,
                                                          pinocchio::LOCAL_WORLD_ALIGNED)};
            const pinocchio::Motion oVf2{getFrameVelocity(system2->robot->pinocchioModel_,
                                                          system2->robot->pinocchioData_,
                                                          frameIndex2,
                                                          pinocchio::LOCAL_WORLD_ALIGNED)};

            // Compute the linear force coupling them
            Eigen::Vector3d dir12{oMf2.translation() - oMf1.translation()};
            const double length{dir12.norm()};
            auto vel12 = oVf2.linear() - oVf1.linear();
            if (length > EPS)
            {
                dir12 /= length;
                auto vel12Proj = vel12.dot(dir12);
                return {(stiffness * (length - restLength) + damping * vel12Proj) * dir12,
                        Eigen::Vector3d::Zero()};
            }
            /* The direction between frames is ill-defined, so applying force in the direction
               of the linear velocity instead. */
            return {damping * vel12, Eigen::Vector3d::Zero()};
        };

        registerCouplingForce(systemName1, systemName2, frameName1, frameName2, forceFunc);
    }

    void EngineMultiRobot::registerViscoelasticDirectionalCouplingForce(
        const std::string & systemName,
        const std::string & frameName1,
        const std::string & frameName2,
        double stiffness,
        double damping,
        double restLength)
    {
        return registerViscoelasticDirectionalCouplingForce(
            systemName, systemName, frameName1, frameName2, stiffness, damping, restLength);
    }

    void EngineMultiRobot::removeCouplingForces(const std::string & systemName1,
                                                const std::string & systemName2)
    {
        // Make sure that no simulation is running
        if (isSimulationRunning_)
        {
            THROW_ERROR(bad_control_flow,
                        "Simulation already running. Stop it before removing coupling forces.");
        }

        // Make sure that the systems exist
        getSystem(systemName1);
        getSystem(systemName2);

        // Remove corresponding coupling forces if any
        couplingForces_.erase(std::remove_if(couplingForces_.begin(),
                                             couplingForces_.end(),
                                             [&systemName1, &systemName2](const auto & force) {
                                                 return (force.systemName1 == systemName1 &&
                                                         force.systemName2 == systemName2);
                                             }),
                              couplingForces_.end());
    }

    void EngineMultiRobot::removeCouplingForces(const std::string & systemName)
    {
        // Make sure that no simulation is running
        if (isSimulationRunning_)
        {
            THROW_ERROR(bad_control_flow,
                        "Simulation already running. Stop it before removing coupling forces.");
        }

        // Make sure that the system exists
        getSystem(systemName);

        // Remove corresponding coupling forces if any
        couplingForces_.erase(std::remove_if(couplingForces_.begin(),
                                             couplingForces_.end(),
                                             [&systemName](const auto & force) {
                                                 return (force.systemName1 == systemName ||
                                                         force.systemName2 == systemName);
                                             }),
                              couplingForces_.end());
    }

    void EngineMultiRobot::removeCouplingForces()
    {
        // Make sure that no simulation is running
        if (isSimulationRunning_)
        {
            THROW_ERROR(bad_control_flow,
                        "Simulation already running. Stop it before removing coupling forces.");
        }

        couplingForces_.clear();
    }

    const CouplingForceVector & EngineMultiRobot::getCouplingForces() const
    {
        return couplingForces_;
    }

    void EngineMultiRobot::removeAllForces()
    {
        removeCouplingForces();
        removeImpulseForces();
        removeProfileForces();
    }

    void EngineMultiRobot::configureTelemetry()
    {
        if (systems_.empty())
        {
            THROW_ERROR(bad_control_flow, "No system added to the engine.");
        }

        if (!isTelemetryConfigured_)
        {
            // Initialize the engine-specific telemetry sender
            telemetrySender_->configure(telemetryData_, ENGINE_TELEMETRY_NAMESPACE);

            auto systemIt = systems_.begin();
            auto systemDataIt = systemDataVec_.begin();
            auto energyIt = energy_.begin();
            for (; systemIt != systems_.end(); ++systemIt, ++systemDataIt, ++energyIt)
            {
                // Generate the log fieldnames
                systemDataIt->logPositionFieldnames =
                    addCircumfix(systemIt->robot->getLogPositionFieldnames(),
                                 systemIt->name,
                                 {},
                                 TELEMETRY_FIELDNAME_DELIMITER);
                systemDataIt->logVelocityFieldnames =
                    addCircumfix(systemIt->robot->getLogVelocityFieldnames(),
                                 systemIt->name,
                                 {},
                                 TELEMETRY_FIELDNAME_DELIMITER);
                systemDataIt->logAccelerationFieldnames =
                    addCircumfix(systemIt->robot->getLogAccelerationFieldnames(),
                                 systemIt->name,
                                 {},
                                 TELEMETRY_FIELDNAME_DELIMITER);
                systemDataIt->logForceExternalFieldnames =
                    addCircumfix(systemIt->robot->getLogForceExternalFieldnames(),
                                 systemIt->name,
                                 {},
                                 TELEMETRY_FIELDNAME_DELIMITER);
                systemDataIt->logCommandFieldnames =
                    addCircumfix(systemIt->robot->getLogCommandFieldnames(),
                                 systemIt->name,
                                 {},
                                 TELEMETRY_FIELDNAME_DELIMITER);
                systemDataIt->logMotorEffortFieldnames =
                    addCircumfix(systemIt->robot->getLogMotorEffortFieldnames(),
                                 systemIt->name,
                                 {},
                                 TELEMETRY_FIELDNAME_DELIMITER);
                systemDataIt->logEnergyFieldname =
                    addCircumfix("energy", systemIt->name, {}, TELEMETRY_FIELDNAME_DELIMITER);

                // Register variables to the telemetry senders
                if (engineOptions_->telemetry.enableConfiguration)
                {
                    telemetrySender_->registerVariable(systemDataIt->logPositionFieldnames,
                                                       systemDataIt->state.q);
                }
                if (engineOptions_->telemetry.enableVelocity)
                {
                    telemetrySender_->registerVariable(systemDataIt->logVelocityFieldnames,
                                                       systemDataIt->state.v);
                }
                if (engineOptions_->telemetry.enableAcceleration)
                {
                    telemetrySender_->registerVariable(systemDataIt->logAccelerationFieldnames,
                                                       systemDataIt->state.a);
                }
                if (engineOptions_->telemetry.enableForceExternal)
                {
                    for (std::size_t i = 1; i < systemDataIt->state.fExternal.size(); ++i)
                    {
                        const auto & fext = systemDataIt->state.fExternal[i].toVector();
                        for (uint8_t j = 0; j < 6U; ++j)
                        {
                            telemetrySender_->registerVariable(
                                systemDataIt->logForceExternalFieldnames[(i - 1) * 6U + j],
                                &fext[j]);
                        }
                    }
                }
                if (engineOptions_->telemetry.enableCommand)
                {
                    telemetrySender_->registerVariable(systemDataIt->logCommandFieldnames,
                                                       systemDataIt->state.command);
                }
                if (engineOptions_->telemetry.enableMotorEffort)
                {
                    telemetrySender_->registerVariable(systemDataIt->logMotorEffortFieldnames,
                                                       systemDataIt->state.uMotor);
                }
                if (engineOptions_->telemetry.enableEnergy)
                {
                    telemetrySender_->registerVariable(systemDataIt->logEnergyFieldname,
                                                       &(*energyIt));
                }
                systemIt->controller->configureTelemetry(telemetryData_, systemIt->name);
                systemIt->robot->configureTelemetry(telemetryData_, systemIt->name);
            }
        }

        isTelemetryConfigured_ = true;
    }

    void EngineMultiRobot::updateTelemetry()
    {
        // Compute the total energy of the system
        auto systemIt = systems_.begin();
        auto energyIt = energy_.begin();
        for (; systemIt != systems_.end(); ++systemIt, ++energyIt)
        {
            *energyIt = systemIt->robot->pinocchioData_.kinetic_energy +
                        systemIt->robot->pinocchioData_.potential_energy;
        }

        // Update system-specific telemetry variables
        for (auto & system : systems_)
        {
            system.controller->updateTelemetry();
            system.robot->updateTelemetry();
        }

        // Update engine-specific telemetry variables
        telemetrySender_->updateValues();

        // Flush the telemetry internal state
        telemetryRecorder_->flushSnapshot(stepperState_.t);
    }

    void EngineMultiRobot::reset(bool resetRandomNumbers, bool removeAllForce)
    {
        // Make sure the simulation is properly stopped
        if (isSimulationRunning_)
        {
            stop();
        }

        // Clear log data buffer
        logData_.reset();

        // Reset the dynamic force register if requested
        if (removeAllForce)
        {
            for (auto & systemData : systemDataVec_)
            {
                systemData.impulseForces.clear();
                systemData.impulseForceBreakpoints.clear();
                systemData.isImpulseForceActiveVec.clear();
                systemData.profileForces.clear();
            }
            // FIXME: replaced `std::get<N>` by placeholder `_` when moving to C++26 (P2169R4)
            stepperUpdatePeriod_ =
                std::get<1>(isGcdIncluded(engineOptions_->stepper.sensorsUpdatePeriod,
                                          engineOptions_->stepper.controllerUpdatePeriod));
        }

        // Reset the random number generators
        if (resetRandomNumbers)
        {
            VectorX<uint32_t> seedSeq = engineOptions_->stepper.randomSeedSeq;
            generator_.seed(std::seed_seq(seedSeq.data(), seedSeq.data() + seedSeq.size()));
        }

        // Reset the internal state of the robot and controller
        for (auto & system : systems_)
        {
            system.robot->reset(generator_);
            system.controller->reset();
        }

        // Clear system state buffers, since the robot kinematic may change
        for (auto & systemData : systemDataVec_)
        {
            systemData.state.clear();
            systemData.statePrev.clear();
        }

        isTelemetryConfigured_ = false;
    }

    struct ForwardKinematicsAccelerationStep :
    public pinocchio::fusion::JointUnaryVisitorBase<ForwardKinematicsAccelerationStep>
    {
        typedef boost::fusion::vector<pinocchio::Data &, const Eigen::VectorXd &> ArgsType;

        template<typename JointModel>
        static void algo(const pinocchio::JointModelBase<JointModel> & jmodel,
                         pinocchio::JointDataBase<typename JointModel::JointDataDerived> & jdata,
                         pinocchio::Data & data,
                         const Eigen::VectorXd & a)
        {
            pinocchio::JointIndex jointIndex = jmodel.id();
            data.a[jointIndex] = jdata.c() + data.v[jointIndex].cross(jdata.v());
            data.a[jointIndex] += jdata.S() * jmodel.jointVelocitySelector(a);
        }
    };

    /// \details This method is optimized to avoid redundant computations.
    ///
    /// \see See `pinocchio::computeAllTerms` for reference:
    ///      https://github.com/stack-of-tasks/pinocchio/blob/a1df23c2f183d84febdc2099e5fbfdbd1fc8018b/src/algorithm/compute-all-terms.hxx
    ///
    /// Copyright (c) 2014-2020, CNRS
    /// Copyright (c) 2018-2020, INRIA
    void computeExtraTerms(System & system, const SystemData & systemData, ForceVector & fExt)
    {
        const pinocchio::Model & model = system.robot->pinocchioModel_;
        pinocchio::Data & data = system.robot->pinocchioData_;

        // Compute the potential and kinematic energy of the system
        pinocchio_overload::computeKineticEnergy(
            model, data, systemData.state.q, systemData.state.v, false);
        pinocchio::computePotentialEnergy(model, data);

        /* Update manually the subtree (apparent) inertia, since it is only computed by crba, which
           is doing more computation than necessary. It will be used here for computing the
           centroidal kinematics, and used later for joint bounds dynamics. Note that, by doing all
           the computations here instead of 'computeForwardKinematics', we are doing the assumption
           that it is varying slowly enough to consider it constant during one integration step. */
        if (!system.robot->hasConstraints())
        {
            for (int i = 1; i < model.njoints; ++i)
            {
                data.Ycrb[i] = model.inertias[i];
            }
            for (int jointIndex = model.njoints - 1; jointIndex > 0; --jointIndex)
            {
                const pinocchio::JointIndex parentJointIndex = model.parents[jointIndex];
                if (parentJointIndex > 0)
                {
                    data.Ycrb[parentJointIndex] +=
                        data.liMi[jointIndex].act(data.Ycrb[jointIndex]);
                }
            }
        }

        /* Neither 'aba' nor 'forwardDynamics' are computing simultaneously the actual joint
           accelerations, joint forces and body forces, so it must be done separately:
           - 1st step: computing the accelerations based on ForwardKinematic algorithm
           - 2nd step: computing the forces based on RNEA algorithm */

        /* Compute the true joint acceleration and the one due to the lone gravity field, then
           the spatial momenta and the total internal and external forces acting on each body. */
        data.h[0].setZero();
        fExt[0].setZero();
        data.f[0].setZero();
        data.a[0].setZero();
        data.a_gf[0] = -model.gravity;
        for (int jointIndex = 1; jointIndex < model.njoints; ++jointIndex)
        {
            ForwardKinematicsAccelerationStep::run(
                model.joints[jointIndex],
                data.joints[jointIndex],
                typename ForwardKinematicsAccelerationStep::ArgsType(data, systemData.state.a));

            const pinocchio::JointIndex parentJointIndex = model.parents[jointIndex];
            data.a_gf[jointIndex] = data.a[jointIndex];
            data.a[jointIndex] += data.liMi[jointIndex].actInv(data.a[parentJointIndex]);
            data.a_gf[jointIndex] += data.liMi[jointIndex].actInv(data.a_gf[parentJointIndex]);

            model.inertias[jointIndex].__mult__(data.v[jointIndex], data.h[jointIndex]);

            model.inertias[jointIndex].__mult__(data.a[jointIndex], fExt[jointIndex]);
            data.f[jointIndex] = data.v[jointIndex].cross(data.h[jointIndex]);
            fExt[jointIndex] += data.f[jointIndex];
            data.f[jointIndex] += model.inertias[jointIndex] * data.a_gf[jointIndex];
            data.f[jointIndex] -= systemData.state.fExternal[jointIndex];
        }
        for (int jointIndex = model.njoints - 1; jointIndex > 0; --jointIndex)
        {
            const pinocchio::JointIndex parentJointIndex = model.parents[jointIndex];
            fExt[parentJointIndex] += data.liMi[jointIndex].act(fExt[jointIndex]);
            data.h[parentJointIndex] += data.liMi[jointIndex].act(data.h[jointIndex]);
            if (parentJointIndex > 0)
            {
                data.f[parentJointIndex] += data.liMi[jointIndex].act(data.f[jointIndex]);
            }
        }

        // Compute the position and velocity of the center of mass of each subtree
        for (int jointIndex = 0; jointIndex < model.njoints; ++jointIndex)
        {
            if (jointIndex > 0)
            {
                data.com[jointIndex] = data.Ycrb[jointIndex].lever();
            }
            data.vcom[jointIndex].noalias() = data.h[jointIndex].linear() / data.mass[jointIndex];
        }
        data.com[0] = data.liMi[1].act(data.com[1]);

        // Compute centroidal dynamics and its derivative
        data.hg = data.h[0];
        data.hg.angular() += data.hg.linear().cross(data.com[0]);
        data.dhg = fExt[0];
        data.dhg.angular() += data.dhg.linear().cross(data.com[0]);
    }

    void computeAllExtraTerms(std::vector<System> & systems,
                              const vector_aligned_t<SystemData> & systemDataVec,
                              vector_aligned_t<ForceVector> & f)
    {
        auto systemIt = systems.begin();
        auto systemDataIt = systemDataVec.begin();
        auto fIt = f.begin();
        for (; systemIt != systems.end(); ++systemIt, ++systemDataIt, ++fIt)
        {
            computeExtraTerms(*systemIt, *systemDataIt, *fIt);
        }
    }

    void syncAccelerationsAndForces(
        const System & system, ForceVector & contactForces, ForceVector & f, MotionVector & a)
    {
        for (std::size_t i = 0; i < system.robot->getContactFrameNames().size(); ++i)
        {
            contactForces[i] = system.robot->contactForces_[i];
        }

        for (int i = 0; i < system.robot->pinocchioModel_.njoints; ++i)
        {
            f[i] = system.robot->pinocchioData_.f[i];
            a[i] = system.robot->pinocchioData_.a[i];
        }
    }

    void syncAllAccelerationsAndForces(const std::vector<System> & systems,
                                       vector_aligned_t<ForceVector> & contactForces,
                                       vector_aligned_t<ForceVector> & f,
                                       vector_aligned_t<MotionVector> & a)
    {
        std::vector<System>::const_iterator systemIt = systems.begin();
        auto contactForceIt = contactForces.begin();
        auto fPrevIt = f.begin();
        auto aPrevIt = a.begin();
        for (; systemIt != systems.end(); ++systemIt, ++contactForceIt, ++fPrevIt, ++aPrevIt)
        {
            syncAccelerationsAndForces(*systemIt, *contactForceIt, *fPrevIt, *aPrevIt);
        }
    }

    void EngineMultiRobot::start(
        const std::map<std::string, Eigen::VectorXd> & qInit,
        const std::map<std::string, Eigen::VectorXd> & vInit,
        const std::optional<std::map<std::string, Eigen::VectorXd>> & aInit)
    {
        // Make sure that no simulation is running
        if (isSimulationRunning_)
        {
            THROW_ERROR(bad_control_flow,
                        "Simulation already running. Stop it before starting again.");
        }

        if (systems_.empty())
        {
            THROW_ERROR(bad_control_flow,
                        "No system to simulate. Please add one before starting a simulation.");
        }

        const std::size_t nSystems = systems_.size();
        if (qInit.size() != nSystems || vInit.size() != nSystems)
        {
            THROW_ERROR(std::invalid_argument,
                        "The number of initial configurations and velocities must "
                        "match the number of systems.");
        }

        // Check the dimension of the initial state associated with every system and order them
        std::vector<Eigen::VectorXd> qSplit;
        std::vector<Eigen::VectorXd> vSplit;
        qSplit.reserve(nSystems);
        vSplit.reserve(nSystems);
        for (const auto & system : systems_)
        {
            auto qInitIt = qInit.find(system.name);
            auto vInitIt = vInit.find(system.name);
            if (qInitIt == qInit.end() || vInitIt == vInit.end())
            {
                THROW_ERROR(std::invalid_argument,
                            "System '",
                            system.name,
                            "'does not have an initial configuration or velocity.");
            }

            const Eigen::VectorXd & q = qInitIt->second;
            const Eigen::VectorXd & v = vInitIt->second;
            if (q.rows() != system.robot->nq() || v.rows() != system.robot->nv())
            {
                THROW_ERROR(std::invalid_argument,
                            "The dimension of the initial configuration or velocity is "
                            "inconsistent with model size for system '",
                            system.name,
                            "'.");
            }

            // Make sure the initial state is normalized
            const bool isValid = isPositionValid(
                system.robot->pinocchioModel_, q, std::numeric_limits<float>::epsilon());
            if (!isValid)
            {
                THROW_ERROR(std::invalid_argument,
                            "The initial configuration is not consistent with the types of joints "
                            "of the model for system '",
                            system.name,
                            "'.");
            }

            /* Check that the initial configuration is not out-of-bounds if appropriate.
               Note that EPS allows to be very slightly out-of-bounds, which may occurs because of
               rounding errors. */
            if ((system.robot->modelOptions_->joints.enablePositionLimit &&
                 (contactModel_ == ContactModelType::CONSTRAINT) &&
                 ((EPS < q.array() - system.robot->getPositionLimitMax().array()).any() ||
                  (EPS < system.robot->getPositionLimitMin().array() - q.array()).any())))
            {
                THROW_ERROR(std::invalid_argument,
                            "The initial configuration is out-of-bounds for system '",
                            system.name,
                            "'.");
            }

            // Check that the initial velocity is not out-of-bounds
            if ((system.robot->modelOptions_->joints.enableVelocityLimit &&
                 (system.robot->getVelocityLimit().array() < v.array().abs()).any()))
            {
                THROW_ERROR(std::invalid_argument,
                            "The initial velocity is out-of-bounds for system '",
                            system.name,
                            "'.");
            }

            /* Make sure the configuration is normalized (as double), since normalization is
               checked using float accuracy rather than double to circumvent float/double casting
               than may occurs because of Python bindings. */
            Eigen::VectorXd qNormalized = q;
            pinocchio::normalize(system.robot->pinocchioModel_, qNormalized);

            qSplit.emplace_back(qNormalized);
            vSplit.emplace_back(v);
        }

        std::vector<Eigen::VectorXd> aSplit;
        aSplit.reserve(nSystems);
        if (aInit)
        {
            // Check the dimension of the initial acceleration associated with every system
            if (aInit->size() != nSystems)
            {
                THROW_ERROR(std::invalid_argument,
                            "If specified, the number of initial accelerations "
                            "must match the number of systems.");
            }

            for (const auto & system : systems_)
            {
                auto aInitIt = aInit->find(system.name);
                if (aInitIt == aInit->end())
                {
                    THROW_ERROR(std::invalid_argument,
                                "System '",
                                system.name,
                                "'does not have an initial acceleration.");
                }

                const Eigen::VectorXd & a = aInitIt->second;
                if (a.rows() != system.robot->nv())
                {
                    THROW_ERROR(std::invalid_argument,
                                "The dimension of the initial acceleration is inconsistent with "
                                "model size for system '",
                                system.name,
                                "'.");
                }

                aSplit.emplace_back(a);
            }
        }
        else
        {
            // Zero acceleration by default
            std::transform(vSplit.begin(),
                           vSplit.end(),
                           std::back_inserter(aSplit),
                           [](const auto & v) -> Eigen::VectorXd
                           { return Eigen::VectorXd::Zero(v.size()); });
        }

        for (auto & system : systems_)
        {
            for (const auto & sensorGroupItem : system.robot->getSensors())
            {
                for (const auto & sensor : sensorGroupItem.second)
                {
                    if (!sensor->getIsInitialized())
                    {
                        THROW_ERROR(bad_control_flow,
                                    "At least a sensor of a robot is not initialized.");
                    }
                }
            }

            for (const auto & motor : system.robot->getMotors())
            {
                if (!motor->getIsInitialized())
                {
                    THROW_ERROR(bad_control_flow,
                                "At least a motor of a robot is not initialized.");
                }
            }
        }

        /* Call reset if the internal state of the engine is not clean. Not doing it systematically
           gives the opportunity to the user to customize the system by resetting first the engine
           manually and then to alter the system before starting a simulation, e.g. to change the
           inertia of a specific body. */
        if (isTelemetryConfigured_)
        {
            reset(false, false);
        }

        // Reset the internal state of the robot and controller
        auto systemIt = systems_.begin();
        auto systemDataIt = systemDataVec_.begin();
        for (; systemIt != systems_.end(); ++systemIt, ++systemDataIt)
        {
            // Propagate the user-defined gravity at robot level
            systemIt->robot->pinocchioModelOrig_.gravity = engineOptions_->world.gravity;
            systemIt->robot->pinocchioModel_.gravity = engineOptions_->world.gravity;

            /* Reinitialize the system state buffers, since the robot kinematic may have changed.
               For example, it may happens if one activates or deactivates the flexibility between
               two successive simulations. */
            systemDataIt->state.initialize(*(systemIt->robot));
            systemDataIt->statePrev.initialize(*(systemIt->robot));
            systemDataIt->successiveSolveFailed = 0U;
        }

        // Initialize the ode solver
        auto systemOde = [this](double t,
                                const std::vector<Eigen::VectorXd> & q,
                                const std::vector<Eigen::VectorXd> & v,
                                std::vector<Eigen::VectorXd> & a) -> void
        {
            this->computeSystemsDynamics(t, q, v, a);
        };
        std::vector<const Robot *> robots;
        robots.reserve(nSystems);
        std::transform(systems_.begin(),
                       systems_.end(),
                       std::back_inserter(robots),
                       [](const auto & sys) -> const Robot * { return sys.robot.get(); });
        if (engineOptions_->stepper.odeSolver == "runge_kutta_dopri5")
        {
            stepper_ = std::unique_ptr<AbstractStepper>(
                new RungeKuttaDOPRIStepper(systemOde,
                                           robots,
                                           engineOptions_->stepper.tolAbs,
                                           engineOptions_->stepper.tolRel));
        }
        else if (engineOptions_->stepper.odeSolver == "runge_kutta_4")
        {
            stepper_ = std::unique_ptr<AbstractStepper>(new RungeKutta4Stepper(systemOde, robots));
        }
        else if (engineOptions_->stepper.odeSolver == "euler_explicit")
        {
            stepper_ =
                std::unique_ptr<AbstractStepper>(new EulerExplicitStepper(systemOde, robots));
        }

        // Initialize the stepper state
        const double t = 0.0;
        stepperState_.reset(SIMULATION_MIN_TIMESTEP, qSplit, vSplit, aSplit);

        // Initialize previous joints forces and accelerations
        contactForcesPrev_.clear();
        contactForcesPrev_.reserve(nSystems);
        fPrev_.clear();
        fPrev_.reserve(nSystems);
        aPrev_.clear();
        aPrev_.reserve(nSystems);
        for (const auto & system : systems_)
        {
            contactForcesPrev_.push_back(system.robot->contactForces_);
            fPrev_.push_back(system.robot->pinocchioData_.f);
            aPrev_.push_back(system.robot->pinocchioData_.a);
        }
        energy_.resize(nSystems, 0.0);

        // Synchronize the individual system states with the global stepper state
        syncSystemsStateWithStepper();

        // Update the frame indices associated with the coupling forces
        for (auto & force : couplingForces_)
        {
            force.frameIndex1 = getFrameIndex(systems_[force.systemIndex1].robot->pinocchioModel_,
                                              force.frameName1);
            force.frameIndex2 = getFrameIndex(systems_[force.systemIndex2].robot->pinocchioModel_,
                                              force.frameName2);
        }

        systemIt = systems_.begin();
        systemDataIt = systemDataVec_.begin();
        for (; systemIt != systems_.end(); ++systemIt, ++systemDataIt)
        {
            // Update the frame indices associated with the impulse forces and force profiles
            for (auto & force : systemDataIt->profileForces)
            {
                force.frameIndex =
                    getFrameIndex(systemIt->robot->pinocchioModel_, force.frameName);
            }
            for (auto & force : systemDataIt->impulseForces)
            {
                force.frameIndex =
                    getFrameIndex(systemIt->robot->pinocchioModel_, force.frameName);
            }

            // Initialize the impulse force breakpoint point iterator
            systemDataIt->impulseForceBreakpointNextIt =
                systemDataIt->impulseForceBreakpoints.begin();

            // Reset the active set of impulse forces
            std::fill(systemDataIt->isImpulseForceActiveVec.begin(),
                      systemDataIt->isImpulseForceActiveVec.end(),
                      false);

            // Activate every force impulse starting at t=0
            auto isImpulseForceActiveIt = systemDataIt->isImpulseForceActiveVec.begin();
            auto impulseForceIt = systemDataIt->impulseForces.begin();
            for (; impulseForceIt != systemDataIt->impulseForces.end();
                 ++isImpulseForceActiveIt, ++impulseForceIt)
            {
                if (impulseForceIt->t < STEPPER_MIN_TIMESTEP)
                {
                    *isImpulseForceActiveIt = true;
                }
            }

            // Compute the forward kinematics for each system
            const Eigen::VectorXd & q = systemDataIt->state.q;
            const Eigen::VectorXd & v = systemDataIt->state.v;
            const Eigen::VectorXd & a = systemDataIt->state.a;
            computeForwardKinematics(*systemIt, q, v, a);

            /* Backup constraint register for fast lookup.
               Internal constraints cannot be added/removed at this point. */
            systemDataIt->constraints = systemIt->robot->getConstraints();

            // Initialize contacts forces in local frame
            const std::vector<pinocchio::FrameIndex> & contactFrameIndices =
                systemIt->robot->getContactFrameIndices();
            systemDataIt->contactFrameForces =
                ForceVector(contactFrameIndices.size(), pinocchio::Force::Zero());
            const std::vector<std::vector<pinocchio::PairIndex>> & collisionPairIndices =
                systemIt->robot->getCollisionPairIndices();
            systemDataIt->collisionBodiesForces.clear();
            systemDataIt->collisionBodiesForces.reserve(collisionPairIndices.size());
            for (const auto & bodyCollisionPairIndices : collisionPairIndices)
            {
                systemDataIt->collisionBodiesForces.emplace_back(bodyCollisionPairIndices.size(),
                                                                 pinocchio::Force::Zero());
            }

            /* Initialize some addition buffers used by impulse contact solver.
               It must be initialized to zero because 'getJointJacobian' will only update non-zero
               coefficients for efficiency. */
            systemDataIt->jointJacobians.resize(
                systemIt->robot->pinocchioModel_.njoints,
                Matrix6Xd::Zero(6, systemIt->robot->pinocchioModel_.nv));

            // Reset the constraints
            systemIt->robot->resetConstraints(q, v);

            /* Set Baumgarte stabilization natural frequency for contact constraints
               Enable all contact constraints by default, it will be disable automatically if not
               in contact. It is useful to start in post-hysteresis state to avoid discontinuities
               at init. */
            systemDataIt->constraints.foreach(
                [&contactModel = contactModel_,
                 &enablePositionLimit = systemIt->robot->modelOptions_->joints.enablePositionLimit,
                 &freq = engineOptions_->contacts.stabilizationFreq](
                    const std::shared_ptr<AbstractConstraintBase> & constraint,
                    ConstraintNodeType node)
                {
                    // Set baumgarte freq for all contact constraints
                    if (node != ConstraintNodeType::USER)
                    {
                        constraint->setBaumgarteFreq(freq);  // It cannot fail
                    }

                    // Enable constraints by default
                    if (contactModel == ContactModelType::CONSTRAINT)
                    {
                        switch (node)
                        {
                        case ConstraintNodeType::BOUNDS_JOINTS:
                            if (!enablePositionLimit)
                            {
                                return;
                            }
                            {
                                auto & jointConstraint =
                                    static_cast<JointConstraint &>(*constraint.get());
                                jointConstraint.setRotationDir(false);
                            }
                            [[fallthrough]];
                        case ConstraintNodeType::CONTACT_FRAMES:
                        case ConstraintNodeType::COLLISION_BODIES:
                            constraint->enable();
                            break;
                        case ConstraintNodeType::USER:
                        default:
                            break;
                        }
                    }
                });

            if (contactModel_ == ContactModelType::SPRING_DAMPER)
            {
                // Make sure that the contact forces are bounded for spring-damper model.
                // TODO: Rather use something like `10 * m * g` instead of a fix threshold.
                double forceMax = 0.0;
                for (std::size_t i = 0; i < contactFrameIndices.size(); ++i)
                {
                    auto & constraint = systemDataIt->constraints.contactFrames[i].second;
                    pinocchio::Force & fextLocal = systemDataIt->contactFrameForces[i];
                    computeContactDynamicsAtFrame(
                        *systemIt, contactFrameIndices[i], constraint, fextLocal);
                    forceMax = std::max(forceMax, fextLocal.linear().norm());
                }

                for (std::size_t i = 0; i < collisionPairIndices.size(); ++i)
                {
                    for (std::size_t j = 0; j < collisionPairIndices[i].size(); ++j)
                    {
                        const pinocchio::PairIndex & collisionPairIndex =
                            collisionPairIndices[i][j];
                        auto & constraint = systemDataIt->constraints.collisionBodies[i][j].second;
                        pinocchio::Force & fextLocal = systemDataIt->collisionBodiesForces[i][j];
                        computeContactDynamicsAtBody(
                            *systemIt, collisionPairIndex, constraint, fextLocal);
                        forceMax = std::max(forceMax, fextLocal.linear().norm());
                    }
                }

                if (forceMax > 1e5)
                {
                    THROW_ERROR(
                        std::invalid_argument,
                        "The initial force exceeds 1e5 for at least one contact point, which is "
                        "forbidden for the sake of numerical stability. Please update the initial "
                        "state.");
                }
            }
        }

        // Lock the robots. At this point, it is no longer possible to change them anymore.
        systemIt = systems_.begin();
        systemDataIt = systemDataVec_.begin();
        for (; systemIt != systems_.end(); ++systemIt, ++systemDataIt)
        {
            systemDataIt->robotLock = systemIt->robot->getLock();
        }

        // Instantiate the desired LCP solver
        systemIt = systems_.begin();
        systemDataIt = systemDataVec_.begin();
        for (; systemIt != systems_.end(); ++systemIt, ++systemDataIt)
        {
            const std::string & constraintSolverType = engineOptions_->constraints.solver;
            switch (CONSTRAINT_SOLVERS_MAP.at(constraintSolverType))
            {
            case ConstraintSolverType::PGS:
                systemDataIt->constraintSolver =
                    std::make_unique<PGSSolver>(&systemIt->robot->pinocchioModel_,
                                                &systemIt->robot->pinocchioData_,
                                                &systemDataIt->constraints,
                                                engineOptions_->contacts.friction,
                                                engineOptions_->contacts.torsion,
                                                engineOptions_->stepper.tolAbs,
                                                engineOptions_->stepper.tolRel,
                                                PGS_MAX_ITERATIONS);
                break;
            case ConstraintSolverType::UNSUPPORTED:
            default:
                break;
            }
        }

        /* Compute the efforts, internal and external forces applied on every systems excluding
           user-specified internal dynamics if any. */
        computeAllTerms(t, qSplit, vSplit);

        // Backup all external forces and internal efforts excluding constraint forces
        vector_aligned_t<ForceVector> fextNoConst;
        std::vector<Eigen::VectorXd> uInternalConst;
        fextNoConst.reserve(nSystems);
        uInternalConst.reserve(nSystems);
        for (const auto & systemData : systemDataVec_)
        {
            fextNoConst.push_back(systemData.state.fExternal);
            uInternalConst.push_back(systemData.state.uInternal);
        }

        /* Solve algebraic coupling between accelerations, sensors and controllers, by
           iterating several times until it (hopefully) converges. */
        bool isFirstIter = true;
        for (uint32_t i = 0; i < INIT_ITERATIONS; ++i)
        {
            systemIt = systems_.begin();
            systemDataIt = systemDataVec_.begin();
            auto fextNoConstIt = fextNoConst.begin();
            auto uInternalConstIt = uInternalConst.begin();
            for (; systemIt != systems_.end();
                 ++systemIt, ++systemDataIt, ++fextNoConstIt, ++uInternalConstIt)
            {
                // Get some system state proxies
                const Eigen::VectorXd & q = systemDataIt->state.q;
                const Eigen::VectorXd & v = systemDataIt->state.v;
                Eigen::VectorXd & a = systemDataIt->state.a;
                Eigen::VectorXd & u = systemDataIt->state.u;
                Eigen::VectorXd & command = systemDataIt->state.command;
                Eigen::VectorXd & uMotor = systemDataIt->state.uMotor;
                Eigen::VectorXd & uInternal = systemDataIt->state.uInternal;
                Eigen::VectorXd & uCustom = systemDataIt->state.uCustom;
                ForceVector & fext = systemDataIt->state.fExternal;

                // Reset the external forces and internal efforts
                fext = *fextNoConstIt;
                uInternal = *uInternalConstIt;

                // Compute dynamics
                a = computeAcceleration(
                    *systemIt, *systemDataIt, q, v, u, fext, !isFirstIter, isFirstIter);

                // Make sure there is no nan at this point
                if ((a.array() != a.array()).any())
                {
                    THROW_ERROR(std::runtime_error,
                                "Impossible to compute the acceleration. Probably a "
                                "subtree has zero inertia along an articulated axis.");
                }

                // Compute all external terms including joints accelerations and forces
                computeAllExtraTerms(systems_, systemDataVec_, fPrev_);

                // Compute the sensor data with the updated effort and acceleration
                systemIt->robot->computeSensorMeasurements(t, q, v, a, uMotor, fext);

                // Compute the actual motor effort
                computeCommand(*systemIt, t, q, v, command);

                // Compute the actual motor effort
                systemIt->robot->computeMotorEfforts(t, q, v, a, command);
                uMotor = systemIt->robot->getMotorEfforts();

                // Compute the internal dynamics
                uCustom.setZero();
                systemIt->controller->internalDynamics(t, q, v, uCustom);

                // Compute the total effort vector
                u = uInternal + uCustom;
                for (const auto & motor : systemIt->robot->getMotors())
                {
                    const std::size_t motorIndex = motor->getIndex();
                    const Eigen::Index motorVelocityIndex = motor->getJointVelocityIndex();
                    u[motorVelocityIndex] += uMotor[motorIndex];
                }
            }
            isFirstIter = false;
        }

        // Update sensor data one last time to take into account the actual motor effort
        systemIt = systems_.begin();
        systemDataIt = systemDataVec_.begin();
        for (; systemIt != systems_.end(); ++systemIt, ++systemDataIt)
        {
            const Eigen::VectorXd & q = systemDataIt->state.q;
            const Eigen::VectorXd & v = systemDataIt->state.v;
            const Eigen::VectorXd & a = systemDataIt->state.a;
            const Eigen::VectorXd & uMotor = systemDataIt->state.uMotor;
            const ForceVector & fext = systemDataIt->state.fExternal;
            systemIt->robot->computeSensorMeasurements(t, q, v, a, uMotor, fext);
        }

        // Backend the updated joint accelerations and forces
        syncAllAccelerationsAndForces(systems_, contactForcesPrev_, fPrev_, aPrev_);

        // Synchronize the global stepper state with the individual system states
        syncStepperStateWithSystems();

        // Initialize the last system states
        for (auto & systemData : systemDataVec_)
        {
            systemData.statePrev = systemData.state;
        }

        // Lock the telemetry. At this point it is not possible to register new variables.
        configureTelemetry();

        // Log systems data
        for (const auto & system : systems_)
        {
            // Backup URDF file
            const std::string telemetryUrdfFile =
                addCircumfix("urdf_file", system.name, {}, TELEMETRY_FIELDNAME_DELIMITER);
            const std::string & urdfFileString = system.robot->getUrdfAsString();
            telemetrySender_->registerConstant(telemetryUrdfFile, urdfFileString);

            // Backup 'has_freeflyer' option
            const std::string telemetrHasFreeflyer =
                addCircumfix("has_freeflyer", system.name, {}, TELEMETRY_FIELDNAME_DELIMITER);
            telemetrySender_->registerConstant(telemetrHasFreeflyer,
                                               toString(system.robot->getHasFreeflyer()));

            // Backup mesh package lookup directories
            const std::string telemetryMeshPackageDirs =
                addCircumfix("mesh_package_dirs", system.name, {}, TELEMETRY_FIELDNAME_DELIMITER);
            std::string meshPackageDirsString;
            std::stringstream meshPackageDirsStream;
            const std::vector<std::string> & meshPackageDirs = system.robot->getMeshPackageDirs();
            copy(meshPackageDirs.begin(),
                 meshPackageDirs.end(),
                 std::ostream_iterator<std::string>(meshPackageDirsStream, ";"));
            if (meshPackageDirsStream.peek() !=
                decltype(meshPackageDirsStream)::traits_type::eof())
            {
                meshPackageDirsString = meshPackageDirsStream.str();
                meshPackageDirsString.pop_back();
            }
            telemetrySender_->registerConstant(telemetryMeshPackageDirs, meshPackageDirsString);

            // Backup the true and theoretical Pinocchio::Model
            std::string key =
                addCircumfix("pinocchio_model", system.name, {}, TELEMETRY_FIELDNAME_DELIMITER);
            std::string value = saveToBinary(system.robot->pinocchioModel_);
            telemetrySender_->registerConstant(key, value);

            /* Backup the Pinocchio GeometryModel for collisions and visuals.
               It may fail because of missing serialization methods for convex, or because it
               cannot fit into memory (return code). Persistent mode is automatically enforced
               if no URDF is associated with the robot.*/
            if (engineOptions_->telemetry.isPersistent || urdfFileString.empty())
            {
                try
                {
                    key = addCircumfix(
                        "collision_model", system.name, {}, TELEMETRY_FIELDNAME_DELIMITER);
                    value = saveToBinary(system.robot->collisionModel_);
                    telemetrySender_->registerConstant(key, value);

                    key = addCircumfix(
                        "visual_model", system.name, {}, TELEMETRY_FIELDNAME_DELIMITER);
                    value = saveToBinary(system.robot->visualModel_);
                    telemetrySender_->registerConstant(key, value);
                }
                catch (const std::exception & e)
                {
                    std::string msg{"Failed to log the collision and/or visual model."};
                    if (urdfFileString.empty())
                    {
                        msg += " It will be impossible to replay log files because no URDF "
                               "file is available as fallback.";
                    }
                    msg += "\nRaised from exception: ";
                    PRINT_WARNING(msg, e.what());
                }
            }
        }

        // Log all options
        GenericConfig allOptions;
        for (const auto & system : systems_)
        {
            const std::string telemetryRobotOptions =
                addCircumfix("system", system.name, {}, TELEMETRY_FIELDNAME_DELIMITER);
            GenericConfig systemOptions;
            systemOptions["robot"] = system.robot->getOptions();
            systemOptions["controller"] = system.controller->getOptions();
            allOptions[telemetryRobotOptions] = systemOptions;
        }
        allOptions["engine"] = engineOptionsGeneric_;
        Json::Value allOptionsJson = convertToJson(allOptions);
        Json::StreamWriterBuilder jsonWriter;
        jsonWriter["indentation"] = "";
        const std::string allOptionsString = Json::writeString(jsonWriter, allOptionsJson);
        telemetrySender_->registerConstant("options", allOptionsString);

        // Write the header: this locks the registration of new variables
        telemetryRecorder_->initialize(telemetryData_.get(), getTelemetryTimeUnit());

        // At this point, consider that the simulation is running
        isSimulationRunning_ = true;
    }

    void EngineMultiRobot::simulate(
        double tEnd,
        const std::map<std::string, Eigen::VectorXd> & qInit,
        const std::map<std::string, Eigen::VectorXd> & vInit,
        const std::optional<std::map<std::string, Eigen::VectorXd>> & aInit)
    {
        if (systems_.empty())
        {
            THROW_ERROR(bad_control_flow,
                        "No system to simulate. Please add one before starting simulation.");
        }

        if (tEnd < 5e-3)
        {
            THROW_ERROR(std::invalid_argument, "Simulation duration cannot be shorter than 5ms.");
        }

        // Reset the robot, controller, and engine
        reset(true, false);

        // Start the simulation
        start(qInit, vInit, aInit);

        // Now that telemetry has been initialized, check simulation duration
        if (tEnd > telemetryRecorder_->getLogDurationMax())
        {
            THROW_ERROR(std::runtime_error,
                        "Time overflow: with the current precision the maximum value that can be "
                        "logged is ",
                        telemetryRecorder_->getLogDurationMax(),
                        "s. Decrease logger precision to simulate for longer than that.");
        }

        // Integration loop based on boost::numeric::odeint::detail::integrate_times
        while (true)
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
            bool isCallbackFalse = false;
            auto systemIt = systems_.begin();
            auto systemDataIt = systemDataVec_.begin();
            for (; systemIt != systems_.end(); ++systemIt, ++systemDataIt)
            {
                if (!systemIt->callback(
                        stepperState_.t, systemDataIt->state.q, systemDataIt->state.v))
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
            if (0U < engineOptions_->stepper.iterMax &&
                engineOptions_->stepper.iterMax <= stepperState_.iter)
            {
                if (engineOptions_->stepper.verbose)
                {
                    std::cout << "Simulation done: maximum number of integration steps exceeded."
                              << std::endl;
                }
                break;
            }

            // Perform a single integration step up to `tEnd`, stopping at `stepperUpdatePeriod_`
            double stepSize;
            if (std::isfinite(stepperUpdatePeriod_))
            {
                stepSize = std::min(stepperUpdatePeriod_, tEnd - stepperState_.t);
            }
            else
            {
                stepSize = std::min(engineOptions_->stepper.dtMax, tEnd - stepperState_.t);
            }
            step(stepSize);  // Automatic dt adjustment
        }

        // Stop the simulation. The lock on the telemetry and the robots are released.
        stop();
    }

    void EngineMultiRobot::step(double stepSize)
    {
        // Check if the simulation has started
        if (!isSimulationRunning_)
        {
            THROW_ERROR(bad_control_flow,
                        "No simulation running. Please start one before using step method.");
        }

        // Clear log data buffer
        logData_.reset();

        // Check if there is something wrong with the integration
        auto qIt = stepperState_.qSplit.begin();
        auto vIt = stepperState_.vSplit.begin();
        auto aIt = stepperState_.aSplit.begin();
        for (; qIt != stepperState_.qSplit.end(); ++qIt, ++vIt, ++aIt)
        {
            if (qIt->hasNaN() || vIt->hasNaN() || aIt->hasNaN())
            {
                THROW_ERROR(std::runtime_error,
                            "Low-level ode solver failed. Consider increasing stepper accuracy.");
            }
        }

        // Check if the desired step size is suitable
        if (stepSize > EPS && stepSize < SIMULATION_MIN_TIMESTEP)
        {
            THROW_ERROR(std::invalid_argument, "Step size out of bounds.");
        }

        /* Set end time: The default step size is equal to the controller update period if
           discrete-time, otherwise it uses the sensor update period if discrete-time, otherwise
           it uses the user-defined parameter dtMax. */
        if (stepSize < EPS)
        {
            const double controllerUpdatePeriod = engineOptions_->stepper.controllerUpdatePeriod;
            if (controllerUpdatePeriod > EPS)
            {
                stepSize = controllerUpdatePeriod;
            }
            else
            {
                const double sensorsUpdatePeriod = engineOptions_->stepper.sensorsUpdatePeriod;
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

        /* Check that end time is not too large for the current logging precision, otherwise abort
           integration. */
        if (stepperState_.t + stepSize > telemetryRecorder_->getLogDurationMax())
        {
            THROW_ERROR(std::runtime_error,
                        "Time overflow: with the current precision the maximum value that can be "
                        "logged is ",
                        telemetryRecorder_->getLogDurationMax(),
                        "s. Decrease logger precision to simulate for longer than that.");
        }

        /* Avoid compounding of error using Kahan algorithm. It consists in keeping track of the
           cumulative rounding error to add it back to the sum when it gets larger than the
           numerical precision, thus avoiding it to grows unbounded. */
        const double stepSizeCorrected = stepSize - stepperState_.tError;
        const double tEnd = stepperState_.t + stepSizeCorrected;
        stepperState_.tError = (tEnd - stepperState_.t) - stepSizeCorrected;

        // Get references to some internal stepper buffers
        double & t = stepperState_.t;
        double & dt = stepperState_.dt;
        double & dtLargest = stepperState_.dtLargest;
        std::vector<Eigen::VectorXd> & qSplit = stepperState_.qSplit;
        std::vector<Eigen::VectorXd> & vSplit = stepperState_.vSplit;
        std::vector<Eigen::VectorXd> & aSplit = stepperState_.aSplit;

        // Monitor iteration failure
        uint32_t successiveIterFailed = 0;
        std::vector<uint32_t> successiveSolveFailedAll(systems_.size(), 0U);
        bool isNan = false;

        /* Flag monitoring if the current time step depends of a breakpoint or the integration
           tolerance. It will be used by the restoration mechanism, if dt gets very small to reach
           a breakpoint, in order to avoid having to perform several steps to stabilize again the
           estimation of the optimal time step. */
        bool isBreakpointReached = false;

        /* Flag monitoring if the dynamics has changed because of impulse forces or the command
           (only in the case of discrete control).

           `tryStep(rhs, x, dxdt, t, dt)` method of error controlled boost steppers leverage the
           FSAL (first same as last) principle. It is implemented by considering at the value of
           (x, dxdt) in argument have been initialized by the user with the system dynamics at
           current time t. Thus, if the system dynamics is discontinuous, one has to manually
           integrate up to t-, then update dxdt to take into the acceleration at t+.

           Note that ONLY the acceleration part of dxdt must be updated since the velocity is not
           supposed to have changed. On top of that, tPrev is invalid at this point because it has
           been updated just after the last successful step.

           TODO: Maybe dt should be reschedule because the dynamics has changed and thereby the
                 previously estimated dt is very meaningful anymore. */
        bool hasDynamicsChanged = false;

        // Start the timer used for timeout handling
        timer_.tic();

        // Perform the integration. Do not simulate extremely small time steps.
        while (tEnd - t >= STEPPER_MIN_TIMESTEP)
        {
            // Initialize next breakpoint time to the one recommended by the stepper
            double tNext = t;

            // Update the active set and get the next breakpoint of impulse forces
            double tImpulseForceNext = INF;
            for (auto & systemData : systemDataVec_)
            {
                /* Update the active set: activate an impulse force as soon as the current time
                   gets close enough of the application time, and deactivate it once the following
                   the same reasoning.

                   Note that breakpoints at the start/end of every impulse forces are already
                   enforced, so that the forces cannot get activated/deactivate too late. */
                auto isImpulseForceActiveIt = systemData.isImpulseForceActiveVec.begin();
                auto impulseForceIt = systemData.impulseForces.begin();
                for (; impulseForceIt != systemData.impulseForces.end();
                     ++isImpulseForceActiveIt, ++impulseForceIt)
                {
                    double tImpulseForce = impulseForceIt->t;
                    double dtImpulseForce = impulseForceIt->dt;

                    if (t > tImpulseForce - STEPPER_MIN_TIMESTEP)
                    {
                        *isImpulseForceActiveIt = true;
                        hasDynamicsChanged = true;
                    }
                    if (t >= tImpulseForce + dtImpulseForce - STEPPER_MIN_TIMESTEP)
                    {
                        *isImpulseForceActiveIt = false;
                        hasDynamicsChanged = true;
                    }
                }

                // Update the breakpoint time iterator if necessary
                auto & tBreakpointNextIt = systemData.impulseForceBreakpointNextIt;
                if (tBreakpointNextIt != systemData.impulseForceBreakpoints.end())
                {
                    if (t >= *tBreakpointNextIt - STEPPER_MIN_TIMESTEP)
                    {
                        // The current breakpoint is behind in time. Switching to the next one.
                        ++tBreakpointNextIt;
                    }
                }

                // Get the next breakpoint time if any
                if (tBreakpointNextIt != systemData.impulseForceBreakpoints.end())
                {
                    tImpulseForceNext = std::min(tImpulseForceNext, *tBreakpointNextIt);
                }
            }

            // Update the external force profiles if necessary (only for finite update frequency)
            if (std::isfinite(stepperUpdatePeriod_))
            {
                auto systemIt = systems_.begin();
                auto systemDataIt = systemDataVec_.begin();
                for (; systemIt != systems_.end(); ++systemIt, ++systemDataIt)
                {
                    for (auto & profileForce : systemDataIt->profileForces)
                    {
                        if (profileForce.updatePeriod > EPS)
                        {
                            double forceUpdatePeriod = profileForce.updatePeriod;
                            double dtNextForceUpdatePeriod =
                                forceUpdatePeriod - std::fmod(t, forceUpdatePeriod);
                            if (dtNextForceUpdatePeriod < SIMULATION_MIN_TIMESTEP ||
                                forceUpdatePeriod - dtNextForceUpdatePeriod < STEPPER_MIN_TIMESTEP)
                            {
                                const Eigen::VectorXd & q = systemDataIt->state.q;
                                const Eigen::VectorXd & v = systemDataIt->state.v;
                                profileForce.force = profileForce.func(t, q, v);
                                hasDynamicsChanged = true;
                            }
                        }
                    }
                }
            }

            // Update the controller command if necessary (only for finite update frequency)
            if (std::isfinite(stepperUpdatePeriod_) &&
                engineOptions_->stepper.controllerUpdatePeriod > EPS)
            {
                double controllerUpdatePeriod = engineOptions_->stepper.controllerUpdatePeriod;
                double dtNextControllerUpdatePeriod =
                    controllerUpdatePeriod - std::fmod(t, controllerUpdatePeriod);
                if (dtNextControllerUpdatePeriod < SIMULATION_MIN_TIMESTEP ||
                    controllerUpdatePeriod - dtNextControllerUpdatePeriod < STEPPER_MIN_TIMESTEP)
                {
                    auto systemIt = systems_.begin();
                    auto systemDataIt = systemDataVec_.begin();
                    for (; systemIt != systems_.end(); ++systemIt, ++systemDataIt)
                    {
                        const Eigen::VectorXd & q = systemDataIt->state.q;
                        const Eigen::VectorXd & v = systemDataIt->state.v;
                        Eigen::VectorXd & command = systemDataIt->state.command;
                        computeCommand(*systemIt, t, q, v, command);
                    }
                    hasDynamicsChanged = true;
                }
            }

            /* Update telemetry if necessary. It monitors the current iteration number, the current
               time, and the systems state, command, and sensors data.

               Note that the acceleration is discontinuous. In particular, it would have different
               values of the same timestep if the command has been updated. There is no way to log
               both the acceleration at the end of the previous step (t-) and at the beginning of
               the next one (t+). Logging the previous acceleration is more natural since it
               preserves the consistency between sensors data and robot state. */
            if (!std::isfinite(stepperUpdatePeriod_) ||
                !engineOptions_->stepper.logInternalStepperSteps)
            {
                bool mustUpdateTelemetry = !std::isfinite(stepperUpdatePeriod_);
                if (!mustUpdateTelemetry)
                {
                    double dtNextStepperUpdatePeriod =
                        stepperUpdatePeriod_ - std::fmod(t, stepperUpdatePeriod_);
                    mustUpdateTelemetry =
                        (dtNextStepperUpdatePeriod < SIMULATION_MIN_TIMESTEP ||
                         stepperUpdatePeriod_ - dtNextStepperUpdatePeriod < STEPPER_MIN_TIMESTEP);
                }
                if (mustUpdateTelemetry)
                {
                    updateTelemetry();
                }
            }

            // Fix the FSAL issue if the dynamics has changed
            if (!std::isfinite(stepperUpdatePeriod_) && hasDynamicsChanged)
            {
                computeSystemsDynamics(t, qSplit, vSplit, aSplit, true);
                syncAllAccelerationsAndForces(systems_, contactForcesPrev_, fPrev_, aPrev_);
                syncSystemsStateWithStepper(true);
                hasDynamicsChanged = false;
            }

            if (std::isfinite(stepperUpdatePeriod_))
            {
                /* Get the time of the next breakpoint for the ODE solver: a breakpoint occurs if:
                   - tEnd has been reached
                   - an external impulse force must be activated/deactivated
                   - the sensor measurements or the controller command must be updated. */
                double dtNextGlobal;  // dt to apply for the next stepper step
                const double dtNextUpdatePeriod =
                    stepperUpdatePeriod_ - std::fmod(t, stepperUpdatePeriod_);
                if (dtNextUpdatePeriod < SIMULATION_MIN_TIMESTEP)
                {
                    /* Step to reach next sensors/controller update is too short: skip one
                       controller update and jump to the next one.

                       Note that in this case, the sensors have already been updated in advance
                       during the previous loop. */
                    dtNextGlobal =
                        std::min(dtNextUpdatePeriod + stepperUpdatePeriod_, tImpulseForceNext - t);
                }
                else
                {
                    dtNextGlobal = std::min(dtNextUpdatePeriod, tImpulseForceNext - t);
                }

                /* Check if the next dt to about equal to the time difference between the current
                   time (it can only be smaller) and enforce next dt to exactly match this value in
                   such a case. */
                if (tEnd - t - STEPPER_MIN_TIMESTEP < dtNextGlobal)
                {
                    dtNextGlobal = tEnd - t;
                }

                // Update next dt
                tNext += dtNextGlobal;

                // Compute the next step using adaptive step method
                while (tNext - t > STEPPER_MIN_TIMESTEP)
                {
                    // Log every stepper state only if the user asked for
                    if (successiveIterFailed == 0 &&
                        engineOptions_->stepper.logInternalStepperSteps)
                    {
                        updateTelemetry();
                    }

                    // Fix the FSAL issue if the dynamics has changed
                    if (hasDynamicsChanged)
                    {
                        computeSystemsDynamics(t, qSplit, vSplit, aSplit, true);
                        syncAllAccelerationsAndForces(
                            systems_, contactForcesPrev_, fPrev_, aPrev_);
                        syncSystemsStateWithStepper(true);
                        hasDynamicsChanged = false;
                    }

                    // Adjust stepsize to end up exactly at the next breakpoint
                    dt = std::min(dt, tNext - t);
                    if (dtLargest > SIMULATION_MIN_TIMESTEP)
                    {
                        if (tNext - (t + dt) < SIMULATION_MIN_TIMESTEP)
                        {
                            dt = tNext - t;
                        }
                    }
                    else
                    {
                        if (tNext - (t + dt) < STEPPER_MIN_TIMESTEP)
                        {
                            dt = tNext - t;
                        }
                    }

                    /* Trying to reach multiples of STEPPER_MIN_TIMESTEP whenever possible. The
                       idea here is to reach only multiples of 1us, making logging easier, given
                       that, 1us can be consider an 'infinitesimal' time in robotics. This
                       arbitrary threshold many not be suited for simulating different, faster
                       dynamics, that require sub-microsecond precision. */
                    if (dt > SIMULATION_MIN_TIMESTEP)
                    {
                        const double dtResidual = std::fmod(dt, SIMULATION_MIN_TIMESTEP);
                        if (dtResidual > STEPPER_MIN_TIMESTEP &&
                            dtResidual < SIMULATION_MIN_TIMESTEP - STEPPER_MIN_TIMESTEP &&
                            dt - dtResidual > STEPPER_MIN_TIMESTEP)
                        {
                            dt -= dtResidual;
                        }
                    }

                    // Break the loop if dt is getting too small. The error code will be set later.
                    if (dt < STEPPER_MIN_TIMESTEP)
                    {
                        break;
                    }

                    // Break the loop in case of timeout. The error code will be set later.
                    if (EPS < engineOptions_->stepper.timeout &&
                        engineOptions_->stepper.timeout < timer_.toc())
                    {
                        break;
                    }

                    // Break the loop in case of too many successive failed inner iteration
                    if (successiveIterFailed > engineOptions_->stepper.successiveIterFailedMax)
                    {
                        break;
                    }

                    /* Backup current number of successive constraint solving failure.
                       It will be restored in case of integration failure. */
                    auto systemDataIt = systemDataVec_.begin();
                    auto successiveSolveFailedIt = successiveSolveFailedAll.begin();
                    for (; systemDataIt != systemDataVec_.end();
                         ++systemDataIt, ++successiveSolveFailedIt)
                    {
                        *successiveSolveFailedIt = systemDataIt->successiveSolveFailed;
                    }

                    // Break the loop in case of too many successive constraint solving failures
                    for (uint32_t successiveSolveFailed : successiveSolveFailedAll)
                    {
                        if (successiveSolveFailed >
                            engineOptions_->stepper.successiveIterFailedMax)
                        {
                            break;
                        }
                    }

                    /* A breakpoint has been reached, causing dt to be smaller that necessary for
                       prescribed integration tol. */
                    isBreakpointReached = (dtLargest > dt);

                    // Set the timestep to be tried by the stepper
                    dtLargest = dt;

                    // Try doing one integration step
                    bool isStepSuccessful =
                        stepper_->tryStep(qSplit, vSplit, aSplit, t, dtLargest);

                    /* Check if the integrator failed miserably even if successfully.
                       It would happen the timestep is fixed and too large, causing the integrator
                       to fail miserably returning nan. */
                    isNan = std::isnan(dtLargest);
                    if (isNan)
                    {
                        break;
                    }

                    // Update buffer if really successful
                    if (isStepSuccessful)
                    {
                        // Reset successive iteration failure counter
                        successiveIterFailed = 0;

                        // Synchronize the position, velocity and acceleration of all systems
                        syncSystemsStateWithStepper();

                        /* Compute all external terms including joints accelerations and forces.
                           Note that it is possible to call this method because `pinocchio::Data`
                           is guaranteed to be up-to-date at this point. */
                        computeAllExtraTerms(systems_, systemDataVec_, fPrev_);

                        // Backend the updated joint accelerations and forces
                        syncAllAccelerationsAndForces(
                            systems_, contactForcesPrev_, fPrev_, aPrev_);

                        // Increment the iteration counter only for successful steps
                        ++stepperState_.iter;

                        /* Restore the step size dt if it has been significantly decreased because
                           of a breakpoint. It is set equal to the last available largest dt to be
                           known, namely the second to last successful step. */
                        if (isBreakpointReached)
                        {
                            /* Restore the step size if and only if:
                               - the next estimated largest step size is larger than the requested
                                 one for the current (successful) step.
                               - the next estimated largest step size is significantly smaller than
                                 the estimated largest step size for the previous step. */
                            double dtRestoreThresholdAbs =
                                stepperState_.dtLargestPrev *
                                engineOptions_->stepper.dtRestoreThresholdRel;
                            if (dt < dtLargest && dtLargest < dtRestoreThresholdAbs)
                            {
                                dtLargest = stepperState_.dtLargestPrev;
                            }
                        }

                        /* Backup the stepper and systems' state on success only:
                           - t at last successful iteration is used to compute dt, which is project
                             the acceleration in the state space instead of SO3^2.
                           - dtLargestPrev is used to restore the largest step size in case of a
                             breakpoint requiring lowering it.
                           - the acceleration and effort at the last successful iteration is used
                             to update the sensors' data in case of continuous sensing. */
                        stepperState_.tPrev = t;
                        stepperState_.dtLargestPrev = dtLargest;
                        for (auto & systemData : systemDataVec_)
                        {
                            systemData.statePrev = systemData.state;
                        }
                    }
                    else
                    {
                        // Increment the failed iteration counters
                        ++successiveIterFailed;
                        ++stepperState_.iterFailed;

                        // Restore number of successive constraint solving failure
                        systemDataIt = systemDataVec_.begin();
                        successiveSolveFailedIt = successiveSolveFailedAll.begin();
                        for (; systemDataIt != systemDataVec_.end();
                             ++systemDataIt, ++successiveSolveFailedIt)
                        {
                            systemDataIt->successiveSolveFailed = *successiveSolveFailedIt;
                        }
                    }

                    // Initialize the next dt
                    dt = std::min(dtLargest, engineOptions_->stepper.dtMax);
                }
            }
            else
            {
                /* Make sure it ends exactly at the tEnd, never exceeds dtMax, and stop to apply
                   impulse forces. */
                dt = std::min({dt, tEnd - t, tImpulseForceNext - t});

                /* A breakpoint has been reached, because dt has been decreased wrt the largest
                   possible dt within integration tol. */
                isBreakpointReached = (dtLargest > dt);

                // Compute the next step using adaptive step method
                bool isStepSuccessful = false;
                while (!isStepSuccessful)
                {
                    // Set the timestep to be tried by the stepper
                    dtLargest = dt;

                    // Break the loop in case of too many successive failed inner iteration
                    if (successiveIterFailed > engineOptions_->stepper.successiveIterFailedMax)
                    {
                        break;
                    }

                    /* Backup current number of successive constraint solving failure.
                       It will be restored in case of integration failure. */
                    auto systemDataIt = systemDataVec_.begin();
                    auto successiveSolveFailedIt = successiveSolveFailedAll.begin();
                    for (; systemDataIt != systemDataVec_.end();
                         ++systemDataIt, ++successiveSolveFailedIt)
                    {
                        *successiveSolveFailedIt = systemDataIt->successiveSolveFailed;
                    }

                    // Break the loop in case of too many successive constraint solving failures
                    for (uint32_t successiveSolveFailed : successiveSolveFailedAll)
                    {
                        if (successiveSolveFailed >
                            engineOptions_->stepper.successiveIterFailedMax)
                        {
                            break;
                        }
                    }

                    // Try to do a step
                    isStepSuccessful = stepper_->tryStep(qSplit, vSplit, aSplit, t, dtLargest);

                    // Check if the integrator failed miserably even if successfully
                    isNan = std::isnan(dtLargest);
                    if (isNan)
                    {
                        break;
                    }

                    if (isStepSuccessful)
                    {
                        // Reset successive iteration failure counter
                        successiveIterFailed = 0;

                        // Synchronize the position, velocity and acceleration of all systems
                        syncSystemsStateWithStepper();

                        // Compute all external terms including joints accelerations and forces
                        computeAllExtraTerms(systems_, systemDataVec_, fPrev_);

                        // Backend the updated joint accelerations and forces
                        syncAllAccelerationsAndForces(
                            systems_, contactForcesPrev_, fPrev_, aPrev_);

                        // Increment the iteration counter
                        ++stepperState_.iter;

                        // Restore the step size if necessary
                        if (isBreakpointReached)
                        {
                            double dtRestoreThresholdAbs =
                                stepperState_.dtLargestPrev *
                                engineOptions_->stepper.dtRestoreThresholdRel;
                            if (dt < dtLargest && dtLargest < dtRestoreThresholdAbs)
                            {
                                dtLargest = stepperState_.dtLargestPrev;
                            }
                        }

                        // Backup the stepper and systems' state
                        stepperState_.tPrev = t;
                        stepperState_.dtLargestPrev = dtLargest;
                        for (auto & systemData : systemDataVec_)
                        {
                            systemData.statePrev = systemData.state;
                        }
                    }
                    else
                    {
                        // Increment the failed iteration counter
                        ++successiveIterFailed;
                        ++stepperState_.iterFailed;

                        // Restore number of successive constraint solving failure
                        systemDataIt = systemDataVec_.begin();
                        successiveSolveFailedIt = successiveSolveFailedAll.begin();
                        for (; systemDataIt != systemDataVec_.end();
                             ++systemDataIt, ++successiveSolveFailedIt)
                        {
                            systemDataIt->successiveSolveFailed = *successiveSolveFailedIt;
                        }
                    }

                    // Initialize the next dt
                    dt = std::min(dtLargest, engineOptions_->stepper.dtMax);
                }
            }

            // Exception handling
            if (isNan)
            {
                THROW_ERROR(std::runtime_error,
                            "Something is wrong with the physics. Aborting integration.");
            }
            if (successiveIterFailed > engineOptions_->stepper.successiveIterFailedMax)
            {
                THROW_ERROR(std::runtime_error,
                            "Too many successive iteration failures. Probably something "
                            "is going wrong with the physics. Aborting integration.");
            }
            for (uint32_t successiveSolveFailed : successiveSolveFailedAll)
            {
                if (successiveSolveFailed > engineOptions_->stepper.successiveIterFailedMax)
                {
                    THROW_ERROR(
                        std::runtime_error,
                        "Too many successive constraint solving failures. Try increasing the "
                        "regularization factor. Aborting integration.");
                }
            }
            if (dt < STEPPER_MIN_TIMESTEP)
            {
                THROW_ERROR(std::runtime_error,
                            "The internal time step is getting too small. Impossible to "
                            "integrate physics further in time. Aborting integration.");
            }
            if (EPS < engineOptions_->stepper.timeout &&
                engineOptions_->stepper.timeout < timer_.toc())
            {
                THROW_ERROR(std::runtime_error, "Step computation timeout. Aborting integration.");
            }

            // Update sensors data if necessary, namely if time-continuous or breakpoint
            const double sensorsUpdatePeriod = engineOptions_->stepper.sensorsUpdatePeriod;
            const double dtNextSensorsUpdatePeriod =
                sensorsUpdatePeriod - std::fmod(t, sensorsUpdatePeriod);
            bool mustUpdateSensors = sensorsUpdatePeriod < EPS;
            if (!mustUpdateSensors)
            {
                mustUpdateSensors = dtNextSensorsUpdatePeriod < SIMULATION_MIN_TIMESTEP ||
                                    sensorsUpdatePeriod - dtNextSensorsUpdatePeriod <
                                        STEPPER_MIN_TIMESTEP;
            }
            if (mustUpdateSensors)
            {
                auto systemIt = systems_.begin();
                auto systemDataIt = systemDataVec_.begin();
                for (; systemIt != systems_.end(); ++systemIt, ++systemDataIt)
                {
                    const Eigen::VectorXd & q = systemDataIt->state.q;
                    const Eigen::VectorXd & v = systemDataIt->state.v;
                    const Eigen::VectorXd & a = systemDataIt->state.a;
                    const Eigen::VectorXd & uMotor = systemDataIt->state.uMotor;
                    const ForceVector & fext = systemDataIt->state.fExternal;
                    systemIt->robot->computeSensorMeasurements(t, q, v, a, uMotor, fext);
                }
            }
        }

        /* Update the final time and dt to make sure it corresponds to the desired values and avoid
           compounding of error. Anyway the user asked for a step of exactly stepSize, so he is
           expecting this value to be reached. */
        t = tEnd;
    }

    void EngineMultiRobot::stop()
    {
        // Release the lock on the robots
        for (auto & systemData : systemDataVec_)
        {
            systemData.robotLock.reset(nullptr);
        }

        // Make sure that a simulation running
        if (!isSimulationRunning_)
        {
            return;
        }

        // Log current buffer content as final point of the log data
        updateTelemetry();

        // Clear log data buffer one last time, now that the final point has been added
        logData_.reset();

        /* Reset the telemetry.
           Note that calling ``stop` or  `reset` does NOT clear the internal data buffer of
           telemetryRecorder_. Clearing is done at init time, so that it remains accessible until
           the next initialization. */
        telemetryRecorder_->reset();
        telemetryData_->reset();

        // Update some internal flags
        isSimulationRunning_ = false;
    }

    void EngineMultiRobot::registerImpulseForce(const std::string & systemName,
                                                const std::string & frameName,
                                                double t,
                                                double dt,
                                                const pinocchio::Force & force)
    {
        if (isSimulationRunning_)
        {
            THROW_ERROR(
                bad_control_flow,
                "Simulation already running. Please stop it before registering new forces.");
        }

        if (dt < STEPPER_MIN_TIMESTEP)
        {
            THROW_ERROR(std::invalid_argument,
                        "Force duration cannot be smaller than ",
                        STEPPER_MIN_TIMESTEP,
                        "s.");
        }

        if (t < 0.0)
        {
            THROW_ERROR(std::invalid_argument, "Force application time must be positive.");
        }

        if (frameName == "universe")
        {
            THROW_ERROR(std::invalid_argument,
                        "Impossible to apply external forces to the universe itself!");
        }

        // TODO: Make sure that the forces do NOT overlap while taking into account dt.

        std::ptrdiff_t systemIndex = getSystemIndex(systemName);
        pinocchio::FrameIndex frameIndex =
            getFrameIndex(systems_[systemIndex].robot->pinocchioModel_, frameName);

        SystemData & systemData = systemDataVec_[systemIndex];
        systemData.impulseForces.emplace_back(frameName, frameIndex, t, dt, force);
        systemData.impulseForceBreakpoints.emplace(t);
        systemData.impulseForceBreakpoints.emplace(t + dt);
        systemData.isImpulseForceActiveVec.emplace_back(false);
    }

    template<typename... Args>
    std::tuple<bool, const double &>
    isGcdIncluded(const vector_aligned_t<SystemData> & systemDataVec, const Args &... values)
    {
        if (systemDataVec.empty())
        {
            return isGcdIncluded(values...);
        }

        const double * valueMinPtr = &INF;
        auto func = [&valueMinPtr, &values...](const SystemData & systemData)
        {
            auto && [isIncluded, value] = isGcdIncluded(
                systemData.profileForces.cbegin(),
                systemData.profileForces.cend(),
                [](const ProfileForce & force) -> const double & { return force.updatePeriod; },
                values...);
            valueMinPtr = &(minClipped(*valueMinPtr, value));
            return isIncluded;
        };
        // FIXME: Order of evaluation is not always respected with MSVC.
        bool isIncluded = std::all_of(systemDataVec.begin(), systemDataVec.end(), func);
        return {isIncluded, *valueMinPtr};
    }

    void EngineMultiRobot::registerProfileForce(const std::string & systemName,
                                                const std::string & frameName,
                                                const ProfileForceFunction & forceFunc,
                                                double updatePeriod)
    {
        if (isSimulationRunning_)
        {
            THROW_ERROR(
                bad_control_flow,
                "Simulation already running. Please stop it before registering new forces.");
        }

        if (frameName == "universe")
        {
            THROW_ERROR(std::invalid_argument,
                        "Impossible to apply external forces to the universe itself!");
        }

        // Get system and frame indices
        std::ptrdiff_t systemIndex = getSystemIndex(systemName);
        pinocchio::FrameIndex frameIndex =
            getFrameIndex(systems_[systemIndex].robot->pinocchioModel_, frameName);

        // Make sure the update period is valid
        if (EPS < updatePeriod && updatePeriod < SIMULATION_MIN_TIMESTEP)
        {
            THROW_ERROR(
                std::invalid_argument,
                "Cannot register external force profile with update period smaller than ",
                SIMULATION_MIN_TIMESTEP,
                "s. Adjust period or switch to continuous mode by setting period to zero.");
        }
        // Make sure the desired update period is a multiple of the stepper period
        auto [isIncluded, updatePeriodMin] =
            isGcdIncluded(systemDataVec_, stepperUpdatePeriod_, updatePeriod);
        if (!isIncluded)
        {
            THROW_ERROR(std::invalid_argument,
                        "In discrete mode, the update period of force profiles and the "
                        "stepper update period (min of controller and sensor update "
                        "periods) must be multiple of each other.");
        }

        // Set breakpoint period during the integration loop
        stepperUpdatePeriod_ = updatePeriodMin;

        // Add force profile to register
        SystemData & systemData = systemDataVec_[systemIndex];
        systemData.profileForces.emplace_back(frameName, frameIndex, updatePeriod, forceFunc);
    }

    void EngineMultiRobot::removeImpulseForces(const std::string & systemName)
    {
        // Make sure that no simulation is running
        if (isSimulationRunning_)
        {
            THROW_ERROR(bad_control_flow,
                        "Simulation already running. Stop it before removing coupling forces.");
        }

        std::ptrdiff_t systemIndex = getSystemIndex(systemName);
        SystemData & systemData = systemDataVec_[systemIndex];
        systemData.impulseForces.clear();
    }

    void EngineMultiRobot::removeImpulseForces()
    {
        // Make sure that no simulation is running
        if (isSimulationRunning_)
        {
            THROW_ERROR(bad_control_flow,
                        "simulation already running. Stop it before removing coupling forces.");
        }

        for (auto & systemData : systemDataVec_)
        {
            systemData.impulseForces.clear();
        }
    }

    void EngineMultiRobot::removeProfileForces(const std::string & systemName)
    {
        // Make sure that no simulation is running
        if (isSimulationRunning_)
        {
            THROW_ERROR(bad_control_flow,
                        "Simulation already running. Stop it before removing coupling forces.");
        }


        // Remove force profile from register
        std::ptrdiff_t systemIndex = getSystemIndex(systemName);
        SystemData & systemData = systemDataVec_[systemIndex];
        systemData.profileForces.clear();

        // Set breakpoint period during the integration loop
        // FIXME: replaced `std::get<N>` by placeholder `_` when moving to C++26 (P2169R4)
        stepperUpdatePeriod_ =
            std::get<1>(isGcdIncluded(systemDataVec_,
                                      engineOptions_->stepper.sensorsUpdatePeriod,
                                      engineOptions_->stepper.controllerUpdatePeriod));
    }

    void EngineMultiRobot::removeProfileForces()
    {
        // Make sure that no simulation is running
        if (isSimulationRunning_)
        {
            THROW_ERROR(bad_control_flow,
                        "Simulation already running. Stop it before removing coupling forces.");
        }

        for (auto & systemData : systemDataVec_)
        {
            systemData.profileForces.clear();
        }
    }

    const ImpulseForceVector & EngineMultiRobot::getImpulseForces(
        const std::string & systemName) const
    {
        std::ptrdiff_t systemIndex = getSystemIndex(systemName);
        const SystemData & systemData = systemDataVec_[systemIndex];
        return systemData.impulseForces;
    }

    const ProfileForceVector & EngineMultiRobot::getProfileForces(
        const std::string & systemName) const
    {
        std::ptrdiff_t systemIndex = getSystemIndex(systemName);
        const SystemData & systemData = systemDataVec_[systemIndex];
        return systemData.profileForces;
    }

    GenericConfig EngineMultiRobot::getOptions() const noexcept
    {
        return engineOptionsGeneric_;
    }

    void EngineMultiRobot::setOptions(const GenericConfig & engineOptions)
    {
        if (isSimulationRunning_)
        {
            THROW_ERROR(bad_control_flow,
                        "Simulation already running. Please stop it before updating the options.");
        }

        // Make sure the dtMax is not out of range
        GenericConfig stepperOptions = boost::get<GenericConfig>(engineOptions.at("stepper"));
        const double dtMax = boost::get<double>(stepperOptions.at("dtMax"));
        if (SIMULATION_MAX_TIMESTEP + EPS < dtMax || dtMax < SIMULATION_MIN_TIMESTEP)
        {
            THROW_ERROR(std::invalid_argument, "'dtMax' option is out of range.");
        }

        // Make sure successiveIterFailedMax is strictly positive
        const uint32_t successiveIterFailedMax =
            boost::get<uint32_t>(stepperOptions.at("successiveIterFailedMax"));
        if (successiveIterFailedMax < 1)
        {
            THROW_ERROR(std::invalid_argument,
                        "'successiveIterFailedMax' must be strictly positive.");
        }

        // Make sure the selected ode solver is available and instantiate it
        const std::string & odeSolver = boost::get<std::string>(stepperOptions.at("odeSolver"));
        if (STEPPERS.find(odeSolver) == STEPPERS.end())
        {
            THROW_ERROR(
                std::invalid_argument, "Requested ODE solver '", odeSolver, "' not available.");
        }

        // Make sure the controller and sensor update periods are valid
        const double sensorsUpdatePeriod =
            boost::get<double>(stepperOptions.at("sensorsUpdatePeriod"));
        const double controllerUpdatePeriod =
            boost::get<double>(stepperOptions.at("controllerUpdatePeriod"));
        auto [isIncluded, updatePeriodMin] =
            isGcdIncluded(systemDataVec_, controllerUpdatePeriod, sensorsUpdatePeriod);
        if ((EPS < sensorsUpdatePeriod && sensorsUpdatePeriod < SIMULATION_MIN_TIMESTEP) ||
            (EPS < controllerUpdatePeriod && controllerUpdatePeriod < SIMULATION_MIN_TIMESTEP))
        {
            THROW_ERROR(
                std::invalid_argument,
                "Cannot simulate a discrete system with update period smaller than ",
                SIMULATION_MIN_TIMESTEP,
                "s. Adjust period or switch to continuous mode by setting period to zero.");
        }
        else if (!isIncluded)
        {
            THROW_ERROR(std::invalid_argument,
                        "In discrete mode, the controller and sensor update "
                        "periods must be multiple of each other.");
        }

        // Make sure the contacts options are fine
        GenericConfig constraintsOptions =
            boost::get<GenericConfig>(engineOptions.at("constraints"));
        const std::string & constraintSolverType =
            boost::get<std::string>(constraintsOptions.at("solver"));
        const auto constraintSolverIt = CONSTRAINT_SOLVERS_MAP.find(constraintSolverType);
        if (constraintSolverIt == CONSTRAINT_SOLVERS_MAP.end())
        {
            THROW_ERROR(std::invalid_argument,
                        "Requested constraint solver '",
                        constraintSolverType,
                        "' not available.");
        }
        double regularization = boost::get<double>(constraintsOptions.at("regularization"));
        if (regularization < 0.0)
        {
            THROW_ERROR(std::invalid_argument,
                        "Constraint option 'regularization' must be positive.");
        }

        // Make sure the contacts options are fine
        GenericConfig contactOptions = boost::get<GenericConfig>(engineOptions.at("contacts"));
        const std::string & contactModel = boost::get<std::string>(contactOptions.at("model"));
        const auto contactModelIt = CONTACT_MODELS_MAP.find(contactModel);
        if (contactModelIt == CONTACT_MODELS_MAP.end())
        {
            THROW_ERROR(std::invalid_argument, "Requested contact model not available.");
        }
        double contactsTransitionEps = boost::get<double>(contactOptions.at("transitionEps"));
        if (contactsTransitionEps < 0.0)
        {
            THROW_ERROR(std::invalid_argument, "Contact option 'transitionEps' must be positive.");
        }
        double transitionVelocity = boost::get<double>(contactOptions.at("transitionVelocity"));
        if (transitionVelocity < EPS)
        {
            THROW_ERROR(std::invalid_argument,
                        "Contact option 'transitionVelocity' must be strictly positive.");
        }
        double stabilizationFreq = boost::get<double>(contactOptions.at("stabilizationFreq"));
        if (stabilizationFreq < 0.0)
        {
            THROW_ERROR(std::invalid_argument,
                        "Contact option 'stabilizationFreq' must be positive.");
        }

        // Make sure the user-defined gravity force has the right dimension
        GenericConfig worldOptions = boost::get<GenericConfig>(engineOptions.at("world"));
        Eigen::VectorXd gravity = boost::get<Eigen::VectorXd>(worldOptions.at("gravity"));
        if (gravity.size() != 6)
        {
            THROW_ERROR(std::invalid_argument, "The size of the gravity force vector must be 6.");
        }

        /* Reset random number generators if setOptions is called for the first time, or if the
           desired random seed has changed. */
        const VectorX<uint32_t> & seedSeq =
            boost::get<VectorX<uint32_t>>(stepperOptions.at("randomSeedSeq"));
        if (!engineOptions_ || seedSeq.size() != engineOptions_->stepper.randomSeedSeq.size() ||
            (seedSeq.array() != engineOptions_->stepper.randomSeedSeq.array()).any())
        {
            generator_.seed(std::seed_seq(seedSeq.data(), seedSeq.data() + seedSeq.size()));
        }

        // Update the internal options
        engineOptionsGeneric_ = engineOptions;

        // Create a fast struct accessor
        engineOptions_ = std::make_unique<const EngineOptions>(engineOptionsGeneric_);

        // Backup contact model as enum for fast check
        contactModel_ = contactModelIt->second;

        // Set breakpoint period during the integration loop
        stepperUpdatePeriod_ = updatePeriodMin;
    }

    std::vector<std::string> EngineMultiRobot::getSystemNames() const
    {
        std::vector<std::string> systemsNames;
        systemsNames.reserve(systems_.size());
        std::transform(systems_.begin(),
                       systems_.end(),
                       std::back_inserter(systemsNames),
                       [](const auto & sys) -> std::string { return sys.name; });
        return systemsNames;
    }

    std::ptrdiff_t EngineMultiRobot::getSystemIndex(const std::string & systemName) const
    {
        auto systemIt = std::find_if(systems_.begin(),
                                     systems_.end(),
                                     [&systemName](const auto & sys)
                                     { return (sys.name == systemName); });
        if (systemIt == systems_.end())
        {
            THROW_ERROR(std::invalid_argument,
                        "No system with this name has been added to the engine.");
        }

        return std::distance(systems_.begin(), systemIt);
    }

    System & EngineMultiRobot::getSystem(const std::string & systemName)
    {
        auto systemIt = std::find_if(systems_.begin(),
                                     systems_.end(),
                                     [&systemName](const auto & sys)
                                     { return (sys.name == systemName); });
        if (systemIt == systems_.end())
        {
            THROW_ERROR(std::invalid_argument,
                        "No system with this name has been added to the engine.");
        }

        return *systemIt;
    }

    const SystemState & EngineMultiRobot::getSystemState(const std::string & systemName) const
    {
        std::ptrdiff_t systemIndex = getSystemIndex(systemName);
        return systemDataVec_[systemIndex].state;
    }

    const StepperState & EngineMultiRobot::getStepperState() const
    {
        return stepperState_;
    }

    const bool & EngineMultiRobot::getIsSimulationRunning() const
    {
        return isSimulationRunning_;
    }

    double EngineMultiRobot::getSimulationDurationMax()
    {
        return TelemetryRecorder::getLogDurationMax(getTelemetryTimeUnit());
    }

    double EngineMultiRobot::getTelemetryTimeUnit()
    {
        return STEPPER_MIN_TIMESTEP;
    }

    // ========================================================
    // =================== Stepper utilities ==================
    // ========================================================

    void EngineMultiRobot::syncStepperStateWithSystems()
    {
        auto qSplitIt = stepperState_.qSplit.begin();
        auto vSplitIt = stepperState_.vSplit.begin();
        auto aSplitIt = stepperState_.aSplit.begin();
        auto systemDataIt = systemDataVec_.begin();
        for (; systemDataIt != systemDataVec_.end();
             ++systemDataIt, ++qSplitIt, ++vSplitIt, ++aSplitIt)
        {
            *qSplitIt = systemDataIt->state.q;
            *vSplitIt = systemDataIt->state.v;
            *aSplitIt = systemDataIt->state.a;
        }
    }

    void EngineMultiRobot::syncSystemsStateWithStepper(bool isStateUpToDate)
    {
        if (isStateUpToDate)
        {
            auto aSplitIt = stepperState_.aSplit.begin();
            auto systemDataIt = systemDataVec_.begin();
            for (; systemDataIt != systemDataVec_.end(); ++systemDataIt, ++aSplitIt)
            {
                systemDataIt->state.a = *aSplitIt;
            }
        }
        else
        {
            auto qSplitIt = stepperState_.qSplit.begin();
            auto vSplitIt = stepperState_.vSplit.begin();
            auto aSplitIt = stepperState_.aSplit.begin();
            auto systemDataIt = systemDataVec_.begin();
            for (; systemDataIt != systemDataVec_.end();
                 ++systemDataIt, ++qSplitIt, ++vSplitIt, ++aSplitIt)
            {
                systemDataIt->state.q = *qSplitIt;
                systemDataIt->state.v = *vSplitIt;
                systemDataIt->state.a = *aSplitIt;
            }
        }
    }

    // ========================================================
    // ================ Core physics utilities ================
    // ========================================================


    void EngineMultiRobot::computeForwardKinematics(System & system,
                                                    const Eigen::VectorXd & q,
                                                    const Eigen::VectorXd & v,
                                                    const Eigen::VectorXd & a)
    {
        // Create proxies for convenience
        const pinocchio::Model & model = system.robot->pinocchioModel_;
        pinocchio::Data & data = system.robot->pinocchioData_;
        const pinocchio::GeometryModel & geomModel = system.robot->collisionModel_;
        pinocchio::GeometryData & geomData = system.robot->collisionData_;

        // Update forward kinematics
        pinocchio::forwardKinematics(model, data, q, v, a);

        // Update frame placements (avoiding redundant computations)
        for (pinocchio::FrameIndex frameIndex = 1;
             frameIndex < static_cast<pinocchio::FrameIndex>(model.nframes);
             ++frameIndex)
        {
            const pinocchio::Frame & frame = model.frames[frameIndex];
            pinocchio::JointIndex parentJointIndex = frame.parent;
            switch (frame.type)
            {
            case pinocchio::FrameType::JOINT:
                /* If the frame is associated with an actual joint, no need to compute anything
                   new, since the frame transform is supposed to be identity. */
                data.oMf[frameIndex] = data.oMi[parentJointIndex];
                break;
            case pinocchio::FrameType::BODY:
                if (model.frames[frame.previousFrame].type == pinocchio::FrameType::FIXED_JOINT)
                {
                    /* BODYs connected via FIXED_JOINT(s) have the same transform than the joint
                       itself, so no need to compute them twice. Here we are doing the assumption
                       that the previous frame transform has already been updated since it is
                       closer to root in kinematic tree. */
                    data.oMf[frameIndex] = data.oMf[frame.previousFrame];
                }
                else
                {
                    /* BODYs connected via JOINT(s) have the identity transform, so copying parent
                       joint transform should be fine. */
                    data.oMf[frameIndex] = data.oMi[parentJointIndex];
                }
                break;
            case pinocchio::FrameType::FIXED_JOINT:
            case pinocchio::FrameType::SENSOR:
            case pinocchio::FrameType::OP_FRAME:
            default:
                // Nothing special, doing the actual computation
                data.oMf[frameIndex] = data.oMi[parentJointIndex] * frame.placement;
            }
        }

        /* Update collision information selectively, ie only for geometries involved in at least
           one collision pair. */
        pinocchio::updateGeometryPlacements(model, data, geomModel, geomData);
        pinocchio::computeCollisions(geomModel, geomData, false);
    }

    void EngineMultiRobot::computeContactDynamicsAtBody(
        const System & system,
        const pinocchio::PairIndex & collisionPairIndex,
        std::shared_ptr<AbstractConstraintBase> & constraint,
        pinocchio::Force & fextLocal) const
    {
        // TODO: It is assumed that the ground is flat. For now ground profile is not supported
        // with body collision. Nevertheless it should not be to hard to generated a collision
        // object simply by sampling points on the profile.

        // Define proxy for convenience
        pinocchio::Data & data = system.robot->pinocchioData_;

        // Get the frame and joint indices
        const pinocchio::GeomIndex & geometryIndex =
            system.robot->collisionModel_.collisionPairs[collisionPairIndex].first;
        pinocchio::JointIndex parentJointIndex =
            system.robot->collisionModel_.geometryObjects[geometryIndex].parentJoint;

        // Extract collision and distance results
        const hpp::fcl::CollisionResult & collisionResult =
            system.robot->collisionData_.collisionResults[collisionPairIndex];

        fextLocal.setZero();

        /* There is no way to get access to the distance from the ground at this point, so it is
           not possible to disable the constraint only if depth > transitionEps. */
        if (constraint)
        {
            constraint->disable();
        }

        for (std::size_t contactIndex = 0; contactIndex < collisionResult.numContacts();
             ++contactIndex)
        {
            /* Extract the contact information.
               Note that there is always a single contact point while computing the collision
               between two shape objects, for instance convex geometry and box primitive. */
            const auto & contact = collisionResult.getContact(contactIndex);
            Eigen::Vector3d nGround = contact.normal.normalized();  // Ground normal in world frame
            double depth = contact.penetration_depth;  // Penetration depth (signed, negative)
            pinocchio::SE3 posContactInWorld = pinocchio::SE3::Identity();
            // TODO double check that it may be between both interfaces
            posContactInWorld.translation() = contact.pos;  //  Point inside the ground

            /* FIXME: Make sure the collision computation didn't failed. If it happens the norm of
               the distance normal is not normalized (usually close to zero). If so, just assume
               there is no collision at all. */
            if (nGround.norm() < 1.0 - EPS)
            {
                continue;
            }

            // Make sure the normal is always pointing upward and the penetration depth is negative
            if (nGround[2] < 0.0)
            {
                nGround *= -1.0;
            }
            if (depth > 0.0)
            {
                depth *= -1.0;
            }

            if (contactModel_ == ContactModelType::SPRING_DAMPER)
            {
                // Compute the linear velocity of the contact point in world frame
                const pinocchio::Motion & motionJointLocal = data.v[parentJointIndex];
                const pinocchio::SE3 & transformJointFrameInWorld = data.oMi[parentJointIndex];
                const pinocchio::SE3 transformJointFrameInContact =
                    posContactInWorld.actInv(transformJointFrameInWorld);
                const Eigen::Vector3d vContactInWorld =
                    transformJointFrameInContact.act(motionJointLocal).linear();

                // Compute the ground reaction force at contact point in world frame
                const pinocchio::Force fextAtContactInGlobal =
                    computeContactDynamics(nGround, depth, vContactInWorld);

                // Move the force at parent frame location
                fextLocal += transformJointFrameInContact.actInv(fextAtContactInGlobal);
            }
            else
            {
                // The constraint is not initialized for geometry shapes that are not supported
                if (constraint)
                {
                    // In case of slippage the contact point has actually moved and must be updated
                    constraint->enable();
                    auto & frameConstraint = static_cast<FrameConstraint &>(*constraint.get());
                    const pinocchio::FrameIndex frameIndex = frameConstraint.getFrameIndex();
                    frameConstraint.setReferenceTransform(
                        {data.oMf[frameIndex].rotation(),
                         data.oMf[frameIndex].translation() - depth * nGround});
                    frameConstraint.setNormal(nGround);

                    // Only one contact constraint per collision body is supported for now
                    break;
                }
            }
        }
    }

    void EngineMultiRobot::computeContactDynamicsAtFrame(
        const System & system,
        pinocchio::FrameIndex frameIndex,
        std::shared_ptr<AbstractConstraintBase> & constraint,
        pinocchio::Force & fextLocal) const
    {
        /* Returns the external force in the contact frame. It must then be converted into a force
           onto the parent joint.
           /!\ Note that the contact dynamics depends only on kinematics data. /!\ */

        // Define proxies for convenience
        const pinocchio::Model & model = system.robot->pinocchioModel_;
        const pinocchio::Data & data = system.robot->pinocchioData_;

        // Get the pose of the frame wrt the world
        const pinocchio::SE3 & transformFrameInWorld = data.oMf[frameIndex];

        // Compute the ground normal and penetration depth at the contact point
        double heightGround;
        Eigen::Vector3d normalGround;
        const Eigen::Vector3d & posFrame = transformFrameInWorld.translation();
        engineOptions_->world.groundProfile(posFrame.head<2>(), heightGround, normalGround);
        normalGround.normalize();  // Make sure the ground normal is normalized

        // First-order projection (exact assuming no curvature)
        const double depth = (posFrame[2] - heightGround) * normalGround[2];

        // Only compute the ground reaction force if the penetration depth is negative
        if (depth < 0.0)
        {
            // Apply the force at the origin of the parent joint frame
            if (contactModel_ == ContactModelType::SPRING_DAMPER)
            {
                // Compute the linear velocity of the contact point in world frame
                const Eigen::Vector3d motionFrameLocal =
                    pinocchio::getFrameVelocity(model, data, frameIndex).linear();
                const Eigen::Matrix3d & rotFrame = transformFrameInWorld.rotation();
                const Eigen::Vector3d vContactInWorld = rotFrame * motionFrameLocal;

                // Compute the ground reaction force in world frame (local world aligned)
                const pinocchio::Force fextAtContactInGlobal =
                    computeContactDynamics(normalGround, depth, vContactInWorld);

                // Deduce the ground reaction force in joint frame
                fextLocal =
                    convertForceGlobalFrameToJoint(model, data, frameIndex, fextAtContactInGlobal);
            }
            else
            {
                // Enable fixed frame constraint
                constraint->enable();
            }
        }
        else
        {
            if (contactModel_ == ContactModelType::SPRING_DAMPER)
            {
                // Not in contact with the ground, thus no force applied
                fextLocal.setZero();
            }
            else if (depth > engineOptions_->contacts.transitionEps)
            {
                /* Disable fixed frame constraint only if the frame move higher `transitionEps` to
                   avoid sporadic contact detection. */
                constraint->disable();
            }
        }

        /* Move the reference position at the surface of the ground.
           Note that it is must done systematically as long as the constraint is enabled because in
           case of slippage the contact point has actually moved. */
        if (constraint->getIsEnabled())
        {
            auto & frameConstraint = static_cast<FrameConstraint &>(*constraint.get());
            frameConstraint.setReferenceTransform(
                {transformFrameInWorld.rotation(), posFrame - depth * normalGround});
            frameConstraint.setNormal(normalGround);
        }
    }

    pinocchio::Force EngineMultiRobot::computeContactDynamics(
        const Eigen::Vector3d & normalGround,
        double depth,
        const Eigen::Vector3d & vContactInWorld) const
    {
        // Initialize the contact force
        Eigen::Vector3d fextInWorld;

        if (depth < 0.0)
        {
            // Extract some proxies
            const ContactOptions & contactOptions_ = engineOptions_->contacts;

            // Compute the penetration speed
            const double vDepth = vContactInWorld.dot(normalGround);

            // Compute normal force
            const double fextNormal = -std::min(
                contactOptions_.stiffness * depth + contactOptions_.damping * vDepth, 0.0);
            fextInWorld.noalias() = fextNormal * normalGround;

            // Compute friction forces
            const Eigen::Vector3d vTangential = vContactInWorld - vDepth * normalGround;
            const double vRatio =
                std::min(vTangential.norm() / contactOptions_.transitionVelocity, 1.0);
            const double fextTangential = contactOptions_.friction * vRatio * fextNormal;
            fextInWorld.noalias() -= fextTangential * vTangential;

            // Add blending factor
            if (contactOptions_.transitionEps > EPS)
            {
                const double blendingFactor = -depth / contactOptions_.transitionEps;
                const double blendingLaw = std::tanh(2.0 * blendingFactor);
                fextInWorld *= blendingLaw;
            }
        }
        else
        {
            fextInWorld.setZero();
        }

        return {fextInWorld, Eigen::Vector3d::Zero()};
    }

    void EngineMultiRobot::computeCommand(System & system,
                                          double t,
                                          const Eigen::VectorXd & q,
                                          const Eigen::VectorXd & v,
                                          Eigen::VectorXd & command)
    {
        // Reinitialize the external forces
        command.setZero();

        // Command the command
        system.controller->computeCommand(t, q, v, command);
    }

    template<template<typename, int, int> class JointModel, typename Scalar, int Options, int axis>
    static std::enable_if_t<
        is_pinocchio_joint_revolute_v<JointModel<Scalar, Options, axis>> ||
            is_pinocchio_joint_revolute_unbounded_v<JointModel<Scalar, Options, axis>>,
        double>
    getSubtreeInertiaProj(const JointModel<Scalar, Options, axis> & /* model */,
                          const pinocchio::Inertia & Isubtree)
    {
        double inertiaProj = Isubtree.inertia()(axis, axis);
        for (Eigen::Index i = 0; i < 3; ++i)
        {
            if (i != axis)
            {
                inertiaProj += Isubtree.mass() * std::pow(Isubtree.lever()[i], 2);
            }
        }
        return inertiaProj;
    }

    template<typename JointModel>
    static std::enable_if_t<is_pinocchio_joint_revolute_unaligned_v<JointModel> ||
                                is_pinocchio_joint_revolute_unbounded_unaligned_v<JointModel>,
                            double>
    getSubtreeInertiaProj(const JointModel & model, const pinocchio::Inertia & Isubtree)
    {
        return model.axis.dot(Isubtree.inertia() * model.axis) +
               Isubtree.mass() * model.axis.cross(Isubtree.lever()).squaredNorm();
    }

    template<typename JointModel>
    static std::enable_if_t<is_pinocchio_joint_prismatic_v<JointModel> ||
                                is_pinocchio_joint_prismatic_unaligned_v<JointModel>,
                            double>
    getSubtreeInertiaProj(const JointModel & /* model */, const pinocchio::Inertia & Isubtree)
    {
        return Isubtree.mass();
    }

    struct computePositionLimitsForcesAlgo :
    public pinocchio::fusion::JointUnaryVisitorBase<computePositionLimitsForcesAlgo>
    {
        typedef boost::fusion::vector<
            const pinocchio::Data & /* pinocchioData */,
            const Eigen::VectorXd & /* q */,
            const Eigen::VectorXd & /* v */,
            const Eigen::VectorXd & /* positionLimitMin */,
            const Eigen::VectorXd & /* positionLimitMax */,
            const std::unique_ptr<const EngineMultiRobot::EngineOptions> & /* engineOptions */,
            ContactModelType /* contactModel */,
            std::shared_ptr<AbstractConstraintBase> & /* constraint */,
            Eigen::VectorXd & /* u */>
            ArgsType;

        template<typename JointModel>
        static std::enable_if_t<is_pinocchio_joint_revolute_v<JointModel> ||
                                    is_pinocchio_joint_revolute_unaligned_v<JointModel> ||
                                    is_pinocchio_joint_prismatic_v<JointModel> ||
                                    is_pinocchio_joint_prismatic_unaligned_v<JointModel>,
                                void>
        algo(const pinocchio::JointModelBase<JointModel> & joint,
             const pinocchio::Data & data,
             const Eigen::VectorXd & q,
             const Eigen::VectorXd & v,
             const Eigen::VectorXd & positionLimitMin,
             const Eigen::VectorXd & positionLimitMax,
             const std::unique_ptr<const EngineMultiRobot::EngineOptions> & engineOptions,
             ContactModelType contactModel,
             std::shared_ptr<AbstractConstraintBase> & constraint,
             Eigen::VectorXd & u)
        {
            // Define some proxies for convenience
            const pinocchio::JointIndex jointIndex = joint.id();
            const Eigen::Index positionIndex = joint.idx_q();
            const Eigen::Index velocityIndex = joint.idx_v();
            const double qJoint = q[positionIndex];
            const double qJointMin = positionLimitMin[positionIndex];
            const double qJointMax = positionLimitMax[positionIndex];
            const double vJoint = v[velocityIndex];
            const double Ia = getSubtreeInertiaProj(joint.derived(), data.Ycrb[jointIndex]);
            const double stiffness = engineOptions->joints.boundStiffness;
            const double damping = engineOptions->joints.boundDamping;
            const double transitionEps = engineOptions->contacts.transitionEps;

            // Check if out-of-bounds
            if (contactModel == ContactModelType::SPRING_DAMPER)
            {
                // Compute the acceleration to apply to move out of the bound
                double accelJoint = 0.0;
                if (qJoint > qJointMax)
                {
                    const double qJointError = qJoint - qJointMax;
                    accelJoint = -std::max(stiffness * qJointError + damping * vJoint, 0.0);
                }
                else if (qJoint < qJointMin)
                {
                    const double qJointError = qJoint - qJointMin;
                    accelJoint = -std::min(stiffness * qJointError + damping * vJoint, 0.0);
                }

                // Apply the resulting force
                u[velocityIndex] += Ia * accelJoint;
            }
            else
            {
                if (qJointMax < qJoint || qJoint < qJointMin)
                {
                    // Enable fixed joint constraint and reset it if it was disable
                    constraint->enable();
                    auto & jointConstraint = static_cast<JointConstraint &>(*constraint.get());
                    jointConstraint.setReferenceConfiguration(
                        Eigen::Matrix<double, 1, 1>(std::clamp(qJoint, qJointMin, qJointMax)));
                    jointConstraint.setRotationDir(qJointMax < qJoint);
                }
                else if (qJointMin + transitionEps < qJoint && qJoint < qJointMax - transitionEps)
                {
                    // Disable fixed joint constraint
                    constraint->disable();
                }
            }
        }

        template<typename JointModel>
        static std::enable_if_t<is_pinocchio_joint_revolute_unbounded_v<JointModel> ||
                                    is_pinocchio_joint_revolute_unbounded_unaligned_v<JointModel>,
                                void>
        algo(const pinocchio::JointModelBase<JointModel> & /* joint */,
             const pinocchio::Data & /* data */,
             const Eigen::VectorXd & /* q */,
             const Eigen::VectorXd & /* v */,
             const Eigen::VectorXd & /* positionLimitMin */,
             const Eigen::VectorXd & /* positionLimitMax */,
             const std::unique_ptr<const EngineMultiRobot::EngineOptions> & /* engineOptions */,
             ContactModelType contactModel,
             std::shared_ptr<AbstractConstraintBase> & constraint,
             Eigen::VectorXd & /* u */)
        {
            if (contactModel == ContactModelType::CONSTRAINT)
            {
                // Disable fixed joint constraint
                constraint->disable();
            }
        }

        template<typename JointModel>
        static std::enable_if_t<is_pinocchio_joint_freeflyer_v<JointModel> ||
                                    is_pinocchio_joint_spherical_v<JointModel> ||
                                    is_pinocchio_joint_spherical_zyx_v<JointModel> ||
                                    is_pinocchio_joint_translation_v<JointModel> ||
                                    is_pinocchio_joint_planar_v<JointModel> ||
                                    is_pinocchio_joint_mimic_v<JointModel> ||
                                    is_pinocchio_joint_composite_v<JointModel>,
                                void>
        algo(const pinocchio::JointModelBase<JointModel> & /* joint */,
             const pinocchio::Data & /* data */,
             const Eigen::VectorXd & /* q */,
             const Eigen::VectorXd & /* v */,
             const Eigen::VectorXd & /* positionLimitMin */,
             const Eigen::VectorXd & /* positionLimitMax */,
             const std::unique_ptr<const EngineMultiRobot::EngineOptions> & /* engineOptions */,
             ContactModelType contactModel,
             std::shared_ptr<AbstractConstraintBase> & constraint,
             Eigen::VectorXd & /* u */)
        {
            PRINT_WARNING("No position bounds implemented for this type of joint.");
            if (contactModel == ContactModelType::CONSTRAINT)
            {
                // Disable fixed joint constraint
                constraint->disable();
            }
        }
    };

    struct computeVelocityLimitsForcesAlgo :
    public pinocchio::fusion::JointUnaryVisitorBase<computeVelocityLimitsForcesAlgo>
    {
        typedef boost::fusion::vector<
            const pinocchio::Data & /* data */,
            const Eigen::VectorXd & /* v */,
            const Eigen::VectorXd & /* velocityLimitMax */,
            const std::unique_ptr<const EngineMultiRobot::EngineOptions> & /* engineOptions */,
            ContactModelType /* contactModel */,
            Eigen::VectorXd & /* u */>
            ArgsType;
        template<typename JointModel>
        static std::enable_if_t<is_pinocchio_joint_revolute_v<JointModel> ||
                                    is_pinocchio_joint_revolute_unaligned_v<JointModel> ||
                                    is_pinocchio_joint_revolute_unbounded_v<JointModel> ||
                                    is_pinocchio_joint_revolute_unbounded_unaligned_v<JointModel> ||
                                    is_pinocchio_joint_prismatic_v<JointModel> ||
                                    is_pinocchio_joint_prismatic_unaligned_v<JointModel>,
                                void>
        algo(const pinocchio::JointModelBase<JointModel> & joint,
             const pinocchio::Data & data,
             const Eigen::VectorXd & v,
             const Eigen::VectorXd & velocityLimitMax,
             const std::unique_ptr<const EngineMultiRobot::EngineOptions> & engineOptions,
             ContactModelType contactModel,
             Eigen::VectorXd & u)
        {
            // Define some proxies for convenience
            const pinocchio::JointIndex jointIndex = joint.id();
            const Eigen::Index velocityIndex = joint.idx_v();
            const double vJoint = v[velocityIndex];
            const double vJointMin = -velocityLimitMax[velocityIndex];
            const double vJointMax = velocityLimitMax[velocityIndex];
            const double Ia = getSubtreeInertiaProj(joint.derived(), data.Ycrb[jointIndex]);
            const double damping = engineOptions->joints.boundDamping;

            // Check if out-of-bounds
            if (contactModel == ContactModelType::SPRING_DAMPER)
            {
                // Compute joint velocity error
                double vJointError = 0.0;
                if (vJoint > vJointMax)
                {
                    vJointError = vJoint - vJointMax;
                }
                else if (vJoint < vJointMin)
                {
                    vJointError = vJoint - vJointMin;
                }
                else
                {
                    return;
                }

                // Generate acceleration in the opposite direction if out-of-bounds
                const double accelJoint = -2.0 * damping * vJointError;

                // Apply the resulting force
                u[velocityIndex] += Ia * accelJoint;
            }
        }

        template<typename JointModel>
        static std::enable_if_t<is_pinocchio_joint_freeflyer_v<JointModel> ||
                                    is_pinocchio_joint_spherical_v<JointModel> ||
                                    is_pinocchio_joint_spherical_zyx_v<JointModel> ||
                                    is_pinocchio_joint_translation_v<JointModel> ||
                                    is_pinocchio_joint_planar_v<JointModel> ||
                                    is_pinocchio_joint_mimic_v<JointModel> ||
                                    is_pinocchio_joint_composite_v<JointModel>,
                                void>
        algo(const pinocchio::JointModelBase<JointModel> & /* joint */,
             const pinocchio::Data & /* data */,
             const Eigen::VectorXd & /* v */,
             const Eigen::VectorXd & /* velocityLimitMax */,
             const std::unique_ptr<const EngineMultiRobot::EngineOptions> & /* engineOptions */,
             ContactModelType /* contactModel */,
             Eigen::VectorXd & /* u */)
        {
            PRINT_WARNING("No velocity bounds implemented for this type of joint.");
        }
    };

    void EngineMultiRobot::computeInternalDynamics(const System & system,
                                                   SystemData & systemData,
                                                   double /* t */,
                                                   const Eigen::VectorXd & q,
                                                   const Eigen::VectorXd & v,
                                                   Eigen::VectorXd & uInternal) const
    {
        // Define some proxies
        const pinocchio::Model & model = system.robot->pinocchioModel_;
        const pinocchio::Data & data = system.robot->pinocchioData_;

        // Enforce the position limit (rigid joints only)
        if (system.robot->modelOptions_->joints.enablePositionLimit)
        {
            const Eigen::VectorXd & positionLimitMin = system.robot->getPositionLimitMin();
            const Eigen::VectorXd & positionLimitMax = system.robot->getPositionLimitMax();
            const std::vector<pinocchio::JointIndex> & rigidJointIndices =
                system.robot->getRigidJointIndices();
            for (std::size_t i = 0; i < rigidJointIndices.size(); ++i)
            {
                auto & constraint = systemData.constraints.boundJoints[i].second;
                computePositionLimitsForcesAlgo::run(
                    model.joints[rigidJointIndices[i]],
                    typename computePositionLimitsForcesAlgo::ArgsType(data,
                                                                       q,
                                                                       v,
                                                                       positionLimitMin,
                                                                       positionLimitMax,
                                                                       engineOptions_,
                                                                       contactModel_,
                                                                       constraint,
                                                                       uInternal));
            }
        }

        // Enforce the velocity limit (rigid joints only)
        if (system.robot->modelOptions_->joints.enableVelocityLimit)
        {
            const Eigen::VectorXd & velocityLimitMax = system.robot->getVelocityLimit();
            for (pinocchio::JointIndex rigidJointIndex : system.robot->getRigidJointIndices())
            {
                computeVelocityLimitsForcesAlgo::run(
                    model.joints[rigidJointIndex],
                    typename computeVelocityLimitsForcesAlgo::ArgsType(
                        data, v, velocityLimitMax, engineOptions_, contactModel_, uInternal));
            }
        }

        // Compute the flexibilities (only support JointModelType::SPHERICAL so far)
        double angle;
        Eigen::Matrix3d rotJlog3;
        const Robot::DynamicsOptions & modelDynOptions = system.robot->modelOptions_->dynamics;
        const std::vector<pinocchio::JointIndex> & flexibilityJointIndices =
            system.robot->getFlexibleJointIndices();
        for (std::size_t i = 0; i < flexibilityJointIndices.size(); ++i)
        {
            const pinocchio::JointIndex jointIndex = flexibilityJointIndices[i];
            const Eigen::Index positionIndex = model.joints[jointIndex].idx_q();
            const Eigen::Index velocityIndex = model.joints[jointIndex].idx_v();
            const Eigen::Vector3d & stiffness = modelDynOptions.flexibilityConfig[i].stiffness;
            const Eigen::Vector3d & damping = modelDynOptions.flexibilityConfig[i].damping;

            const Eigen::Map<const Eigen::Quaterniond> quat(q.segment<4>(positionIndex).data());
            const Eigen::Vector3d angleAxis = pinocchio::quaternion::log3(quat, angle);
            assert((angle < 0.95 * M_PI) &&
                   "Flexible joint angle must be smaller than 0.95 * pi.");
            pinocchio::Jlog3(angle, angleAxis, rotJlog3);
            uInternal.segment<3>(velocityIndex) -=
                rotJlog3 * (stiffness.array() * angleAxis.array()).matrix();
            uInternal.segment<3>(velocityIndex).array() -=
                damping.array() * v.segment<3>(velocityIndex).array();
        }
    }

    void EngineMultiRobot::computeCollisionForces(const System & system,
                                                  SystemData & systemData,
                                                  ForceVector & fext,
                                                  bool isStateUpToDate) const
    {
        // Compute the forces at contact points
        const std::vector<pinocchio::FrameIndex> & contactFrameIndices =
            system.robot->getContactFrameIndices();
        for (std::size_t i = 0; i < contactFrameIndices.size(); ++i)
        {
            // Compute force at the given contact frame.
            const pinocchio::FrameIndex frameIndex = contactFrameIndices[i];
            auto & constraint = systemData.constraints.contactFrames[i].second;
            pinocchio::Force & fextLocal = systemData.contactFrameForces[i];
            if (!isStateUpToDate)
            {
                computeContactDynamicsAtFrame(system, frameIndex, constraint, fextLocal);
            }

            // Apply the force at the origin of the parent joint frame, in local joint frame
            const pinocchio::JointIndex parentJointIndex =
                system.robot->pinocchioModel_.frames[frameIndex].parent;
            fext[parentJointIndex] += fextLocal;

            // Convert contact force from global frame to local frame to store it in contactForces_
            const pinocchio::SE3 & transformContactInJoint =
                system.robot->pinocchioModel_.frames[frameIndex].placement;
            system.robot->contactForces_[i] = transformContactInJoint.actInv(fextLocal);
        }

        // Compute the force at collision bodies
        const std::vector<pinocchio::FrameIndex> & collisionBodyIndices =
            system.robot->getCollisionBodyIndices();
        const std::vector<std::vector<pinocchio::PairIndex>> & collisionPairIndices =
            system.robot->getCollisionPairIndices();
        for (std::size_t i = 0; i < collisionBodyIndices.size(); ++i)
        {
            /* Compute force at the given collision body.
               It returns the force applied at the origin of parent joint frame in global frame. */
            const pinocchio::FrameIndex frameIndex = collisionBodyIndices[i];
            const pinocchio::JointIndex parentJointIndex =
                system.robot->pinocchioModel_.frames[frameIndex].parent;
            for (std::size_t j = 0; j < collisionPairIndices[i].size(); ++j)
            {
                pinocchio::Force & fextLocal = systemData.collisionBodiesForces[i][j];
                if (!isStateUpToDate)
                {
                    const pinocchio::PairIndex & collisionPairIndex = collisionPairIndices[i][j];
                    auto & constraint = systemData.constraints.collisionBodies[i][j].second;
                    computeContactDynamicsAtBody(
                        system, collisionPairIndex, constraint, fextLocal);
                }

                // Apply the force at the origin of the parent joint frame, in local joint frame
                fext[parentJointIndex] += fextLocal;
            }
        }
    }

    void EngineMultiRobot::computeExternalForces(const System & system,
                                                 SystemData & systemData,
                                                 double t,
                                                 const Eigen::VectorXd & q,
                                                 const Eigen::VectorXd & v,
                                                 ForceVector & fext)
    {
        // Add the effect of user-defined external impulse forces
        auto isImpulseForceActiveIt = systemData.isImpulseForceActiveVec.begin();
        auto impulseForceIt = systemData.impulseForces.begin();
        for (; impulseForceIt != systemData.impulseForces.end();
             ++isImpulseForceActiveIt, ++impulseForceIt)
        {
            /* Do not check if the force is active at this point. This is managed at stepper level
               to get around the ambiguous t- versus t+. */
            if (*isImpulseForceActiveIt)
            {
                const pinocchio::FrameIndex frameIndex = impulseForceIt->frameIndex;
                const pinocchio::JointIndex parentJointIndex =
                    system.robot->pinocchioModel_.frames[frameIndex].parent;
                fext[parentJointIndex] +=
                    convertForceGlobalFrameToJoint(system.robot->pinocchioModel_,
                                                   system.robot->pinocchioData_,
                                                   frameIndex,
                                                   impulseForceIt->force);
            }
        }

        // Add the effect of time-continuous external force profiles
        for (auto & profileForce : systemData.profileForces)
        {
            const pinocchio::FrameIndex frameIndex = profileForce.frameIndex;
            const pinocchio::JointIndex parentJointIndex =
                system.robot->pinocchioModel_.frames[frameIndex].parent;
            if (profileForce.updatePeriod < EPS)
            {
                profileForce.force = profileForce.func(t, q, v);
            }
            fext[parentJointIndex] += convertForceGlobalFrameToJoint(system.robot->pinocchioModel_,
                                                                     system.robot->pinocchioData_,
                                                                     frameIndex,
                                                                     profileForce.force);
        }
    }

    void EngineMultiRobot::computeCouplingForces(double t,
                                                 const std::vector<Eigen::VectorXd> & qSplit,
                                                 const std::vector<Eigen::VectorXd> & vSplit)
    {
        for (auto & couplingForce : couplingForces_)
        {
            // Extract info about the first system involved
            const std::ptrdiff_t systemIndex1 = couplingForce.systemIndex1;
            const System & system1 = systems_[systemIndex1];
            const Eigen::VectorXd & q1 = qSplit[systemIndex1];
            const Eigen::VectorXd & v1 = vSplit[systemIndex1];
            const pinocchio::FrameIndex frameIndex1 = couplingForce.frameIndex1;
            ForceVector & fext1 = systemDataVec_[systemIndex1].state.fExternal;

            // Extract info about the second system involved
            const std::ptrdiff_t systemIndex2 = couplingForce.systemIndex2;
            const System & system2 = systems_[systemIndex2];
            const Eigen::VectorXd & q2 = qSplit[systemIndex2];
            const Eigen::VectorXd & v2 = vSplit[systemIndex2];
            const pinocchio::FrameIndex frameIndex2 = couplingForce.frameIndex2;
            ForceVector & fext2 = systemDataVec_[systemIndex2].state.fExternal;

            // Compute the coupling force
            pinocchio::Force force = couplingForce.func(t, q1, v1, q2, v2);
            const pinocchio::JointIndex parentJointIndex1 =
                system1.robot->pinocchioModel_.frames[frameIndex1].parent;
            fext1[parentJointIndex1] += convertForceGlobalFrameToJoint(
                system1.robot->pinocchioModel_, system1.robot->pinocchioData_, frameIndex1, force);

            // Move force from frame1 to frame2 to apply it to the second system
            force.toVector() *= -1;
            const pinocchio::JointIndex parentJointIndex2 =
                system2.robot->pinocchioModel_.frames[frameIndex2].parent;
            const Eigen::Vector3d offset =
                system2.robot->pinocchioData_.oMf[frameIndex2].translation() -
                system1.robot->pinocchioData_.oMf[frameIndex1].translation();
            force.angular() -= offset.cross(force.linear());
            fext2[parentJointIndex2] += convertForceGlobalFrameToJoint(
                system2.robot->pinocchioModel_, system2.robot->pinocchioData_, frameIndex2, force);
        }
    }

    void EngineMultiRobot::computeAllTerms(double t,
                                           const std::vector<Eigen::VectorXd> & qSplit,
                                           const std::vector<Eigen::VectorXd> & vSplit,
                                           bool isStateUpToDate)
    {
        // Reinitialize the external forces and internal efforts
        for (auto & systemData : systemDataVec_)
        {
            for (pinocchio::Force & fext_i : systemData.state.fExternal)
            {
                fext_i.setZero();
            }
            if (!isStateUpToDate)
            {
                systemData.state.uInternal.setZero();
            }
        }

        // Compute the internal forces
        computeCouplingForces(t, qSplit, vSplit);

        // Compute each individual system dynamics
        auto systemIt = systems_.begin();
        auto systemDataIt = systemDataVec_.begin();
        auto qIt = qSplit.begin();
        auto vIt = vSplit.begin();
        for (; systemIt != systems_.end(); ++systemIt, ++systemDataIt, ++qIt, ++vIt)
        {
            // Define some proxies
            ForceVector & fext = systemDataIt->state.fExternal;
            Eigen::VectorXd & uInternal = systemDataIt->state.uInternal;

            /* Compute internal dynamics, namely the efforts in joint space associated with
               position/velocity bounds dynamics, and flexibility dynamics. */
            if (!isStateUpToDate)
            {
                computeInternalDynamics(*systemIt, *systemDataIt, t, *qIt, *vIt, uInternal);
            }

            /* Compute the collision forces and estimated time at which the contact state will
               changed (Take-off / Touch-down). */
            computeCollisionForces(*systemIt, *systemDataIt, fext, isStateUpToDate);

            // Compute the external contact forces.
            computeExternalForces(*systemIt, *systemDataIt, t, *qIt, *vIt, fext);
        }
    }

    void EngineMultiRobot::computeSystemsDynamics(double t,
                                                  const std::vector<Eigen::VectorXd> & qSplit,
                                                  const std::vector<Eigen::VectorXd> & vSplit,
                                                  std::vector<Eigen::VectorXd> & aSplit,
                                                  bool isStateUpToDate)
    {
        /* Note that the position of the free flyer is in world frame, whereas the velocities and
           accelerations are relative to the parent body frame. */

        // Make sure that a simulation is running
        if (!isSimulationRunning_)
        {
            THROW_ERROR(std::logic_error,
                        "No simulation running. Please start one before calling this method.");
        }

        // Make sure memory has been allocated for the output acceleration
        aSplit.resize(vSplit.size());

        if (!isStateUpToDate)
        {
            // Update kinematics for each system
            auto systemIt = systems_.begin();
            auto systemDataIt = systemDataVec_.begin();
            auto qIt = qSplit.begin();
            auto vIt = vSplit.begin();
            for (; systemIt != systems_.end(); ++systemIt, ++systemDataIt, ++qIt, ++vIt)
            {
                const Eigen::VectorXd & aPrev = systemDataIt->statePrev.a;
                computeForwardKinematics(*systemIt, *qIt, *vIt, aPrev);
            }
        }

        /* Compute internal and external forces and efforts applied on every systems, excluding
           user-specified internal dynamics if any.

           Note that one must call this method BEFORE updating the sensors since the force sensor
           measurements rely on robot_->contactForces_. */
        computeAllTerms(t, qSplit, vSplit, isStateUpToDate);

        // Compute each individual system dynamics
        auto systemIt = systems_.begin();
        auto systemDataIt = systemDataVec_.begin();
        auto qIt = qSplit.begin();
        auto vIt = vSplit.begin();
        auto contactForcesPrevIt = contactForcesPrev_.begin();
        auto fPrevIt = fPrev_.begin();
        auto aPrevIt = aPrev_.begin();
        auto aIt = aSplit.begin();
        for (; systemIt != systems_.end(); ++systemIt,
                                           ++systemDataIt,
                                           ++qIt,
                                           ++vIt,
                                           ++aIt,
                                           ++contactForcesPrevIt,
                                           ++fPrevIt,
                                           ++aPrevIt)
        {
            // Define some proxies
            Eigen::VectorXd & u = systemDataIt->state.u;
            Eigen::VectorXd & command = systemDataIt->state.command;
            Eigen::VectorXd & uMotor = systemDataIt->state.uMotor;
            Eigen::VectorXd & uInternal = systemDataIt->state.uInternal;
            Eigen::VectorXd & uCustom = systemDataIt->state.uCustom;
            ForceVector & fext = systemDataIt->state.fExternal;
            const Eigen::VectorXd & aPrev = systemDataIt->statePrev.a;
            const Eigen::VectorXd & uMotorPrev = systemDataIt->statePrev.uMotor;
            const ForceVector & fextPrev = systemDataIt->statePrev.fExternal;

            /* Update the sensor data if necessary (only for infinite update frequency).
               Note that it is impossible to have access to the current accelerations and efforts
               since they depend on the sensor values themselves. */
            if (!isStateUpToDate && engineOptions_->stepper.sensorsUpdatePeriod < EPS)
            {
                // Roll back to forces and accelerations computed at previous iteration
                contactForcesPrevIt->swap(systemIt->robot->contactForces_);
                fPrevIt->swap(systemIt->robot->pinocchioData_.f);
                aPrevIt->swap(systemIt->robot->pinocchioData_.a);

                // Update sensors based on previous accelerations and forces
                systemIt->robot->computeSensorMeasurements(
                    t, *qIt, *vIt, aPrev, uMotorPrev, fextPrev);

                // Restore current forces and accelerations
                contactForcesPrevIt->swap(systemIt->robot->contactForces_);
                fPrevIt->swap(systemIt->robot->pinocchioData_.f);
                aPrevIt->swap(systemIt->robot->pinocchioData_.a);
            }

            /* Update the controller command if necessary (only for infinite update frequency).
               Make sure that the sensor state has been updated beforehand. */
            if (engineOptions_->stepper.controllerUpdatePeriod < EPS)
            {
                computeCommand(*systemIt, t, *qIt, *vIt, command);
            }

            /* Compute the actual motor effort.
               Note that it is impossible to have access to the current accelerations. */
            systemIt->robot->computeMotorEfforts(t, *qIt, *vIt, aPrev, command);
            uMotor = systemIt->robot->getMotorEfforts();

            /* Compute the user-defined internal dynamics.
               Make sure that the sensor state has been updated beforehand since the user-defined
               internal dynamics may rely on it. */
            uCustom.setZero();
            systemIt->controller->internalDynamics(t, *qIt, *vIt, uCustom);

            // Compute the total effort vector
            u = uInternal + uCustom;
            for (const auto & motor : systemIt->robot->getMotors())
            {
                const std::size_t motorIndex = motor->getIndex();
                const Eigen::Index motorVelocityIndex = motor->getJointVelocityIndex();
                u[motorVelocityIndex] += uMotor[motorIndex];
            }

            // Compute the dynamics
            *aIt = computeAcceleration(
                *systemIt, *systemDataIt, *qIt, *vIt, u, fext, isStateUpToDate);
        }
    }

    const Eigen::VectorXd & EngineMultiRobot::computeAcceleration(System & system,
                                                                  SystemData & systemData,
                                                                  const Eigen::VectorXd & q,
                                                                  const Eigen::VectorXd & v,
                                                                  const Eigen::VectorXd & u,
                                                                  ForceVector & fext,
                                                                  bool isStateUpToDate,
                                                                  bool ignoreBounds)
    {
        const pinocchio::Model & model = system.robot->pinocchioModel_;
        pinocchio::Data & data = system.robot->pinocchioData_;

        if (system.robot->hasConstraints())
        {
            if (!isStateUpToDate)
            {
                // Compute kinematic constraints. It will take care of updating the joint Jacobian.
                system.robot->computeConstraints(q, v);

                // Compute non-linear effects
                pinocchio::nonLinearEffects(model, data, q, v);
            }

            // Project external forces from cartesian space to joint space.
            data.u = u;
            for (int i = 1; i < model.njoints; ++i)
            {
                /* Skip computation if force is zero for efficiency. It should be the case most
                   often. */
                if ((fext[i].toVector().array().abs() > EPS).any())
                {
                    if (!isStateUpToDate)
                    {
                        pinocchio::getJointJacobian(
                            model, data, i, pinocchio::LOCAL, systemData.jointJacobians[i]);
                    }
                    data.u.noalias() +=
                        systemData.jointJacobians[i].transpose() * fext[i].toVector();
                }
            }

            // Call forward dynamics
            bool isSucess = systemData.constraintSolver->SolveBoxedForwardDynamics(
                engineOptions_->constraints.regularization, isStateUpToDate, ignoreBounds);

            /* Monitor number of successive constraint solving failure. Exception raising is
               delegated to the 'step' method since this method is supposed to always succeed. */
            if (isSucess)
            {
                systemData.successiveSolveFailed = 0U;
            }
            else
            {
                if (engineOptions_->stepper.verbose)
                {
                    std::cout << "Constraint solver failure." << std::endl;
                }
                ++systemData.successiveSolveFailed;
            }

            // Restore contact frame forces and bounds internal efforts
            systemData.constraints.foreach(
                ConstraintNodeType::BOUNDS_JOINTS,
                [&u = systemData.state.u,
                 &uInternal = systemData.state.uInternal,
                 &joints = const_cast<pinocchio::Model::JointModelVector &>(model.joints)](
                    std::shared_ptr<AbstractConstraintBase> & constraint,
                    ConstraintNodeType /* node */)
                {
                    if (!constraint->getIsEnabled())
                    {
                        return;
                    }

                    Eigen::VectorXd & uJoint = constraint->lambda_;
                    const auto & jointConstraint =
                        static_cast<const JointConstraint &>(*constraint.get());
                    const auto & jointModel = joints[jointConstraint.getJointIndex()];
                    jointModel.jointVelocitySelector(uInternal) += uJoint;
                    jointModel.jointVelocitySelector(u) += uJoint;
                });

            auto constraintIt = systemData.constraints.contactFrames.begin();
            auto forceIt = system.robot->contactForces_.begin();
            for (; constraintIt != systemData.constraints.contactFrames.end();
                 ++constraintIt, ++forceIt)
            {
                auto & constraint = *constraintIt->second.get();
                if (!constraint.getIsEnabled())
                {
                    continue;
                }
                const auto & frameConstraint = static_cast<const FrameConstraint &>(constraint);

                // Extract force in local reference-frame-aligned from lagrangian multipliers
                pinocchio::Force fextInLocal(frameConstraint.lambda_.head<3>(),
                                             frameConstraint.lambda_[3] *
                                                 Eigen::Vector3d::UnitZ());

                // Compute force in local world aligned frame
                const Eigen::Matrix3d & rotationLocal = frameConstraint.getLocalFrame();
                const pinocchio::Force fextInWorld({
                    rotationLocal * fextInLocal.linear(),
                    rotationLocal * fextInLocal.angular(),
                });

                // Convert the force from local world aligned frame to local frame
                const pinocchio::FrameIndex frameIndex = frameConstraint.getFrameIndex();
                const auto rotationWorldInContact = data.oMf[frameIndex].rotation().transpose();
                forceIt->linear().noalias() = rotationWorldInContact * fextInWorld.linear();
                forceIt->angular().noalias() = rotationWorldInContact * fextInWorld.angular();

                // Convert the force from local world aligned to local parent joint
                pinocchio::JointIndex parentJointIndex = model.frames[frameIndex].parent;
                fext[parentJointIndex] +=
                    convertForceGlobalFrameToJoint(model, data, frameIndex, fextInWorld);
            }

            systemData.constraints.foreach(
                ConstraintNodeType::COLLISION_BODIES,
                [&fext, &model, &data](std::shared_ptr<AbstractConstraintBase> & constraint,
                                       ConstraintNodeType /* node */)
                {
                    if (!constraint->getIsEnabled())
                    {
                        return;
                    }
                    const auto & frameConstraint =
                        static_cast<const FrameConstraint &>(*constraint.get());

                    // Extract force in world frame from lagrangian multipliers
                    pinocchio::Force fextInLocal(frameConstraint.lambda_.head<3>(),
                                                 frameConstraint.lambda_[3] *
                                                     Eigen::Vector3d::UnitZ());

                    // Compute force in world frame
                    const Eigen::Matrix3d & rotationLocal = frameConstraint.getLocalFrame();
                    const pinocchio::Force fextInWorld({
                        rotationLocal * fextInLocal.linear(),
                        rotationLocal * fextInLocal.angular(),
                    });

                    // Convert the force from local world aligned to local parent joint
                    const pinocchio::FrameIndex frameIndex = frameConstraint.getFrameIndex();
                    const pinocchio::JointIndex parentJointIndex = model.frames[frameIndex].parent;
                    fext[parentJointIndex] +=
                        convertForceGlobalFrameToJoint(model, data, frameIndex, fextInWorld);
                });

            return data.ddq;
        }
        else
        {
            // No kinematic constraint: run aba algorithm
            return pinocchio_overload::aba(model, data, q, v, u, fext);
        }
    }

    // ===================================================================
    // ================ Log reading and writing utilities ================
    // ===================================================================

    std::shared_ptr<const LogData> EngineMultiRobot::getLog()
    {
        // Update internal log data buffer if uninitialized
        if (!logData_)
        {
            logData_ = std::make_shared<LogData>(telemetryRecorder_->getLog());
        }

        // Return shared pointer to internal log data buffer
        return std::const_pointer_cast<const LogData>(logData_);
    }

    LogData readLogHdf5(const std::string & filename)
    {
        LogData logData{};

        // Open HDF5 logfile
        std::unique_ptr<H5::H5File> file;
        try
        {
            /* Specifying `H5F_CLOSE_STRONG` is necessary to completely close the file (including
               all open objects) before returning. See:
               https://docs.hdfgroup.org/hdf5/v1_12/group___f_a_p_l.html#ga60e3567f677fd3ade75b909b636d7b9c
            */
            H5::FileAccPropList access_plist;
            access_plist.setFcloseDegree(H5F_CLOSE_STRONG);
            file = std::make_unique<H5::H5File>(
                filename, H5F_ACC_RDONLY, H5::FileCreatPropList::DEFAULT, access_plist);
        }
        catch (const H5::FileIException &)
        {
            THROW_ERROR(std::runtime_error,
                        "Impossible to open the log file. Make sure it exists and "
                        "you have reading permissions.");
        }

        // Extract all constants. There is no ordering among them, unlike variables.
        H5::Group constantsGroup = file->openGroup("/constants");
        file->iterateElems(
            "/constants",
            NULL,
            [](hid_t group, const char * name, void * op_data) -> herr_t
            {
                LogData * logDataPtr = static_cast<LogData *>(op_data);
                H5::Group _constantsGroup(group);
                const H5::DataSet constantDataSet = _constantsGroup.openDataSet(name);
                const H5::DataSpace constantSpace = H5::DataSpace(H5S_SCALAR);
                const H5::StrType constantDataType = constantDataSet.getStrType();
                const hssize_t numBytes = constantDataType.getSize();
                H5::StrType stringType(H5::PredType::C_S1, numBytes);
                stringType.setStrpad(H5T_str_t::H5T_STR_NULLPAD);
                std::string value(numBytes, '\0');
                constantDataSet.read(value.data(), stringType, constantSpace);
                logDataPtr->constants.emplace_back(name, std::move(value));
                return 0;
            },
            static_cast<void *>(&logData));

        // Extract the times
        const H5::DataSet globalTimeDataSet = file->openDataSet(std::string{GLOBAL_TIME});
        const H5::DataSpace timeSpace = globalTimeDataSet.getSpace();
        const hssize_t numData = timeSpace.getSimpleExtentNpoints();
        logData.times.resize(numData);
        globalTimeDataSet.read(logData.times.data(), H5::PredType::NATIVE_INT64);

        // Add "unit" attribute to GLOBAL_TIME vector
        const H5::Attribute unitAttrib = globalTimeDataSet.openAttribute("unit");
        unitAttrib.read(H5::PredType::NATIVE_DOUBLE, &logData.timeUnit);

        // Get the (partitioned) number of variables
        H5::Group variablesGroup = file->openGroup("/variables");
        int64_t numInt = 0, numFloat = 0;
        std::pair<int64_t &, int64_t &> numVar{numInt, numFloat};
        H5Literate(
            variablesGroup.getId(),
            H5_INDEX_CRT_ORDER,
            H5_ITER_INC,
            NULL,
            [](hid_t group, const char * name, const H5L_info_t * /* oinfo */, void * op_data)
                -> herr_t
            {
                auto & [_numInt, _numFloat] =
                    *static_cast<std::pair<int64_t &, int64_t &> *>(op_data);
                H5::Group fieldGroup = H5::Group(group).openGroup(name);
                const H5::DataSet valueDataset = fieldGroup.openDataSet("value");
                const H5T_class_t valueType = valueDataset.getTypeClass();
                if (valueType == H5T_FLOAT)
                {
                    ++_numFloat;
                }
                else
                {
                    ++_numInt;
                }
                return 0;
            },
            static_cast<void *>(&numVar));

        // Pre-allocate memory
        logData.integerValues.resize(numInt, numData);
        logData.floatValues.resize(numFloat, numData);
        VectorX<int64_t> intVector(numData);
        VectorX<double> floatVector(numData);
        logData.variableNames.reserve(1 + numInt + numFloat);
        logData.variableNames.emplace_back(GLOBAL_TIME);

        // Read all variables while preserving ordering
        using opDataT = std::tuple<LogData &, VectorX<int64_t> &, VectorX<double> &>;
        opDataT opData{logData, intVector, floatVector};
        H5Literate(
            variablesGroup.getId(),
            H5_INDEX_CRT_ORDER,
            H5_ITER_INC,
            NULL,
            [](hid_t group, const char * name, const H5L_info_t * /* oinfo */, void * op_data)
                -> herr_t
            {
                auto & [logDataIn, intVectorIn, floatVectorIn] = *static_cast<opDataT *>(op_data);
                const Eigen::Index varIndex = logDataIn.variableNames.size() - 1;
                const int64_t numIntIn = logDataIn.integerValues.rows();
                H5::Group fieldGroup = H5::Group(group).openGroup(name);
                const H5::DataSet valueDataset = fieldGroup.openDataSet("value");
                if (varIndex < numIntIn)
                {
                    valueDataset.read(intVectorIn.data(), H5::PredType::NATIVE_INT64);
                    logDataIn.integerValues.row(varIndex) = intVectorIn;
                }
                else
                {
                    valueDataset.read(floatVectorIn.data(), H5::PredType::NATIVE_DOUBLE);
                    logDataIn.floatValues.row(varIndex - numIntIn) = floatVectorIn;
                }
                logDataIn.variableNames.push_back(name);
                return 0;
            },
            static_cast<void *>(&opData));

        // Close file once done
        file->close();

        return logData;
    }

    LogData EngineMultiRobot::readLog(const std::string & filename, const std::string & format)
    {
        if (format == "binary")
        {
            return TelemetryRecorder::readLog(filename);
        }
        if (format == "hdf5")
        {
            return readLogHdf5(filename);
        }
        THROW_ERROR(std::invalid_argument,
                    "Format '",
                    format,
                    "' not recognized. It must be either 'binary' or 'hdf5'.");
    }

    void writeLogHdf5(const std::string & filename, const std::shared_ptr<const LogData> & logData)
    {
        // Open HDF5 logfile
        std::unique_ptr<H5::H5File> file;
        try
        {
            H5::FileAccPropList access_plist;
            access_plist.setFcloseDegree(H5F_CLOSE_STRONG);
            file = std::make_unique<H5::H5File>(
                filename, H5F_ACC_TRUNC, H5::FileCreatPropList::DEFAULT, access_plist);
        }
        catch (const H5::FileIException & open_file)
        {
            THROW_ERROR(std::runtime_error,
                        "Impossible to create the log file. Make sure the root folder "
                        "exists and you have writing permissions.");
        }

        // Add "VERSION" attribute
        const H5::DataSpace versionSpace = H5::DataSpace(H5S_SCALAR);
        const H5::Attribute versionAttrib =
            file->createAttribute("VERSION", H5::PredType::NATIVE_INT32, versionSpace);
        versionAttrib.write(H5::PredType::NATIVE_INT32, &logData->version);

        // Add "START_TIME" attribute
        int64_t time = std::time(nullptr);
        const H5::DataSpace startTimeSpace = H5::DataSpace(H5S_SCALAR);
        const H5::Attribute startTimeAttrib =
            file->createAttribute("START_TIME", H5::PredType::NATIVE_INT64, startTimeSpace);
        startTimeAttrib.write(H5::PredType::NATIVE_INT64, &time);

        // Add GLOBAL_TIME vector
        const hsize_t timeDims[1] = {hsize_t(logData->times.size())};
        const H5::DataSpace globalTimeSpace = H5::DataSpace(1, timeDims);
        const H5::DataSet globalTimeDataSet = file->createDataSet(
            std::string{GLOBAL_TIME}, H5::PredType::NATIVE_INT64, globalTimeSpace);
        globalTimeDataSet.write(logData->times.data(), H5::PredType::NATIVE_INT64);

        // Add "unit" attribute to GLOBAL_TIME vector
        const H5::DataSpace unitSpace = H5::DataSpace(H5S_SCALAR);
        const H5::Attribute unitAttrib =
            globalTimeDataSet.createAttribute("unit", H5::PredType::NATIVE_DOUBLE, unitSpace);
        unitAttrib.write(H5::PredType::NATIVE_DOUBLE, &logData->timeUnit);

        // Add group "constants"
        H5::Group constantsGroup(file->createGroup("constants"));
        for (const auto & [key, value] : logData->constants)
        {
            // Define a dataset with a single string of fixed length
            const H5::DataSpace constantSpace = H5::DataSpace(H5S_SCALAR);
            H5::StrType stringType(H5::PredType::C_S1, std::max(value.size(), std::size_t(1)));

            // To tell parser continue reading if '\0' is encountered
            stringType.setStrpad(H5T_str_t::H5T_STR_NULLPAD);

            // Write the constant
            H5::DataSet constantDataSet =
                constantsGroup.createDataSet(key, stringType, constantSpace);
            constantDataSet.write(value, stringType);
        }

        // Temporary contiguous storage for variables
        VectorX<int64_t> intVector;
        VectorX<double> floatVector;

        // Get the number of integer and float variables
        const Eigen::Index numInt = logData->integerValues.rows();
        const Eigen::Index numFloat = logData->floatValues.rows();

        /* Add group "variables".
           C++ helper `file->createGroup("variables")` cannot be used
           because we want to preserve order. */
        hid_t group_creation_plist = H5Pcreate(H5P_GROUP_CREATE);
        H5Pset_link_creation_order(group_creation_plist,
                                   H5P_CRT_ORDER_TRACKED | H5P_CRT_ORDER_INDEXED);
        hid_t group_id = H5Gcreate(
            file->getId(), "/variables/", H5P_DEFAULT, group_creation_plist, H5P_DEFAULT);
        H5::Group variablesGroup(group_id);

        // Store all integers
        for (Eigen::Index i = 0; i < numInt; ++i)
        {
            const std::string & key = logData->variableNames[i];

            // Create group for field
            H5::Group fieldGroup = variablesGroup.createGroup(key);

            // Enable compression and shuffling
            H5::DSetCreatPropList plist;
            const hsize_t chunkSize[1] = {std::max(hsize_t(1), hsize_t(logData->times.size()))};
            plist.setChunk(1, chunkSize);  // Read the whole vector at once.
            plist.setShuffle();
            plist.setDeflate(4);

            // Create time dataset using symbolic link
            fieldGroup.link(H5L_TYPE_HARD, toString("/", GLOBAL_TIME), "time");

            // Create variable dataset
            H5::DataSpace valueSpace = H5::DataSpace(1, timeDims);
            H5::DataSet valueDataset =
                fieldGroup.createDataSet("value", H5::PredType::NATIVE_INT64, valueSpace, plist);

            // Write values in one-shot for efficiency
            intVector = logData->integerValues.row(i);
            valueDataset.write(intVector.data(), H5::PredType::NATIVE_INT64);
        }

        // Store all floats
        for (Eigen::Index i = 0; i < numFloat; ++i)
        {
            const std::string & key = logData->variableNames[i + 1 + numInt];

            // Create group for field
            H5::Group fieldGroup(variablesGroup.createGroup(key));

            // Enable compression and shuffling
            H5::DSetCreatPropList plist;
            const hsize_t chunkSize[1] = {std::max(hsize_t(1), hsize_t(logData->times.size()))};
            plist.setChunk(1, chunkSize);  // Read the whole vector at once.
            plist.setShuffle();
            plist.setDeflate(4);

            // Create time dataset using symbolic link
            fieldGroup.link(H5L_TYPE_HARD, toString("/", GLOBAL_TIME), "time");

            // Create variable dataset
            H5::DataSpace valueSpace = H5::DataSpace(1, timeDims);
            H5::DataSet valueDataset =
                fieldGroup.createDataSet("value", H5::PredType::NATIVE_DOUBLE, valueSpace, plist);

            // Write values
            floatVector = logData->floatValues.row(i);
            valueDataset.write(floatVector.data(), H5::PredType::NATIVE_DOUBLE);
        }

        // Close file once done
        file->close();
    }

    void EngineMultiRobot::writeLog(const std::string & filename, const std::string & format)
    {
        // Make sure there is something to write
        if (!isTelemetryConfigured_)
        {
            THROW_ERROR(bad_control_flow,
                        "Telemetry not configured. Please start a simulation before writing log.");
        }

        // Pick the appropriate format
        if (format == "binary")
        {
            telemetryRecorder_->writeLog(filename);
        }
        else if (format == "hdf5")
        {
            // Extract log data
            std::shared_ptr<const LogData> logData = getLog();

            // Write log data
            writeLogHdf5(filename, logData);
        }
        else
        {
            THROW_ERROR(std::invalid_argument,
                        "Format '",
                        format,
                        "' not recognized. It must be either 'binary' or 'hdf5'.");
        }
    }
}
