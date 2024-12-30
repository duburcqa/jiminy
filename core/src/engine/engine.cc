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
#include "pinocchio/algorithm/joint-configuration.hpp"  // `pinocchio::neutral`, `pinocchio::normalize`
#include "pinocchio/algorithm/geometry.hpp"             // `pinocchio::computeCollisions`

#include "H5Cpp.h"
#include "json/json.h"

#include "jiminy/core/telemetry/fwd.h"
#include "jiminy/core/hardware/fwd.h"
#include "jiminy/core/utilities/helpers.h"
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
#include "jiminy/core/solver/constraint_solvers.h"
#include "jiminy/core/stepper/abstract_stepper.h"
#include "jiminy/core/stepper/euler_explicit_stepper.h"
#include "jiminy/core/stepper/runge_kutta_dopri_stepper.h"
#include "jiminy/core/stepper/runge_kutta4_stepper.h"

#include "jiminy/core/engine/engine.h"

namespace jiminy
{
    inline constexpr uint32_t INIT_ITERATIONS{4U};
    inline constexpr uint32_t PGS_MAX_ITERATIONS{100U};

    // ******************************** External force functors ******************************** //

    ProfileForce::ProfileForce(const std::string & frameNameIn,
                               pinocchio::FrameIndex frameIndexIn,
                               double updatePeriodIn,
                               const ProfileForceFunction & forceFuncIn) noexcept :
    frameName{frameNameIn},
    frameIndex{frameIndexIn},
    updatePeriod{updatePeriodIn},
    func{forceFuncIn}
    {
    }

    ImpulseForce::ImpulseForce(const std::string & frameNameIn,
                               pinocchio::FrameIndex frameIndexIn,
                               double tIn,
                               double dtIn,
                               const pinocchio::Force & forceIn) noexcept :
    frameName{frameNameIn},
    frameIndex{frameIndexIn},
    t{tIn},
    dt{dtIn},
    force{forceIn}
    {
    }

    CouplingForce::CouplingForce(const std::string & robotName1In,
                                 std::ptrdiff_t robotIndex1In,
                                 const std::string & robotName2In,
                                 std::ptrdiff_t robotIndex2In,
                                 const std::string & frameName1In,
                                 pinocchio::FrameIndex frameIndex1In,
                                 const std::string & frameName2In,
                                 pinocchio::FrameIndex frameIndex2In,
                                 const CouplingForceFunction & forceFuncIn) noexcept :
    robotName1{robotName1In},
    robotIndex1{robotIndex1In},
    robotName2{robotName2In},
    robotIndex2{robotIndex2In},
    frameName1{frameName1In},
    frameIndex1{frameIndex1In},
    frameName2{frameName2In},
    frameIndex2{frameIndex2In},
    func{forceFuncIn}
    {
    }

    // ************************************** System state ************************************* //

    void RobotState::initialize(const Robot & robot)
    {
        if (!robot.getIsInitialized())
        {
            JIMINY_THROW(bad_control_flow, "Robot not initialized.");
        }

        Eigen::Index nv = robot.nv();
        std::size_t nMotors = robot.nmotors();
        std::size_t nJoints = robot.pinocchioModel_.njoints;
        q = pinocchio::neutral(robot.pinocchioModel_);
        v.setZero(nv);
        a.setZero(nv);
        command.setZero(nMotors);
        u.setZero(nv);
        uMotor.setZero(nMotors);
        uTransmission.setZero(nMotors);
        uInternal.setZero(nv);
        uCustom.setZero(nv);
        fExternal = ForceVector(nJoints, pinocchio::Force::Zero());
        isInitialized_ = true;
    }

    bool RobotState::getIsInitialized() const
    {
        return isInitialized_;
    }

    void RobotState::clear()
    {
        q.resize(0);
        v.resize(0);
        a.resize(0);
        command.resize(0);
        u.resize(0);
        uMotor.resize(0);
        uTransmission.resize(0);
        uInternal.resize(0);
        uCustom.resize(0);
        fExternal.clear();
    }

    RobotData::RobotData() = default;
    RobotData::RobotData(RobotData &&) = default;
    RobotData & RobotData::operator=(RobotData &&) = default;
    RobotData::~RobotData() = default;

    Engine::Engine() noexcept :
    generator_{std::seed_seq{std::random_device{}()}},
    telemetrySender_{std::make_unique<TelemetrySender>()},
    telemetryData_{std::make_shared<TelemetryData>()},
    telemetryRecorder_{std::make_unique<TelemetryRecorder>()}
    {
        // Initialize the configuration options to the default.
        simulationOptionsGeneric_["engine"] = getDefaultEngineOptions();
        setOptions(getOptions());
    }

    // ************************************ Engine *********************************** //

    // Cannot be default in the header since some types are incomplete at this point
    Engine::~Engine() = default;

    void Engine::addRobot(std::shared_ptr<Robot> robotIn)
    {
        // Make sure that no simulation is running
        if (isSimulationRunning_)
        {
            JIMINY_THROW(bad_control_flow,
                         "Simulation already running. Stop it before adding new robot.");
        }

        if (!robotIn)
        {
            JIMINY_THROW(std::invalid_argument, "No robot specified.");
        }

        if (!robotIn->getIsInitialized())
        {
            JIMINY_THROW(bad_control_flow, "Robot not initialized.");
        }

        /* All the robots must have a unique name. The latter will be used as prefix of telemetry
           constants and variables in the log file. As an exception, the first robot added to the
           engine is allowed to have no name. In such a case, no prefix will be added to telemetry
           variables for this specific robot. This does not prevent adding other robots with
           qualified names later on. This branching adds complexity internally, but simplifies
           single-robot simulation for the end-user, which is by far the most common use-case.
           Similarly, the name 'robot' is reserved for the first robot only. */
        const std::string & robotName = robotIn->getName();
        if (!robots_.empty() && (robotName == "" || robotName == "robot"))
        {
            JIMINY_THROW(std::invalid_argument,
                         "All robots, except the first, must have a non-empty name other than "
                         "'robot'.");
        }

        // Check if a robot with the same name already exists
        auto robotIt = std::find_if(robots_.begin(),
                                    robots_.end(),
                                    [&robotName](const auto & robot)
                                    { return (robot->getName() == robotName); });
        if (robotIt != robots_.end())
        {
            JIMINY_THROW(std::invalid_argument,
                         "Robot with name '",
                         robotName,
                         "' already added to the engine.");
        }

        robots_.push_back(std::move(robotIn));
        robotDataVec_.resize(robots_.size());
    }

    void Engine::removeRobot(const std::string & robotName)
    {
        // Make sure that no simulation is running
        if (isSimulationRunning_)
        {
            JIMINY_THROW(bad_control_flow,
                         "Simulation already running. Stop it before removing a robot.");
        }

        /* Remove every coupling forces involving the robot.
           Note that it is already checking that the robot exists. */
        removeCouplingForces(robotName);

        // Get robot index
        std::ptrdiff_t robotIndex = getRobotIndex(robotName);

        // Update the robots' indices for the remaining coupling forces
        for (auto & force : couplingForces_)
        {
            if (force.robotIndex1 > robotIndex)
            {
                --force.robotIndex1;
            }
            if (force.robotIndex2 > robotIndex)
            {
                --force.robotIndex2;
            }
        }

        // Remove the robot from the list
        robots_.erase(robots_.begin() + robotIndex);
        robotDataVec_.erase(robotDataVec_.begin() + robotIndex);

        // Remove robot from generic options
        std::string robotOptionsKey = robotName;
        if (robotOptionsKey.empty())
        {
            robotOptionsKey = "robot";
        }
        simulationOptionsGeneric_.erase(robotOptionsKey);
    }

    void Engine::registerCouplingForce(const std::string & robotName1,
                                       const std::string & robotName2,
                                       const std::string & frameName1,
                                       const std::string & frameName2,
                                       const CouplingForceFunction & forceFunc)
    {
        // Make sure that no simulation is running
        if (isSimulationRunning_)
        {
            JIMINY_THROW(bad_control_flow,
                         "Simulation already running. Stop it before adding coupling forces.");
        }

        // Get robot and frame indices
        const std::ptrdiff_t robotIndex1 = getRobotIndex(robotName1);
        const std::ptrdiff_t robotIndex2 = getRobotIndex(robotName2);
        const pinocchio::FrameIndex frameIndex1 =
            getFrameIndex(robots_[robotIndex1]->pinocchioModel_, frameName1);
        const pinocchio::FrameIndex frameIndex2 =
            getFrameIndex(robots_[robotIndex2]->pinocchioModel_, frameName2);

        // Make sure it is not coupling the exact same frame
        if (robotIndex1 == robotIndex2 && frameIndex1 == frameIndex2)
        {
            JIMINY_THROW(std::invalid_argument,
                         "A coupling force must involve two different frames.");
        }

        couplingForces_.emplace_back(robotName1,
                                     robotIndex1,
                                     robotName2,
                                     robotIndex2,
                                     frameName1,
                                     frameIndex1,
                                     frameName2,
                                     frameIndex2,
                                     forceFunc);
    }

    void Engine::registerViscoelasticCouplingForce(const std::string & robotName1,
                                                   const std::string & robotName2,
                                                   const std::string & frameName1,
                                                   const std::string & frameName2,
                                                   const Vector6d & stiffness,
                                                   const Vector6d & damping,
                                                   double alpha)
    {
        // Make sure that the input arguments are valid
        if ((stiffness.array() < 0.0).any() || (damping.array() < 0.0).any())
        {
            JIMINY_THROW(std::invalid_argument,
                         "Stiffness and damping parameters must be positive.");
        }

        // Get robot and frame indices
        Robot * robot1 = getRobot(robotName1).get();
        Robot * robot2 = getRobot(robotName2).get();
        pinocchio::FrameIndex frameIndex1 = getFrameIndex(robot1->pinocchioModel_, frameName1);
        pinocchio::FrameIndex frameIndex2 = getFrameIndex(robot2->pinocchioModel_, frameName2);

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
            /* One must keep track of robot pointers and frames indices internally and update
               them at reset since the robots may have changed between simulations. Note that
               `isSimulationRunning_` is false when called for the first time in `start` method
               before locking the robot, so it is the right time to update those proxies. */
            if (!isSimulationRunning_)
            {
                robot1 = getRobot(robotName1).get();
                robot2 = getRobot(robotName2).get();
                frameIndex1 = getFrameIndex(robot1->pinocchioModel_, frameName1);
                frameIndex2 = getFrameIndex(robot2->pinocchioModel_, frameName2);
            }

            // Get the frames positions and velocities in world
            const pinocchio::SE3 & oMf1{robot1->pinocchioData_.oMf[frameIndex1]};
            const pinocchio::SE3 & oMf2{robot2->pinocchioData_.oMf[frameIndex2]};
            const pinocchio::Motion oVf1{getFrameVelocity(robot1->pinocchioModel_,
                                                          robot1->pinocchioData_,
                                                          frameIndex1,
                                                          pinocchio::LOCAL_WORLD_ALIGNED)};
            const pinocchio::Motion oVf2{getFrameVelocity(robot2->pinocchioModel_,
                                                          robot2->pinocchioData_,
                                                          frameIndex2,
                                                          pinocchio::LOCAL_WORLD_ALIGNED)};

            // Compute intermediary quantities
            rot12.noalias() = oMf1.rotation().transpose() * oMf2.rotation();
            rotLog12 = pinocchio::log3(rot12, angle);
            if (angle > 0.95 * M_PI)
            {
                JIMINY_THROW(std::runtime_error,
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

        registerCouplingForce(robotName1, robotName2, frameName1, frameName2, forceFunc);
    }

    void Engine::registerViscoelasticCouplingForce(const std::string & robotName,
                                                   const std::string & frameName1,
                                                   const std::string & frameName2,
                                                   const Vector6d & stiffness,
                                                   const Vector6d & damping,
                                                   double alpha)
    {
        return registerViscoelasticCouplingForce(
            robotName, robotName, frameName1, frameName2, stiffness, damping, alpha);
    }

    void Engine::registerViscoelasticDirectionalCouplingForce(const std::string & robotName1,
                                                              const std::string & robotName2,
                                                              const std::string & frameName1,
                                                              const std::string & frameName2,
                                                              double stiffness,
                                                              double damping,
                                                              double restLength)
    {
        // Make sure that the input arguments are valid
        if (stiffness < 0.0 || damping < 0.0)
        {
            JIMINY_THROW(std::invalid_argument,
                         "The stiffness and damping parameters must be positive.");
        }

        // Get robot and frame indices
        Robot * robot1 = getRobot(robotName1).get();
        Robot * robot2 = getRobot(robotName2).get();
        pinocchio::FrameIndex frameIndex1 = getFrameIndex(robot1->pinocchioModel_, frameName1);
        pinocchio::FrameIndex frameIndex2 = getFrameIndex(robot2->pinocchioModel_, frameName2);

        auto forceFunc = [=](double /* t */,
                             const Eigen::VectorXd & /* q1 */,
                             const Eigen::VectorXd & /* v1 */,
                             const Eigen::VectorXd & /* q2 */,
                             const Eigen::VectorXd & /* v2 */) mutable -> pinocchio::Force
        {
            /* One must keep track of robot pointers and frames indices internally and update
               them at reset since the robots may have changed between simulations. Note that
               `isSimulationRunning_` is false when called for the first time in `start` method
               before locking the robot, so it is the right time to update those proxies. */
            if (!isSimulationRunning_)
            {
                robot1 = getRobot(robotName1).get();
                robot2 = getRobot(robotName2).get();
                frameIndex1 = getFrameIndex(robot1->pinocchioModel_, frameName1);
                frameIndex2 = getFrameIndex(robot2->pinocchioModel_, frameName2);
            }

            // Get the frames positions and velocities in world
            const pinocchio::SE3 & oMf1{robot1->pinocchioData_.oMf[frameIndex1]};
            const pinocchio::SE3 & oMf2{robot2->pinocchioData_.oMf[frameIndex2]};
            const pinocchio::Motion oVf1{getFrameVelocity(robot1->pinocchioModel_,
                                                          robot1->pinocchioData_,
                                                          frameIndex1,
                                                          pinocchio::LOCAL_WORLD_ALIGNED)};
            const pinocchio::Motion oVf2{getFrameVelocity(robot2->pinocchioModel_,
                                                          robot2->pinocchioData_,
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

        registerCouplingForce(robotName1, robotName2, frameName1, frameName2, forceFunc);
    }

    void Engine::registerViscoelasticDirectionalCouplingForce(const std::string & robotName,
                                                              const std::string & frameName1,
                                                              const std::string & frameName2,
                                                              double stiffness,
                                                              double damping,
                                                              double restLength)
    {
        return registerViscoelasticDirectionalCouplingForce(
            robotName, robotName, frameName1, frameName2, stiffness, damping, restLength);
    }

    void Engine::removeCouplingForces(const std::string & robotName1,
                                      const std::string & robotName2)
    {
        // Make sure that no simulation is running
        if (isSimulationRunning_)
        {
            JIMINY_THROW(bad_control_flow,
                         "Simulation already running. Stop it before removing coupling forces.");
        }

        // Make sure that the robots exist
        getRobot(robotName1);
        getRobot(robotName2);

        // Remove corresponding coupling forces if any
        couplingForces_.erase(std::remove_if(couplingForces_.begin(),
                                             couplingForces_.end(),
                                             [&robotName1, &robotName2](const auto & force)
                                             {
                                                 return (force.robotName1 == robotName1 &&
                                                         force.robotName2 == robotName2);
                                             }),
                              couplingForces_.end());
    }

    void Engine::removeCouplingForces(const std::string & robotName)
    {
        // Make sure that no simulation is running
        if (isSimulationRunning_)
        {
            JIMINY_THROW(bad_control_flow,
                         "Simulation already running. Stop it before removing coupling forces.");
        }

        // Make sure that the robot exists
        getRobot(robotName);

        // Remove corresponding coupling forces if any
        couplingForces_.erase(std::remove_if(couplingForces_.begin(),
                                             couplingForces_.end(),
                                             [&robotName](const auto & force)
                                             {
                                                 return (force.robotName1 == robotName ||
                                                         force.robotName2 == robotName);
                                             }),
                              couplingForces_.end());
    }

    void Engine::removeCouplingForces()
    {
        // Make sure that no simulation is running
        if (isSimulationRunning_)
        {
            JIMINY_THROW(bad_control_flow,
                         "Simulation already running. Stop it before removing coupling forces.");
        }

        couplingForces_.clear();
    }

    const CouplingForceVector & Engine::getCouplingForces() const
    {
        return couplingForces_;
    }

    void Engine::removeAllForces()
    {
        removeCouplingForces();
        removeImpulseForces();
        removeProfileForces();
    }

    void Engine::configureTelemetry()
    {
        if (robots_.empty())
        {
            JIMINY_THROW(bad_control_flow, "No robot added to the engine.");
        }

        if (!isTelemetryConfigured_)
        {
            // Initialize the engine-specific telemetry sender
            telemetrySender_->configure(telemetryData_, ENGINE_TELEMETRY_NAMESPACE);

            auto robotIt = robots_.begin();
            auto robotDataIt = robotDataVec_.begin();
            auto energyIt = energy_.begin();
            for (; robotIt != robots_.end(); ++robotIt, ++robotDataIt, ++energyIt)
            {
                // Define proxy for convenience
                const std::string & robotName = (*robotIt)->getName();

                // Generate the log fieldnames
                robotDataIt->logPositionFieldnames =
                    addCircumfix((*robotIt)->getLogPositionFieldnames(),
                                 robotName,
                                 {},
                                 TELEMETRY_FIELDNAME_DELIMITER);
                robotDataIt->logVelocityFieldnames =
                    addCircumfix((*robotIt)->getLogVelocityFieldnames(),
                                 robotName,
                                 {},
                                 TELEMETRY_FIELDNAME_DELIMITER);
                robotDataIt->logAccelerationFieldnames =
                    addCircumfix((*robotIt)->getLogAccelerationFieldnames(),
                                 robotName,
                                 {},
                                 TELEMETRY_FIELDNAME_DELIMITER);
                robotDataIt->logEffortFieldnames =
                    addCircumfix((*robotIt)->getLogEffortFieldnames(),
                                 robotName,
                                 {},
                                 TELEMETRY_FIELDNAME_DELIMITER);
                robotDataIt->logForceExternalFieldnames =
                    addCircumfix((*robotIt)->getLogForceExternalFieldnames(),
                                 robotName,
                                 {},
                                 TELEMETRY_FIELDNAME_DELIMITER);
                robotDataIt->logConstraintFieldnames =
                    addCircumfix((*robotIt)->getLogConstraintFieldnames(),
                                 robotName,
                                 {},
                                 TELEMETRY_FIELDNAME_DELIMITER);
                robotDataIt->logCommandFieldnames =
                    addCircumfix((*robotIt)->getLogCommandFieldnames(),
                                 robotName,
                                 {},
                                 TELEMETRY_FIELDNAME_DELIMITER);
                robotDataIt->logEnergyFieldname =
                    addCircumfix("energy", robotName, {}, TELEMETRY_FIELDNAME_DELIMITER);

                // Register variables to the telemetry senders
                if (engineOptions_->telemetry.enableConfiguration)
                {
                    telemetrySender_->registerVariable(robotDataIt->logPositionFieldnames,
                                                       robotDataIt->state.q);
                }
                if (engineOptions_->telemetry.enableVelocity)
                {
                    telemetrySender_->registerVariable(robotDataIt->logVelocityFieldnames,
                                                       robotDataIt->state.v);
                }
                if (engineOptions_->telemetry.enableAcceleration)
                {
                    telemetrySender_->registerVariable(robotDataIt->logAccelerationFieldnames,
                                                       robotDataIt->state.a);
                }
                if (engineOptions_->telemetry.enableEffort)
                {
                    telemetrySender_->registerVariable(robotDataIt->logEffortFieldnames,
                                                       robotDataIt->state.u);
                }
                if (engineOptions_->telemetry.enableForceExternal)
                {
                    for (std::size_t i = 1; i < robotDataIt->state.fExternal.size(); ++i)
                    {
                        const auto & fext = robotDataIt->state.fExternal[i].toVector();
                        for (uint8_t j = 0; j < 6U; ++j)
                        {
                            telemetrySender_->registerVariable(
                                robotDataIt->logForceExternalFieldnames[(i - 1) * 6U + j],
                                &fext[j]);
                        }
                    }
                }
                if (engineOptions_->telemetry.enableConstraint)
                {
                    const ConstraintTree & constraints = (*robotIt)->getConstraints();
                    // FIXME: Remove explicit `telemetrySender` capture when moving to C++20
                    constraints.foreach(
                        [&telemetrySender = telemetrySender_, &robotData = *robotDataIt, i = 0](
                            const std::shared_ptr<AbstractConstraintBase> & constraint,
                            ConstraintRegistryType /* type */) mutable
                        {
                            for (uint8_t j = 0; j < constraint->getSize(); ++j)
                            {
                                telemetrySender->registerVariable(
                                    robotData.logConstraintFieldnames[i++],
                                    &constraint->lambda_[j]);
                            }
                        });
                }
                if (engineOptions_->telemetry.enableCommand)
                {
                    telemetrySender_->registerVariable(robotDataIt->logCommandFieldnames,
                                                       robotDataIt->state.command);
                }
                if (engineOptions_->telemetry.enableEnergy)
                {
                    telemetrySender_->registerVariable(robotDataIt->logEnergyFieldname,
                                                       &(*energyIt));
                }
                (*robotIt)->configureTelemetry(telemetryData_);
            }
        }

        isTelemetryConfigured_ = true;
    }

    void Engine::updateTelemetry()
    {
        // Compute the total energy of the robot
        auto robotIt = robots_.begin();
        auto energyIt = energy_.begin();
        for (; robotIt != robots_.end(); ++robotIt, ++energyIt)
        {
            *energyIt = (*robotIt)->pinocchioData_.kinetic_energy +
                        (*robotIt)->pinocchioData_.potential_energy;
        }

        // Update robot-specific telemetry variables
        for (auto & robot : robots_)
        {
            robot->updateTelemetry();
        }

        // Update engine-specific telemetry variables
        telemetrySender_->updateValues();

        // Flush the telemetry internal state
        telemetryRecorder_->flushSnapshot(stepperState_.t);
    }

    void Engine::reset(bool resetRandomNumbers, bool removeAllForce)
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
            for (auto & robotData : robotDataVec_)
            {
                robotData.impulseForces.clear();
                robotData.impulseForceBreakpoints.clear();
                robotData.isImpulseForceActiveVec.clear();
                robotData.profileForces.clear();
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

        // Reset the internal state of the robot
        for (auto & robot : robots_)
        {
            robot->reset(generator_);
        }

        // Clear robot state buffers, since the robot kinematic may change
        for (auto & robotData : robotDataVec_)
        {
            robotData.state.clear();
            robotData.statePrev.clear();
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
    void computeExtraTerms(
        std::shared_ptr<Robot> & robot, const RobotData & robotData, ForceVector & fExt)
    {
        // Define some proxies for convenience
        const pinocchio::Model & model = robot->pinocchioModel_;
        pinocchio::Data & data = robot->pinocchioData_;

        // Compute the potential and kinematic energy of the robot
        pinocchio_overload::computeKineticEnergy(
            model, data, robotData.state.q, robotData.state.v, false);
        pinocchio::computePotentialEnergy(model, data);

        /* Update manually the subtree (apparent) inertia, since it is only computed by CRBA, which
           is doing more computation than necessary. It will be used here for computing the
           centroidal kinematics, and used later for joint bounds dynamics. Note that, by doing all
           the computations here instead of 'computeForwardKinematics', we are doing the assumption
           that it is varying slowly enough to consider it constant during one integration step. */
        if (!robot->hasConstraints())
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

        /* The objective here is to compute the actual joint accelerations and the joint internal
           forces. The latter are not involved in dynamic computations, but are useful for analysis
           of the mechanical design. Indeed, it brings information about stresses and strains
           applied to the mechanical structure, which may cause unexpected fatigue wear. In
           addition, the body external forces are also evaluated, as an intermediate quantity for
           computing the centroidal dynamics, ie the spatial momentum of the whole robot at the
           Center of Mass along with its temporal derivative.

           Neither 'aba' nor 'forwardDynamics' are computing simultaneously the actual joint
           accelerations, the joint internal forces and the body external forces. Hence, it is done
           manually, following a two computation steps procedure:
           * joint accelerations based on ForwardKinematic algorithm
           * joint internal forces and body external forces based on RNEA algorithm */

        /* Compute the true joint acceleration and the one due to the lone gravity field, then
           the spatial momenta and the total internal and external forces acting on each body.
           * `fExt` is used as a buffer for storing the total body external forces. It serves
             no purpose other than being an intermediate quantity for other computations.
           * `data.f` stores the joint internal forces */
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
                typename ForwardKinematicsAccelerationStep::ArgsType(data, robotData.state.a));

            const pinocchio::JointIndex parentJointIndex = model.parents[jointIndex];
            data.a_gf[jointIndex] = data.a[jointIndex];
            data.a[jointIndex] += data.liMi[jointIndex].actInv(data.a[parentJointIndex]);
            data.a_gf[jointIndex] += data.liMi[jointIndex].actInv(data.a_gf[parentJointIndex]);

            model.inertias[jointIndex].__mult__(data.v[jointIndex], data.h[jointIndex]);

            model.inertias[jointIndex].__mult__(data.a[jointIndex], fExt[jointIndex]);
            data.f[jointIndex] = data.v[jointIndex].cross(data.h[jointIndex]);
            fExt[jointIndex] += data.f[jointIndex];
            data.f[jointIndex] += model.inertias[jointIndex] * data.a_gf[jointIndex];
            data.f[jointIndex] -= robotData.state.fExternal[jointIndex];
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

    static void computeAllExtraTerms(std::vector<std::shared_ptr<Robot>> & robots,
                                     const vector_aligned_t<RobotData> & robotDataVec,
                                     vector_aligned_t<ForceVector> & f)
    {
        auto robotIt = robots.begin();
        auto robotDataIt = robotDataVec.begin();
        auto fIt = f.begin();
        for (; robotIt != robots.end(); ++robotIt, ++robotDataIt, ++fIt)
        {
            computeExtraTerms(*robotIt, *robotDataIt, *fIt);
        }
    }

    static void syncAccelerationsAndForces(const std::shared_ptr<Robot> & robot,
                                           ForceVector & contactForces,
                                           ForceVector & f,
                                           MotionVector & a)
    {
        for (std::size_t i = 0; i < robot->getContactFrameNames().size(); ++i)
        {
            contactForces[i] = robot->contactForces_[i];
        }

        for (int i = 0; i < robot->pinocchioModel_.njoints; ++i)
        {
            f[i] = robot->pinocchioData_.f[i];
            a[i] = robot->pinocchioData_.a[i];
        }
    }

    static void syncAllAccelerationsAndForces(const std::vector<std::shared_ptr<Robot>> & robots,
                                              vector_aligned_t<ForceVector> & contactForces,
                                              vector_aligned_t<ForceVector> & f,
                                              vector_aligned_t<MotionVector> & a)
    {
        std::vector<std::shared_ptr<Robot>>::const_iterator robotIt = robots.begin();
        auto contactForceIt = contactForces.begin();
        auto fPrevIt = f.begin();
        auto aPrevIt = a.begin();
        for (; robotIt != robots.end(); ++robotIt, ++contactForceIt, ++fPrevIt, ++aPrevIt)
        {
            syncAccelerationsAndForces(*robotIt, *contactForceIt, *fPrevIt, *aPrevIt);
        }
    }

    void Engine::start(const std::map<std::string, Eigen::VectorXd> & qInit,
                       const std::map<std::string, Eigen::VectorXd> & vInit,
                       const std::optional<std::map<std::string, Eigen::VectorXd>> & aInit)
    {
        // Make sure that no simulation is running
        if (isSimulationRunning_)
        {
            JIMINY_THROW(bad_control_flow,
                         "Simulation already running. Please stop it before starting a new one.");
        }

        if (robots_.empty())
        {
            JIMINY_THROW(bad_control_flow,
                         "No robot to simulate. Please add one before starting a simulation.");
        }

        const std::size_t nRobots = robots_.size();
        if (qInit.size() != nRobots || vInit.size() != nRobots)
        {
            JIMINY_THROW(
                std::invalid_argument,
                "Mismatching between number of robots and initial configurations or velocities.");
        }

        // Check the dimension of the initial state associated with every robot and order them
        std::vector<Eigen::VectorXd> qSplit;
        std::vector<Eigen::VectorXd> vSplit;
        qSplit.reserve(nRobots);
        vSplit.reserve(nRobots);
        for (const auto & robot : robots_)
        {
            const std::string & robotName = robot->getName();

            auto qInitIt = qInit.find(robotName);
            auto vInitIt = vInit.find(robotName);
            if (qInitIt == qInit.end() || vInitIt == vInit.end())
            {
                JIMINY_THROW(std::invalid_argument,
                             "Robot '",
                             robotName,
                             "' does not have an initial configuration or velocity.");
            }

            const Eigen::VectorXd & q = qInitIt->second;
            const Eigen::VectorXd & v = vInitIt->second;
            if (q.rows() != robot->nq() || v.rows() != robot->nv())
            {
                JIMINY_THROW(std::invalid_argument,
                             "The dimension of the initial configuration or velocity is "
                             "inconsistent with model size for robot '",
                             robotName,
                             "'.");
            }

            // Make sure the initial state is normalized
            const bool isValid =
                isPositionValid(robot->pinocchioModel_, q, std::numeric_limits<float>::epsilon());
            if (!isValid)
            {
                JIMINY_THROW(std::invalid_argument,
                             "Initial configuration not consistent with model for robot '",
                             robotName,
                             "'.");
            }

            /* Check that the initial configuration is not out-of-bounds if appropriate.
               Note that EPS allows to be very slightly out-of-bounds, which may occurs because of
               rounding errors. */
            if ((EPS < q.array() - robot->pinocchioModel_.upperPositionLimit.array()).any() ||
                (EPS < robot->pinocchioModel_.lowerPositionLimit.array() - q.array()).any())
            {
                JIMINY_THROW(std::invalid_argument,
                             "Initial configuration out-of-bounds for robot '",
                             robotName,
                             "'.");
            }

            // Check that the initial velocity is not out-of-bounds
            if ((robot->pinocchioModel_.velocityLimit.array() < v.array().abs()).any())
            {
                JIMINY_THROW(std::invalid_argument,
                             "Initial velocity out-of-bounds for robot '",
                             robotName,
                             "'.");
            }

            /* Make sure the configuration is normalized (as double), since normalization is
               checked using float accuracy rather than double to circumvent float/double casting
               than may occurs because of Python bindings. */
            Eigen::VectorXd qNormalized = q;
            pinocchio::normalize(robot->pinocchioModel_, qNormalized);

            qSplit.emplace_back(qNormalized);
            vSplit.emplace_back(v);
        }

        std::vector<Eigen::VectorXd> aSplit;
        aSplit.reserve(nRobots);
        if (aInit)
        {
            // Check the dimension of the initial acceleration associated with every robot
            if (aInit->size() != nRobots)
            {
                JIMINY_THROW(std::invalid_argument,
                             "If specified, the number of initial accelerations must match the "
                             "number of robots.");
            }

            for (const auto & robot : robots_)
            {
                auto aInitIt = aInit->find(robot->getName());
                if (aInitIt == aInit->end())
                {
                    JIMINY_THROW(std::invalid_argument,
                                 "Robot '",
                                 robot->getName(),
                                 "'does not have an initial acceleration.");
                }

                const Eigen::VectorXd & a = aInitIt->second;
                if (a.rows() != robot->nv())
                {
                    JIMINY_THROW(
                        std::invalid_argument,
                        "Dimension of initial acceleration inconsistent with model for robot '",
                        robot->getName(),
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

        for (auto & robot : robots_)
        {
            for (const auto & sensorGroupItem : robot->getSensors())
            {
                for (const auto & sensor : sensorGroupItem.second)
                {
                    if (!sensor->getIsInitialized())
                    {
                        JIMINY_THROW(bad_control_flow,
                                     "At least a sensor of a robot is not initialized.");
                    }
                }
            }

            for (const auto & motor : robot->getMotors())
            {
                if (!motor->getIsInitialized())
                {
                    JIMINY_THROW(bad_control_flow,
                                 "At least a motor of a robot is not initialized.");
                }
            }
        }

        /* Call reset if the internal state of the engine is not clean. Not doing it systematically
           gives the opportunity to the user to customize the robot by resetting first the engine
           manually and then to alter the robot before starting a simulation, e.g. to change the
           inertia of a specific body. */
        if (isTelemetryConfigured_)
        {
            reset(false, false);
        }

        // Reset the internal state of the robot
        auto robotIt = robots_.begin();
        auto robotDataIt = robotDataVec_.begin();
        for (; robotIt != robots_.end(); ++robotIt, ++robotDataIt)
        {
            // Propagate the user-defined gravity at robot level
            (*robotIt)->pinocchioModelTh_.gravity = engineOptions_->world.gravity;
            (*robotIt)->pinocchioModel_.gravity = engineOptions_->world.gravity;

            /* Reinitialize the robot state buffers, since the robot kinematic may have changed.
               For example, it may happens if one activates or deactivates the flexibility between
               two successive simulations. */
            robotDataIt->state.initialize(*(*robotIt));
            robotDataIt->statePrev.initialize(*(*robotIt));
            robotDataIt->successiveSolveFailed = 0U;
        }

        // Initialize the ode solver
        auto robotOde = [this](double t,
                               const std::vector<Eigen::VectorXd> & q,
                               const std::vector<Eigen::VectorXd> & v,
                               std::vector<Eigen::VectorXd> & a) -> void
        {
            this->computeRobotsDynamics(t, q, v, a);
        };
        std::vector<const Robot *> robots;
        robots.reserve(nRobots);
        std::transform(robots_.begin(),
                       robots_.end(),
                       std::back_inserter(robots),
                       [](const auto & robot) { return robot.get(); });
        if (engineOptions_->stepper.odeSolver == "runge_kutta_dopri")
        {
            stepper_ = std::unique_ptr<AbstractStepper>(new RungeKuttaDOPRIStepper(
                robotOde, robots, engineOptions_->stepper.tolAbs, engineOptions_->stepper.tolRel));
        }
        else if (engineOptions_->stepper.odeSolver == "runge_kutta_4")
        {
            stepper_ = std::unique_ptr<AbstractStepper>(new RungeKutta4Stepper(robotOde, robots));
        }
        else if (engineOptions_->stepper.odeSolver == "euler_explicit")
        {
            stepper_ =
                std::unique_ptr<AbstractStepper>(new EulerExplicitStepper(robotOde, robots));
        }

        // Initialize the stepper state
        const double t = 0.0;
        stepperState_.reset(SIMULATION_MIN_TIMESTEP, qSplit, vSplit, aSplit);

        // Initialize previous joints forces and accelerations
        contactForcesPrev_.clear();
        contactForcesPrev_.reserve(nRobots);
        fPrev_.clear();
        fPrev_.reserve(nRobots);
        aPrev_.clear();
        aPrev_.reserve(nRobots);
        for (const auto & robot : robots_)
        {
            contactForcesPrev_.push_back(robot->contactForces_);
            fPrev_.push_back(robot->pinocchioData_.f);
            aPrev_.push_back(robot->pinocchioData_.a);
        }
        energy_.resize(nRobots, 0.0);

        // Synchronize the individual robot states with the global stepper state
        syncRobotsStateWithStepper();

        // Update the frame indices associated with the coupling forces
        for (auto & force : couplingForces_)
        {
            force.frameIndex1 =
                getFrameIndex(robots_[force.robotIndex1]->pinocchioModel_, force.frameName1);
            force.frameIndex2 =
                getFrameIndex(robots_[force.robotIndex2]->pinocchioModel_, force.frameName2);
        }

        robotIt = robots_.begin();
        robotDataIt = robotDataVec_.begin();
        for (; robotIt != robots_.end(); ++robotIt, ++robotDataIt)
        {
            // Update the frame indices associated with the impulse forces and force profiles
            for (auto & force : robotDataIt->profileForces)
            {
                force.frameIndex = getFrameIndex((*robotIt)->pinocchioModel_, force.frameName);
            }
            for (auto & force : robotDataIt->impulseForces)
            {
                force.frameIndex = getFrameIndex((*robotIt)->pinocchioModel_, force.frameName);
            }

            // Initialize the impulse force breakpoint point iterator
            robotDataIt->impulseForceBreakpointNextIt =
                robotDataIt->impulseForceBreakpoints.begin();

            // Reset the active set of impulse forces
            std::fill(robotDataIt->isImpulseForceActiveVec.begin(),
                      robotDataIt->isImpulseForceActiveVec.end(),
                      false);

            // Activate every force impulse starting at t=0
            auto isImpulseForceActiveIt = robotDataIt->isImpulseForceActiveVec.begin();
            auto impulseForceIt = robotDataIt->impulseForces.begin();
            for (; impulseForceIt != robotDataIt->impulseForces.end();
                 ++isImpulseForceActiveIt, ++impulseForceIt)
            {
                if (impulseForceIt->t < STEPPER_MIN_TIMESTEP)
                {
                    *isImpulseForceActiveIt = true;
                }
            }

            // Compute the forward kinematics for each robot
            const Eigen::VectorXd & q = robotDataIt->state.q;
            const Eigen::VectorXd & v = robotDataIt->state.v;
            const Eigen::VectorXd & a = robotDataIt->state.a;
            computeForwardKinematics(*robotIt, q, v, a);

            // Initialize contacts forces in local frame
            const std::vector<pinocchio::FrameIndex> & contactFrameIndices =
                (*robotIt)->getContactFrameIndices();
            robotDataIt->contactFrameForces =
                ForceVector(contactFrameIndices.size(), pinocchio::Force::Zero());
            const std::vector<std::vector<pinocchio::PairIndex>> & collisionPairIndices =
                (*robotIt)->getCollisionPairIndices();
            robotDataIt->collisionBodiesForces.clear();
            robotDataIt->collisionBodiesForces.reserve(collisionPairIndices.size());
            for (const auto & bodyCollisionPairIndices : collisionPairIndices)
            {
                robotDataIt->collisionBodiesForces.emplace_back(bodyCollisionPairIndices.size(),
                                                                pinocchio::Force::Zero());
            }

            /* Initialize some addition buffers used by impulse contact solver.
               It must be initialized to zero because 'getJointJacobian' will only update non-zero
               coefficients for efficiency. */
            robotDataIt->jointJacobians.resize((*robotIt)->pinocchioModel_.njoints,
                                               Matrix6Xd::Zero(6, (*robotIt)->pinocchioModel_.nv));

            // Reset the constraints
            (*robotIt)->resetConstraints(q, v);

            /* Set Baumgarte stabilization natural frequency for contact constraints
               Enable all contact constraints by default, it will be disable automatically if not
               in contact. It is useful to start in post-hysteresis state to avoid discontinuities
               at init. */
            const ConstraintTree & constraints = (*robotIt)->getConstraints();
            constraints.foreach(
                [&contactModel = contactModel_,
                 &freq = engineOptions_->contacts.stabilizationFreq](
                    const std::shared_ptr<AbstractConstraintBase> & constraint,
                    ConstraintRegistryType type)
                {
                    // Set baumgarte freq for all contact constraints
                    if (type != ConstraintRegistryType::USER)
                    {
                        constraint->setBaumgarteFreq(freq);  // It cannot fail
                    }

                    // Enable constraints by default
                    if (contactModel == ContactModelType::CONSTRAINT)
                    {
                        switch (type)
                        {
                        case ConstraintRegistryType::BOUNDS_JOINTS:
                        {
                            auto & jointConstraint =
                                static_cast<JointConstraint &>(*constraint.get());
                            jointConstraint.setRotationDir(false);
                        }
                            [[fallthrough]];
                        case ConstraintRegistryType::CONTACT_FRAMES:
                        case ConstraintRegistryType::COLLISION_BODIES:
                            constraint->enable();
                            break;
                        case ConstraintRegistryType::USER:
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
                    auto & constraint = constraints.contactFrames[i].second;
                    pinocchio::Force & fextLocal = robotDataIt->contactFrameForces[i];
                    computeContactDynamicsAtFrame(
                        *robotIt, contactFrameIndices[i], constraint, fextLocal);
                    forceMax = std::max(forceMax, fextLocal.linear().norm());
                }

                for (std::size_t i = 0; i < collisionPairIndices.size(); ++i)
                {
                    for (std::size_t j = 0; j < collisionPairIndices[i].size(); ++j)
                    {
                        const pinocchio::PairIndex & collisionPairIndex =
                            collisionPairIndices[i][j];
                        auto & constraint = constraints.collisionBodies[i][j].second;
                        pinocchio::Force & fextLocal = robotDataIt->collisionBodiesForces[i][j];
                        computeContactDynamicsAtBody(
                            *robotIt, collisionPairIndex, constraint, fextLocal);
                        forceMax = std::max(forceMax, fextLocal.linear().norm());
                    }
                }

                if (forceMax > 1e5)
                {
                    JIMINY_THROW(
                        std::invalid_argument,
                        "The initial force exceeds 1e5 for at least one contact point, which is "
                        "forbidden for the sake of numerical stability. Please update the initial "
                        "state.");
                }
            }
        }

        // Lock the robots. At this point, it is no longer possible to change them anymore.
        robotIt = robots_.begin();
        robotDataIt = robotDataVec_.begin();
        for (; robotIt != robots_.end(); ++robotIt, ++robotDataIt)
        {
            robotDataIt->robotLock = (*robotIt)->getLock();
        }

        // Instantiate the desired LCP solver
        robotIt = robots_.begin();
        robotDataIt = robotDataVec_.begin();
        for (; robotIt != robots_.end(); ++robotIt, ++robotDataIt)
        {
            const std::string & constraintSolverType = engineOptions_->constraints.solver;
            switch (CONSTRAINT_SOLVERS_MAP.at(constraintSolverType))
            {
            case ConstraintSolverType::PGS:
                robotDataIt->constraintSolver =
                    std::make_unique<PGSSolver>(&((*robotIt)->pinocchioModel_),
                                                &((*robotIt)->pinocchioData_),
                                                &((*robotIt)->getConstraints()),
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

        /* Compute the efforts, internal and external forces applied on every robots excluding
           user-specified internal dynamics if any. */
        computeAllTerms(t, qSplit, vSplit);

        // Backup all external forces and internal efforts excluding constraint forces
        vector_aligned_t<ForceVector> fextNoConst;
        std::vector<Eigen::VectorXd> uInternalConst;
        fextNoConst.reserve(nRobots);
        uInternalConst.reserve(nRobots);
        for (const auto & robotData : robotDataVec_)
        {
            fextNoConst.push_back(robotData.state.fExternal);
            uInternalConst.push_back(robotData.state.uInternal);
        }

        /* Solve algebraic coupling between accelerations, sensors and controllers, by
           iterating several times until it (hopefully) converges. */
        bool isFirstIter = true;
        for (uint32_t i = 0; i < INIT_ITERATIONS; ++i)
        {
            robotIt = robots_.begin();
            robotDataIt = robotDataVec_.begin();
            auto fextNoConstIt = fextNoConst.begin();
            auto uInternalConstIt = uInternalConst.begin();
            for (; robotIt != robots_.end();
                 ++robotIt, ++robotDataIt, ++fextNoConstIt, ++uInternalConstIt)
            {
                // Get some robot state proxies
                const Eigen::VectorXd & q = robotDataIt->state.q;
                const Eigen::VectorXd & v = robotDataIt->state.v;
                Eigen::VectorXd & a = robotDataIt->state.a;
                Eigen::VectorXd & u = robotDataIt->state.u;
                Eigen::VectorXd & command = robotDataIt->state.command;
                Eigen::VectorXd & uMotor = robotDataIt->state.uMotor;
                Eigen::VectorXd & uTransmission = robotDataIt->state.uTransmission;
                Eigen::VectorXd & uInternal = robotDataIt->state.uInternal;
                Eigen::VectorXd & uCustom = robotDataIt->state.uCustom;
                ForceVector & fext = robotDataIt->state.fExternal;

                // Reset the external forces and internal efforts
                fext = *fextNoConstIt;
                uInternal = *uInternalConstIt;

                // Compute dynamics
                a = computeAcceleration(
                    *robotIt, *robotDataIt, q, v, u, fext, !isFirstIter, isFirstIter);

                // Make sure there is no nan at this point
                if ((a.array() != a.array()).any())
                {
                    JIMINY_THROW(std::runtime_error,
                                 "Impossible to compute the acceleration. Probably a "
                                 "subtree has zero inertia along an articulated axis.");
                }

                // Compute all external terms including joints accelerations and forces
                computeAllExtraTerms(robots_, robotDataVec_, fPrev_);

                // Compute the sensor data with the updated effort and acceleration
                (*robotIt)->computeSensorMeasurements(t, q, v, a, uMotor, fext);

                // Compute the actual motor effort
                computeCommand(*robotIt, t, q, v, command);

                // Compute the actual motor effort
                (*robotIt)->computeMotorEfforts(t, q, v, a, command);
                const auto & uMotorAndJoint = (*robotIt)->getMotorEfforts();
                std::tie(uMotor, uTransmission) = uMotorAndJoint;

                // Compute the internal dynamics
                uCustom.setZero();
                (*robotIt)->getController()->internalDynamics(t, q, v, uCustom);

                // Compute the total effort vector
                u = uInternal + uCustom;
                for (const auto & motor : (*robotIt)->getMotors())
                {
                    const std::size_t motorIndex = motor->getIndex();
                    const pinocchio::JointIndex jointIndex = motor->getJointIndex();
                    const Eigen::Index motorVelocityIndex =
                        (*robotIt)->pinocchioModel_.joints[jointIndex].idx_v();
                    u[motorVelocityIndex] += uTransmission[motorIndex];
                }
            }
            isFirstIter = false;
        }

        // Update sensor data one last time to take into account the actual motor effort
        robotIt = robots_.begin();
        robotDataIt = robotDataVec_.begin();
        for (; robotIt != robots_.end(); ++robotIt, ++robotDataIt)
        {
            const Eigen::VectorXd & q = robotDataIt->state.q;
            const Eigen::VectorXd & v = robotDataIt->state.v;
            const Eigen::VectorXd & a = robotDataIt->state.a;
            const Eigen::VectorXd & uMotor = robotDataIt->state.uMotor;
            const ForceVector & fext = robotDataIt->state.fExternal;
            (*robotIt)->computeSensorMeasurements(t, q, v, a, uMotor, fext);
        }

        // Backend the updated joint accelerations and forces
        syncAllAccelerationsAndForces(robots_, contactForcesPrev_, fPrev_, aPrev_);

        // Synchronize the global stepper state with the individual robot states
        syncStepperStateWithRobots();

        // Initialize the last robot states
        for (auto & robotData : robotDataVec_)
        {
            robotData.statePrev = robotData.state;
        }

        /* Register all engine and robots variables, then lock the telemetry.
           At this point it is not possible for the user to register new variables. */
        configureTelemetry();

        // Log robots
        for (const auto & robot : robots_)
        {
            const bool isPersistent = engineOptions_->telemetry.isPersistent ||
                                      robot->getUrdfAsString().empty();
            const std::string key =
                addCircumfix("robot", robot->getName(), {}, TELEMETRY_FIELDNAME_DELIMITER);
            telemetrySender_->registerConstant(key, saveToBinary(robot, isPersistent));
        }

        // Log all options
        Json::Value simulationOptionsJson = convertToJson(getSimulationOptions());
        Json::StreamWriterBuilder jsonWriter;
        jsonWriter["indentation"] = "";
        const std::string simulationOptionsString =
            Json::writeString(jsonWriter, simulationOptionsJson);
        telemetrySender_->registerConstant("options", simulationOptionsString);

        // Write the header: this locks the registration of new variables
        telemetryRecorder_->initialize(telemetryData_.get(), getTelemetryTimeUnit());

        // Make sure tha the simulation options are not considered refreshed anymore
        areSimulationOptionsRefreshed_ = false;

        // At this point, consider that the simulation is running
        isSimulationRunning_ = true;
    }

    std::tuple<std::map<std::string, Eigen::VectorXd>,
               std::map<std::string, Eigen::VectorXd>,
               std::optional<std::map<std::string, Eigen::VectorXd>>>
    sanitizeInitialData(const std::shared_ptr<Robot> & robot,
                        bool isStateTheoretical,
                        const Eigen::VectorXd & qInit,
                        const Eigen::VectorXd & vInit,
                        const std::optional<Eigen::VectorXd> & aInit)
    {
        // Extract robot name for convenience
        const std::string & robotName = robot->getName();

        // Process initial configuration
        std::map<std::string, Eigen::VectorXd> qInitMap;
        if (isStateTheoretical)
        {
            Eigen::VectorXd q0;
            robot->getExtendedPositionFromTheoretical(qInit, q0);
            qInitMap.emplace(robotName, std::move(q0));
        }
        else
        {
            qInitMap.emplace(robotName, qInit);
        }

        // Process initial velocity
        std::map<std::string, Eigen::VectorXd> vInitMap;
        if (isStateTheoretical)
        {
            Eigen::VectorXd v0;
            robot->getExtendedVelocityFromTheoretical(vInit, v0);
            vInitMap.emplace(robotName, std::move(v0));
        }
        else
        {
            vInitMap.emplace(robotName, vInit);
        }

        // Process initial acceleration
        std::optional<std::map<std::string, Eigen::VectorXd>> aInitMap;
        if (aInit)
        {
            aInitMap.emplace();
            if (isStateTheoretical)
            {
                Eigen::VectorXd a0;
                robot->getExtendedVelocityFromTheoretical(*aInit, a0);
                aInitMap->emplace(robotName, std::move(a0));
            }
            else
            {
                aInitMap->emplace(robotName, aInit.value());
            }
        }

        return {qInitMap, vInitMap, aInitMap};
    }

    void Engine::start(const Eigen::VectorXd & qInit,
                       const Eigen::VectorXd & vInit,
                       const std::optional<Eigen::VectorXd> & aInit,
                       bool isStateTheoretical)
    {
        // Make sure that a single robot has been added to the engine
        if (robots_.size() != 1)
        {
            JIMINY_THROW(bad_control_flow,
                         "Multi-robot simulation requires specifying the initial state of each "
                         "robot individually.");
        }

        // Pre-process initial state
        auto [qInitMap, vInitMap, aInitMap] =
            sanitizeInitialData(robots_[0], isStateTheoretical, qInit, vInit, aInit);

        // Start simulation
        start(qInitMap, vInitMap, aInitMap);
    }

    void Engine::simulate(double tEnd,
                          const std::map<std::string, Eigen::VectorXd> & qInit,
                          const std::map<std::string, Eigen::VectorXd> & vInit,
                          const std::optional<std::map<std::string, Eigen::VectorXd>> & aInit,
                          const AbortSimulationFunction & callback)
    {
        // Make sure that no simulation is already running
        if (robots_.empty())
        {
            JIMINY_THROW(bad_control_flow,
                         "No robot to simulate. Please add one before starting simulation.");
        }

        // Make sure that the user-specified simulation duration is long enough
        if (tEnd < 5e-3)
        {
            JIMINY_THROW(std::invalid_argument, "Simulation duration cannot be shorter than 5ms.");
        }

        // Reset the engine and all the robots
        reset(true, false);

        // Start the simulation
        start(qInit, vInit, aInit);

        // Now that telemetry has been initialized, check simulation duration
        if (tEnd > telemetryRecorder_->getLogDurationMax())
        {
            JIMINY_THROW(std::runtime_error,
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

            // Stop the simulation if callback returns false
            if (!callback())
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

    void Engine::simulate(double tEnd,
                          const Eigen::VectorXd & qInit,
                          const Eigen::VectorXd & vInit,
                          const std::optional<Eigen::VectorXd> & aInit,
                          bool isStateTheoretical,
                          const AbortSimulationFunction & callback)
    {
        // Make sure that a single robot has been added to the engine
        if (robots_.size() != 1)
        {
            JIMINY_THROW(bad_control_flow,
                         "Multi-robot simulation requires specifying the initial state of each "
                         "robot individually.");
        }

        // Pre-process initial state
        auto [qInitMap, vInitMap, aInitMap] =
            sanitizeInitialData(robots_[0], isStateTheoretical, qInit, vInit, aInit);

        // Run simulation
        simulate(tEnd, qInitMap, vInitMap, aInitMap, callback);
    }

    void Engine::step(double stepSize)
    {
        // Check if the simulation has started
        if (!isSimulationRunning_)
        {
            JIMINY_THROW(bad_control_flow,
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
                JIMINY_THROW(std::runtime_error,
                             "Low-level ode solver failed. Consider increasing stepper accuracy.");
            }
        }

        // Check if the desired step size is suitable
        if (stepSize > EPS && stepSize < SIMULATION_MIN_TIMESTEP)
        {
            JIMINY_THROW(std::invalid_argument, "Step size out of bounds.");
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
            JIMINY_THROW(std::runtime_error,
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
        uint32_t successiveIterTooLarge = 0;
        uint32_t successiveIterFailed = 0;
        std::vector<uint32_t> successiveSolveFailedAll(robots_.size(), 0U);
        stepper::StatusInfo status{stepper::ReturnCode::IS_SUCCESS, {}};

        /* Flag monitoring if the current time step depends of a breakpoint or the integration
           tolerance. It will be used by the restoration mechanism, if dt gets very small to reach
           a breakpoint, in order to avoid having to perform several steps to stabilize again the
           estimation of the optimal time step. */
        bool isBreakpointReached = false;

        /* Flag monitoring if the dynamics has changed between t- and t+ because of impulse forces
           or command update in the case of discrete controllers.

           `tryStep(rhs, x, dxdt, t, dt)` method of error controlled boost steppers leverages the
           FSAL (First Same As Last) principle, which consists in assuming that the value of
           `(x, dxdt)` in argument have been initialized by the user with the robot dynamics at
           current time t. Thus, if the robot dynamics is discontinuous, one has to manually
           integrate up to t-, then update dxdt to take into the acceleration at t+.

           Note that ONLY the acceleration part of dxdt must be updated since the velocity is not
           supposed to have changed. On top of that, tPrev is invalid at this point because it has
           been updated just after the last successful step.

           TODO: In theory, dt should be reschedule because the dynamics has changed and thereby
           the previously estimated dt is not appropriate anymore. */
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
            for (auto & robotData : robotDataVec_)
            {
                /* Update the active set: activate an impulse force as soon as the current time
                   gets close enough of the application time, and deactivate it the same way.

                   Note that breakpoints at the start/end of every impulse forces are already
                   enforced, so that the forces cannot get activated/deactivate too late. */
                auto isImpulseForceActiveIt = robotData.isImpulseForceActiveVec.begin();
                auto impulseForceIt = robotData.impulseForces.begin();
                for (; impulseForceIt != robotData.impulseForces.end();
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
                auto & tBreakpointNextIt = robotData.impulseForceBreakpointNextIt;
                if (tBreakpointNextIt != robotData.impulseForceBreakpoints.end())
                {
                    if (t >= *tBreakpointNextIt - STEPPER_MIN_TIMESTEP)
                    {
                        // The current breakpoint is behind in time. Switching to the next one.
                        ++tBreakpointNextIt;
                    }
                }

                // Get the next breakpoint time if any
                if (tBreakpointNextIt != robotData.impulseForceBreakpoints.end())
                {
                    tImpulseForceNext = std::min(tImpulseForceNext, *tBreakpointNextIt);
                }
            }

            // Update the external force profiles if necessary (only for finite update frequency)
            if (std::isfinite(stepperUpdatePeriod_))
            {
                auto robotIt = robots_.begin();
                auto robotDataIt = robotDataVec_.begin();
                for (; robotIt != robots_.end(); ++robotIt, ++robotDataIt)
                {
                    for (auto & profileForce : robotDataIt->profileForces)
                    {
                        if (profileForce.updatePeriod > EPS)
                        {
                            double forceUpdatePeriod = profileForce.updatePeriod;
                            double dtNextForceUpdatePeriod =
                                forceUpdatePeriod - std::fmod(t, forceUpdatePeriod);
                            if (dtNextForceUpdatePeriod < SIMULATION_MIN_TIMESTEP ||
                                forceUpdatePeriod - dtNextForceUpdatePeriod < STEPPER_MIN_TIMESTEP)
                            {
                                const Eigen::VectorXd & q = robotDataIt->state.q;
                                const Eigen::VectorXd & v = robotDataIt->state.v;
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
                    auto robotIt = robots_.begin();
                    auto robotDataIt = robotDataVec_.begin();
                    for (; robotIt != robots_.end(); ++robotIt, ++robotDataIt)
                    {
                        const Eigen::VectorXd & q = robotDataIt->state.q;
                        const Eigen::VectorXd & v = robotDataIt->state.v;
                        Eigen::VectorXd & command = robotDataIt->state.command;
                        computeCommand(*robotIt, t, q, v, command);
                    }
                    hasDynamicsChanged = true;
                }
            }

            /* Update telemetry if necessary. It monitors the current iteration number, the current
               time, and the robots state, command, and sensors data.

               Note that the acceleration is discontinuous. In particular, it would have different
               values of the same timestep if the command has been updated. Logging the previous
               acceleration is more natural since it preserves the consistency between sensors data
               and robot state. However, it is not the one that will be taken into account for
               integrating the physics at the current step. As a result, it is necessary to log the
               acceleration both at the end of the previous step (t-) and at the beginning of the
               next one (t+) to make sure that no information is lost in log data. This means that
               the same timestep will be logged twice, but this is permitted by the telemetry. */
            bool mustUpdateTelemetry = false;
            if (!std::isfinite(stepperUpdatePeriod_) ||
                !engineOptions_->telemetry.logInternalStepperSteps)
            {
                mustUpdateTelemetry = !std::isfinite(stepperUpdatePeriod_);
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

            // Fix the FSAL issue if the dynamics has changed, and update the telemetry accordingly
            if (!std::isfinite(stepperUpdatePeriod_) && hasDynamicsChanged)
            {
                computeRobotsDynamics(t, qSplit, vSplit, aSplit, true);
                syncAllAccelerationsAndForces(robots_, contactForcesPrev_, fPrev_, aPrev_);
                syncRobotsStateWithStepper(true);
                hasDynamicsChanged = false;
                if (mustUpdateTelemetry && engineOptions_->telemetry.logInternalStepperSteps)
                {
                    updateTelemetry();
                }
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
                    mustUpdateTelemetry = successiveIterFailed == 0 &&
                                          engineOptions_->telemetry.logInternalStepperSteps;
                    if (mustUpdateTelemetry)
                    {
                        updateTelemetry();
                    }

                    // Fix the FSAL issue if the dynamics has changed and update the telemetry
                    if (hasDynamicsChanged)
                    {
                        computeRobotsDynamics(t, qSplit, vSplit, aSplit, true);
                        syncAllAccelerationsAndForces(robots_, contactForcesPrev_, fPrev_, aPrev_);
                        syncRobotsStateWithStepper(true);
                        hasDynamicsChanged = false;
                        if (mustUpdateTelemetry)
                        {
                            updateTelemetry();
                        }
                    }

                    /* Break the loop if the prescribed timestep 'dt' is getting too small.
                       At this point, it corresponds to the estimated maximum timestep 'dtLargest'
                       in case of adaptive steppers, which is equal to +INF for fixed timestep.
                       An exception will be raised later. */
                    if (dt < STEPPER_MIN_TIMESTEP)
                    {
                        break;
                    }

                    /* Adjust stepsize to end up exactly at the next breakpoint if it is reasonable
                       to expect that integration will be successful, namely:
                       - If the next breakpoint is closer than the estimated maximum step size
                       OR
                       - If the next breakpoint is farther but not so far away compared to the
                         estimated maximum step size, AND the previous integration trial was not
                         a failure due to over-optimistic attempted timestep. This last condition
                         is essential to prevent infinite loop of slightly increasing the step
                         size, failing to integrate, then try again and again until triggering
                         maximum successive iteration failure exception. */
                    double dtResidualThr = STEPPER_MIN_TIMESTEP;
                    if (successiveIterTooLarge == 0)
                    {
                        dtResidualThr =
                            std::clamp(0.1 * dt, STEPPER_MIN_TIMESTEP, SIMULATION_MIN_TIMESTEP);
                    }
                    if (tNext - t < dt ||
                        (successiveIterTooLarge <= 1 && tNext - t < dt + dtResidualThr))
                    {
                        dt = tNext - t;
                    }

                    /* Trying to reach multiples of SIMULATION_MIN_TIMESTEP whenever possible. The
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

                    // Break the loop in case of timeout. Exception will be raised later.
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
                    auto robotDataIt = robotDataVec_.begin();
                    auto successiveSolveFailedIt = successiveSolveFailedAll.begin();
                    for (; robotDataIt != robotDataVec_.end();
                         ++robotDataIt, ++successiveSolveFailedIt)
                    {
                        *successiveSolveFailedIt = robotDataIt->successiveSolveFailed;
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
                    status = stepper_->tryStep(qSplit, vSplit, aSplit, t, dtLargest);
                    bool isStepSuccessful = status.returnCode == stepper::ReturnCode::IS_SUCCESS;

                    // Update buffer if really successful
                    if (isStepSuccessful)
                    {
                        // Reset successive iteration failure counter
                        successiveIterTooLarge = 0;
                        successiveIterFailed = 0;

                        // Synchronize the position, velocity and acceleration of all robots
                        syncRobotsStateWithStepper();

                        /* Compute all external terms including joints accelerations and forces.
                           Note that it is possible to call this method because `pinocchio::Data`
                           is guaranteed to be up-to-date at this point. */
                        computeAllExtraTerms(robots_, robotDataVec_, fPrev_);

                        // Backend the updated joint accelerations and forces
                        syncAllAccelerationsAndForces(robots_, contactForcesPrev_, fPrev_, aPrev_);

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

                        /* Backup the stepper and robots' state on success only:
                           - t at last successful iteration is used to compute dt, which is project
                             the acceleration in the state space instead of SO3^2.
                           - dtLargestPrev is used to restore the largest step size in case of a
                             breakpoint requiring lowering it.
                           - the acceleration and effort at the last successful iteration is used
                             to update the sensors' data in case of continuous sensing. */
                        stepperState_.tPrev = t;
                        stepperState_.dtLargestPrev = dtLargest;
                        for (auto & robotData : robotDataVec_)
                        {
                            robotData.statePrev = robotData.state;
                        }
                    }
                    else
                    {
                        /* Check if the integrator raised an exception. This typically happens
                           when the timestep is fixed and too large, causing the integrator to
                           fail miserably returning nan. In such a case, adjust the timestep
                           manually as a recovery mechanism based on a simple heuristic.
                           Note that it has no effect for fixed-timestep integrator since
                           `dtLargest` should be INF already. */
                        if (status.returnCode == stepper::ReturnCode::IS_ERROR)
                        {
                            dtLargest *= 0.1;
                        }

                        // Increment the failed iteration counters
                        if (status.returnCode == stepper::ReturnCode::IS_FAILURE)
                        {
                            ++successiveIterTooLarge;
                        }
                        ++successiveIterFailed;
                        ++stepperState_.iterFailed;

                        // Restore number of successive constraint solving failure
                        robotDataIt = robotDataVec_.begin();
                        successiveSolveFailedIt = successiveSolveFailedAll.begin();
                        for (; robotDataIt != robotDataVec_.end();
                             ++robotDataIt, ++successiveSolveFailedIt)
                        {
                            robotDataIt->successiveSolveFailed = *successiveSolveFailedIt;
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
                    // Break the loop in case of too many successive failed inner iteration
                    if (successiveIterFailed > engineOptions_->stepper.successiveIterFailedMax)
                    {
                        break;
                    }

                    /* Backup current number of successive constraint solving failure.
                       It will be restored in case of integration failure. */
                    auto robotDataIt = robotDataVec_.begin();
                    auto successiveSolveFailedIt = successiveSolveFailedAll.begin();
                    for (; robotDataIt != robotDataVec_.end();
                         ++robotDataIt, ++successiveSolveFailedIt)
                    {
                        *successiveSolveFailedIt = robotDataIt->successiveSolveFailed;
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

                    // Set the timestep to be tried by the stepper
                    dtLargest = dt;

                    // Try to do a step
                    status = stepper_->tryStep(qSplit, vSplit, aSplit, t, dtLargest);
                    isStepSuccessful = status.returnCode == stepper::ReturnCode::IS_SUCCESS;

                    if (isStepSuccessful)
                    {
                        // Reset successive iteration failure counter
                        successiveIterTooLarge = 0;
                        successiveIterFailed = 0;

                        // Synchronize the position, velocity and acceleration of all robots
                        syncRobotsStateWithStepper();

                        // Compute all external terms including joints accelerations and forces
                        computeAllExtraTerms(robots_, robotDataVec_, fPrev_);

                        // Backend the updated joint accelerations and forces
                        syncAllAccelerationsAndForces(robots_, contactForcesPrev_, fPrev_, aPrev_);

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

                        // Backup the stepper and robots' state
                        stepperState_.tPrev = t;
                        stepperState_.dtLargestPrev = dtLargest;
                        for (auto & robotData : robotDataVec_)
                        {
                            robotData.statePrev = robotData.state;
                        }
                    }
                    else
                    {
                        // Adjust timestep manually if necessary
                        if (status.returnCode == stepper::ReturnCode::IS_ERROR)
                        {
                            dtLargest *= 0.1;
                        }

                        // Increment the failed iteration counter
                        if (status.returnCode == stepper::ReturnCode::IS_FAILURE)
                        {
                            ++successiveIterTooLarge;
                        }
                        ++successiveIterFailed;
                        ++stepperState_.iterFailed;

                        // Restore number of successive constraint solving failure
                        robotDataIt = robotDataVec_.begin();
                        successiveSolveFailedIt = successiveSolveFailedAll.begin();
                        for (; robotDataIt != robotDataVec_.end();
                             ++robotDataIt, ++successiveSolveFailedIt)
                        {
                            robotDataIt->successiveSolveFailed = *successiveSolveFailedIt;
                        }
                    }

                    // Initialize the next dt
                    dt = std::min(dtLargest, engineOptions_->stepper.dtMax);
                }
            }

            // Exception handling
            if (successiveIterFailed > engineOptions_->stepper.successiveIterFailedMax)
            {
                if (status.exception)
                {
                    try
                    {
                        std::rethrow_exception(status.exception);
                    }
                    catch (const std::exception & e)
                    {
                        // TODO: Support `std::throw_with_nested` in JIMINY_THROW instead
                        JIMINY_THROW(
                            std::runtime_error,
                            "Something is wrong with the physics. Try using an adaptive stepper. "
                            "Aborting integration.\nRaised from exception: ",
                            e.what());
                    }
                }
                JIMINY_THROW(std::runtime_error,
                             "Too many successive iteration failures. Probably something is wrong "
                             "with the physics. Aborting integration.");
            }
            for (uint32_t successiveSolveFailed : successiveSolveFailedAll)
            {
                if (successiveSolveFailed > engineOptions_->stepper.successiveIterFailedMax)
                {
                    JIMINY_THROW(
                        std::runtime_error,
                        "Too many successive constraint solving failures. Try increasing the "
                        "regularization factor. Aborting integration.");
                }
            }
            if (dt < STEPPER_MIN_TIMESTEP)
            {
                JIMINY_THROW(std::runtime_error,
                             "The internal time step is getting too small. Impossible to "
                             "integrate physics further in time. Aborting integration.");
            }
            if (EPS < engineOptions_->stepper.timeout &&
                engineOptions_->stepper.timeout < timer_.toc())
            {
                JIMINY_THROW(std::runtime_error,
                             "Step computation timeout. Aborting integration.");
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
                auto robotIt = robots_.begin();
                auto robotDataIt = robotDataVec_.begin();
                for (; robotIt != robots_.end(); ++robotIt, ++robotDataIt)
                {
                    const Eigen::VectorXd & q = robotDataIt->state.q;
                    const Eigen::VectorXd & v = robotDataIt->state.v;
                    const Eigen::VectorXd & a = robotDataIt->state.a;
                    const Eigen::VectorXd & uMotor = robotDataIt->state.uMotor;
                    const ForceVector & fext = robotDataIt->state.fExternal;
                    (*robotIt)->computeSensorMeasurements(t, q, v, a, uMotor, fext);
                }
            }
        }

        /* Update the final time and dt to make sure it corresponds to the desired values and avoid
           compounding of error. Anyway the user asked for a step of exactly stepSize, so he is
           expecting this value to be reached. */
        t = tEnd;
    }

    void Engine::stop()
    {
        // Release the lock on the robots
        for (auto & robotData : robotDataVec_)
        {
            robotData.robotLock.reset(nullptr);
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

    void Engine::registerImpulseForce(const std::string & robotName,
                                      const std::string & frameName,
                                      double t,
                                      double dt,
                                      const pinocchio::Force & force)
    {
        if (isSimulationRunning_)
        {
            JIMINY_THROW(
                bad_control_flow,
                "Simulation already running. Please stop it before registering new forces.");
        }

        if (dt < STEPPER_MIN_TIMESTEP)
        {
            JIMINY_THROW(std::invalid_argument,
                         "Force duration cannot be smaller than ",
                         STEPPER_MIN_TIMESTEP,
                         "s.");
        }

        if (t < 0.0)
        {
            JIMINY_THROW(std::invalid_argument, "Force application time must be positive.");
        }

        if (frameName == "universe")
        {
            JIMINY_THROW(std::invalid_argument,
                         "Impossible to apply external forces to the universe itself!");
        }

        // TODO: Make sure that the forces do NOT overlap while taking into account dt.

        std::ptrdiff_t robotIndex = getRobotIndex(robotName);
        pinocchio::FrameIndex frameIndex =
            getFrameIndex(robots_[robotIndex]->pinocchioModel_, frameName);

        RobotData & robotData = robotDataVec_[robotIndex];
        robotData.impulseForces.emplace_back(frameName, frameIndex, t, dt, force);
        robotData.impulseForceBreakpoints.emplace(t);
        robotData.impulseForceBreakpoints.emplace(t + dt);
        robotData.isImpulseForceActiveVec.emplace_back(false);
    }

    template<typename... Args>
    std::tuple<bool, const double &>
    isGcdIncluded(const vector_aligned_t<RobotData> & robotDataVec, const Args &... values)
    {
        if (robotDataVec.empty())
        {
            return isGcdIncluded(values...);
        }

        const double * valueMinPtr = &INF;
        auto func = [&valueMinPtr, &values...](const RobotData & robotData)
        {
            auto && [isIncluded, value] = isGcdIncluded(
                robotData.profileForces.cbegin(),
                robotData.profileForces.cend(),
                [](const ProfileForce & force) -> const double & { return force.updatePeriod; },
                values...);
            valueMinPtr = &(minClipped(*valueMinPtr, value));
            return isIncluded;
        };
        // FIXME: Order of evaluation is not always respected with MSVC.
        bool isIncluded = std::all_of(robotDataVec.begin(), robotDataVec.end(), func);
        return {isIncluded, *valueMinPtr};
    }

    void Engine::registerProfileForce(const std::string & robotName,
                                      const std::string & frameName,
                                      const ProfileForceFunction & forceFunc,
                                      double updatePeriod)
    {
        if (isSimulationRunning_)
        {
            JIMINY_THROW(
                bad_control_flow,
                "Simulation already running. Please stop it before registering new forces.");
        }

        if (frameName == "universe")
        {
            JIMINY_THROW(std::invalid_argument,
                         "Impossible to apply external forces to the universe itself!");
        }

        // Get robot and frame indices
        std::ptrdiff_t robotIndex = getRobotIndex(robotName);
        pinocchio::FrameIndex frameIndex =
            getFrameIndex(robots_[robotIndex]->pinocchioModel_, frameName);

        // Make sure the update period is valid
        if (EPS < updatePeriod && updatePeriod < SIMULATION_MIN_TIMESTEP)
        {
            JIMINY_THROW(
                std::invalid_argument,
                "Cannot register external force profile with update period smaller than ",
                SIMULATION_MIN_TIMESTEP,
                "s. Adjust period or switch to continuous mode by setting period to zero.");
        }
        // Make sure the desired update period is a multiple of the stepper period
        auto [isIncluded, updatePeriodMin] =
            isGcdIncluded(robotDataVec_, stepperUpdatePeriod_, updatePeriod);
        if (!isIncluded)
        {
            JIMINY_THROW(std::invalid_argument,
                         "In discrete mode, the update period of force profiles and the "
                         "stepper update period (min of controller and sensor update "
                         "periods) must be multiple of each other.");
        }

        // Set breakpoint period during the integration loop
        stepperUpdatePeriod_ = updatePeriodMin;

        // Add force profile to register
        RobotData & robotData = robotDataVec_[robotIndex];
        robotData.profileForces.emplace_back(frameName, frameIndex, updatePeriod, forceFunc);
    }

    void Engine::removeImpulseForces(const std::string & robotName)
    {
        // Make sure that no simulation is running
        if (isSimulationRunning_)
        {
            JIMINY_THROW(bad_control_flow,
                         "Simulation already running. Stop it before removing impulse forces.");
        }

        std::ptrdiff_t robotIndex = getRobotIndex(robotName);
        RobotData & robotData = robotDataVec_[robotIndex];
        robotData.impulseForces.clear();
    }

    void Engine::removeImpulseForces()
    {
        // Make sure that no simulation is running
        if (isSimulationRunning_)
        {
            JIMINY_THROW(bad_control_flow,
                         "simulation already running. Stop it before removing impulse forces.");
        }

        for (auto & robotData : robotDataVec_)
        {
            robotData.impulseForces.clear();
        }
    }

    void Engine::removeProfileForces(const std::string & robotName)
    {
        // Make sure that no simulation is running
        if (isSimulationRunning_)
        {
            JIMINY_THROW(bad_control_flow,
                         "Simulation already running. Stop it before removing profile forces.");
        }


        // Remove force profile from register
        std::ptrdiff_t robotIndex = getRobotIndex(robotName);
        RobotData & robotData = robotDataVec_[robotIndex];
        robotData.profileForces.clear();

        // Set breakpoint period during the integration loop
        // FIXME: replaced `std::get<N>` by placeholder `_` when moving to C++26 (P2169R4)
        stepperUpdatePeriod_ =
            std::get<1>(isGcdIncluded(robotDataVec_,
                                      engineOptions_->stepper.sensorsUpdatePeriod,
                                      engineOptions_->stepper.controllerUpdatePeriod));
    }

    void Engine::removeProfileForces()
    {
        // Make sure that no simulation is running
        if (isSimulationRunning_)
        {
            JIMINY_THROW(bad_control_flow,
                         "Simulation already running. Stop it before removing profile forces.");
        }

        for (auto & robotData : robotDataVec_)
        {
            robotData.profileForces.clear();
        }
    }

    const ImpulseForceVector & Engine::getImpulseForces(const std::string & robotName) const
    {
        std::ptrdiff_t robotIndex = getRobotIndex(robotName);
        const RobotData & robotData = robotDataVec_[robotIndex];
        return robotData.impulseForces;
    }

    const ProfileForceVector & Engine::getProfileForces(const std::string & robotName) const
    {
        std::ptrdiff_t robotIndex = getRobotIndex(robotName);
        const RobotData & robotData = robotDataVec_[robotIndex];
        return robotData.profileForces;
    }

    void Engine::setOptions(const GenericConfig & engineOptions)
    {
        if (isSimulationRunning_)
        {
            JIMINY_THROW(
                bad_control_flow,
                "Simulation already running. Please stop it before updating the options.");
        }

        // Make sure the dtMax is not out of range
        const GenericConfig & stepperOptions =
            boost::get<GenericConfig>(engineOptions.at("stepper"));
        const double dtMax = boost::get<double>(stepperOptions.at("dtMax"));
        if (SIMULATION_MAX_TIMESTEP + EPS < dtMax || dtMax < SIMULATION_MIN_TIMESTEP)
        {
            JIMINY_THROW(std::invalid_argument,
                         "'dtMax' option must bge in range [",
                         SIMULATION_MIN_TIMESTEP,
                         ", ",
                         SIMULATION_MAX_TIMESTEP,
                         "].");
        }

        // Make sure successiveIterFailedMax is strictly positive
        const uint32_t successiveIterFailedMax =
            boost::get<uint32_t>(stepperOptions.at("successiveIterFailedMax"));
        if (successiveIterFailedMax < 1)
        {
            JIMINY_THROW(std::invalid_argument,
                         "'successiveIterFailedMax' must be strictly positive.");
        }

        // Make sure the selected ode solver is available and instantiate it
        const std::string & odeSolver = boost::get<std::string>(stepperOptions.at("odeSolver"));
        if (STEPPERS.find(odeSolver) == STEPPERS.end())
        {
            JIMINY_THROW(
                std::invalid_argument, "Requested ODE solver '", odeSolver, "' not available.");
        }

        // Make sure the controller and sensor update periods are valid
        const double sensorsUpdatePeriod =
            boost::get<double>(stepperOptions.at("sensorsUpdatePeriod"));
        const double controllerUpdatePeriod =
            boost::get<double>(stepperOptions.at("controllerUpdatePeriod"));
        auto [isIncluded, updatePeriodMin] =
            isGcdIncluded(robotDataVec_, controllerUpdatePeriod, sensorsUpdatePeriod);
        if ((EPS < sensorsUpdatePeriod && sensorsUpdatePeriod < SIMULATION_MIN_TIMESTEP) ||
            (EPS < controllerUpdatePeriod && controllerUpdatePeriod < SIMULATION_MIN_TIMESTEP))
        {
            JIMINY_THROW(
                std::invalid_argument,
                "Cannot simulate a discrete robot with update period smaller than ",
                SIMULATION_MIN_TIMESTEP,
                "s. Adjust period or switch to continuous mode by setting period to zero.");
        }
        else if (!isIncluded)
        {
            JIMINY_THROW(std::invalid_argument,
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
            JIMINY_THROW(std::invalid_argument,
                         "Requested constraint solver '",
                         constraintSolverType,
                         "' not available.");
        }
        double regularization = boost::get<double>(constraintsOptions.at("regularization"));
        if (regularization < 0.0)
        {
            JIMINY_THROW(std::invalid_argument,
                         "Constraint option 'regularization' must be positive.");
        }

        // Make sure the contacts options are fine
        const GenericConfig & contactOptions =
            boost::get<GenericConfig>(engineOptions.at("contacts"));
        const std::string & contactModel = boost::get<std::string>(contactOptions.at("model"));
        const auto contactModelIt = CONTACT_MODELS_MAP.find(contactModel);
        if (contactModelIt == CONTACT_MODELS_MAP.end())
        {
            JIMINY_THROW(std::invalid_argument, "Requested contact model not available.");
        }
        double contactsTransitionEps = boost::get<double>(contactOptions.at("transitionEps"));
        if (contactsTransitionEps < 0.0)
        {
            JIMINY_THROW(std::invalid_argument,
                         "Contact option 'transitionEps' must be positive.");
        }
        double transitionVelocity = boost::get<double>(contactOptions.at("transitionVelocity"));
        if (transitionVelocity < EPS)
        {
            JIMINY_THROW(std::invalid_argument,
                         "Contact option 'transitionVelocity' must be strictly positive.");
        }
        double stabilizationFreq = boost::get<double>(contactOptions.at("stabilizationFreq"));
        if (stabilizationFreq < 0.0)
        {
            JIMINY_THROW(std::invalid_argument,
                         "Contact option 'stabilizationFreq' must be positive.");
        }

        // Make sure the user-defined gravity force has the right dimension
        const GenericConfig & worldOptions = boost::get<GenericConfig>(engineOptions.at("world"));
        Eigen::VectorXd gravity = boost::get<Eigen::VectorXd>(worldOptions.at("gravity"));
        if (gravity.size() != 6)
        {
            JIMINY_THROW(std::invalid_argument, "The size of the gravity force vector must be 6.");
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

        // Update class-specific "strongly typed" accessor for fast and convenient access
        engineOptions_ = std::make_unique<const EngineOptions>(engineOptions);

        // Update inherited polymorphic accessor
        deepUpdate(boost::get<GenericConfig>(simulationOptionsGeneric_.at("engine")),
                   engineOptions);

        // Backup contact model as enum for fast check
        contactModel_ = contactModelIt->second;

        // Set breakpoint period during the integration loop
        stepperUpdatePeriod_ = updatePeriodMin;
    }

    const GenericConfig & Engine::getOptions() const noexcept
    {
        return boost::get<GenericConfig>(simulationOptionsGeneric_.at("engine"));
    }

    void Engine::setSimulationOptions(const GenericConfig & simulationOptions)
    {
        // Set engine options
        GenericConfig::const_iterator engineOptionsIt = simulationOptions.find("engine");
        if (engineOptionsIt == simulationOptions.end())
        {
            JIMINY_THROW(std::invalid_argument, "Engine options key 'engine' is missing.");
        }
        setOptions(boost::get<GenericConfig>(engineOptionsIt->second));

        // Set options for each robot sequentially
        for (auto & robot : robots_)
        {
            std::string robotOptionsKey = robot->getName();
            if (robotOptionsKey.empty())
            {
                robotOptionsKey = "robot";
            }
            GenericConfig::const_iterator robotOptionsIt = simulationOptions.find(robotOptionsKey);
            if (robotOptionsIt == simulationOptions.end())
            {
                JIMINY_THROW(std::invalid_argument,
                             "Robot options key '",
                             robotOptionsKey,
                             "' is missing.");
            }
            robot->setOptions(boost::get<GenericConfig>(robotOptionsIt->second));
        }
    }

    GenericConfig Engine::getSimulationOptions() const noexcept
    {
        /* Return options without refreshing all options if and only if the same simulation is
           still running since the last time they were considered valid. */
        if (areSimulationOptionsRefreshed_ && isSimulationRunning_)
        {
            return simulationOptionsGeneric_;
        }

        // Refresh robot options
        for (const auto & robot : robots_)
        {
            std::string robotOptionsKey = robot->getName();
            if (robotOptionsKey.empty())
            {
                robotOptionsKey = "robot";
            }
            simulationOptionsGeneric_[robotOptionsKey] = robot->getOptions();
        }

        // Options are now considered "valid"
        areSimulationOptionsRefreshed_ = true;

        return simulationOptionsGeneric_;
    }

    std::ptrdiff_t Engine::getRobotIndex(const std::string & robotName) const
    {
        auto robotIt = std::find_if(robots_.begin(),
                                    robots_.end(),
                                    [&robotName](const auto & robot)
                                    { return (robot->getName() == robotName); });
        if (robotIt == robots_.end())
        {
            JIMINY_THROW(std::invalid_argument,
                         "No robot with name '",
                         robotName,
                         "' has been added to the engine.");
        }

        return std::distance(robots_.begin(), robotIt);
    }

    std::shared_ptr<Robot> Engine::getRobot(const std::string & robotName)
    {
        std::ptrdiff_t robotIndex = getRobotIndex(robotName);
        return robots_[robotIndex];
    }

    const RobotState & Engine::getRobotState(const std::string & robotName) const
    {
        std::ptrdiff_t robotIndex = getRobotIndex(robotName);
        return robotDataVec_[robotIndex].state;
    }

    const StepperState & Engine::getStepperState() const
    {
        return stepperState_;
    }

    const bool & Engine::getIsSimulationRunning() const
    {
        return isSimulationRunning_;
    }

    double Engine::getSimulationDurationMax()
    {
        return TelemetryRecorder::getLogDurationMax(getTelemetryTimeUnit());
    }

    double Engine::getTelemetryTimeUnit()
    {
        return STEPPER_MIN_TIMESTEP;
    }

    // ========================================================
    // =================== Stepper utilities ==================
    // ========================================================

    void Engine::syncStepperStateWithRobots()
    {
        auto qSplitIt = stepperState_.qSplit.begin();
        auto vSplitIt = stepperState_.vSplit.begin();
        auto aSplitIt = stepperState_.aSplit.begin();
        auto robotDataIt = robotDataVec_.begin();
        for (; robotDataIt != robotDataVec_.end();
             ++robotDataIt, ++qSplitIt, ++vSplitIt, ++aSplitIt)
        {
            *qSplitIt = robotDataIt->state.q;
            *vSplitIt = robotDataIt->state.v;
            *aSplitIt = robotDataIt->state.a;
        }
    }

    void Engine::syncRobotsStateWithStepper(bool isStateUpToDate)
    {
        if (isStateUpToDate)
        {
            auto aSplitIt = stepperState_.aSplit.begin();
            auto robotDataIt = robotDataVec_.begin();
            for (; robotDataIt != robotDataVec_.end(); ++robotDataIt, ++aSplitIt)
            {
                robotDataIt->state.a = *aSplitIt;
            }
        }
        else
        {
            auto qSplitIt = stepperState_.qSplit.begin();
            auto vSplitIt = stepperState_.vSplit.begin();
            auto aSplitIt = stepperState_.aSplit.begin();
            auto robotDataIt = robotDataVec_.begin();
            for (; robotDataIt != robotDataVec_.end();
                 ++robotDataIt, ++qSplitIt, ++vSplitIt, ++aSplitIt)
            {
                robotDataIt->state.q = *qSplitIt;
                robotDataIt->state.v = *vSplitIt;
                robotDataIt->state.a = *aSplitIt;
            }
        }
    }

    // ========================================================
    // ================ Core physics utilities ================
    // ========================================================

    void Engine::computeForwardKinematics(std::shared_ptr<Robot> & robot,
                                          const Eigen::VectorXd & q,
                                          const Eigen::VectorXd & v,
                                          const Eigen::VectorXd & a)
    {
        // Create proxies for convenience
        const pinocchio::Model & model = robot->pinocchioModel_;
        pinocchio::Data & data = robot->pinocchioData_;
        const pinocchio::GeometryModel & geomModel = robot->collisionModel_;
        pinocchio::GeometryData & geomData = robot->collisionData_;

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

    void Engine::computeContactDynamicsAtBody(
        const std::shared_ptr<Robot> & robot,
        const pinocchio::PairIndex & collisionPairIndex,
        const std::shared_ptr<AbstractConstraintBase> & constraint,
        pinocchio::Force & fextLocal) const
    {
        // TODO: It is assumed that the ground is flat. For now ground profile is not supported
        // with body collision. Nevertheless it should not be to hard to generated a collision
        // object simply by sampling points on the profile.

        // Define proxy for convenience
        pinocchio::Data & data = robot->pinocchioData_;

        // Get the frame and joint indices
        const pinocchio::GeomIndex & geometryIndex =
            robot->collisionModel_.collisionPairs[collisionPairIndex].first;
        pinocchio::JointIndex parentJointIndex =
            robot->collisionModel_.geometryObjects[geometryIndex].parentJoint;

        // Extract collision and distance results
        const hpp::fcl::CollisionResult & collisionResult =
            robot->collisionData_.collisionResults[collisionPairIndex];

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

    void Engine::computeContactDynamicsAtFrame(
        const std::shared_ptr<Robot> & robot,
        pinocchio::FrameIndex frameIndex,
        const std::shared_ptr<AbstractConstraintBase> & constraint,
        pinocchio::Force & fextLocal) const
    {
        /* Returns the external force in the contact frame. It must then be converted into a force
           onto the parent joint.
           /!\ Note that the contact dynamics depends only on kinematics data. /!\ */

        // Define proxies for convenience
        const pinocchio::Model & model = robot->pinocchioModel_;
        const pinocchio::Data & data = robot->pinocchioData_;

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
           case of slippage the contact point has physically moved. */
        if (constraint->getIsEnabled())
        {
            auto & frameConstraint = static_cast<FrameConstraint &>(*constraint.get());
            frameConstraint.setReferenceTransform(
                {transformFrameInWorld.rotation(), posFrame - depth * normalGround});
            frameConstraint.setNormal(normalGround);
        }
    }

    pinocchio::Force Engine::computeContactDynamics(const Eigen::Vector3d & normalGround,
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

    void Engine::computeCommand(std::shared_ptr<Robot> & robot,
                                double t,
                                const Eigen::VectorXd & q,
                                const Eigen::VectorXd & v,
                                Eigen::VectorXd & command)
    {
        // Reinitialize the external forces
        command.setZero();

        // Command the command
        robot->getController()->computeCommand(t, q, v, command);
    }

    struct computePositionLimitsForcesAlgo :
    public pinocchio::fusion::JointUnaryVisitorBase<computePositionLimitsForcesAlgo>
    {
        typedef boost::fusion::vector<
            const Eigen::VectorXd & /* q */,
            const Eigen::VectorXd & /* positionLimitLower */,
            const Eigen::VectorXd & /* positionLimitUpper */,
            const std::unique_ptr<const Engine::EngineOptions> & /* engineOptions */,
            const std::shared_ptr<AbstractConstraintBase> & /* constraint */>
            ArgsType;

        template<typename JointModel>
        static std::enable_if_t<is_pinocchio_joint_revolute_v<JointModel> ||
                                    is_pinocchio_joint_revolute_unaligned_v<JointModel> ||
                                    is_pinocchio_joint_prismatic_v<JointModel> ||
                                    is_pinocchio_joint_prismatic_unaligned_v<JointModel>,
                                void>
        algo(const pinocchio::JointModelBase<JointModel> & joint,
             const Eigen::VectorXd & q,
             const Eigen::VectorXd & positionLimitLower,
             const Eigen::VectorXd & positionLimitUpper,
             const std::unique_ptr<const Engine::EngineOptions> & engineOptions,
             const std::shared_ptr<AbstractConstraintBase> & constraint)
        {
            // Define some proxies for convenience
            const Eigen::Index positionIndex = joint.idx_q();
            const double qJoint = q[positionIndex];
            const double qJointMin = positionLimitLower[positionIndex];
            const double qJointMax = positionLimitUpper[positionIndex];
            const double transitionEps = engineOptions->contacts.transitionEps;

            // Check if out-of-bounds
            if (qJointMax < qJoint || qJoint < qJointMin)
            {
                // Enable fixed joint constraint
                auto & jointConstraint = static_cast<JointConstraint &>(*constraint.get());
                jointConstraint.setReferenceConfiguration(
                    Eigen::Matrix<double, 1, 1>(std::clamp(qJoint, qJointMin, qJointMax)));
                jointConstraint.setRotationDir(qJointMax < qJoint);
                constraint->enable();
            }
            else if (qJointMin + transitionEps < qJoint && qJoint < qJointMax - transitionEps)
            {
                // Disable fixed joint constraint
                constraint->disable();
            }
        }

        template<typename JointModel>
        static std::enable_if_t<is_pinocchio_joint_revolute_unbounded_v<JointModel> ||
                                    is_pinocchio_joint_revolute_unbounded_unaligned_v<JointModel>,
                                void>
        algo(const pinocchio::JointModelBase<JointModel> & /* joint */,
             const Eigen::VectorXd & /* q */,
             const Eigen::VectorXd & /* positionLimitLower */,
             const Eigen::VectorXd & /* positionLimitUpper */,
             const std::unique_ptr<const Engine::EngineOptions> & /* engineOptions */,
             const std::shared_ptr<AbstractConstraintBase> & constraint)
        {
            // Disable fixed joint constraint
            constraint->disable();
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
             const Eigen::VectorXd & /* q */,
             const Eigen::VectorXd & /* positionLimitLower */,
             const Eigen::VectorXd & /* positionLimitUpper */,
             const std::unique_ptr<const Engine::EngineOptions> & /* engineOptions */,
             const std::shared_ptr<AbstractConstraintBase> & constraint)
        {
#ifndef NDEBUG
            JIMINY_WARNING("Position bounds not implemented for this type of joint.");
#endif
            // Disable fixed joint constraint
            constraint->disable();
        }
    };

    void Engine::computeInternalDynamics(const std::shared_ptr<Robot> & robot,
                                         double /* t */,
                                         const Eigen::VectorXd & q,
                                         const Eigen::VectorXd & v,
                                         Eigen::VectorXd & uInternal) const
    {
        // Define some proxies
        const pinocchio::Model & model = robot->pinocchioModel_;
        const ConstraintTree & constraints = robot->getConstraints();

        /* Enforce position limits for all joints having bounds constraints, ie mechanical and
           backlash joints. */
        const Eigen::VectorXd & positionLimitLower = robot->pinocchioModel_.lowerPositionLimit;
        const Eigen::VectorXd & positionLimitUpper = robot->pinocchioModel_.upperPositionLimit;
        for (auto & constraintItem : constraints.boundJoints)
        {
            auto & constraint = constraintItem.second;
            const auto jointConstraint = std::static_pointer_cast<JointConstraint>(constraint);
            const pinocchio::JointIndex jointIndex = jointConstraint->getJointIndex();
            computePositionLimitsForcesAlgo::run(
                model.joints[jointIndex],
                typename computePositionLimitsForcesAlgo::ArgsType(
                    q, positionLimitLower, positionLimitUpper, engineOptions_, constraint));
        }

        // Compute the flexibilities (only support `JointModelType::SPHERICAL` so far)
        double angle;
        Eigen::Matrix3d rotJlog3;
        const Robot::DynamicsOptions & modelDynOptions = robot->modelOptions_->dynamics;
        const std::vector<pinocchio::JointIndex> & flexibilityJointIndices =
            robot->getFlexibilityJointIndices();
        for (std::size_t i = 0; i < flexibilityJointIndices.size(); ++i)
        {
            const pinocchio::JointIndex jointIndex = flexibilityJointIndices[i];
            const Eigen::Index positionIndex = model.joints[jointIndex].idx_q();
            const Eigen::Index velocityIndex = model.joints[jointIndex].idx_v();
            const Eigen::Vector3d & stiffness = modelDynOptions.flexibilityConfig[i].stiffness;
            const Eigen::Vector3d & damping = modelDynOptions.flexibilityConfig[i].damping;

            const Eigen::Map<const Eigen::Quaterniond> quat(q.segment<4>(positionIndex).data());
            const Eigen::Vector3d angleAxis = pinocchio::quaternion::log3(quat, angle);
            if (angle > 0.95 * M_PI)  // Angle is always positive
            {
                JIMINY_THROW(std::runtime_error,
                             "Flexible joint angle must be smaller than 0.95 * pi.");
            }
            pinocchio::Jlog3(angle, angleAxis, rotJlog3);
            uInternal.segment<3>(velocityIndex) -=
                rotJlog3 * (stiffness.array() * angleAxis.array()).matrix();
            uInternal.segment<3>(velocityIndex).array() -=
                damping.array() * v.segment<3>(velocityIndex).array();
        }
    }

    void Engine::computeCollisionForces(const std::shared_ptr<Robot> & robot,
                                        RobotData & robotData,
                                        ForceVector & fext,
                                        bool isStateUpToDate) const
    {
        // Define proxy for convenience
        const ConstraintTree & constraints = robot->getConstraints();

        // Compute the forces at contact points
        const std::vector<pinocchio::FrameIndex> & contactFrameIndices =
            robot->getContactFrameIndices();
        for (std::size_t i = 0; i < contactFrameIndices.size(); ++i)
        {
            // Compute force at the given contact frame.
            const pinocchio::FrameIndex frameIndex = contactFrameIndices[i];
            auto & constraint = constraints.contactFrames[i].second;
            pinocchio::Force & fextLocal = robotData.contactFrameForces[i];
            if (!isStateUpToDate)
            {
                computeContactDynamicsAtFrame(robot, frameIndex, constraint, fextLocal);
            }

            // Apply the force at the origin of the parent joint frame, in local joint frame
            const pinocchio::JointIndex parentJointIndex =
                robot->pinocchioModel_.frames[frameIndex].parent;
            fext[parentJointIndex] += fextLocal;

            // Convert contact force from global frame to local frame to store it in contactForces_
            const pinocchio::SE3 & transformContactInJoint =
                robot->pinocchioModel_.frames[frameIndex].placement;
            robot->contactForces_[i] = transformContactInJoint.actInv(fextLocal);
        }

        // Compute the force at collision bodies
        const std::vector<pinocchio::FrameIndex> & collisionBodyIndices =
            robot->getCollisionBodyIndices();
        const std::vector<std::vector<pinocchio::PairIndex>> & collisionPairIndices =
            robot->getCollisionPairIndices();
        for (std::size_t i = 0; i < collisionBodyIndices.size(); ++i)
        {
            /* Compute force at the given collision body.
               It returns the force applied at the origin of parent joint frame in global frame. */
            const pinocchio::FrameIndex frameIndex = collisionBodyIndices[i];
            const pinocchio::JointIndex parentJointIndex =
                robot->pinocchioModel_.frames[frameIndex].parent;
            for (std::size_t j = 0; j < collisionPairIndices[i].size(); ++j)
            {
                pinocchio::Force & fextLocal = robotData.collisionBodiesForces[i][j];
                if (!isStateUpToDate)
                {
                    const pinocchio::PairIndex & collisionPairIndex = collisionPairIndices[i][j];
                    auto & constraint = constraints.collisionBodies[i][j].second;
                    computeContactDynamicsAtBody(robot, collisionPairIndex, constraint, fextLocal);
                }

                // Apply the force at the origin of the parent joint frame, in local joint frame
                fext[parentJointIndex] += fextLocal;
            }
        }
    }

    void Engine::computeExternalForces(const std::shared_ptr<Robot> & robot,
                                       RobotData & robotData,
                                       double t,
                                       const Eigen::VectorXd & q,
                                       const Eigen::VectorXd & v,
                                       ForceVector & fext)
    {
        // Add the effect of user-defined external impulse forces
        auto isImpulseForceActiveIt = robotData.isImpulseForceActiveVec.begin();
        auto impulseForceIt = robotData.impulseForces.begin();
        for (; impulseForceIt != robotData.impulseForces.end();
             ++isImpulseForceActiveIt, ++impulseForceIt)
        {
            /* Do not check if the force is active at this point. This is managed at stepper level
               to be able to disambiguate t- versus t+. */
            if (*isImpulseForceActiveIt)
            {
                const pinocchio::FrameIndex frameIndex = impulseForceIt->frameIndex;
                const pinocchio::JointIndex parentJointIndex =
                    robot->pinocchioModel_.frames[frameIndex].parent;
                fext[parentJointIndex] += convertForceGlobalFrameToJoint(robot->pinocchioModel_,
                                                                         robot->pinocchioData_,
                                                                         frameIndex,
                                                                         impulseForceIt->force);
            }
        }

        // Add the effect of time-continuous external force profiles
        for (auto & profileForce : robotData.profileForces)
        {
            const pinocchio::FrameIndex frameIndex = profileForce.frameIndex;
            const pinocchio::JointIndex parentJointIndex =
                robot->pinocchioModel_.frames[frameIndex].parent;
            if (profileForce.updatePeriod < EPS)
            {
                profileForce.force = profileForce.func(t, q, v);
            }
            fext[parentJointIndex] += convertForceGlobalFrameToJoint(
                robot->pinocchioModel_, robot->pinocchioData_, frameIndex, profileForce.force);
        }
    }

    void Engine::computeCouplingForces(double t,
                                       const std::vector<Eigen::VectorXd> & qSplit,
                                       const std::vector<Eigen::VectorXd> & vSplit)
    {
        for (auto & couplingForce : couplingForces_)
        {
            // Extract info about the first robot involved
            const std::ptrdiff_t robotIndex1 = couplingForce.robotIndex1;
            const std::shared_ptr<Robot> & robot1 = robots_[robotIndex1];
            const Eigen::VectorXd & q1 = qSplit[robotIndex1];
            const Eigen::VectorXd & v1 = vSplit[robotIndex1];
            const pinocchio::FrameIndex frameIndex1 = couplingForce.frameIndex1;
            ForceVector & fext1 = robotDataVec_[robotIndex1].state.fExternal;

            // Extract info about the second robot involved
            const std::ptrdiff_t robotIndex2 = couplingForce.robotIndex2;
            const std::shared_ptr<Robot> & robot2 = robots_[robotIndex2];
            const Eigen::VectorXd & q2 = qSplit[robotIndex2];
            const Eigen::VectorXd & v2 = vSplit[robotIndex2];
            const pinocchio::FrameIndex frameIndex2 = couplingForce.frameIndex2;
            ForceVector & fext2 = robotDataVec_[robotIndex2].state.fExternal;

            // Compute the coupling force
            pinocchio::Force force = couplingForce.func(t, q1, v1, q2, v2);
            const pinocchio::JointIndex parentJointIndex1 =
                robot1->pinocchioModel_.frames[frameIndex1].parent;
            fext1[parentJointIndex1] += convertForceGlobalFrameToJoint(
                robot1->pinocchioModel_, robot1->pinocchioData_, frameIndex1, force);

            // Move force from frame1 to frame2 to apply it to the second robot
            force.toVector() *= -1;
            const pinocchio::JointIndex parentJointIndex2 =
                robot2->pinocchioModel_.frames[frameIndex2].parent;
            const Eigen::Vector3d offset = robot2->pinocchioData_.oMf[frameIndex2].translation() -
                                           robot1->pinocchioData_.oMf[frameIndex1].translation();
            force.angular() -= offset.cross(force.linear());
            fext2[parentJointIndex2] += convertForceGlobalFrameToJoint(
                robot2->pinocchioModel_, robot2->pinocchioData_, frameIndex2, force);
        }
    }

    void Engine::computeAllTerms(double t,
                                 const std::vector<Eigen::VectorXd> & qSplit,
                                 const std::vector<Eigen::VectorXd> & vSplit,
                                 bool isStateUpToDate)
    {
        // Reinitialize the external forces and internal efforts
        for (auto & robotData : robotDataVec_)
        {
            for (pinocchio::Force & fext_i : robotData.state.fExternal)
            {
                fext_i.setZero();
            }
            robotData.state.uInternal.setZero();
        }

        // Compute the internal forces
        computeCouplingForces(t, qSplit, vSplit);

        // Compute each individual robot dynamics
        auto robotIt = robots_.begin();
        auto robotDataIt = robotDataVec_.begin();
        auto qIt = qSplit.begin();
        auto vIt = vSplit.begin();
        for (; robotIt != robots_.end(); ++robotIt, ++robotDataIt, ++qIt, ++vIt)
        {
            // Define some proxies
            ForceVector & fext = robotDataIt->state.fExternal;
            Eigen::VectorXd & uInternal = robotDataIt->state.uInternal;

            /* Compute internal dynamics, namely the efforts in joint space associated with
               position/velocity bounds dynamics, and flexibility dynamics.
               Note that they must be recomputed systematically, even if the state did not change
               since the last time they were computed, because we are not tracking their value but
               only aggregating them as part of the internal joint efforts. The latter will be
               updated when computing the system acceleration to account for the efforts associated
               with constraints at the joint-level no matter if the state is up to date. */
            computeInternalDynamics(*robotIt, t, *qIt, *vIt, uInternal);

            /* Compute the collision forces and estimated time at which the contact state will
               changed (Take-off / Touch-down). */
            computeCollisionForces(*robotIt, *robotDataIt, fext, isStateUpToDate);

            // Compute the external contact forces.
            computeExternalForces(*robotIt, *robotDataIt, t, *qIt, *vIt, fext);
        }
    }

    void Engine::computeRobotsDynamics(double t,
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
            JIMINY_THROW(bad_control_flow,
                         "No simulation running. Please start one before calling this method.");
        }

        // Make sure memory has been allocated for the output acceleration
        aSplit.resize(vSplit.size());

        if (!isStateUpToDate)
        {
            // Update kinematics for each robot
            auto robotIt = robots_.begin();
            auto robotDataIt = robotDataVec_.begin();
            auto qIt = qSplit.begin();
            auto vIt = vSplit.begin();
            for (; robotIt != robots_.end(); ++robotIt, ++robotDataIt, ++qIt, ++vIt)
            {
                const Eigen::VectorXd & aPrev = robotDataIt->statePrev.a;
                computeForwardKinematics(*robotIt, *qIt, *vIt, aPrev);
            }
        }

        /* Compute internal and external forces and efforts applied on every robots, excluding
           user-specified internal dynamics if any.

           Note that one must call this method BEFORE updating the sensors since the force sensor
           measurements rely on robot_->contactForces_. */
        computeAllTerms(t, qSplit, vSplit, isStateUpToDate);

        // Compute each individual robot dynamics
        auto robotIt = robots_.begin();
        auto robotDataIt = robotDataVec_.begin();
        auto qIt = qSplit.begin();
        auto vIt = vSplit.begin();
        auto contactForcesPrevIt = contactForcesPrev_.begin();
        auto fPrevIt = fPrev_.begin();
        auto aPrevIt = aPrev_.begin();
        auto aIt = aSplit.begin();
        for (; robotIt != robots_.end(); ++robotIt,
                                         ++robotDataIt,
                                         ++qIt,
                                         ++vIt,
                                         ++aIt,
                                         ++contactForcesPrevIt,
                                         ++fPrevIt,
                                         ++aPrevIt)
        {
            // Define some proxies
            Eigen::VectorXd & u = robotDataIt->state.u;
            Eigen::VectorXd & command = robotDataIt->state.command;
            Eigen::VectorXd & uMotor = robotDataIt->state.uMotor;
            Eigen::VectorXd & uTransmission = robotDataIt->state.uTransmission;
            Eigen::VectorXd & uInternal = robotDataIt->state.uInternal;
            Eigen::VectorXd & uCustom = robotDataIt->state.uCustom;
            ForceVector & fext = robotDataIt->state.fExternal;
            const Eigen::VectorXd & aPrev = robotDataIt->statePrev.a;
            const Eigen::VectorXd & uMotorPrev = robotDataIt->statePrev.uMotor;
            const ForceVector & fextPrev = robotDataIt->statePrev.fExternal;

            /* Update the sensor data if necessary (only for infinite update frequency).
               Note that it is impossible to have access to the current accelerations and efforts
               since they depend on the sensor values themselves. */
            if (!isStateUpToDate && engineOptions_->stepper.sensorsUpdatePeriod < EPS)
            {
                // Roll back to forces and accelerations computed at previous iteration
                contactForcesPrevIt->swap((*robotIt)->contactForces_);
                fPrevIt->swap((*robotIt)->pinocchioData_.f);
                aPrevIt->swap((*robotIt)->pinocchioData_.a);

                // Update sensors based on previous accelerations and forces
                (*robotIt)->computeSensorMeasurements(t, *qIt, *vIt, aPrev, uMotorPrev, fextPrev);

                // Restore current forces and accelerations
                contactForcesPrevIt->swap((*robotIt)->contactForces_);
                fPrevIt->swap((*robotIt)->pinocchioData_.f);
                aPrevIt->swap((*robotIt)->pinocchioData_.a);
            }

            /* Update the controller command if necessary (only for infinite update frequency).
               Make sure that the sensor state has been updated beforehand. */
            if (engineOptions_->stepper.controllerUpdatePeriod < EPS)
            {
                computeCommand(*robotIt, t, *qIt, *vIt, command);
            }

            /* Compute the actual motor effort.
               Note that it is impossible to have access to the current accelerations. */
            (*robotIt)->computeMotorEfforts(t, *qIt, *vIt, aPrev, command);
            const auto & uMotorAndJoint = (*robotIt)->getMotorEfforts();
            std::tie(uMotor, uTransmission) = uMotorAndJoint;

            /* Compute the user-defined internal dynamics.
               Make sure that the sensor state has been updated beforehand since the user-defined
               internal dynamics may rely on it. */
            uCustom.setZero();
            (*robotIt)->getController()->internalDynamics(t, *qIt, *vIt, uCustom);

            // Compute the total effort vector
            u = uInternal + uCustom;
            for (const auto & motor : (*robotIt)->getMotors())
            {
                const std::size_t motorIndex = motor->getIndex();
                const pinocchio::JointIndex jointIndex = motor->getJointIndex();
                const Eigen::Index motorVelocityIndex =
                    (*robotIt)->pinocchioModel_.joints[jointIndex].idx_v();
                u[motorVelocityIndex] += uTransmission[motorIndex];
            }

            // Compute the dynamics
            *aIt =
                computeAcceleration(*robotIt, *robotDataIt, *qIt, *vIt, u, fext, isStateUpToDate);
        }
    }

    const Eigen::VectorXd & Engine::computeAcceleration(std::shared_ptr<Robot> & robot,
                                                        RobotData & robotData,
                                                        const Eigen::VectorXd & q,
                                                        const Eigen::VectorXd & v,
                                                        const Eigen::VectorXd & u,
                                                        ForceVector & fext,
                                                        bool isStateUpToDate,
                                                        bool ignoreBounds)
    {
        const pinocchio::Model & model = robot->pinocchioModel_;
        pinocchio::Data & data = robot->pinocchioData_;

        if (robot->hasConstraints())
        {
            if (!isStateUpToDate)
            {
                // Compute kinematic constraints. It will take care of updating the joint Jacobian.
                robot->computeConstraints(q, v);

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
                            model, data, i, pinocchio::LOCAL, robotData.jointJacobians[i]);
                    }
                    data.u.noalias() +=
                        robotData.jointJacobians[i].transpose() * fext[i].toVector();
                }
            }

            // Call forward dynamics
            bool isSuccess = robotData.constraintSolver->SolveBoxedForwardDynamics(
                engineOptions_->constraints.regularization, isStateUpToDate, ignoreBounds);

            /* Monitor number of successive constraint solving failure. Exception raising is
               delegated to the 'step' method since this method is supposed to always succeed. */
            if (isSuccess)
            {
                robotData.successiveSolveFailed = 0U;
            }
            else
            {
                if (engineOptions_->stepper.verbose)
                {
                    JIMINY_WARNING("Constraint solver failure.");
                }
                ++robotData.successiveSolveFailed;
            }

            // Restore contact frame forces and bounds internal efforts
            const ConstraintTree & constraints = robot->getConstraints();
            constraints.foreach(
                ConstraintRegistryType::BOUNDS_JOINTS,
                [&u = robotData.state.u,
                 &uInternal = robotData.state.uInternal,
                 &joints = const_cast<pinocchio::Model::JointModelVector &>(model.joints)](
                    const std::shared_ptr<AbstractConstraintBase> & constraint,
                    ConstraintRegistryType /* type */)
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

            auto constraintIt = constraints.contactFrames.begin();
            auto forceIt = robot->contactForces_.begin();
            for (; constraintIt != constraints.contactFrames.end(); ++constraintIt, ++forceIt)
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

            constraints.foreach(
                ConstraintRegistryType::COLLISION_BODIES,
                [&fext, &model, &data](const std::shared_ptr<AbstractConstraintBase> & constraint,
                                       ConstraintRegistryType /* type */)
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

    std::shared_ptr<const LogData> Engine::getLog()
    {
        // Update internal log data buffer if uninitialized
        if (!logData_)
        {
            logData_ = std::make_shared<LogData>(telemetryRecorder_->getLog());
        }

        // Return shared pointer to internal log data buffer
        return std::const_pointer_cast<const LogData>(logData_);
    }

    static LogData readLogHdf5(const std::string & filename)
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
            JIMINY_THROW(std::runtime_error,
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

    LogData Engine::readLog(const std::string & filename, const std::string & format)
    {
        if (format == "binary")
        {
            return TelemetryRecorder::readLog(filename);
        }
        if (format == "hdf5")
        {
            return readLogHdf5(filename);
        }
        JIMINY_THROW(std::invalid_argument,
                     "Format '",
                     format,
                     "' not recognized. It must be either 'binary' or 'hdf5'.");
    }

    static void writeLogHdf5(const std::string & filename,
                             const std::shared_ptr<const LogData> & logData)
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
            JIMINY_THROW(std::runtime_error,
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

    void Engine::writeLog(const std::string & filename, const std::string & format)
    {
        // Make sure there is something to write
        if (!isTelemetryConfigured_)
        {
            JIMINY_THROW(
                bad_control_flow,
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
            JIMINY_THROW(std::invalid_argument,
                         "Format '",
                         format,
                         "' not recognized. It must be either 'binary' or 'hdf5'.");
        }
    }
}
