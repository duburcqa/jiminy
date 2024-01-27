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

    hresult_t EngineMultiRobot::addSystem(const std::string & systemName,
                                          std::shared_ptr<Robot> robot,
                                          std::shared_ptr<AbstractController> controller,
                                          CallbackFunctor callbackFct)
    {
        // Make sure that no simulation is running
        if (isSimulationRunning_)
        {
            PRINT_ERROR("A simulation is already running. Stop it before adding a new system.");
            return hresult_t::ERROR_GENERIC;
        }

        if (!robot)
        {
            PRINT_ERROR("Robot unspecified.");
            return hresult_t::ERROR_INIT_FAILED;
        }

        if (!robot->getIsInitialized())
        {
            PRINT_ERROR("Robot not initialized.");
            return hresult_t::ERROR_INIT_FAILED;
        }

        if (!controller)
        {
            PRINT_ERROR("Controller unspecified.");
            return hresult_t::ERROR_INIT_FAILED;
        }

        if (!controller->getIsInitialized())
        {
            PRINT_ERROR("Controller not initialized.");
            return hresult_t::ERROR_INIT_FAILED;
        }

        auto robot_controller = controller->robot_.lock();
        if (!robot_controller)
        {
            PRINT_ERROR("Controller's robot expired or unset.");
            return hresult_t::ERROR_INIT_FAILED;
        }

        if (robot != robot_controller)
        {
            PRINT_ERROR("Controller not initialized for specified robot.");
            return hresult_t::ERROR_INIT_FAILED;
        }

        // TODO: Check that the callback function is working as expected
        // FIXME: remove constructor call to `std::string` when moving to C++20
        systems_.emplace_back(
            systemHolder_t{systemName, robot, controller, std::move(callbackFct)});
        systemsDataHolder_.resize(systems_.size());

        return hresult_t::SUCCESS;
    }

    hresult_t EngineMultiRobot::addSystem(
        const std::string & systemName, std::shared_ptr<Robot> robot, CallbackFunctor callbackFct)
    {
        // Make sure an actual robot is specified
        if (!robot)
        {
            PRINT_ERROR("Robot unspecified.");
            return hresult_t::ERROR_INIT_FAILED;
        }

        // Make sure the robot is properly initialized
        if (!robot->getIsInitialized())
        {
            PRINT_ERROR("Robot not initialized.");
            return hresult_t::ERROR_INIT_FAILED;
        }

        // When using several robots the robots names are specified
        // as a circumfix in the log, for consistency they must all
        // have a name
        if (systems_.size() && systemName == "")
        {
            PRINT_ERROR("All systems but the first one must have a name.");
            return hresult_t::ERROR_GENERIC;
        }

        // Check if a system with the same name already exists
        auto systemIt = std::find_if(systems_.begin(),
                                     systems_.end(),
                                     [&systemName](const auto & sys)
                                     { return (sys.name == systemName); });
        if (systemIt != systems_.end())
        {
            PRINT_ERROR("A system with this name has already been added to the engine.");
            return hresult_t::ERROR_BAD_INPUT;
        }

        // Make sure none of the existing system is referring to the same robot
        systemIt = std::find_if(systems_.begin(),
                                systems_.end(),
                                [&robot](const auto & sys) { return (sys.robot == robot); });
        if (systemIt != systems_.end())
        {
            PRINT_ERROR("The system '", systemIt->name, "' is already referring to this robot.");
            return hresult_t::ERROR_BAD_INPUT;
        }

        // Create and initialize a controller doing nothing
        auto noopFunctor = +[](double /* t */,
                               const Eigen::VectorXd & /* q */,
                               const Eigen::VectorXd & /* v */,
                               const SensorsDataMap & /* sensorsData */,
                               Eigen::VectorXd & /* out */) {
        };
        auto controller =
            std::make_shared<ControllerFunctor<decltype(noopFunctor), decltype(noopFunctor)>>(
                noopFunctor, noopFunctor);
        controller->initialize(robot);

        return addSystem(systemName, robot, controller, std::move(callbackFct));
    }

    hresult_t EngineMultiRobot::removeSystem(const std::string & systemName)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        // Make sure that no simulation is running
        if (isSimulationRunning_)
        {
            PRINT_ERROR("A simulation is already running. Stop it before removing a system.");
            returnCode = hresult_t::ERROR_GENERIC;
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            /* Remove every coupling forces involving the system.
               Note that it is already checking that the system exists. */
            returnCode = removeForcesCoupling(systemName);
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            // Get the system index
            std::ptrdiff_t systemIdx{};
            getSystemIdx(systemName, systemIdx);  // Cannot fail at this point

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

            // Remove the system from the list
            systems_.erase(systems_.begin() + systemIdx);
            systemsDataHolder_.erase(systemsDataHolder_.begin() + systemIdx);
        }

        return returnCode;
    }

    hresult_t EngineMultiRobot::setController(const std::string & systemName,
                                              std::shared_ptr<AbstractController> controller)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        // Make sure that no simulation is running
        if (isSimulationRunning_)
        {
            PRINT_ERROR("A simulation is already running. Stop it before setting a new controller "
                        "for a system.");
            returnCode = hresult_t::ERROR_GENERIC;
        }

        // Make sure that the controller is initialized
        if (returnCode == hresult_t::SUCCESS)
        {
            if (!controller->getIsInitialized())
            {
                PRINT_ERROR("Controller not initialized.");
                returnCode = hresult_t::ERROR_INIT_FAILED;
            }
        }

        auto robot_controller = controller->robot_.lock();
        if (returnCode == hresult_t::SUCCESS)
        {
            if (!robot_controller)
            {
                PRINT_ERROR("Controller's robot expired or unset.");
                returnCode = hresult_t::ERROR_INIT_FAILED;
            }
        }

        // Make sure that the system for which to set the controller exists
        systemHolder_t * system;
        if (returnCode == hresult_t::SUCCESS)
        {
            returnCode = getSystem(systemName, system);
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            if (system->robot != robot_controller)
            {
                PRINT_ERROR(
                    "Controller not initialized for robot associated with specified system.");
                returnCode = hresult_t::ERROR_INIT_FAILED;
            }
        }

        // Set the controller
        if (returnCode == hresult_t::SUCCESS)
        {
            system->controller = controller;
        }

        return returnCode;
    }

    hresult_t EngineMultiRobot::registerForceCoupling(const std::string & systemName1,
                                                      const std::string & systemName2,
                                                      const std::string & frameName1,
                                                      const std::string & frameName2,
                                                      ForceCouplingFunctor forceFct)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        // Make sure that no simulation is running
        if (isSimulationRunning_)
        {
            PRINT_ERROR("A simulation is already running. Stop it before adding coupling forces.");
            returnCode = hresult_t::ERROR_GENERIC;
        }

        // Get system and frame indices
        std::ptrdiff_t systemIdx1;
        if (returnCode == hresult_t::SUCCESS)
        {
            returnCode = getSystemIdx(systemName1, systemIdx1);
        }

        std::ptrdiff_t systemIdx2;
        if (returnCode == hresult_t::SUCCESS)
        {
            returnCode = getSystemIdx(systemName2, systemIdx2);
        }

        pinocchio::FrameIndex frameIdx1;
        if (returnCode == hresult_t::SUCCESS)
        {
            returnCode = getFrameIdx(systems_[systemIdx1].robot->pncModel_, frameName1, frameIdx1);
        }

        pinocchio::FrameIndex frameIdx2;
        if (returnCode == hresult_t::SUCCESS)
        {
            returnCode = getFrameIdx(systems_[systemIdx2].robot->pncModel_, frameName2, frameIdx2);
        }

        // Make sure it is not coupling the exact same frame
        if (returnCode == hresult_t::SUCCESS)
        {
            if (systemIdx1 == systemIdx2 && frameIdx1 == frameIdx2)
            {
                PRINT_ERROR("A coupling force requires different frames.");
                returnCode = hresult_t::ERROR_GENERIC;
            }
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            forcesCoupling_.emplace_back(systemName1,
                                         systemIdx1,
                                         systemName2,
                                         systemIdx2,
                                         frameName1,
                                         frameIdx1,
                                         frameName2,
                                         frameIdx2,
                                         std::move(forceFct));
        }

        return returnCode;
    }

    hresult_t EngineMultiRobot::registerViscoelasticForceCoupling(const std::string & systemName1,
                                                                  const std::string & systemName2,
                                                                  const std::string & frameName1,
                                                                  const std::string & frameName2,
                                                                  const Vector6d & stiffness,
                                                                  const Vector6d & damping,
                                                                  double alpha)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if ((stiffness.array() < 0.0).any() || (damping.array() < 0.0).any())
        {
            PRINT_ERROR("The stiffness and damping parameters must be positive.");
            returnCode = hresult_t::ERROR_GENERIC;
        }

        systemHolder_t * system1;
        if (returnCode == hresult_t::SUCCESS)
        {
            returnCode = getSystem(systemName1, system1);
        }

        systemHolder_t * system2;
        if (returnCode == hresult_t::SUCCESS)
        {
            returnCode = getSystem(systemName2, system2);
        }

        pinocchio::FrameIndex frameIdx1, frameIdx2;
        if (returnCode == hresult_t::SUCCESS)
        {
            getFrameIdx(system1->robot->pncModel_, frameName1, frameIdx1);
            getFrameIdx(system2->robot->pncModel_, frameName2, frameIdx2);
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            // Allocate memory
            double angle{0.0};
            Eigen::Matrix3d rot12{}, rotJLog12{}, rotJExp12{}, rotRef12{}, omega{};
            Eigen::Vector3d rotLog12{}, pos12{}, posLocal12{}, fLin{}, fAng{};

            auto forceFct = [=](double /*t*/,
                                const Eigen::VectorXd & /*q_1*/,
                                const Eigen::VectorXd & /*v_1*/,
                                const Eigen::VectorXd & /*q_2*/,
                                const Eigen::VectorXd & /*v_2*/) mutable -> pinocchio::Force
            {
                /* One must keep track of system pointers and frames indices internally and update
                   them at reset since the systems may have changed between simulations. Note that
                   `isSimulationRunning_` is false when called for the first time in `start` method
                   before locking the robot, so it is the right time to update those proxies. */
                if (!isSimulationRunning_)
                {
                    getSystem(systemName1, system1);
                    getFrameIdx(system1->robot->pncModel_, frameName1, frameIdx1);
                    getSystem(systemName2, system2);
                    getFrameIdx(system2->robot->pncModel_, frameName2, frameIdx2);
                }

                // Get the frames positions and velocities in world
                const pinocchio::SE3 & oMf1{system1->robot->pncData_.oMf[frameIdx1]};
                const pinocchio::SE3 & oMf2{system2->robot->pncData_.oMf[frameIdx2]};
                const pinocchio::Motion oVf1{getFrameVelocity(system1->robot->pncModel_,
                                                              system1->robot->pncData_,
                                                              frameIdx1,
                                                              pinocchio::LOCAL_WORLD_ALIGNED)};
                const pinocchio::Motion oVf2{getFrameVelocity(system2->robot->pncModel_,
                                                              system2->robot->pncData_,
                                                              frameIdx2,
                                                              pinocchio::LOCAL_WORLD_ALIGNED)};

                // Compute intermediary quantities
                rot12.noalias() = oMf1.rotation().transpose() * oMf2.rotation();
                rotLog12 = pinocchio::log3(rot12, angle);
                assert((angle < 0.95 * M_PI) &&
                       "Relative angle between reference frames of viscoelastic coupling must be "
                       "smaller than 0.95 * pi.");
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
                f.angular() -=
                    oMf2.rotation() * omega.colwise().cross(posLocal12).transpose() * fLin;
                f.angular() += oMf1.rotation() * rotJLog12 * fAng;

                // Deduce the force acting on frame 1 from action-reaction law
                f.angular() += pos12.cross(f.linear());

                return f;
            };

            returnCode =
                registerForceCoupling(systemName1, systemName2, frameName1, frameName2, forceFct);
        }

        return returnCode;
    }

    hresult_t EngineMultiRobot::registerViscoelasticForceCoupling(const std::string & systemName,
                                                                  const std::string & frameName1,
                                                                  const std::string & frameName2,
                                                                  const Vector6d & stiffness,
                                                                  const Vector6d & damping,
                                                                  double alpha)
    {
        return registerViscoelasticForceCoupling(
            systemName, systemName, frameName1, frameName2, stiffness, damping, alpha);
    }

    hresult_t EngineMultiRobot::registerViscoelasticDirectionalForceCoupling(
        const std::string & systemName1,
        const std::string & systemName2,
        const std::string & frameName1,
        const std::string & frameName2,
        double stiffness,
        double damping,
        double restLength)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (stiffness < 0.0 || damping < 0.0)
        {
            PRINT_ERROR("The stiffness and damping parameters must be positive.");
            returnCode = hresult_t::ERROR_GENERIC;
        }

        systemHolder_t * system1;
        if (returnCode == hresult_t::SUCCESS)
        {
            returnCode = getSystem(systemName1, system1);
        }

        pinocchio::FrameIndex frameIdx1;
        if (returnCode == hresult_t::SUCCESS)
        {
            returnCode = getFrameIdx(system1->robot->pncModel_, frameName1, frameIdx1);
        }

        systemHolder_t * system2;
        if (returnCode == hresult_t::SUCCESS)
        {
            returnCode = getSystem(systemName2, system2);
        }

        pinocchio::FrameIndex frameIdx2;
        if (returnCode == hresult_t::SUCCESS)
        {
            returnCode = getFrameIdx(system2->robot->pncModel_, frameName2, frameIdx2);
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            auto forceFct = [=](double /*t*/,
                                const Eigen::VectorXd & /*q_1*/,
                                const Eigen::VectorXd & /*v_1*/,
                                const Eigen::VectorXd & /*q_2*/,
                                const Eigen::VectorXd & /*v_2*/) mutable -> pinocchio::Force
            {
                /* One must keep track of system pointers and frames indices internally and update
                   them at reset since the systems may have changed between simulations. Note that
                   `isSimulationRunning_` is false when called for the first time in `start` method
                   before locking the robot, so it is the right time to update those proxies. */
                if (!isSimulationRunning_)
                {
                    getSystem(systemName1, system1);
                    getFrameIdx(system1->robot->pncModel_, frameName1, frameIdx1);
                    getSystem(systemName2, system2);
                    getFrameIdx(system2->robot->pncModel_, frameName2, frameIdx2);
                }

                // Get the frames positions and velocities in world
                const pinocchio::SE3 & oMf1{system1->robot->pncData_.oMf[frameIdx1]};
                const pinocchio::SE3 & oMf2{system2->robot->pncData_.oMf[frameIdx2]};
                const pinocchio::Motion oVf1{getFrameVelocity(system1->robot->pncModel_,
                                                              system1->robot->pncData_,
                                                              frameIdx1,
                                                              pinocchio::LOCAL_WORLD_ALIGNED)};
                const pinocchio::Motion oVf2{getFrameVelocity(system2->robot->pncModel_,
                                                              system2->robot->pncData_,
                                                              frameIdx2,
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

            returnCode =
                registerForceCoupling(systemName1, systemName2, frameName1, frameName2, forceFct);
        }

        return returnCode;
    }

    hresult_t EngineMultiRobot::registerViscoelasticDirectionalForceCoupling(
        const std::string & systemName,
        const std::string & frameName1,
        const std::string & frameName2,
        double stiffness,
        double damping,
        double restLength)
    {
        return registerViscoelasticDirectionalForceCoupling(
            systemName, systemName, frameName1, frameName2, stiffness, damping, restLength);
    }

    hresult_t EngineMultiRobot::removeForcesCoupling(const std::string & systemName1,
                                                     const std::string & systemName2)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        // Make sure that no simulation is running
        if (isSimulationRunning_)
        {
            PRINT_ERROR(
                "A simulation is already running. Stop it before removing coupling forces.");
            returnCode = hresult_t::ERROR_GENERIC;
        }

        systemHolder_t * system1;
        if (returnCode == hresult_t::SUCCESS)
        {
            returnCode = getSystem(systemName1, system1);
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            systemHolder_t * system2;
            returnCode = getSystem(systemName2, system2);
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            forcesCoupling_.erase(std::remove_if(forcesCoupling_.begin(),
                                                 forcesCoupling_.end(),
                                                 [&systemName1, &systemName2](const auto & force) {
                                                     return (force.systemName1 == systemName1 &&
                                                             force.systemName2 == systemName2);
                                                 }),
                                  forcesCoupling_.end());
        }

        return returnCode;
    }

    hresult_t EngineMultiRobot::removeForcesCoupling(const std::string & systemName)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        // Make sure that no simulation is running
        if (isSimulationRunning_)
        {
            PRINT_ERROR(
                "A simulation is already running. Stop it before removing coupling forces.");
            returnCode = hresult_t::ERROR_GENERIC;
        }

        systemHolder_t * system;
        if (returnCode == hresult_t::SUCCESS)
        {
            returnCode = getSystem(systemName, system);
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            forcesCoupling_.erase(std::remove_if(forcesCoupling_.begin(),
                                                 forcesCoupling_.end(),
                                                 [&systemName](const auto & force) {
                                                     return (force.systemName1 == systemName ||
                                                             force.systemName2 == systemName);
                                                 }),
                                  forcesCoupling_.end());
        }

        return returnCode;
    }

    hresult_t EngineMultiRobot::removeForcesCoupling()
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        // Make sure that no simulation is running
        if (isSimulationRunning_)
        {
            PRINT_ERROR(
                "A simulation is already running. Stop it before removing coupling forces.");
            returnCode = hresult_t::ERROR_GENERIC;
        }

        forcesCoupling_.clear();

        return returnCode;
    }

    const ForceCouplingRegister & EngineMultiRobot::getForcesCoupling() const
    {
        return forcesCoupling_;
    }

    hresult_t EngineMultiRobot::removeAllForces()
    {
        hresult_t returnCode = hresult_t::SUCCESS;
        returnCode = removeForcesCoupling();
        if (returnCode == hresult_t::SUCCESS)
        {
            returnCode = removeForcesImpulse();
        }
        if (returnCode == hresult_t::SUCCESS)
        {
            returnCode = removeForcesProfile();
        }
        return returnCode;
    }

    hresult_t EngineMultiRobot::configureTelemetry()
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (systems_.empty())
        {
            PRINT_ERROR("No system added to the engine.");
            returnCode = hresult_t::ERROR_INIT_FAILED;
        }

        if (!isTelemetryConfigured_)
        {
            // Initialize the engine-specific telemetry sender
            telemetrySender_->configureObject(telemetryData_, ENGINE_TELEMETRY_NAMESPACE);

            auto systemIt = systems_.begin();
            auto systemDataIt = systemsDataHolder_.begin();
            auto energyIt = energy_.begin();
            for (; systemIt != systems_.end(); ++systemIt, ++systemDataIt, ++energyIt)
            {
                // Generate the log fieldnames
                systemDataIt->logFieldnamesPosition =
                    addCircumfix(systemIt->robot->getLogFieldnamesPosition(),
                                 systemIt->name,
                                 {},
                                 TELEMETRY_FIELDNAME_DELIMITER);
                systemDataIt->logFieldnamesVelocity =
                    addCircumfix(systemIt->robot->getLogFieldnamesVelocity(),
                                 systemIt->name,
                                 {},
                                 TELEMETRY_FIELDNAME_DELIMITER);
                systemDataIt->logFieldnamesAcceleration =
                    addCircumfix(systemIt->robot->getLogFieldnamesAcceleration(),
                                 systemIt->name,
                                 {},
                                 TELEMETRY_FIELDNAME_DELIMITER);
                systemDataIt->logFieldnamesForceExternal =
                    addCircumfix(systemIt->robot->getLogFieldnamesForceExternal(),
                                 systemIt->name,
                                 {},
                                 TELEMETRY_FIELDNAME_DELIMITER);
                systemDataIt->logFieldnamesCommand =
                    addCircumfix(systemIt->robot->getCommandFieldnames(),
                                 systemIt->name,
                                 {},
                                 TELEMETRY_FIELDNAME_DELIMITER);
                systemDataIt->logFieldnamesMotorEffort =
                    addCircumfix(systemIt->robot->getMotorEffortFieldnames(),
                                 systemIt->name,
                                 {},
                                 TELEMETRY_FIELDNAME_DELIMITER);
                systemDataIt->logFieldnameEnergy =
                    addCircumfix("energy", systemIt->name, {}, TELEMETRY_FIELDNAME_DELIMITER);

                // Register variables to the telemetry senders
                if (returnCode == hresult_t::SUCCESS)
                {
                    if (engineOptions_->telemetry.enableConfiguration)
                    {
                        returnCode = telemetrySender_->registerVariable(
                            systemDataIt->logFieldnamesPosition, systemDataIt->state.q);
                    }
                }
                if (returnCode == hresult_t::SUCCESS)
                {
                    if (engineOptions_->telemetry.enableVelocity)
                    {
                        returnCode = telemetrySender_->registerVariable(
                            systemDataIt->logFieldnamesVelocity, systemDataIt->state.v);
                    }
                }
                if (returnCode == hresult_t::SUCCESS)
                {
                    if (engineOptions_->telemetry.enableAcceleration)
                    {
                        returnCode = telemetrySender_->registerVariable(
                            systemDataIt->logFieldnamesAcceleration, systemDataIt->state.a);
                    }
                }
                if (engineOptions_->telemetry.enableForceExternal)
                {
                    for (std::size_t i = 1; i < systemDataIt->state.fExternal.size(); ++i)
                    {
                        const auto & fext = systemDataIt->state.fExternal[i].toVector();
                        for (uint8_t j = 0; j < 6U; ++j)
                        {
                            returnCode = telemetrySender_->registerVariable(
                                systemDataIt->logFieldnamesForceExternal[(i - 1) * 6U + j],
                                &fext[j]);
                        }
                    }
                }
                if (returnCode == hresult_t::SUCCESS)
                {
                    if (engineOptions_->telemetry.enableCommand)
                    {
                        returnCode = telemetrySender_->registerVariable(
                            systemDataIt->logFieldnamesCommand, systemDataIt->state.command);
                    }
                }
                if (returnCode == hresult_t::SUCCESS)
                {
                    if (engineOptions_->telemetry.enableMotorEffort)
                    {
                        returnCode = telemetrySender_->registerVariable(
                            systemDataIt->logFieldnamesMotorEffort, systemDataIt->state.uMotor);
                    }
                }
                if (returnCode == hresult_t::SUCCESS)
                {
                    if (engineOptions_->telemetry.enableEnergy)
                    {
                        returnCode = telemetrySender_->registerVariable(
                            systemDataIt->logFieldnameEnergy, &(*energyIt));
                    }
                }

                if (returnCode == hresult_t::SUCCESS)
                {
                    returnCode =
                        systemIt->controller->configureTelemetry(telemetryData_, systemIt->name);
                }
                if (returnCode == hresult_t::SUCCESS)
                {
                    returnCode =
                        systemIt->robot->configureTelemetry(telemetryData_, systemIt->name);
                }
            }
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            isTelemetryConfigured_ = true;
        }

        return returnCode;
    }

    void EngineMultiRobot::updateTelemetry()
    {
        // Compute the total energy of the system
        auto systemIt = systems_.begin();
        auto energyIt = energy_.begin();
        for (; systemIt != systems_.end(); ++systemIt, ++energyIt)
        {
            *energyIt = systemIt->robot->pncData_.kinetic_energy +
                        systemIt->robot->pncData_.potential_energy;
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
        telemetryRecorder_->flushDataSnapshot(stepperState_.t);
    }

    void EngineMultiRobot::reset(bool resetRandomNumbers, bool removeAllForce)
    {
        // Make sure the simulation is properly stopped
        if (isSimulationRunning_)
        {
            stop();
        }

        // Clear log data buffer
        logData_ = nullptr;

        // Reset the dynamic force register if requested
        if (removeAllForce)
        {
            for (auto & systemData : systemsDataHolder_)
            {
                systemData.forcesImpulse.clear();
                systemData.forcesImpulseBreaks.clear();
                systemData.forcesImpulseActive.clear();
                systemData.forcesProfile.clear();
            }
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
        for (auto & systemData : systemsDataHolder_)
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
            pinocchio::JointIndex i = jmodel.id();
            data.a[i] = jdata.c() + data.v[i].cross(jdata.v());
            data.a[i] += jdata.S() * jmodel.jointVelocitySelector(a);
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
        systemHolder_t & system, const systemDataHolder_t & systemData, ForceVector & fExt)
    {
        const pinocchio::Model & model = system.robot->pncModel_;
        pinocchio::Data & data = system.robot->pncData_;

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
            for (int i = model.njoints - 1; i > 0; --i)
            {
                const pinocchio::JointIndex jointModelIdx = model.joints[i].id();
                const pinocchio::JointIndex parentJointModelIdx = model.parents[jointModelIdx];
                if (parentJointModelIdx > 0)
                {
                    data.Ycrb[parentJointModelIdx] +=
                        data.liMi[jointModelIdx].act(data.Ycrb[jointModelIdx]);
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
        for (int i = 1; i < model.njoints; ++i)
        {
            const auto & jmodel = model.joints[i];
            const pinocchio::JointIndex jointModelIdx = jmodel.id();
            const pinocchio::JointIndex parentJointModelIdx = model.parents[jointModelIdx];

            ForwardKinematicsAccelerationStep::run(
                jmodel,
                data.joints[i],
                typename ForwardKinematicsAccelerationStep::ArgsType(data, systemData.state.a));
            data.a_gf[jointModelIdx] = data.a[jointModelIdx];
            data.a[jointModelIdx] += data.liMi[jointModelIdx].actInv(data.a[parentJointModelIdx]);
            data.a_gf[jointModelIdx] +=
                data.liMi[jointModelIdx].actInv(data.a_gf[parentJointModelIdx]);

            model.inertias[jointModelIdx].__mult__(data.v[jointModelIdx], data.h[jointModelIdx]);

            model.inertias[jointModelIdx].__mult__(data.a[jointModelIdx], fExt[jointModelIdx]);
            data.f[jointModelIdx] = data.v[jointModelIdx].cross(data.h[jointModelIdx]);
            fExt[jointModelIdx] += data.f[jointModelIdx];
            data.f[jointModelIdx] += model.inertias[jointModelIdx] * data.a_gf[jointModelIdx];
            data.f[jointModelIdx] -= systemData.state.fExternal[jointModelIdx];
        }
        for (int i = model.njoints - 1; i > 0; --i)
        {
            const auto & jmodel = model.joints[i];
            const pinocchio::JointIndex jointModelIdx = jmodel.id();
            const pinocchio::JointIndex parentJointModelIdx = model.parents[jointModelIdx];

            fExt[parentJointModelIdx] += data.liMi[jointModelIdx].act(fExt[jointModelIdx]);
            data.h[parentJointModelIdx] += data.liMi[jointModelIdx].act(data.h[jointModelIdx]);
            if (parentJointModelIdx > 0)
            {
                data.f[parentJointModelIdx] += data.liMi[jointModelIdx].act(data.f[jointModelIdx]);
            }
        }

        // Compute the position and velocity of the center of mass of each subtree
        for (int i = 0; i < model.njoints; ++i)
        {
            data.com[i] = data.Ycrb[i].lever();
            data.vcom[i].noalias() = data.h[i].linear() / data.mass[i];
        }
        data.com[0] = data.liMi[1].act(data.com[1]);
        data.vcom[0].noalias() = data.h[0].linear() / data.mass[0];

        // Compute centroidal dynamics and its derivative
        data.hg = data.h[0];
        data.hg.angular() += data.hg.linear().cross(data.com[0]);
        data.dhg = fExt[0];
        data.dhg.angular() += data.dhg.linear().cross(data.com[0]);
    }

    void computeAllExtraTerms(std::vector<systemHolder_t> & systems,
                              const vector_aligned_t<systemDataHolder_t> & systemsDataHolder,
                              vector_aligned_t<ForceVector> & f)
    {
        auto systemIt = systems.begin();
        auto systemDataIt = systemsDataHolder.begin();
        auto fIt = f.begin();
        for (; systemIt != systems.end(); ++systemIt, ++systemDataIt, ++fIt)
        {
            computeExtraTerms(*systemIt, *systemDataIt, *fIt);
        }
    }

    void syncAccelerationsAndForces(const systemHolder_t & system,
                                    ForceVector & contactForces,
                                    ForceVector & f,
                                    MotionVector & a)
    {
        for (std::size_t i = 0; i < system.robot->getContactFramesNames().size(); ++i)
        {
            contactForces[i] = system.robot->contactForces_[i];
        }

        for (int i = 0; i < system.robot->pncModel_.njoints; ++i)
        {
            f[i] = system.robot->pncData_.f[i];
            a[i] = system.robot->pncData_.a[i];
        }
    }

    void syncAllAccelerationsAndForces(const std::vector<systemHolder_t> & systems,
                                       vector_aligned_t<ForceVector> & contactForces,
                                       vector_aligned_t<ForceVector> & f,
                                       vector_aligned_t<MotionVector> & a)
    {
        std::vector<systemHolder_t>::const_iterator systemIt = systems.begin();
        auto contactForcesIt = contactForces.begin();
        auto fPrevIt = f.begin();
        auto aPrevIt = a.begin();
        for (; systemIt != systems.end(); ++systemIt, ++contactForcesIt, ++fPrevIt, ++aPrevIt)
        {
            syncAccelerationsAndForces(*systemIt, *contactForcesIt, *fPrevIt, *aPrevIt);
        }
    }

    hresult_t EngineMultiRobot::start(
        const std::map<std::string, Eigen::VectorXd> & qInit,
        const std::map<std::string, Eigen::VectorXd> & vInit,
        const std::optional<std::map<std::string, Eigen::VectorXd>> & aInit)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        // Make sure that no simulation is running
        if (isSimulationRunning_)
        {
            PRINT_ERROR("A simulation is already running. Stop it before starting again.");
            return hresult_t::ERROR_GENERIC;
        }

        if (systems_.empty())
        {
            PRINT_ERROR("No system to simulate. Please add one before starting a simulation.");
            return hresult_t::ERROR_INIT_FAILED;
        }

        const std::size_t nSystems = systems_.size();
        if (qInit.size() != nSystems || vInit.size() != nSystems)
        {
            PRINT_ERROR("The number of initial configurations and velocities must match the "
                        "number of systems.");
            return hresult_t::ERROR_BAD_INPUT;
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
                PRINT_ERROR("System '",
                            system.name,
                            "'does not have an initial configuration or velocity.");
                return hresult_t::ERROR_BAD_INPUT;
            }

            const Eigen::VectorXd & q = qInitIt->second;
            const Eigen::VectorXd & v = vInitIt->second;
            if (q.rows() != system.robot->nq() || v.rows() != system.robot->nv())
            {
                PRINT_ERROR("The dimension of the initial configuration or velocity is "
                            "inconsistent with model size for system '",
                            system.name,
                            "'.");
                return hresult_t::ERROR_BAD_INPUT;
            }

            // Make sure the initial state is normalized
            bool isValid;
            isPositionValid(system.robot->pncModel_,
                            q,
                            isValid,
                            std::numeric_limits<float>::epsilon());  // Cannot throw exception
            if (!isValid)
            {
                PRINT_ERROR("The initial configuration is not consistent with the types of "
                            "joints of the model for system '",
                            system.name,
                            "'.");
                return hresult_t::ERROR_BAD_INPUT;
            }

            /* Check that the initial configuration is not out-of-bounds if appropriate.
               Note that EPS allows to be very slightly out-of-bounds, which may occurs because of
               rounding errors. */
            if ((system.robot->mdlOptions_->joints.enablePositionLimit &&
                 (contactModel_ == contactModel_t::CONSTRAINT) &&
                 ((EPS < q.array() - system.robot->getPositionLimitMax().array()).any() ||
                  (EPS < system.robot->getPositionLimitMin().array() - q.array()).any())))
            {
                PRINT_ERROR(
                    "The initial configuration is out-of-bounds for system '", system.name, "'.");
                return hresult_t::ERROR_BAD_INPUT;
            }

            // Check that the initial velocity is not out-of-bounds
            if ((system.robot->mdlOptions_->joints.enableVelocityLimit &&
                 (system.robot->getVelocityLimit().array() < v.array().abs()).any()))
            {
                PRINT_ERROR(
                    "The initial velocity is out-of-bounds for system '", system.name, "'.");
                return hresult_t::ERROR_BAD_INPUT;
            }

            /* Make sure the configuration is normalized (as double), since normalization is
               checked using float accuracy rather than double to circumvent float/double casting
               than may occurs because of Python bindings. */
            Eigen::VectorXd qNormalized = q;
            pinocchio::normalize(system.robot->pncModel_, qNormalized);

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
                PRINT_ERROR("If specified, the number of initial accelerations must match the "
                            "number of systems.");
                return hresult_t::ERROR_BAD_INPUT;
            }

            for (const auto & system : systems_)
            {
                auto aInitIt = aInit->find(system.name);
                if (aInitIt == aInit->end())
                {
                    PRINT_ERROR(
                        "System '", system.name, "'does not have an initial acceleration.");
                    return hresult_t::ERROR_BAD_INPUT;
                }

                const Eigen::VectorXd & a = aInitIt->second;
                if (a.rows() != system.robot->nv())
                {
                    PRINT_ERROR("The dimension of the initial acceleration is inconsistent "
                                "with model size for system '",
                                system.name,
                                "'.");
                    return hresult_t::ERROR_BAD_INPUT;
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
            for (const auto & sensorsGroupItem : system.robot->getSensors())
            {
                for (const auto & sensor : sensorsGroupItem.second)
                {
                    if (!sensor->getIsInitialized())
                    {
                        PRINT_ERROR("At least a sensor of a robot is not initialized.");
                        return hresult_t::ERROR_INIT_FAILED;
                    }
                }
            }

            for (const auto & motor : system.robot->getMotors())
            {
                if (!motor->getIsInitialized())
                {
                    PRINT_ERROR("At least a motor of a robot is not initialized.");
                    return hresult_t::ERROR_INIT_FAILED;
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
        auto systemDataIt = systemsDataHolder_.begin();
        for (; systemIt != systems_.end(); ++systemIt, ++systemDataIt)
        {
            // Propagate the user-defined gravity at robot level
            systemIt->robot->pncModelOrig_.gravity = engineOptions_->world.gravity;
            systemIt->robot->pncModel_.gravity = engineOptions_->world.gravity;

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
        fPrev_.clear();
        aPrev_.clear();
        contactForcesPrev_.reserve(nSystems);
        fPrev_.reserve(nSystems);
        aPrev_.reserve(nSystems);
        for (const auto & system : systems_)
        {
            contactForcesPrev_.push_back(system.robot->contactForces_);
            fPrev_.push_back(system.robot->pncData_.f);
            aPrev_.push_back(system.robot->pncData_.a);
        }
        energy_.resize(nSystems, 0.0);

        // Synchronize the individual system states with the global stepper state
        syncSystemsStateWithStepper();

        // Update the frame indices associated with the coupling forces
        for (auto & force : forcesCoupling_)
        {
            getFrameIdx(
                systems_[force.systemIdx1].robot->pncModel_, force.frameName1, force.frameIdx1);
            getFrameIdx(
                systems_[force.systemIdx2].robot->pncModel_, force.frameName2, force.frameIdx2);
        }

        systemIt = systems_.begin();
        systemDataIt = systemsDataHolder_.begin();
        for (; systemIt != systems_.end(); ++systemIt, ++systemDataIt)
        {
            // Update the frame indices associated with the impulse forces and force profiles
            for (auto & force : systemDataIt->forcesProfile)
            {
                getFrameIdx(systemIt->robot->pncModel_, force.frameName, force.frameIdx);
            }
            for (auto & force : systemDataIt->forcesImpulse)
            {
                getFrameIdx(systemIt->robot->pncModel_, force.frameName, force.frameIdx);
            }

            // Initialize the impulse force breakpoint point iterator
            systemDataIt->forcesImpulseBreakNextIt = systemDataIt->forcesImpulseBreaks.begin();

            // Reset the active set of impulse forces
            std::fill(systemDataIt->forcesImpulseActive.begin(),
                      systemDataIt->forcesImpulseActive.end(),
                      false);

            // Activate every force impulse starting at t=0
            auto forcesImpulseActiveIt = systemDataIt->forcesImpulseActive.begin();
            auto forcesImpulseIt = systemDataIt->forcesImpulse.begin();
            for (; forcesImpulseIt != systemDataIt->forcesImpulse.end();
                 ++forcesImpulseActiveIt, ++forcesImpulseIt)
            {
                if (forcesImpulseIt->t < STEPPER_MIN_TIMESTEP)
                {
                    *forcesImpulseActiveIt = true;
                }
            }

            // Compute the forward kinematics for each system
            const Eigen::VectorXd & q = systemDataIt->state.q;
            const Eigen::VectorXd & v = systemDataIt->state.v;
            const Eigen::VectorXd & a = systemDataIt->state.a;
            computeForwardKinematics(*systemIt, q, v, a);

            /* Backup constraint register for fast lookup.
               Internal constraints cannot be added/removed at this point. */
            systemDataIt->constraintsHolder = systemIt->robot->getConstraints();

            // Initialize contacts forces in local frame
            const std::vector<pinocchio::FrameIndex> & contactFramesIdx =
                systemIt->robot->getContactFramesIdx();
            systemDataIt->contactFramesForces =
                ForceVector(contactFramesIdx.size(), pinocchio::Force::Zero());
            const std::vector<std::vector<pinocchio::PairIndex>> & collisionPairsIdx =
                systemIt->robot->getCollisionPairsIdx();
            systemDataIt->collisionBodiesForces.clear();
            systemDataIt->collisionBodiesForces.reserve(collisionPairsIdx.size());
            for (std::size_t i = 0; i < collisionPairsIdx.size(); ++i)
            {
                systemDataIt->collisionBodiesForces.emplace_back(collisionPairsIdx[i].size(),
                                                                 pinocchio::Force::Zero());
            }

            /* Initialize some addition buffers used by impulse contact solver.
               It must be initialized to zero because 'getJointJacobian' will only update non-zero
               coefficients for efficiency. */
            systemDataIt->jointsJacobians.resize(
                systemIt->robot->pncModel_.njoints,
                Matrix6Xd::Zero(6, systemIt->robot->pncModel_.nv));

            // Reset the constraints
            returnCode = systemIt->robot->resetConstraints(q, v);

            /* Set Baumgarte stabilization natural frequency for contact constraints
               Enable all contact constraints by default, it will be disable automatically if not
               in contact. It is useful to start in post-hysteresis state to avoid discontinuities
               at init. */
            systemDataIt->constraintsHolder.foreach(
                [&contactModel = contactModel_,
                 &enablePositionLimit = systemIt->robot->mdlOptions_->joints.enablePositionLimit,
                 &freq = engineOptions_->contacts.stabilizationFreq](
                    const std::shared_ptr<AbstractConstraintBase> & constraint,
                    constraintsHolderType_t holderType)
                {
                    // Set baumgarte freq for all contact constraints
                    if (holderType != constraintsHolderType_t::USER)
                    {
                        constraint->setBaumgarteFreq(freq);  // It cannot fail
                    }

                    // Enable constraints by default
                    if (contactModel == contactModel_t::CONSTRAINT)
                    {
                        switch (holderType)
                        {
                        case constraintsHolderType_t::BOUNDS_JOINTS:
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
                        case constraintsHolderType_t::CONTACT_FRAMES:
                        case constraintsHolderType_t::COLLISION_BODIES:
                            constraint->enable();
                            break;
                        case constraintsHolderType_t::USER:
                        default:
                            break;
                        }
                    }
                });

            if (contactModel_ == contactModel_t::SPRING_DAMPER)
            {
                // Make sure that the contact forces are bounded for spring-damper model.
                // TODO: Rather use something like `10 * m * g` instead of a fix threshold.
                double forceMax = 0.0;
                for (std::size_t i = 0; i < contactFramesIdx.size(); ++i)
                {
                    auto & constraint = systemDataIt->constraintsHolder.contactFrames[i].second;
                    pinocchio::Force & fextLocal = systemDataIt->contactFramesForces[i];
                    computeContactDynamicsAtFrame(
                        *systemIt, contactFramesIdx[i], constraint, fextLocal);
                    forceMax = std::max(forceMax, fextLocal.linear().norm());
                }

                for (std::size_t i = 0; i < collisionPairsIdx.size(); ++i)
                {
                    for (std::size_t j = 0; j < collisionPairsIdx[i].size(); ++j)
                    {
                        const pinocchio::PairIndex & collisionPairIdx = collisionPairsIdx[i][j];
                        auto & constraint =
                            systemDataIt->constraintsHolder.collisionBodies[i][j].second;
                        pinocchio::Force & fextLocal = systemDataIt->collisionBodiesForces[i][j];
                        computeContactDynamicsAtBody(
                            *systemIt, collisionPairIdx, constraint, fextLocal);
                        forceMax = std::max(forceMax, fextLocal.linear().norm());
                    }
                }

                if (forceMax > 1e5)
                {
                    PRINT_ERROR("The initial force exceeds 1e5 for at least one contact point, "
                                "which is forbidden for the sake of numerical stability. Please "
                                "update the initial state.");
                    return hresult_t::ERROR_BAD_INPUT;
                }
            }
        }

        systemIt = systems_.begin();
        systemDataIt = systemsDataHolder_.begin();
        for (; systemIt != systems_.end(); ++systemIt, ++systemDataIt)
        {
            if (returnCode == hresult_t::SUCCESS)
            {
                // Lock the robot. At this point, it is no longer possible to change it anymore.
                returnCode = systemIt->robot->getLock(systemDataIt->robotLock);
            }
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            // Instantiate the desired LCP solver
            systemIt = systems_.begin();
            systemDataIt = systemsDataHolder_.begin();
            for (; systemIt != systems_.end(); ++systemIt, ++systemDataIt)
            {
                const std::string & constraintSolverType = engineOptions_->constraints.solver;
                switch (CONSTRAINT_SOLVERS_MAP.at(constraintSolverType))
                {
                case constraintSolver_t::PGS:
                    systemDataIt->constraintSolver =
                        std::make_unique<PGSSolver>(&systemIt->robot->pncModel_,
                                                    &systemIt->robot->pncData_,
                                                    &systemDataIt->constraintsHolder,
                                                    engineOptions_->contacts.friction,
                                                    engineOptions_->contacts.torsion,
                                                    engineOptions_->stepper.tolAbs,
                                                    engineOptions_->stepper.tolRel,
                                                    PGS_MAX_ITERATIONS);
                    break;
                case constraintSolver_t::UNSUPPORTED:
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
            for (const auto & systemData : systemsDataHolder_)
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
                systemDataIt = systemsDataHolder_.begin();
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
                        PRINT_ERROR("Impossible to compute the acceleration. Probably a subtree "
                                    "has zero inertia along an articulated axis.");
                        return hresult_t::ERROR_GENERIC;
                    }

                    // Compute all external terms including joints accelerations and forces
                    computeAllExtraTerms(systems_, systemsDataHolder_, fPrev_);

                    // Compute the sensor data with the updated effort and acceleration
                    systemIt->robot->setSensorsData(t, q, v, a, uMotor, fext);

                    // Compute the actual motor effort
                    computeCommand(*systemIt, t, q, v, command);

                    // Compute the actual motor effort
                    systemIt->robot->computeMotorsEfforts(t, q, v, a, command);
                    uMotor = systemIt->robot->getMotorsEfforts();

                    // Compute the internal dynamics
                    uCustom.setZero();
                    systemIt->controller->internalDynamics(t, q, v, uCustom);

                    // Compute the total effort vector
                    u = uInternal + uCustom;
                    for (const auto & motor : systemIt->robot->getMotors())
                    {
                        const std::size_t motorIdx = motor->getIdx();
                        const Eigen::Index motorVelocityIdx = motor->getJointVelocityIdx();
                        u[motorVelocityIdx] += uMotor[motorIdx];
                    }
                }
                isFirstIter = false;
            }

            // Update sensor data one last time to take into account the actual motor effort
            systemIt = systems_.begin();
            systemDataIt = systemsDataHolder_.begin();
            for (; systemIt != systems_.end(); ++systemIt, ++systemDataIt)
            {
                const Eigen::VectorXd & q = systemDataIt->state.q;
                const Eigen::VectorXd & v = systemDataIt->state.v;
                const Eigen::VectorXd & a = systemDataIt->state.a;
                const Eigen::VectorXd & uMotor = systemDataIt->state.uMotor;
                const ForceVector & fext = systemDataIt->state.fExternal;
                systemIt->robot->setSensorsData(t, q, v, a, uMotor, fext);
            }

            // Backend the updated joint accelerations and forces
            syncAllAccelerationsAndForces(systems_, contactForcesPrev_, fPrev_, aPrev_);

            // Synchronize the global stepper state with the individual system states
            syncStepperStateWithSystems();

            // Initialize the last system states
            for (auto & systemData : systemsDataHolder_)
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
                const std::string telemetryMeshPackageDirs = addCircumfix(
                    "mesh_package_dirs", system.name, {}, TELEMETRY_FIELDNAME_DELIMITER);
                std::string meshPackageDirsString;
                std::stringstream meshPackageDirsStream;
                const std::vector<std::string> & meshPackageDirs =
                    system.robot->getMeshPackageDirs();
                copy(meshPackageDirs.begin(),
                     meshPackageDirs.end(),
                     std::ostream_iterator<std::string>(meshPackageDirsStream, ";"));
                if (meshPackageDirsStream.peek() !=
                    decltype(meshPackageDirsStream)::traits_type::eof())
                {
                    meshPackageDirsString = meshPackageDirsStream.str();
                    meshPackageDirsString.pop_back();
                }
                telemetrySender_->registerConstant(telemetryMeshPackageDirs,
                                                   meshPackageDirsString);

                // Backup the true and theoretical Pinocchio::Model
                std::string key = addCircumfix(
                    "pinocchio_model", system.name, {}, TELEMETRY_FIELDNAME_DELIMITER);
                std::string value = saveToBinary(system.robot->pncModel_);
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
                        PRINT_ERROR(msg, e.what());
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
            allOptions["engine"] = engineOptionsHolder_;
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

        return returnCode;
    }

    hresult_t EngineMultiRobot::simulate(
        double tEnd,
        const std::map<std::string, Eigen::VectorXd> & qInit,
        const std::map<std::string, Eigen::VectorXd> & vInit,
        const std::optional<std::map<std::string, Eigen::VectorXd>> & aInit)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (systems_.empty())
        {
            PRINT_ERROR("No system to simulate. Please add one before starting a simulation.");
            returnCode = hresult_t::ERROR_INIT_FAILED;
        }

        if (tEnd < 5e-3)
        {
            PRINT_ERROR("The duration of the simulation cannot be shorter than 5ms.");
            returnCode = hresult_t::ERROR_BAD_INPUT;
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            // Reset the robot, controller, and engine
            reset(true, false);

            // Start the simulation
            returnCode = start(qInit, vInit, aInit);
        }

        // Now that telemetry has been initialized, check simulation duration
        if (tEnd > telemetryRecorder_->getMaximumLogTime())
        {
            PRINT_ERROR("Time overflow: with the current precision the maximum value that can be "
                        "logged is ",
                        telemetryRecorder_->getMaximumLogTime(),
                        "s. Decrease logger precision to simulate for longer than that.");
            returnCode = hresult_t::ERROR_BAD_INPUT;
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
            bool isCallbackFalse = false;
            auto systemIt = systems_.begin();
            auto systemDataIt = systemsDataHolder_.begin();
            for (; systemIt != systems_.end(); ++systemIt, ++systemDataIt)
            {
                if (!systemIt->callbackFct(
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
            returnCode = step(stepSize);  // Automatic dt adjustment
        }

        // Stop the simulation. The lock on the telemetry and the robots are released.
        stop();

        return returnCode;
    }

    hresult_t EngineMultiRobot::step(double stepSize)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        // Check if the simulation has started
        if (!isSimulationRunning_)
        {
            PRINT_ERROR("No simulation running. Please start it before using step method.");
            return hresult_t::ERROR_GENERIC;
        }

        // Clear log data buffer
        logData_ = nullptr;

        // Check if there is something wrong with the integration
        auto qIt = stepperState_.qSplit.begin();
        auto vIt = stepperState_.vSplit.begin();
        auto aIt = stepperState_.aSplit.begin();
        for (; qIt != stepperState_.qSplit.end(); ++qIt, ++vIt, ++aIt)
        {
            if (qIt->hasNaN() || vIt->hasNaN() || aIt->hasNaN())
            {
                PRINT_ERROR(
                    "The low-level ode solver failed. Consider increasing the stepper accuracy.");
                return hresult_t::ERROR_GENERIC;
            }
        }

        // Check if the desired step size is suitable
        if (stepSize > EPS && stepSize < SIMULATION_MIN_TIMESTEP)
        {
            PRINT_ERROR("The requested step size is out of bounds.");
            return hresult_t::ERROR_BAD_INPUT;
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
        if (stepperState_.t + stepSize > telemetryRecorder_->getMaximumLogTime())
        {
            PRINT_ERROR("Time overflow: with the current precision the maximum value that "
                        "can be logged is ",
                        telemetryRecorder_->getMaximumLogTime(),
                        "s. Decrease logger precision to simulate for longer than that.");
            return hresult_t::ERROR_GENERIC;
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
        while ((tEnd - t >= STEPPER_MIN_TIMESTEP) && (returnCode == hresult_t::SUCCESS))
        {
            // Initialize next breakpoint time to the one recommended by the stepper
            double tNext = t;

            // Update the active set and get the next breakpoint of impulse forces
            double tForceImpulseNext = INF;
            for (auto & systemData : systemsDataHolder_)
            {
                /* Update the active set: activate an impulse force as soon as the current time
                   gets close enough of the application time, and deactivate it once the following
                   the same reasoning.

                   Note that breakpoints at the start/end of every impulse forces are already
                   enforced, so that the forces cannot get activated/deactivate too late. */
                auto forcesImpulseActiveIt = systemData.forcesImpulseActive.begin();
                auto forcesImpulseIt = systemData.forcesImpulse.begin();
                for (; forcesImpulseIt != systemData.forcesImpulse.end();
                     ++forcesImpulseActiveIt, ++forcesImpulseIt)
                {
                    double tForceImpulse = forcesImpulseIt->t;
                    double dtForceImpulse = forcesImpulseIt->dt;

                    if (t > tForceImpulse - STEPPER_MIN_TIMESTEP)
                    {
                        *forcesImpulseActiveIt = true;
                        hasDynamicsChanged = true;
                    }
                    if (t >= tForceImpulse + dtForceImpulse - STEPPER_MIN_TIMESTEP)
                    {
                        *forcesImpulseActiveIt = false;
                        hasDynamicsChanged = true;
                    }
                }

                // Update the breakpoint time iterator if necessary
                auto & tBreakNextIt = systemData.forcesImpulseBreakNextIt;
                if (tBreakNextIt != systemData.forcesImpulseBreaks.end())
                {
                    if (t >= *tBreakNextIt - STEPPER_MIN_TIMESTEP)
                    {
                        // The current breakpoint is behind in time. Switching to the next one.
                        ++tBreakNextIt;
                    }
                }

                // Get the next breakpoint time if any
                if (tBreakNextIt != systemData.forcesImpulseBreaks.end())
                {
                    tForceImpulseNext = std::min(tForceImpulseNext, *tBreakNextIt);
                }
            }

            // Update the external force profiles if necessary (only for finite update frequency)
            if (std::isfinite(stepperUpdatePeriod_))
            {
                auto systemIt = systems_.begin();
                auto systemDataIt = systemsDataHolder_.begin();
                for (; systemIt != systems_.end(); ++systemIt, ++systemDataIt)
                {
                    for (auto & forceProfile : systemDataIt->forcesProfile)
                    {
                        if (forceProfile.updatePeriod > EPS)
                        {
                            double forceUpdatePeriod = forceProfile.updatePeriod;
                            double dtNextForceUpdatePeriod =
                                forceUpdatePeriod - std::fmod(t, forceUpdatePeriod);
                            if (dtNextForceUpdatePeriod < SIMULATION_MIN_TIMESTEP ||
                                forceUpdatePeriod - dtNextForceUpdatePeriod < STEPPER_MIN_TIMESTEP)
                            {
                                const Eigen::VectorXd & q = systemDataIt->state.q;
                                const Eigen::VectorXd & v = systemDataIt->state.v;
                                forceProfile.forcePrev = forceProfile.forceFct(t, q, v);
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
                    auto systemDataIt = systemsDataHolder_.begin();
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
                        std::min(dtNextUpdatePeriod + stepperUpdatePeriod_, tForceImpulseNext - t);
                }
                else
                {
                    dtNextGlobal = std::min(dtNextUpdatePeriod, tForceImpulseNext - t);
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
                    auto systemDataIt = systemsDataHolder_.begin();
                    auto successiveSolveFailedIt = successiveSolveFailedAll.begin();
                    for (; systemDataIt != systemsDataHolder_.end();
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
                        computeAllExtraTerms(systems_, systemsDataHolder_, fPrev_);

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
                        for (auto & systemData : systemsDataHolder_)
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
                        systemDataIt = systemsDataHolder_.begin();
                        successiveSolveFailedIt = successiveSolveFailedAll.begin();
                        for (; systemDataIt != systemsDataHolder_.end();
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
                dt = std::min({dt, tEnd - t, tForceImpulseNext - t});

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
                    auto systemDataIt = systemsDataHolder_.begin();
                    auto successiveSolveFailedIt = successiveSolveFailedAll.begin();
                    for (; systemDataIt != systemsDataHolder_.end();
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
                        computeAllExtraTerms(systems_, systemsDataHolder_, fPrev_);

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
                        for (auto & systemData : systemsDataHolder_)
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
                        systemDataIt = systemsDataHolder_.begin();
                        successiveSolveFailedIt = successiveSolveFailedAll.begin();
                        for (; systemDataIt != systemsDataHolder_.end();
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
                PRINT_ERROR("Something is wrong with the physics. Aborting integration.");
                returnCode = hresult_t::ERROR_GENERIC;
            }
            if (successiveIterFailed > engineOptions_->stepper.successiveIterFailedMax)
            {
                PRINT_ERROR("Too many successive iteration failures. Probably something is "
                            "going wrong with the physics. Aborting integration.");
                returnCode = hresult_t::ERROR_GENERIC;
            }
            for (uint32_t successiveSolveFailed : successiveSolveFailedAll)
            {
                if (successiveSolveFailed > engineOptions_->stepper.successiveIterFailedMax)
                {
                    PRINT_ERROR("Too many successive constraint solving failures. Try increasing "
                                "the regularization factor. Aborting integration.");
                    return hresult_t::ERROR_GENERIC;
                }
            }
            if (dt < STEPPER_MIN_TIMESTEP)
            {
                PRINT_ERROR("The internal time step is getting too small. Impossible to "
                            "integrate physics further in time. Aborting integration.");
                returnCode = hresult_t::ERROR_GENERIC;
            }
            if (EPS < engineOptions_->stepper.timeout &&
                engineOptions_->stepper.timeout < timer_.toc())
            {
                PRINT_ERROR("Step computation timeout. Aborting integration.");
                returnCode = hresult_t::ERROR_GENERIC;
            }

            // Update sensors data if necessary, namely if time-continuous or breakpoint
            if (returnCode == hresult_t::SUCCESS)
            {
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
                    auto systemDataIt = systemsDataHolder_.begin();
                    for (; systemIt != systems_.end(); ++systemIt, ++systemDataIt)
                    {
                        const Eigen::VectorXd & q = systemDataIt->state.q;
                        const Eigen::VectorXd & v = systemDataIt->state.v;
                        const Eigen::VectorXd & a = systemDataIt->state.a;
                        const Eigen::VectorXd & uMotor = systemDataIt->state.uMotor;
                        const ForceVector & fext = systemDataIt->state.fExternal;
                        systemIt->robot->setSensorsData(t, q, v, a, uMotor, fext);
                    }
                }
            }
        }

        /* Update the final time and dt to make sure it corresponds to the desired values and avoid
           compounding of error. Anyway the user asked for a step of exactly stepSize, so he is
           expecting this value to be reached. */
        if (returnCode == hresult_t::SUCCESS)
        {
            t = tEnd;
        }

        return returnCode;
    }

    void EngineMultiRobot::stop()
    {
        // Release the lock on the robots
        for (auto & systemData : systemsDataHolder_)
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
        logData_ = nullptr;

        /* Reset the telemetry.
           Note that calling ``stop` or  `reset` does NOT clear the internal data buffer of
           telemetryRecorder_. Clearing is done at init time, so that it remains accessible until
           the next initialization. */
        telemetryRecorder_->reset();
        telemetryData_->reset();

        // Update some internal flags
        isSimulationRunning_ = false;
    }

    hresult_t EngineMultiRobot::registerForceImpulse(const std::string & systemName,
                                                     const std::string & frameName,
                                                     double t,
                                                     double dt,
                                                     const pinocchio::Force & F)
    {
        // Make sure that the forces do NOT overlap while taking into account dt.

        hresult_t returnCode = hresult_t::SUCCESS;

        if (isSimulationRunning_)
        {
            PRINT_ERROR("A simulation is running. Please stop it before registering new forces.");
            returnCode = hresult_t::ERROR_GENERIC;
        }

        if (dt < STEPPER_MIN_TIMESTEP)
        {
            PRINT_ERROR("The force duration cannot be smaller than ", STEPPER_MIN_TIMESTEP, ".");
            returnCode = hresult_t::ERROR_BAD_INPUT;
        }

        if (t < 0.0)
        {
            PRINT_ERROR("The force application time must be positive.");
            returnCode = hresult_t::ERROR_BAD_INPUT;
        }

        if (frameName == "universe")
        {
            PRINT_ERROR("Impossible to apply external forces to the universe itself!");
            returnCode = hresult_t::ERROR_GENERIC;
        }

        std::ptrdiff_t systemIdx;
        if (returnCode == hresult_t::SUCCESS)
        {
            returnCode = getSystemIdx(systemName, systemIdx);
        }

        pinocchio::FrameIndex frameIdx;
        if (returnCode == hresult_t::SUCCESS)
        {
            const systemHolder_t & system = systems_[systemIdx];
            returnCode = getFrameIdx(system.robot->pncModel_, frameName, frameIdx);
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            systemDataHolder_t & systemData = systemsDataHolder_[systemIdx];
            systemData.forcesImpulse.emplace_back(frameName, frameIdx, t, dt, F);
            systemData.forcesImpulseBreaks.emplace(t);
            systemData.forcesImpulseBreaks.emplace(t + dt);
            systemData.forcesImpulseActive.emplace_back(false);
        }

        return hresult_t::SUCCESS;
    }

    template<typename... Args>
    std::tuple<bool, const double &> isGcdIncluded(
        const vector_aligned_t<systemDataHolder_t> & systemsDataHolder, const Args &... values)
    {
        if (systemsDataHolder.empty())
        {
            return isGcdIncluded(values...);
        }

        const double * minValuePtr = &INF;
        auto lambda = [&minValuePtr, &values...](const systemDataHolder_t & systemData)
        {
            auto && [isIncluded, value] = isGcdIncluded(
                systemData.forcesProfile.cbegin(),
                systemData.forcesProfile.cend(),
                [](ForceProfile const & force) -> const double & { return force.updatePeriod; },
                values...);
            minValuePtr = &(minClipped(*minValuePtr, value));
            return isIncluded;
        };
        // FIXME: Order of evaluation is not always respected with MSVC.
        bool isIncluded = std::all_of(systemsDataHolder.begin(), systemsDataHolder.end(), lambda);
        return {isIncluded, *minValuePtr};
    }

    hresult_t EngineMultiRobot::registerForceProfile(const std::string & systemName,
                                                     const std::string & frameName,
                                                     const ForceProfileFunctor & forceFct,
                                                     double updatePeriod)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (isSimulationRunning_)
        {
            PRINT_ERROR("A simulation is running. Please stop it before registering new forces.");
            returnCode = hresult_t::ERROR_GENERIC;
        }

        std::ptrdiff_t systemIdx;
        if (returnCode == hresult_t::SUCCESS)
        {
            returnCode = getSystemIdx(systemName, systemIdx);
        }

        if (frameName == "universe")
        {
            PRINT_ERROR("Impossible to apply external forces to the universe itself!");
            returnCode = hresult_t::ERROR_GENERIC;
        }

        pinocchio::FrameIndex frameIdx;
        if (returnCode == hresult_t::SUCCESS)
        {
            const systemHolder_t & system = systems_[systemIdx];
            returnCode = getFrameIdx(system.robot->pncModel_, frameName, frameIdx);
        }

        // Make sure the update period is valid
        if (returnCode == hresult_t::SUCCESS)
        {
            if (EPS < updatePeriod && updatePeriod < SIMULATION_MIN_TIMESTEP)
            {
                PRINT_ERROR(
                    "Cannot register external force profile with update period smaller than ",
                    SIMULATION_MIN_TIMESTEP,
                    "s. Adjust period or switch to continuous mode "
                    "by setting period to zero.");
                returnCode = hresult_t::ERROR_BAD_INPUT;
            }
        }

        // Make sure the desired update period is a multiple of the stepper period
        auto [isIncluded, minUpdatePeriod] =
            isGcdIncluded(systemsDataHolder_, stepperUpdatePeriod_, updatePeriod);
        if (returnCode == hresult_t::SUCCESS)
        {
            if (!isIncluded)
            {
                PRINT_ERROR(
                    "In discrete mode, the update period of force profiles and the stepper "
                    "update period (min of controller and sensor update periods) must be "
                    "multiple of each other.");
                returnCode = hresult_t::ERROR_BAD_INPUT;
            }
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            // Set breakpoint period during the integration loop
            stepperUpdatePeriod_ = minUpdatePeriod;

            // Add force profile to register
            systemDataHolder_t & systemData = systemsDataHolder_[systemIdx];
            systemData.forcesProfile.emplace_back(frameName, frameIdx, updatePeriod, forceFct);
        }

        return returnCode;
    }

    hresult_t EngineMultiRobot::removeForcesImpulse(const std::string & systemName)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        // Make sure that no simulation is running
        if (isSimulationRunning_)
        {
            PRINT_ERROR(
                "A simulation is already running. Stop it before removing coupling forces.");
            returnCode = hresult_t::ERROR_GENERIC;
        }

        std::ptrdiff_t systemIdx;
        if (returnCode == hresult_t::SUCCESS)
        {
            returnCode = getSystemIdx(systemName, systemIdx);
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            systemDataHolder_t & systemData = systemsDataHolder_[systemIdx];
            systemData.forcesImpulse.clear();
        }

        return hresult_t::SUCCESS;
    }

    hresult_t EngineMultiRobot::removeForcesImpulse()
    {
        // Make sure that no simulation is running
        if (isSimulationRunning_)
        {
            PRINT_ERROR(
                "A simulation is already running. Stop it before removing coupling forces.");
            return hresult_t::ERROR_GENERIC;
        }

        for (auto & systemData : systemsDataHolder_)
        {
            systemData.forcesImpulse.clear();
        }

        return hresult_t::SUCCESS;
    }

    hresult_t EngineMultiRobot::removeForcesProfile(const std::string & systemName)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        // Make sure that no simulation is running
        if (isSimulationRunning_)
        {
            PRINT_ERROR(
                "A simulation is already running. Stop it before removing coupling forces.");
            return hresult_t::ERROR_GENERIC;
        }

        std::ptrdiff_t systemIdx;
        if (returnCode == hresult_t::SUCCESS)
        {
            returnCode = getSystemIdx(systemName, systemIdx);
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            // Remove force profile from register
            systemDataHolder_t & systemData = systemsDataHolder_[systemIdx];
            systemData.forcesProfile.clear();

            // Set breakpoint period during the integration loop
            stepperUpdatePeriod_ =
                std::get<1>(isGcdIncluded(systemsDataHolder_,
                                          engineOptions_->stepper.sensorsUpdatePeriod,
                                          engineOptions_->stepper.controllerUpdatePeriod));
        }

        return hresult_t::SUCCESS;
    }

    hresult_t EngineMultiRobot::removeForcesProfile()
    {
        // Make sure that no simulation is running
        if (isSimulationRunning_)
        {
            PRINT_ERROR(
                "A simulation is already running. Stop it before removing coupling forces.");
            return hresult_t::ERROR_GENERIC;
        }

        for (auto & systemData : systemsDataHolder_)
        {
            systemData.forcesProfile.clear();
        }

        return hresult_t::SUCCESS;
    }

    hresult_t EngineMultiRobot::getForcesImpulse(
        const std::string & systemName, const ForceImpulseRegister *& forcesImpulsePtr) const
    {
        static ForceImpulseRegister forcesImpuseDummy;

        hresult_t returnCode = hresult_t::SUCCESS;

        forcesImpulsePtr = &forcesImpuseDummy;

        std::ptrdiff_t systemIdx;
        returnCode = getSystemIdx(systemName, systemIdx);

        if (returnCode == hresult_t::SUCCESS)
        {
            const systemDataHolder_t & systemData = systemsDataHolder_[systemIdx];
            forcesImpulsePtr = &systemData.forcesImpulse;
        }

        return returnCode;
    }

    hresult_t EngineMultiRobot::getForcesProfile(
        const std::string & systemName, const ForceProfileRegister *& forcesProfilePtr) const
    {
        static ForceProfileRegister forcesRegisterDummy;

        hresult_t returnCode = hresult_t::SUCCESS;

        forcesProfilePtr = &forcesRegisterDummy;

        std::ptrdiff_t systemIdx;
        returnCode = getSystemIdx(systemName, systemIdx);

        if (returnCode == hresult_t::SUCCESS)
        {
            const systemDataHolder_t & systemData = systemsDataHolder_[systemIdx];
            forcesProfilePtr = &systemData.forcesProfile;
        }

        return returnCode;
    }

    GenericConfig EngineMultiRobot::getOptions() const noexcept
    {
        return engineOptionsHolder_;
    }

    hresult_t EngineMultiRobot::setOptions(const GenericConfig & engineOptions)
    {
        if (isSimulationRunning_)
        {
            PRINT_ERROR("A simulation is running. Please stop it before updating the options.");
            return hresult_t::ERROR_GENERIC;
        }

        // Make sure the dtMax is not out of range
        GenericConfig stepperOptions = boost::get<GenericConfig>(engineOptions.at("stepper"));
        const double dtMax = boost::get<double>(stepperOptions.at("dtMax"));
        if (SIMULATION_MAX_TIMESTEP + EPS < dtMax || dtMax < SIMULATION_MIN_TIMESTEP)
        {
            PRINT_ERROR("'dtMax' option is out of range.");
            return hresult_t::ERROR_BAD_INPUT;
        }

        // Make sure successiveIterFailedMax is strictly positive
        const uint32_t successiveIterFailedMax =
            boost::get<uint32_t>(stepperOptions.at("successiveIterFailedMax"));
        if (successiveIterFailedMax < 1)
        {
            PRINT_ERROR("'successiveIterFailedMax' must be strictly positive.");
            return hresult_t::ERROR_BAD_INPUT;
        }

        // Make sure the selected ode solver is available and instantiate it
        const std::string & odeSolver = boost::get<std::string>(stepperOptions.at("odeSolver"));
        if (STEPPERS.find(odeSolver) == STEPPERS.end())
        {
            PRINT_ERROR("The requested ODE solver is not available.");
            return hresult_t::ERROR_BAD_INPUT;
        }

        // Make sure the controller and sensor update periods are valid
        const double sensorsUpdatePeriod =
            boost::get<double>(stepperOptions.at("sensorsUpdatePeriod"));
        const double controllerUpdatePeriod =
            boost::get<double>(stepperOptions.at("controllerUpdatePeriod"));
        auto [isIncluded, minUpdatePeriod] =
            isGcdIncluded(systemsDataHolder_, controllerUpdatePeriod, sensorsUpdatePeriod);
        if ((EPS < sensorsUpdatePeriod && sensorsUpdatePeriod < SIMULATION_MIN_TIMESTEP) ||
            (EPS < controllerUpdatePeriod && controllerUpdatePeriod < SIMULATION_MIN_TIMESTEP))
        {
            PRINT_ERROR(
                "Cannot simulate a discrete system with update period smaller than ",
                SIMULATION_MIN_TIMESTEP,
                "s. Adjust period or switch to continuous mode by setting period to zero.");
            return hresult_t::ERROR_BAD_INPUT;
        }
        else if (!isIncluded)
        {
            PRINT_ERROR("In discrete mode, the controller and sensor update periods must be "
                        "multiple of each other.");
            return hresult_t::ERROR_BAD_INPUT;
        }

        // Make sure the contacts options are fine
        GenericConfig constraintsOptions =
            boost::get<GenericConfig>(engineOptions.at("constraints"));
        const std::string & constraintSolverType =
            boost::get<std::string>(constraintsOptions.at("solver"));
        const auto constraintSolverIt = CONSTRAINT_SOLVERS_MAP.find(constraintSolverType);
        if (constraintSolverIt == CONSTRAINT_SOLVERS_MAP.end())
        {
            PRINT_ERROR("The requested constraint solver is not available.");
            return hresult_t::ERROR_BAD_INPUT;
        }
        double regularization = boost::get<double>(constraintsOptions.at("regularization"));
        if (regularization < 0.0)
        {
            PRINT_ERROR("The constraints option 'regularization' must be positive.");
            return hresult_t::ERROR_BAD_INPUT;
        }

        // Make sure the contacts options are fine
        GenericConfig contactsOptions = boost::get<GenericConfig>(engineOptions.at("contacts"));
        const std::string & contactModel = boost::get<std::string>(contactsOptions.at("model"));
        const auto contactModelIt = CONTACT_MODELS_MAP.find(contactModel);
        if (contactModelIt == CONTACT_MODELS_MAP.end())
        {
            PRINT_ERROR("The requested contact model is not available.");
            return hresult_t::ERROR_BAD_INPUT;
        }
        double contactsTransitionEps = boost::get<double>(contactsOptions.at("transitionEps"));
        if (contactsTransitionEps < 0.0)
        {
            PRINT_ERROR("The contacts option 'transitionEps' must be positive.");
            return hresult_t::ERROR_BAD_INPUT;
        }
        double transitionVelocity = boost::get<double>(contactsOptions.at("transitionVelocity"));
        if (transitionVelocity < EPS)
        {
            PRINT_ERROR("The contacts option 'transitionVelocity' must be strictly positive.");
            return hresult_t::ERROR_BAD_INPUT;
        }
        double stabilizationFreq = boost::get<double>(contactsOptions.at("stabilizationFreq"));
        if (stabilizationFreq < 0.0)
        {
            PRINT_ERROR("The contacts option 'stabilizationFreq' must be positive.");
            return hresult_t::ERROR_BAD_INPUT;
        }

        // Make sure the user-defined gravity force has the right dimension
        GenericConfig worldOptions = boost::get<GenericConfig>(engineOptions.at("world"));
        Eigen::VectorXd gravity = boost::get<Eigen::VectorXd>(worldOptions.at("gravity"));
        if (gravity.size() != 6)
        {
            PRINT_ERROR("The size of the gravity force vector must be 6.");
            return hresult_t::ERROR_BAD_INPUT;
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
        engineOptionsHolder_ = engineOptions;

        // Create a fast struct accessor
        engineOptions_ = std::make_unique<const engineOptions_t>(engineOptionsHolder_);

        // Backup contact model as enum for fast check
        contactModel_ = contactModelIt->second;

        // Set breakpoint period during the integration loop
        stepperUpdatePeriod_ = minUpdatePeriod;

        return hresult_t::SUCCESS;
    }

    std::vector<std::string> EngineMultiRobot::getSystemsNames() const
    {
        std::vector<std::string> systemsNames;
        systemsNames.reserve(systems_.size());
        std::transform(systems_.begin(),
                       systems_.end(),
                       std::back_inserter(systemsNames),
                       [](const auto & sys) -> std::string { return sys.name; });
        return systemsNames;
    }

    hresult_t EngineMultiRobot::getSystemIdx(const std::string & systemName,
                                             std::ptrdiff_t & systemIdx) const
    {
        auto systemIt = std::find_if(systems_.begin(),
                                     systems_.end(),
                                     [&systemName](const auto & sys)
                                     { return (sys.name == systemName); });
        if (systemIt == systems_.end())
        {
            PRINT_ERROR("No system with this name has been added to the engine.");
            return hresult_t::ERROR_BAD_INPUT;
        }

        systemIdx = std::distance(systems_.begin(), systemIt);

        return hresult_t::SUCCESS;
    }

    hresult_t EngineMultiRobot::getSystem(const std::string & systemName, systemHolder_t *& system)
    {
        static systemHolder_t systemEmpty;

        hresult_t returnCode = hresult_t::SUCCESS;

        auto systemIt = std::find_if(systems_.begin(),
                                     systems_.end(),
                                     [&systemName](const auto & sys)
                                     { return (sys.name == systemName); });
        if (systemIt == systems_.end())
        {
            PRINT_ERROR("No system with this name has been added to the engine.");
            returnCode = hresult_t::ERROR_BAD_INPUT;
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            system = &(*systemIt);
            return returnCode;
        }

        system = &systemEmpty;

        return returnCode;
    }

    hresult_t EngineMultiRobot::getSystemState(const std::string & systemName,
                                               const systemState_t *& systemState) const
    {
        static const systemState_t systemStateEmpty;

        hresult_t returnCode = hresult_t::SUCCESS;

        std::ptrdiff_t systemIdx;
        returnCode = getSystemIdx(systemName, systemIdx);
        if (returnCode == hresult_t::SUCCESS)
        {
            systemState = &(systemsDataHolder_[systemIdx].state);
            return returnCode;
        }

        systemState = &systemStateEmpty;
        return returnCode;
    }

    const StepperState & EngineMultiRobot::getStepperState() const
    {
        return stepperState_;
    }

    const bool & EngineMultiRobot::getIsSimulationRunning() const
    {
        return isSimulationRunning_;
    }

    double EngineMultiRobot::getMaxSimulationDuration()
    {
        return TelemetryRecorder::getMaximumLogTime(getTelemetryTimeUnit());
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
        auto systemDataIt = systemsDataHolder_.begin();
        for (; systemDataIt != systemsDataHolder_.end();
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
            auto systemDataIt = systemsDataHolder_.begin();
            for (; systemDataIt != systemsDataHolder_.end(); ++systemDataIt, ++aSplitIt)
            {
                systemDataIt->state.a = *aSplitIt;
            }
        }
        else
        {
            auto qSplitIt = stepperState_.qSplit.begin();
            auto vSplitIt = stepperState_.vSplit.begin();
            auto aSplitIt = stepperState_.aSplit.begin();
            auto systemDataIt = systemsDataHolder_.begin();
            for (; systemDataIt != systemsDataHolder_.end();
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


    void EngineMultiRobot::computeForwardKinematics(systemHolder_t & system,
                                                    const Eigen::VectorXd & q,
                                                    const Eigen::VectorXd & v,
                                                    const Eigen::VectorXd & a)
    {
        // Create proxies for convenience
        const pinocchio::Model & model = system.robot->pncModel_;
        pinocchio::Data & data = system.robot->pncData_;
        const pinocchio::GeometryModel & geomModel = system.robot->collisionModel_;
        pinocchio::GeometryData & geomData = system.robot->collisionData_;

        // Update forward kinematics
        pinocchio::forwardKinematics(model, data, q, v, a);

        // Update frame placements (avoiding redundant computations)
        for (int frameIdx = 1; frameIdx < model.nframes; ++frameIdx)
        {
            const pinocchio::Frame & frame = model.frames[frameIdx];
            pinocchio::JointIndex parentJointModelIdx = frame.parent;
            switch (frame.type)
            {
            case pinocchio::FrameType::JOINT:
                /* If the frame is associated with an actual joint, no need to compute anything
                   new, since the frame transform is supposed to be identity. */
                data.oMf[frameIdx] = data.oMi[parentJointModelIdx];
                break;
            case pinocchio::FrameType::BODY:
                if (model.frames[frame.previousFrame].type == pinocchio::FrameType::FIXED_JOINT)
                {
                    /* BODYs connected via FIXED_JOINT(s) have the same transform than the joint
                       itself, so no need to compute them twice. Here we are doing the assumption
                       that the previous frame transform has already been updated since it is
                       closer to root in kinematic tree. */
                    data.oMf[frameIdx] = data.oMf[frame.previousFrame];
                }
                else
                {
                    /* BODYs connected via JOINT(s) have the identity transform, so copying parent
                       joint transform should be fine. */
                    data.oMf[frameIdx] = data.oMi[parentJointModelIdx];
                }
                break;
            case pinocchio::FrameType::FIXED_JOINT:
            case pinocchio::FrameType::SENSOR:
            case pinocchio::FrameType::OP_FRAME:
            default:
                // Nothing special, doing the actual computation
                data.oMf[frameIdx] = data.oMi[parentJointModelIdx] * frame.placement;
            }
        }

        /* Update collision information selectively, ie only for geometries involved in at least
           one collision pair. */
        pinocchio::updateGeometryPlacements(model, data, geomModel, geomData);
        pinocchio::computeCollisions(geomModel, geomData, false);
    }

    void EngineMultiRobot::computeContactDynamicsAtBody(
        const systemHolder_t & system,
        const pinocchio::PairIndex & collisionPairIdx,
        std::shared_ptr<AbstractConstraintBase> & constraint,
        pinocchio::Force & fextLocal) const
    {
        // TODO: It is assumed that the ground is flat. For now ground profile is not supported
        // with body collision. Nevertheless it should not be to hard to generated a collision
        // object simply by sampling points on the profile.

        // Get the frame and joint indices
        const pinocchio::GeomIndex & geometryIdx =
            system.robot->collisionModel_.collisionPairs[collisionPairIdx].first;
        pinocchio::JointIndex parentJointModelIdx =
            system.robot->collisionModel_.geometryObjects[geometryIdx].parentJoint;

        // Extract collision and distance results
        const hpp::fcl::CollisionResult & collisionResult =
            system.robot->collisionData_.collisionResults[collisionPairIdx];

        fextLocal.setZero();

        /* There is no way to get access to the distance from the ground at this point, so it is
           not possible to disable the constraint only if depth > transitionEps. */
        if (constraint)
        {
            constraint->disable();
        }

        for (std::size_t contactIdx = 0; contactIdx < collisionResult.numContacts(); ++contactIdx)
        {
            /* Extract the contact information.
               Note that there is always a single contact point while computing the collision
               between two shape objects, for instance convex geometry and box primitive. */
            const auto & contact = collisionResult.getContact(contactIdx);
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

            if (contactModel_ == contactModel_t::SPRING_DAMPER)
            {
                // Compute the linear velocity of the contact point in world frame
                const pinocchio::Motion & motionJointLocal =
                    system.robot->pncData_.v[parentJointModelIdx];
                const pinocchio::SE3 & transformJointFrameInWorld =
                    system.robot->pncData_.oMi[parentJointModelIdx];
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
                    const pinocchio::FrameIndex frameIdx = frameConstraint.getFrameIdx();
                    frameConstraint.setReferenceTransform(
                        {system.robot->pncData_.oMf[frameIdx].rotation(),
                         system.robot->pncData_.oMf[frameIdx].translation() - depth * nGround});
                    frameConstraint.setNormal(nGround);

                    // Only one contact constraint per collision body is supported for now
                    break;
                }
            }
        }
    }

    void EngineMultiRobot::computeContactDynamicsAtFrame(
        const systemHolder_t & system,
        pinocchio::FrameIndex frameIdx,
        std::shared_ptr<AbstractConstraintBase> & constraint,
        pinocchio::Force & fextLocal) const
    {
        /* Returns the external force in the contact frame. It must then be converted into a force
           onto the parent joint.
           /!\ Note that the contact dynamics depends only on kinematics data. /!\ */

        // Define proxies for convenience
        const pinocchio::Model & model = system.robot->pncModel_;
        const pinocchio::Data & data = system.robot->pncData_;

        // Get the pose of the frame wrt the world
        const pinocchio::SE3 & transformFrameInWorld = data.oMf[frameIdx];

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
            if (contactModel_ == contactModel_t::SPRING_DAMPER)
            {
                // Compute the linear velocity of the contact point in world frame
                const Eigen::Vector3d motionFrameLocal =
                    pinocchio::getFrameVelocity(model, data, frameIdx).linear();
                const Eigen::Matrix3d & rotFrame = transformFrameInWorld.rotation();
                const Eigen::Vector3d vContactInWorld = rotFrame * motionFrameLocal;

                // Compute the ground reaction force in world frame (local world aligned)
                const pinocchio::Force fextAtContactInGlobal =
                    computeContactDynamics(normalGround, depth, vContactInWorld);

                // Deduce the ground reaction force in joint frame
                fextLocal =
                    convertForceGlobalFrameToJoint(model, data, frameIdx, fextAtContactInGlobal);
            }
            else
            {
                // Enable fixed frame constraint
                constraint->enable();
            }
        }
        else
        {
            if (contactModel_ == contactModel_t::SPRING_DAMPER)
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
            const contactOptions_t & contactOptions_ = engineOptions_->contacts;

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

    void EngineMultiRobot::computeCommand(systemHolder_t & system,
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
            const pinocchio::Data & /* pncData */,
            const Eigen::VectorXd & /* q */,
            const Eigen::VectorXd & /* v */,
            const Eigen::VectorXd & /* positionLimitMin */,
            const Eigen::VectorXd & /* positionLimitMax */,
            const std::unique_ptr<const EngineMultiRobot::engineOptions_t> & /* engineOptions */,
            contactModel_t /* contactModel */,
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
             const pinocchio::Data & pncData,
             const Eigen::VectorXd & q,
             const Eigen::VectorXd & v,
             const Eigen::VectorXd & positionLimitMin,
             const Eigen::VectorXd & positionLimitMax,
             const std::unique_ptr<const EngineMultiRobot::engineOptions_t> & engineOptions,
             contactModel_t contactModel,
             std::shared_ptr<AbstractConstraintBase> & constraint,
             Eigen::VectorXd & u)
        {
            // Define some proxies for convenience
            const pinocchio::JointIndex jointModelIdx = joint.id();
            const Eigen::Index positionIdx = joint.idx_q();
            const Eigen::Index velocityIdx = joint.idx_v();
            const double qJoint = q[positionIdx];
            const double qJointMin = positionLimitMin[positionIdx];
            const double qJointMax = positionLimitMax[positionIdx];
            const double vJoint = v[velocityIdx];
            const double Ia = getSubtreeInertiaProj(joint.derived(), pncData.Ycrb[jointModelIdx]);
            const double stiffness = engineOptions->joints.boundStiffness;
            const double damping = engineOptions->joints.boundDamping;
            const double transitionEps = engineOptions->contacts.transitionEps;

            // Check if out-of-bounds
            if (contactModel == contactModel_t::SPRING_DAMPER)
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
                u[velocityIdx] += Ia * accelJoint;
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
             const pinocchio::Data & /* pncData */,
             const Eigen::VectorXd & /* q */,
             const Eigen::VectorXd & /* v */,
             const Eigen::VectorXd & /* positionLimitMin */,
             const Eigen::VectorXd & /* positionLimitMax */,
             const std::unique_ptr<const EngineMultiRobot::engineOptions_t> & /* engineOptions */,
             contactModel_t contactModel,
             std::shared_ptr<AbstractConstraintBase> & constraint,
             Eigen::VectorXd & /* u */)
        {
            if (contactModel == contactModel_t::CONSTRAINT)
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
             const pinocchio::Data & /* pncData */,
             const Eigen::VectorXd & /* q */,
             const Eigen::VectorXd & /* v */,
             const Eigen::VectorXd & /* positionLimitMin */,
             const Eigen::VectorXd & /* positionLimitMax */,
             const std::unique_ptr<const EngineMultiRobot::engineOptions_t> & /* engineOptions */,
             contactModel_t contactModel,
             std::shared_ptr<AbstractConstraintBase> & constraint,
             Eigen::VectorXd & /* u */)
        {
            PRINT_WARNING("No position bounds implemented for this type of joint.");
            if (contactModel == contactModel_t::CONSTRAINT)
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
            const pinocchio::Data & /* pncData */,
            const Eigen::VectorXd & /* v */,
            const Eigen::VectorXd & /* velocityLimitMax */,
            const std::unique_ptr<const EngineMultiRobot::engineOptions_t> & /* engineOptions */,
            contactModel_t /* contactModel */,
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
             const pinocchio::Data & pncData,
             const Eigen::VectorXd & v,
             const Eigen::VectorXd & velocityLimitMax,
             const std::unique_ptr<const EngineMultiRobot::engineOptions_t> & engineOptions,
             contactModel_t contactModel,
             Eigen::VectorXd & u)
        {
            // Define some proxies for convenience
            const pinocchio::JointIndex jointModelIdx = joint.id();
            const Eigen::Index velocityIdx = joint.idx_v();
            const double vJoint = v[velocityIdx];
            const double vJointMin = -velocityLimitMax[velocityIdx];
            const double vJointMax = velocityLimitMax[velocityIdx];
            const double Ia = getSubtreeInertiaProj(joint.derived(), pncData.Ycrb[jointModelIdx]);
            const double damping = engineOptions->joints.boundDamping;

            // Check if out-of-bounds
            if (contactModel == contactModel_t::SPRING_DAMPER)
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
                u[velocityIdx] += Ia * accelJoint;
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
             const pinocchio::Data & /* pncData */,
             const Eigen::VectorXd & /* v */,
             const Eigen::VectorXd & /* velocityLimitMax */,
             const std::unique_ptr<const EngineMultiRobot::engineOptions_t> & /* engineOptions */,
             contactModel_t /* contactModel */,
             Eigen::VectorXd & /* u */)
        {
            PRINT_WARNING("No velocity bounds implemented for this type of joint.");
        }
    };

    void EngineMultiRobot::computeInternalDynamics(const systemHolder_t & system,
                                                   systemDataHolder_t & systemData,
                                                   double /* t */,
                                                   const Eigen::VectorXd & q,
                                                   const Eigen::VectorXd & v,
                                                   Eigen::VectorXd & uInternal) const
    {
        // Define some proxies
        const pinocchio::Model & pncModel = system.robot->pncModel_;
        const pinocchio::Data & pncData = system.robot->pncData_;

        // Enforce the position limit (rigid joints only)
        if (system.robot->mdlOptions_->joints.enablePositionLimit)
        {
            const Eigen::VectorXd & positionLimitMin = system.robot->getPositionLimitMin();
            const Eigen::VectorXd & positionLimitMax = system.robot->getPositionLimitMax();
            const std::vector<pinocchio::JointIndex> & rigidJointModelIndices =
                system.robot->getRigidJointsModelIdx();
            for (std::size_t i = 0; i < rigidJointModelIndices.size(); ++i)
            {
                auto & constraint = systemData.constraintsHolder.boundJoints[i].second;
                computePositionLimitsForcesAlgo::run(
                    pncModel.joints[rigidJointModelIndices[i]],
                    typename computePositionLimitsForcesAlgo::ArgsType(pncData,
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
        if (system.robot->mdlOptions_->joints.enableVelocityLimit)
        {
            const Eigen::VectorXd & velocityLimitMax = system.robot->getVelocityLimit();
            for (pinocchio::JointIndex rigidJointModelIdx : system.robot->getRigidJointsModelIdx())
            {
                computeVelocityLimitsForcesAlgo::run(
                    pncModel.joints[rigidJointModelIdx],
                    typename computeVelocityLimitsForcesAlgo::ArgsType(
                        pncData, v, velocityLimitMax, engineOptions_, contactModel_, uInternal));
            }
        }

        // Compute the flexibilities (only support JointModelType::SPHERICAL so far)
        double angle;
        Eigen::Matrix3d rotJlog3;
        const Robot::dynamicsOptions_t & mdlDynOptions = system.robot->mdlOptions_->dynamics;
        const std::vector<pinocchio::JointIndex> & flexibilityJointModelIndices =
            system.robot->getFlexibleJointsModelIdx();
        for (std::size_t i = 0; i < flexibilityJointModelIndices.size(); ++i)
        {
            const pinocchio::JointIndex jointModelIdx = flexibilityJointModelIndices[i];
            const Eigen::Index positionIdx = pncModel.joints[jointModelIdx].idx_q();
            const Eigen::Index velocityIdx = pncModel.joints[jointModelIdx].idx_v();
            const Eigen::Vector3d & stiffness = mdlDynOptions.flexibilityConfig[i].stiffness;
            const Eigen::Vector3d & damping = mdlDynOptions.flexibilityConfig[i].damping;

            const Eigen::Map<const Eigen::Quaterniond> quat(q.segment<4>(positionIdx).data());
            const Eigen::Vector3d angleAxis = pinocchio::quaternion::log3(quat, angle);
            assert((angle < 0.95 * M_PI) &&
                   "Flexible joint angle must be smaller than 0.95 * pi.");
            pinocchio::Jlog3(angle, angleAxis, rotJlog3);
            uInternal.segment<3>(velocityIdx) -=
                rotJlog3 * (stiffness.array() * angleAxis.array()).matrix();
            uInternal.segment<3>(velocityIdx).array() -=
                damping.array() * v.segment<3>(velocityIdx).array();
        }
    }

    void EngineMultiRobot::computeCollisionForces(const systemHolder_t & system,
                                                  systemDataHolder_t & systemData,
                                                  ForceVector & fext,
                                                  bool isStateUpToDate) const
    {
        // Compute the forces at contact points
        const std::vector<pinocchio::FrameIndex> & contactFramesIdx =
            system.robot->getContactFramesIdx();
        for (std::size_t i = 0; i < contactFramesIdx.size(); ++i)
        {
            // Compute force at the given contact frame.
            const pinocchio::FrameIndex frameIdx = contactFramesIdx[i];
            auto & constraint = systemData.constraintsHolder.contactFrames[i].second;
            pinocchio::Force & fextLocal = systemData.contactFramesForces[i];
            if (!isStateUpToDate)
            {
                computeContactDynamicsAtFrame(system, frameIdx, constraint, fextLocal);
            }

            // Apply the force at the origin of the parent joint frame, in local joint frame
            const pinocchio::JointIndex parentJointModelIdx =
                system.robot->pncModel_.frames[frameIdx].parent;
            fext[parentJointModelIdx] += fextLocal;

            // Convert contact force from global frame to local frame to store it in contactForces_
            const pinocchio::SE3 & transformContactInJoint =
                system.robot->pncModel_.frames[frameIdx].placement;
            system.robot->contactForces_[i] = transformContactInJoint.actInv(fextLocal);
        }

        // Compute the force at collision bodies
        const std::vector<pinocchio::FrameIndex> & collisionBodiesIdx =
            system.robot->getCollisionBodiesIdx();
        const std::vector<std::vector<pinocchio::PairIndex>> & collisionPairsIdx =
            system.robot->getCollisionPairsIdx();
        for (std::size_t i = 0; i < collisionBodiesIdx.size(); ++i)
        {
            /* Compute force at the given collision body.
               It returns the force applied at the origin of parent joint frame in global frame. */
            const pinocchio::FrameIndex frameIdx = collisionBodiesIdx[i];
            const pinocchio::JointIndex parentJointModelIdx =
                system.robot->pncModel_.frames[frameIdx].parent;
            for (std::size_t j = 0; j < collisionPairsIdx[i].size(); ++j)
            {
                pinocchio::Force & fextLocal = systemData.collisionBodiesForces[i][j];
                if (!isStateUpToDate)
                {
                    const pinocchio::PairIndex & collisionPairIdx = collisionPairsIdx[i][j];
                    auto & constraint = systemData.constraintsHolder.collisionBodies[i][j].second;
                    computeContactDynamicsAtBody(system, collisionPairIdx, constraint, fextLocal);
                }

                // Apply the force at the origin of the parent joint frame, in local joint frame
                fext[parentJointModelIdx] += fextLocal;
            }
        }
    }

    void EngineMultiRobot::computeExternalForces(const systemHolder_t & system,
                                                 systemDataHolder_t & systemData,
                                                 double t,
                                                 const Eigen::VectorXd & q,
                                                 const Eigen::VectorXd & v,
                                                 ForceVector & fext)
    {
        // Add the effect of user-defined external impulse forces
        auto forcesImpulseActiveIt = systemData.forcesImpulseActive.begin();
        auto forcesImpulseIt = systemData.forcesImpulse.begin();
        for (; forcesImpulseIt != systemData.forcesImpulse.end();
             ++forcesImpulseActiveIt, ++forcesImpulseIt)
        {
            /* Do not check if the force is active at this point. This is managed at stepper level
               to get around the ambiguous t- versus t+. */
            if (*forcesImpulseActiveIt)
            {
                const pinocchio::FrameIndex frameIdx = forcesImpulseIt->frameIdx;
                const pinocchio::JointIndex parentJointModelIdx =
                    system.robot->pncModel_.frames[frameIdx].parent;
                const pinocchio::Force & F = forcesImpulseIt->F;

                fext[parentJointModelIdx] += convertForceGlobalFrameToJoint(
                    system.robot->pncModel_, system.robot->pncData_, frameIdx, F);
            }
        }

        // Add the effect of time-continuous external force profiles
        for (auto & forceProfile : systemData.forcesProfile)
        {
            const pinocchio::FrameIndex frameIdx = forceProfile.frameIdx;
            const pinocchio::JointIndex parentJointModelIdx =
                system.robot->pncModel_.frames[frameIdx].parent;
            if (forceProfile.updatePeriod < EPS)
            {
                forceProfile.forcePrev = forceProfile.forceFct(t, q, v);
            }
            fext[parentJointModelIdx] += convertForceGlobalFrameToJoint(
                system.robot->pncModel_, system.robot->pncData_, frameIdx, forceProfile.forcePrev);
        }
    }

    void EngineMultiRobot::computeForcesCoupling(double t,
                                                 const std::vector<Eigen::VectorXd> & qSplit,
                                                 const std::vector<Eigen::VectorXd> & vSplit)
    {
        for (auto & forceCoupling : forcesCoupling_)
        {
            // Extract info about the first system involved
            const std::ptrdiff_t systemIdx1 = forceCoupling.systemIdx1;
            const systemHolder_t & system1 = systems_[systemIdx1];
            const Eigen::VectorXd & q1 = qSplit[systemIdx1];
            const Eigen::VectorXd & v1 = vSplit[systemIdx1];
            const pinocchio::FrameIndex frameIdx1 = forceCoupling.frameIdx1;
            ForceVector & fext1 = systemsDataHolder_[systemIdx1].state.fExternal;

            // Extract info about the second system involved
            const std::ptrdiff_t systemIdx2 = forceCoupling.systemIdx2;
            const systemHolder_t & system2 = systems_[systemIdx2];
            const Eigen::VectorXd & q2 = qSplit[systemIdx2];
            const Eigen::VectorXd & v2 = vSplit[systemIdx2];
            const pinocchio::FrameIndex frameIdx2 = forceCoupling.frameIdx2;
            ForceVector & fext2 = systemsDataHolder_[systemIdx2].state.fExternal;

            // Compute the coupling force
            pinocchio::Force force = forceCoupling.forceFct(t, q1, v1, q2, v2);
            const pinocchio::JointIndex parentJointModelIdx1 =
                system1.robot->pncModel_.frames[frameIdx1].parent;
            fext1[parentJointModelIdx1] += convertForceGlobalFrameToJoint(
                system1.robot->pncModel_, system1.robot->pncData_, frameIdx1, force);

            // Move force from frame1 to frame2 to apply it to the second system
            force.toVector() *= -1;
            const pinocchio::JointIndex parentJointModelIdx2 =
                system2.robot->pncModel_.frames[frameIdx2].parent;
            const Eigen::Vector3d offset = system2.robot->pncData_.oMf[frameIdx2].translation() -
                                           system1.robot->pncData_.oMf[frameIdx1].translation();
            force.angular() -= offset.cross(force.linear());
            fext2[parentJointModelIdx2] += convertForceGlobalFrameToJoint(
                system2.robot->pncModel_, system2.robot->pncData_, frameIdx2, force);
        }
    }

    void EngineMultiRobot::computeAllTerms(double t,
                                           const std::vector<Eigen::VectorXd> & qSplit,
                                           const std::vector<Eigen::VectorXd> & vSplit,
                                           bool isStateUpToDate)
    {
        // Reinitialize the external forces and internal efforts
        for (auto & systemData : systemsDataHolder_)
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
        computeForcesCoupling(t, qSplit, vSplit);

        // Compute each individual system dynamics
        auto systemIt = systems_.begin();
        auto systemDataIt = systemsDataHolder_.begin();
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

    hresult_t EngineMultiRobot::computeSystemsDynamics(double t,
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
            PRINT_ERROR("No simulation running. Please start it before calling this method.");
            return hresult_t::ERROR_INIT_FAILED;
        }

        // Make sure memory has been allocated for the output acceleration
        aSplit.resize(vSplit.size());

        if (!isStateUpToDate)
        {
            // Update kinematics for each system
            auto systemIt = systems_.begin();
            auto systemDataIt = systemsDataHolder_.begin();
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
        auto systemDataIt = systemsDataHolder_.begin();
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
                fPrevIt->swap(systemIt->robot->pncData_.f);
                aPrevIt->swap(systemIt->robot->pncData_.a);

                // Update sensors based on previous accelerations and forces
                systemIt->robot->setSensorsData(t, *qIt, *vIt, aPrev, uMotorPrev, fextPrev);

                // Restore current forces and accelerations
                contactForcesPrevIt->swap(systemIt->robot->contactForces_);
                fPrevIt->swap(systemIt->robot->pncData_.f);
                aPrevIt->swap(systemIt->robot->pncData_.a);
            }

            /* Update the controller command if necessary (only for infinite update frequency).
               Make sure that the sensor state has been updated beforehand. */
            if (engineOptions_->stepper.controllerUpdatePeriod < EPS)
            {
                computeCommand(*systemIt, t, *qIt, *vIt, command);
            }

            /* Compute the actual motor effort.
               Note that it is impossible to have access to the current accelerations. */
            systemIt->robot->computeMotorsEfforts(t, *qIt, *vIt, aPrev, command);
            uMotor = systemIt->robot->getMotorsEfforts();

            /* Compute the user-defined internal dynamics.
               Make sure that the sensor state has been updated beforehand since the user-defined
               internal dynamics may rely on it. */
            uCustom.setZero();
            systemIt->controller->internalDynamics(t, *qIt, *vIt, uCustom);

            // Compute the total effort vector
            u = uInternal + uCustom;
            for (const auto & motor : systemIt->robot->getMotors())
            {
                const std::size_t motorIdx = motor->getIdx();
                const Eigen::Index motorVelocityIdx = motor->getJointVelocityIdx();
                u[motorVelocityIdx] += uMotor[motorIdx];
            }

            // Compute the dynamics
            *aIt = computeAcceleration(
                *systemIt, *systemDataIt, *qIt, *vIt, u, fext, isStateUpToDate);
        }

        return hresult_t::SUCCESS;
    }

    const Eigen::VectorXd & EngineMultiRobot::computeAcceleration(systemHolder_t & system,
                                                                  systemDataHolder_t & systemData,
                                                                  const Eigen::VectorXd & q,
                                                                  const Eigen::VectorXd & v,
                                                                  const Eigen::VectorXd & u,
                                                                  ForceVector & fext,
                                                                  bool isStateUpToDate,
                                                                  bool ignoreBounds)
    {
        const pinocchio::Model & model = system.robot->pncModel_;
        pinocchio::Data & data = system.robot->pncData_;

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
                            model, data, i, pinocchio::LOCAL, systemData.jointsJacobians[i]);
                    }
                    data.u.noalias() +=
                        systemData.jointsJacobians[i].transpose() * fext[i].toVector();
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
            systemData.constraintsHolder.foreach(
                constraintsHolderType_t::BOUNDS_JOINTS,
                [&u = systemData.state.u,
                 &uInternal = systemData.state.uInternal,
                 &joints = const_cast<pinocchio::Model::JointModelVector &>(model.joints)](
                    std::shared_ptr<AbstractConstraintBase> & constraint,
                    constraintsHolderType_t /* holderType */)
                {
                    if (!constraint->getIsEnabled())
                    {
                        return;
                    }

                    Eigen::VectorXd & uJoint = constraint->lambda_;
                    const auto & jointConstraint =
                        static_cast<JointConstraint const &>(*constraint.get());
                    const auto & jointModel = joints[jointConstraint.getJointModelIdx()];
                    jointModel.jointVelocitySelector(uInternal) += uJoint;
                    jointModel.jointVelocitySelector(u) += uJoint;
                });

            auto constraintIt = systemData.constraintsHolder.contactFrames.begin();
            auto forceIt = system.robot->contactForces_.begin();
            for (; constraintIt != systemData.constraintsHolder.contactFrames.end();
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
                const pinocchio::FrameIndex frameIdx = frameConstraint.getFrameIdx();
                const auto rotationWorldInContact = data.oMf[frameIdx].rotation().transpose();
                forceIt->linear().noalias() = rotationWorldInContact * fextInWorld.linear();
                forceIt->angular().noalias() = rotationWorldInContact * fextInWorld.angular();

                // Convert the force from local world aligned to local parent joint
                pinocchio::JointIndex jointIdx = model.frames[frameIdx].parent;
                fext[jointIdx] +=
                    convertForceGlobalFrameToJoint(model, data, frameIdx, fextInWorld);
            }

            systemData.constraintsHolder.foreach(
                constraintsHolderType_t::COLLISION_BODIES,
                [&fext, &model, &data](std::shared_ptr<AbstractConstraintBase> & constraint,
                                       constraintsHolderType_t /* holderType */)
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
                    const pinocchio::FrameIndex frameIdx = frameConstraint.getFrameIdx();
                    const pinocchio::JointIndex jointIdx = model.frames[frameIdx].parent;
                    fext[jointIdx] +=
                        convertForceGlobalFrameToJoint(model, data, frameIdx, fextInWorld);
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

    hresult_t EngineMultiRobot::getLog(std::shared_ptr<const LogData> & logData)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        // Update internal log data buffer if uninitialized
        if (!logData_)
        {
            logData_ = std::make_shared<LogData>();
            returnCode = telemetryRecorder_->getLog(*logData_);
        }

        // Return shared pointer to internal log data buffer
        logData = std::const_pointer_cast<const LogData>(logData_);

        return returnCode;
    }

    hresult_t readLogHdf5(const std::string & filename, LogData & logData)
    {
        // Clear log data if any
        logData = {};

        // Open HDF5 logfile
        std::unique_ptr<H5::H5File> file;
        try
        {
            /* Specifying `H5F_CLOSE_STRONG` must be specified to completely close the file
               (including all open objects) before returning. See:
               https://docs.hdfgroup.org/hdf5/v1_12/group___f_a_p_l.html#ga60e3567f677fd3ade75b909b636d7b9c
            */
            H5::FileAccPropList access_plist;
            access_plist.setFcloseDegree(H5F_CLOSE_STRONG);
            file = std::make_unique<H5::H5File>(
                filename, H5F_ACC_RDONLY, H5::FileCreatPropList::DEFAULT, access_plist);
        }
        catch (const H5::FileIException & open_file)
        {
            PRINT_ERROR("Impossible to open the log file. Make sure it exists and you have "
                        "reading permissions.");
            return hresult_t::ERROR_BAD_INPUT;
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
                auto & [_logData, _intVector, _floatVector] = *static_cast<opDataT *>(op_data);
                const Eigen::Index varIdx = _logData.variableNames.size() - 1;
                const int64_t _numInt = _logData.integerValues.rows();
                H5::Group fieldGroup = H5::Group(group).openGroup(name);
                const H5::DataSet valueDataset = fieldGroup.openDataSet("value");
                if (varIdx < _numInt)
                {
                    valueDataset.read(_intVector.data(), H5::PredType::NATIVE_INT64);
                    _logData.integerValues.row(varIdx) = _intVector;
                }
                else
                {
                    valueDataset.read(_floatVector.data(), H5::PredType::NATIVE_DOUBLE);
                    _logData.floatValues.row(varIdx - _numInt) = _floatVector;
                }
                _logData.variableNames.push_back(name);
                return 0;
            },
            static_cast<void *>(&opData));

        // Close file once done
        file->close();

        return hresult_t::SUCCESS;
    }

    hresult_t EngineMultiRobot::readLog(
        const std::string & filename, const std::string & format, LogData & logData)
    {
        if (format == "binary")
        {
            return TelemetryRecorder::readLog(filename, logData);
        }
        else if (format == "hdf5")
        {
            return readLogHdf5(filename, logData);
        }

        PRINT_ERROR("Format '", format, "' not recognized. It must be either 'binary' or 'hdf5'.");
        return hresult_t::ERROR_BAD_INPUT;
    }

    hresult_t writeLogHdf5(const std::string & filename,
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
            PRINT_ERROR("Impossible to create the log file. Make sure the root folder exists and "
                        "you have writing permissions.");
            return hresult_t::ERROR_BAD_INPUT;
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

        return hresult_t::SUCCESS;
    }

    hresult_t EngineMultiRobot::writeLog(const std::string & filename, const std::string & format)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        // Make sure there is something to write
        if (!isTelemetryConfigured_)
        {
            PRINT_ERROR("Telemetry not configured. Please start a simulation before writing log.");
            returnCode = hresult_t::ERROR_BAD_INPUT;
        }

        // Pick the appropriate format
        if (returnCode == hresult_t::SUCCESS)
        {
            if (format == "binary")
            {
                returnCode = telemetryRecorder_->writeLog(filename);
            }
            else if (format == "hdf5")
            {
                // Extract log data
                std::shared_ptr<const LogData> logData;
                returnCode = getLog(logData);

                // Write log data
                if (returnCode == hresult_t::SUCCESS)
                {
                    returnCode = writeLogHdf5(filename, logData);
                }
            }
            else
            {
                PRINT_ERROR(
                    "Format '", format, "' not recognized. It must be either 'binary' or 'hdf5'.");
                returnCode = hresult_t::ERROR_BAD_INPUT;
            }
        }

        return returnCode;
    }
}
