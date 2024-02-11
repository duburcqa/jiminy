#include "jiminy/core/robot/robot.h"
#include "jiminy/core/control/abstract_controller.h"

#include "jiminy/core/engine/engine.h"


namespace jiminy
{
    void Engine::initializeImpl(std::shared_ptr<Robot> robot,
                                std::shared_ptr<AbstractController> controller,
                                const AbortSimulationFunction & callback)
    {
        // Make sure the simulation is properly stopped
        if (isSimulationRunning_)
        {
            stop();
        }

        // Remove the existing system if already initialized
        if (isInitialized_)
        {
            EngineMultiRobot::removeSystem("");
            robot_ = nullptr;
            controller_ = nullptr;
            isInitialized_ = false;
        }

        // Add the system with empty name since it is irrelevant for a single robot engine
        if (controller)
        {
            EngineMultiRobot::addSystem("", robot, controller, callback);
        }
        else
        {
            EngineMultiRobot::addSystem("", robot, callback);
        }

        // Get some convenience proxies
        robot_ = systems_.begin()->robot.get();
        controller_ = systems_.begin()->controller.get();

        // Set the initialization flag
        isInitialized_ = true;
    }

    void Engine::initialize(std::shared_ptr<Robot> robot,
                            std::shared_ptr<AbstractController> controller,
                            const AbortSimulationFunction & callback)
    {
        return initializeImpl(robot, controller, callback);
    }

    void Engine::initialize(std::shared_ptr<Robot> robot, const AbortSimulationFunction & callback)
    {
        return initializeImpl(robot, std::shared_ptr<AbstractController>(), callback);
    }

    void Engine::setController(std::shared_ptr<AbstractController> controller)
    {
        return EngineMultiRobot::setController("", controller);
    }

    void Engine::addSystem(const std::string & /* systemName */,
                           std::shared_ptr<Robot> /* robot */,
                           std::shared_ptr<AbstractController> /* controller */)
    {
        THROW_ERROR(
            not_implemented_error,
            "This method is not supported by this class. Please call `initialize` instead to set "
            "the model, or use `EngineMultiRobot` class directly to simulate multiple robots "
            "simultaneously.");
    }

    void Engine::removeSystem(const std::string & /* systemName */)
    {
        THROW_ERROR(
            not_implemented_error,
            "This method is not supported by this class. Please call `initialize` instead to set "
            "the model, or use `EngineMultiRobot` class directly to simulate multiple robots "
            "simultaneously.");
    }

    void singleToMultipleSystemsInitialData(
        const Robot & robot,
        bool isStateTheoretical,
        const Eigen::VectorXd & qInit,
        const Eigen::VectorXd & vInit,
        const std::optional<Eigen::VectorXd> & aInit,
        std::map<std::string, Eigen::VectorXd> & qInitList,
        std::map<std::string, Eigen::VectorXd> & vInitList,
        std::optional<std::map<std::string, Eigen::VectorXd>> & aInitList)
    {
        if (isStateTheoretical && robot.modelOptions_->dynamics.enableFlexibleModel)
        {
            Eigen::VectorXd q0;
            robot.getFlexiblePositionFromRigid(qInit, q0);
            qInitList.emplace("", std::move(q0));
        }
        else
        {
            qInitList.emplace("", qInit);
        }

        if (isStateTheoretical && robot.modelOptions_->dynamics.enableFlexibleModel)
        {
            Eigen::VectorXd v0;
            robot.getFlexibleVelocityFromRigid(vInit, v0);
            vInitList.emplace("", std::move(v0));
        }
        else
        {
            vInitList.emplace("", vInit);
        }

        if (aInit)
        {
            aInitList.emplace();
            if (isStateTheoretical && robot.modelOptions_->dynamics.enableFlexibleModel)
            {
                Eigen::VectorXd a0;
                robot.getFlexibleVelocityFromRigid(*aInit, a0);
                aInitList->emplace("", std::move(a0));
            }
            else
            {
                aInitList->emplace("", *aInit);
            }
        }
    }

    void Engine::start(const Eigen::VectorXd & qInit,
                       const Eigen::VectorXd & vInit,
                       const std::optional<Eigen::VectorXd> & aInit,
                       bool isStateTheoretical)
    {
        if (!isInitialized_)
        {
            THROW_ERROR(bad_control_flow, "Engine not initialized.");
        }

        std::map<std::string, Eigen::VectorXd> qInitList;
        std::map<std::string, Eigen::VectorXd> vInitList;
        std::optional<std::map<std::string, Eigen::VectorXd>> aInitList = std::nullopt;
        singleToMultipleSystemsInitialData(
            *robot_, isStateTheoretical, qInit, vInit, aInit, qInitList, vInitList, aInitList);

        EngineMultiRobot::start(qInitList, vInitList, aInitList);
    }

    void Engine::simulate(double tEnd,
                          const Eigen::VectorXd & qInit,
                          const Eigen::VectorXd & vInit,
                          const std::optional<Eigen::VectorXd> & aInit,
                          bool isStateTheoretical)
    {
        if (!isInitialized_)
        {
            THROW_ERROR(bad_control_flow, "Engine not initialized.");
        }

        std::map<std::string, Eigen::VectorXd> qInitList;
        std::map<std::string, Eigen::VectorXd> vInitList;
        std::optional<std::map<std::string, Eigen::VectorXd>> aInitList = std::nullopt;
        singleToMultipleSystemsInitialData(
            *robot_, isStateTheoretical, qInit, vInit, aInit, qInitList, vInitList, aInitList);

        EngineMultiRobot::simulate(tEnd, qInitList, vInitList, aInitList);
    }

    void Engine::registerImpulseForce(
        const std::string & frameName, double t, double dt, const pinocchio::Force & force)
    {
        return EngineMultiRobot::registerImpulseForce("", frameName, t, dt, force);
    }

    void Engine::registerProfileForce(
        const std::string & frameName, const ProfileForceFunction & forceFunc, double updatePeriod)
    {
        return EngineMultiRobot::registerProfileForce("", frameName, forceFunc, updatePeriod);
    }

    void Engine::removeImpulseForces()
    {
        return EngineMultiRobot::removeImpulseForces("");
    }

    void Engine::removeProfileForces()
    {
        return EngineMultiRobot::removeProfileForces("");
    }

    const ImpulseForceVector & Engine::getImpulseForces() const
    {
        return EngineMultiRobot::getImpulseForces("");
    }

    const ProfileForceVector & Engine::getProfileForces() const
    {
        return EngineMultiRobot::getProfileForces("");
    }

    void Engine::registerCouplingForce(const std::string & frameName1,
                                       const std::string & frameName2,
                                       const ProfileForceFunction & forceFunc)
    {
        auto couplingForceFun = [forceFunc](double t,
                                            const Eigen::VectorXd & q1,
                                            const Eigen::VectorXd & v1,
                                            const Eigen::VectorXd & /* q2 */,
                                            const Eigen::VectorXd & /* v2 */)
        {
            return forceFunc(t, q1, v1);
        };
        return EngineMultiRobot::registerCouplingForce(
            "", "", frameName1, frameName2, couplingForceFun);
    }

    void Engine::registerViscoelasticCouplingForce(const std::string & frameName1,
                                                   const std::string & frameName2,
                                                   const Vector6d & stiffness,
                                                   const Vector6d & damping,
                                                   double alpha)
    {
        return EngineMultiRobot::registerViscoelasticCouplingForce(
            "", "", frameName1, frameName2, stiffness, damping, alpha);
    }

    void Engine::registerViscoelasticDirectionalCouplingForce(const std::string & frameName1,
                                                              const std::string & frameName2,
                                                              double stiffness,
                                                              double damping,
                                                              double restLength)
    {
        return EngineMultiRobot::registerViscoelasticDirectionalCouplingForce(
            "", "", frameName1, frameName2, stiffness, damping, restLength);
    }

    void Engine::removeCouplingForces()
    {
        return EngineMultiRobot::removeCouplingForces("");
    }

    bool Engine::getIsInitialized() const
    {
        return isInitialized_;
    }

    System & Engine::getSystem()
    {
        if (!isInitialized_)
        {
            THROW_ERROR(bad_control_flow, "Engine not initialized.");
        }

        return *systems_.begin();
    }

    std::shared_ptr<Robot> Engine::getRobot()
    {
        return getSystem().robot;
    }

    std::shared_ptr<AbstractController> Engine::getController()
    {
        return getSystem().controller;
    }

    const SystemState & Engine::getSystemState() const
    {
        if (!isInitialized_)
        {
            THROW_ERROR(bad_control_flow, "Engine not initialized.");
        }
        return EngineMultiRobot::getSystemState("");
    }
}
