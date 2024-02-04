#include "jiminy/core/robot/robot.h"
#include "jiminy/core/control/abstract_controller.h"

#include "jiminy/core/engine/engine.h"


namespace jiminy
{
    hresult_t Engine::initializeImpl(std::shared_ptr<Robot> robot,
                                     std::shared_ptr<AbstractController> controller,
                                     const AbortSimulationFunction & callback)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        // Make sure the simulation is properly stopped
        if (isSimulationRunning_)
        {
            stop();
        }

        // Remove the existing system if already initialized
        if (isInitialized_)
        {
            EngineMultiRobot::removeSystem("");  // Cannot fail at this point
            robot_ = nullptr;
            controller_ = nullptr;
            isInitialized_ = false;
        }

        // Add the system with empty name since it is irrelevant for a single robot engine
        if (controller)
        {
            returnCode = EngineMultiRobot::addSystem("", robot, controller, callback);
        }
        else
        {
            returnCode = EngineMultiRobot::addSystem("", robot, callback);
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            // Get some convenience proxies
            robot_ = systems_.begin()->robot.get();
            controller_ = systems_.begin()->controller.get();

            // Set the initialization flag
            isInitialized_ = true;
        }

        return returnCode;
    }

    hresult_t Engine::initialize(std::shared_ptr<Robot> robot,
                                 std::shared_ptr<AbstractController> controller,
                                 const AbortSimulationFunction & callback)
    {
        return initializeImpl(robot, controller, callback);
    }

    hresult_t Engine::initialize(std::shared_ptr<Robot> robot,
                                 const AbortSimulationFunction & callback)
    {
        return initializeImpl(robot, std::shared_ptr<AbstractController>(), callback);
    }

    hresult_t Engine::setController(std::shared_ptr<AbstractController> controller)
    {
        return EngineMultiRobot::setController("", controller);
    }

    hresult_t Engine::addSystem(const std::string & /* systemName */,
                                std::shared_ptr<Robot> /* robot */,
                                std::shared_ptr<AbstractController> /* controller */)
    {
        PRINT_ERROR("This method is not supported by this class. Please call "
                    "`initialize` instead to set the model, or use `EngineMultiRobot` "
                    "class directly to simulate multiple robots simultaneously.");
        return hresult_t::ERROR_GENERIC;
    }

    hresult_t Engine::removeSystem(const std::string & /* systemName */)
    {
        PRINT_ERROR("This method is not supported by this class. Please call "
                    "`initialize` instead to set the model, or use `EngineMultiRobot` "
                    "class directly to simulate multiple robots simultaneously.");
        return hresult_t::ERROR_GENERIC;
    }

    hresult_t singleToMultipleSystemsInitialData(
        const Robot & robot,
        bool isStateTheoretical,
        const Eigen::VectorXd & qInit,
        const Eigen::VectorXd & vInit,
        const std::optional<Eigen::VectorXd> & aInit,
        std::map<std::string, Eigen::VectorXd> & qInitList,
        std::map<std::string, Eigen::VectorXd> & vInitList,
        std::optional<std::map<std::string, Eigen::VectorXd>> & aInitList)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (isStateTheoretical && robot.modelOptions_->dynamics.enableFlexibleModel)
        {
            Eigen::VectorXd q0;
            returnCode = robot.getFlexiblePositionFromRigid(qInit, q0);
            qInitList.emplace("", std::move(q0));
        }
        else
        {
            qInitList.emplace("", qInit);
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            if (isStateTheoretical && robot.modelOptions_->dynamics.enableFlexibleModel)
            {
                Eigen::VectorXd v0;
                returnCode = robot.getFlexibleVelocityFromRigid(vInit, v0);
                vInitList.emplace("", std::move(v0));
            }
            else
            {
                vInitList.emplace("", vInit);
            }
        }
        if (returnCode == hresult_t::SUCCESS)
        {
            if (aInit)
            {
                aInitList.emplace();
                if (isStateTheoretical && robot.modelOptions_->dynamics.enableFlexibleModel)
                {
                    Eigen::VectorXd a0;
                    returnCode = robot.getFlexibleVelocityFromRigid(*aInit, a0);
                    aInitList->emplace("", std::move(a0));
                }
                else
                {
                    aInitList->emplace("", *aInit);
                }
            }
        }

        return returnCode;
    }

    hresult_t Engine::start(const Eigen::VectorXd & qInit,
                            const Eigen::VectorXd & vInit,
                            const std::optional<Eigen::VectorXd> & aInit,
                            bool isStateTheoretical)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (!isInitialized_)
        {
            PRINT_ERROR("The engine is not initialized.");
            returnCode = hresult_t::ERROR_INIT_FAILED;
        }

        std::map<std::string, Eigen::VectorXd> qInitList;
        std::map<std::string, Eigen::VectorXd> vInitList;
        std::optional<std::map<std::string, Eigen::VectorXd>> aInitList = std::nullopt;
        if (returnCode == hresult_t::SUCCESS)
        {
            returnCode = singleToMultipleSystemsInitialData(
                *robot_, isStateTheoretical, qInit, vInit, aInit, qInitList, vInitList, aInitList);
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            returnCode = EngineMultiRobot::start(qInitList, vInitList, aInitList);
        }

        return returnCode;
    }

    hresult_t Engine::simulate(double tEnd,
                               const Eigen::VectorXd & qInit,
                               const Eigen::VectorXd & vInit,
                               const std::optional<Eigen::VectorXd> & aInit,
                               bool isStateTheoretical)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (!isInitialized_)
        {
            PRINT_ERROR("The engine is not initialized.");
            returnCode = hresult_t::ERROR_INIT_FAILED;
        }

        std::map<std::string, Eigen::VectorXd> qInitList;
        std::map<std::string, Eigen::VectorXd> vInitList;
        std::optional<std::map<std::string, Eigen::VectorXd>> aInitList = std::nullopt;
        if (returnCode == hresult_t::SUCCESS)
        {
            returnCode = singleToMultipleSystemsInitialData(
                *robot_, isStateTheoretical, qInit, vInit, aInit, qInitList, vInitList, aInitList);
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            returnCode = EngineMultiRobot::simulate(tEnd, qInitList, vInitList, aInitList);
        }

        return returnCode;
    }

    hresult_t Engine::registerImpulseForce(
        const std::string & frameName, double t, double dt, const pinocchio::Force & force)
    {
        return EngineMultiRobot::registerImpulseForce("", frameName, t, dt, force);
    }

    hresult_t Engine::registerProfileForce(
        const std::string & frameName, const ProfileForceFunction & forceFunc, double updatePeriod)
    {
        return EngineMultiRobot::registerProfileForce("", frameName, forceFunc, updatePeriod);
    }

    hresult_t Engine::removeImpulseForces()
    {
        return EngineMultiRobot::removeImpulseForces("");
    }

    hresult_t Engine::removeProfileForces()
    {
        return EngineMultiRobot::removeProfileForces("");
    }

    const ImpulseForceVector & Engine::getImpulseForces() const
    {
        const ImpulseForceVector * impulseForces;
        EngineMultiRobot::getImpulseForces("", impulseForces);
        return *impulseForces;
    }

    const ProfileForceVector & Engine::getProfileForces() const
    {
        const ProfileForceVector * profileForces;
        EngineMultiRobot::getProfileForces("", profileForces);
        return *profileForces;
    }

    hresult_t Engine::registerCouplingForce(const std::string & frameName1,
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

    hresult_t Engine::registerViscoelasticCouplingForce(const std::string & frameName1,
                                                        const std::string & frameName2,
                                                        const Vector6d & stiffness,
                                                        const Vector6d & damping,
                                                        double alpha)
    {
        return EngineMultiRobot::registerViscoelasticCouplingForce(
            "", "", frameName1, frameName2, stiffness, damping, alpha);
    }

    hresult_t Engine::registerViscoelasticDirectionalCouplingForce(const std::string & frameName1,
                                                                   const std::string & frameName2,
                                                                   double stiffness,
                                                                   double damping,
                                                                   double restLength)
    {
        return EngineMultiRobot::registerViscoelasticDirectionalCouplingForce(
            "", "", frameName1, frameName2, stiffness, damping, restLength);
    }

    hresult_t Engine::removeCouplingForces()
    {
        return EngineMultiRobot::removeCouplingForces("");
    }

    bool Engine::getIsInitialized() const
    {
        return isInitialized_;
    }

    hresult_t Engine::getSystem(System *& system)
    {
        static System systemEmpty;

        hresult_t returnCode = hresult_t::SUCCESS;

        if (!isInitialized_)
        {
            PRINT_ERROR("The engine is not initialized.");
            returnCode = hresult_t::ERROR_BAD_INPUT;
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            system = &(*systems_.begin());
            return returnCode;
        }

        system = &systemEmpty;
        return returnCode;
    }

    hresult_t Engine::getRobot(std::shared_ptr<Robot> & robot)
    {
        System * system;

        hresult_t returnCode = hresult_t::SUCCESS;

        returnCode = getSystem(system);
        robot = system->robot;

        return returnCode;
    }

    hresult_t Engine::getController(std::shared_ptr<AbstractController> & controller)
    {
        System * system;

        hresult_t returnCode = hresult_t::SUCCESS;

        returnCode = getSystem(system);
        controller = system->controller;

        return returnCode;
    }

    hresult_t Engine::getSystemState(const SystemState *& systemState) const
    {
        static const SystemState systemStateEmpty;

        hresult_t returnCode = hresult_t::SUCCESS;

        if (!isInitialized_)
        {
            PRINT_ERROR("The engine is not initialized.");
            returnCode = hresult_t::ERROR_BAD_INPUT;
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            EngineMultiRobot::getSystemState("", systemState);  // Cannot fail at this point
            return returnCode;
        }

        systemState = &systemStateEmpty;
        return returnCode;
    }
}
