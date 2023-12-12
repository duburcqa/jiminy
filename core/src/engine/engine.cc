#include "jiminy/core/exceptions.h"
#include "jiminy/core/robot/robot.h"
#include "jiminy/core/control/abstract_controller.h"

#include "jiminy/core/engine/engine.h"


namespace jiminy
{
    hresult_t Engine::initializeImpl(std::shared_ptr<Robot> robot,
                                     std::shared_ptr<AbstractController> controller,
                                     CallbackFunctor callbackFct)
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
            returnCode =
                EngineMultiRobot::addSystem("", robot, controller, std::move(callbackFct));
        }
        else
        {
            returnCode = EngineMultiRobot::addSystem("", robot, std::move(callbackFct));
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
                                 CallbackFunctor callbackFct)
    {
        return initializeImpl(robot, controller, callbackFct);
    }

    hresult_t Engine::initialize(std::shared_ptr<Robot> robot, CallbackFunctor callbackFct)
    {
        return initializeImpl(robot, std::shared_ptr<AbstractController>(), callbackFct);
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
        const bool_t & isStateTheoretical,
        const Eigen::VectorXd & qInit,
        const Eigen::VectorXd & vInit,
        const std::optional<Eigen::VectorXd> & aInit,
        std::map<std::string, Eigen::VectorXd> & qInitList,
        std::map<std::string, Eigen::VectorXd> & vInitList,
        std::optional<std::map<std::string, Eigen::VectorXd>> & aInitList)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (isStateTheoretical && robot.mdlOptions_->dynamics.enableFlexibleModel)
        {
            Eigen::VectorXd q0;
            returnCode = robot.getFlexibleConfigurationFromRigid(qInit, q0);
            qInitList.emplace("", std::move(q0));
        }
        else
        {
            qInitList.emplace("", qInit);
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            if (isStateTheoretical && robot.mdlOptions_->dynamics.enableFlexibleModel)
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
                if (isStateTheoretical && robot.mdlOptions_->dynamics.enableFlexibleModel)
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
                            const bool_t & isStateTheoretical)
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

    hresult_t Engine::simulate(const float64_t & tEnd,
                               const Eigen::VectorXd & qInit,
                               const Eigen::VectorXd & vInit,
                               const std::optional<Eigen::VectorXd> & aInit,
                               const bool_t & isStateTheoretical)
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

    hresult_t Engine::registerForceImpulse(const std::string & frameName,
                                           const float64_t & t,
                                           const float64_t & dt,
                                           const pinocchio::Force & F)
    {
        return EngineMultiRobot::registerForceImpulse("", frameName, t, dt, F);
    }

    hresult_t Engine::registerForceProfile(const std::string & frameName,
                                           const ForceProfileFunctor & forceFct,
                                           const float64_t & updatePeriod)
    {
        return EngineMultiRobot::registerForceProfile("", frameName, forceFct, updatePeriod);
    }

    hresult_t Engine::removeForcesImpulse()
    {
        return EngineMultiRobot::removeForcesImpulse("");
    }

    hresult_t Engine::removeForcesProfile()
    {
        return EngineMultiRobot::removeForcesProfile("");
    }

    const ForceImpulseRegister & Engine::getForcesImpulse() const
    {
        const ForceImpulseRegister * forcesImpulse;
        EngineMultiRobot::getForcesImpulse("", forcesImpulse);
        return *forcesImpulse;
    }

    const ForceProfileRegister & Engine::getForcesProfile() const
    {
        const ForceProfileRegister * forcesProfile;
        EngineMultiRobot::getForcesProfile("", forcesProfile);
        return *forcesProfile;
    }

    hresult_t Engine::registerForceCoupling(const std::string & frameName1,
                                            const std::string & frameName2,
                                            ForceProfileFunctor forceFct)
    {
        auto forceCouplingFct = [forceFct](const float64_t & t,
                                           const Eigen::VectorXd & q1,
                                           const Eigen::VectorXd & v1,
                                           const Eigen::VectorXd & /* q2 */,
                                           const Eigen::VectorXd & /* v2 */)
        {
            return forceFct(t, q1, v1);
        };
        return EngineMultiRobot::registerForceCoupling(
            "", "", frameName1, frameName2, forceCouplingFct);
    }

    hresult_t Engine::registerViscoelasticForceCoupling(const std::string & frameName1,
                                                        const std::string & frameName2,
                                                        const Vector6d & stiffness,
                                                        const Vector6d & damping,
                                                        const float64_t & alpha)
    {
        return EngineMultiRobot::registerViscoelasticForceCoupling(
            "", "", frameName1, frameName2, stiffness, damping, alpha);
    }

    hresult_t Engine::registerViscoelasticDirectionalForceCoupling(const std::string & frameName1,
                                                                   const std::string & frameName2,
                                                                   const float64_t & stiffness,
                                                                   const float64_t & damping,
                                                                   const float64_t & restLength)
    {
        return EngineMultiRobot::registerViscoelasticDirectionalForceCoupling(
            "", "", frameName1, frameName2, stiffness, damping, restLength);
    }

    hresult_t Engine::removeForcesCoupling()
    {
        return EngineMultiRobot::removeForcesCoupling("");
    }

    const bool_t & Engine::getIsInitialized() const
    {
        return isInitialized_;
    }

    hresult_t Engine::getSystem(systemHolder_t *& system)
    {
        static systemHolder_t systemEmpty;

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
        systemHolder_t * system;

        hresult_t returnCode = hresult_t::SUCCESS;

        returnCode = getSystem(system);
        robot = system->robot;

        return returnCode;
    }

    hresult_t Engine::getController(std::shared_ptr<AbstractController> & controller)
    {
        systemHolder_t * system;

        hresult_t returnCode = hresult_t::SUCCESS;

        returnCode = getSystem(system);
        controller = system->controller;

        return returnCode;
    }

    hresult_t Engine::getSystemState(const systemState_t *& systemState) const
    {
        static const systemState_t systemStateEmpty;

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
