#include <iostream>

#include "jiminy/core/robot/Robot.h"
#include "jiminy/core/control/AbstractController.h"

#include "jiminy/core/engine/Engine.h"


namespace jiminy
{
    Engine::Engine(void):
    EngineMultiRobot(),
    isInitialized_(false),
    robot_(nullptr),
    controller_(nullptr)
    {
        // Empty on purpose.
    }

    hresult_t Engine::initialize(std::shared_ptr<Robot>              robot,
                                 std::shared_ptr<AbstractController> controller,
                                 callbackFunctor_t                   callbackFct)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        /* Add the system without associated name, since
           it is irrelevant for a single robot engine. */
        returnCode = addSystem("", std::move(robot),
                               std::move(controller),
                               std::move(callbackFct));

        if (returnCode == hresult_t::SUCCESS)
        {
            // Get some convenience proxies
            robot_ = systemsDataHolder_.begin()->robot.get();
            controller_ = systemsDataHolder_.begin()->controller.get();

            // Set the initialization flag
            isInitialized_ = true;
        }

        return returnCode;
    }

    hresult_t Engine::initialize(std::shared_ptr<Robot> robot,
                                 callbackFunctor_t      callbackFct)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        /* Add the system without associated name, since
           it is irrelevant for a single robot engine. */
        returnCode = addSystem("", std::move(robot), std::move(callbackFct));

        if (returnCode == hresult_t::SUCCESS)
        {
            // Get some convenience proxies
            robot_ = systemsDataHolder_.begin()->robot.get();
            controller_ = systemsDataHolder_.begin()->controller.get();

            // Set the initialization flag
            isInitialized_ = true;
        }

        return returnCode;
    }

    hresult_t Engine::setController(std::shared_ptr<AbstractController> controller)
    {
        return setController("", controller);
    }

    hresult_t Engine::start(vectorN_t const & xInit,
                            bool_t    const & isStateTheoretical,
                            bool_t    const & resetRandomNumbers,
                            bool_t    const & resetDynamicForceRegister)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (!isInitialized_)
        {
            std::cout << "Error - Engine::start - The engine is not initialized." << std::endl;
            returnCode = hresult_t::ERROR_INIT_FAILED;
        }

        std::map<std::string, vectorN_t> xInitList;
        if (returnCode == hresult_t::SUCCESS)
        {
            if (isStateTheoretical && robot_->mdlOptions_->dynamics.enableFlexibleModel)
            {
                vectorN_t x0;
                returnCode = robot_->getFlexibleStateFromRigid(xInit, x0);
                xInitList.emplace("", std::move(x0));
            }
            else
            {
                xInitList.emplace("", xInit);
            }
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            returnCode = EngineMultiRobot::start(
                xInitList, resetRandomNumbers, resetDynamicForceRegister);
        }

        return returnCode;
    }

    hresult_t Engine::simulate(float64_t const & tEnd,
                               vectorN_t const & xInit,
                               bool_t    const & isStateTheoretical)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (!isInitialized_)
        {
            std::cout << "Error - Engine::simulate - The engine is not initialized." << std::endl;
            returnCode = hresult_t::ERROR_INIT_FAILED;
        }

        std::map<std::string, vectorN_t> xInitList;
        if (returnCode == hresult_t::SUCCESS)
        {
            if (isStateTheoretical && robot_->mdlOptions_->dynamics.enableFlexibleModel)
            {
                vectorN_t x0;
                returnCode = robot_->getFlexibleStateFromRigid(xInit, x0);
                xInitList.emplace("", std::move(x0));
            }
            else
            {
                xInitList.emplace("", xInit);
            }
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            returnCode = EngineMultiRobot::simulate(tEnd, xInitList);
        }

        return returnCode;
    }

    hresult_t Engine::registerForceImpulse(std::string      const & frameName,
                                           float64_t        const & t,
                                           float64_t        const & dt,
                                           pinocchio::Force const & F)
    {
        return EngineMultiRobot::registerForceImpulse("", frameName, t, dt, F);
    }

    hresult_t Engine::registerForceProfile(std::string           const & frameName,
                                           forceProfileFunctor_t         forceFct)
    {
        return EngineMultiRobot::registerForceProfile("", frameName, forceFct);
    }

    bool_t const & Engine::getIsInitialized(void) const
    {
        return isInitialized_;
    }

    hresult_t Engine::getRobot(std::shared_ptr<Robot> & robot)
    {
        if (!isInitialized_)
        {
            std::cout << "Error - Engine::getRobot - The engine is not initialized." << std::endl;
            return hresult_t::ERROR_BAD_INPUT;
        }

        robot = systemsDataHolder_.begin()->robot;

        return hresult_t::SUCCESS;
    }

    hresult_t Engine::getController(std::shared_ptr<AbstractController> & controller)
    {
        if (!isInitialized_)
        {
            std::cout << "Error - Engine::getRobot - The engine is not initialized." << std::endl;
            return hresult_t::ERROR_BAD_INPUT;
        }

        controller = systemsDataHolder_.begin()->controller;

        return hresult_t::SUCCESS;
    }

    hresult_t Engine::getSystemState(systemState_t const * & systemState) const
    {
        static systemState_t const systemStateEmpty;

        hresult_t returnCode = hresult_t::SUCCESS;

        if (!isInitialized_)
        {
            std::cout << "Error - Engine::getRobot - The engine is not initialized." << std::endl;
            returnCode = hresult_t::ERROR_BAD_INPUT;
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            EngineMultiRobot::getSystemState("", systemState);  // It cannot fail at this point
            return returnCode;
        }

        systemState = &systemStateEmpty;
        return returnCode;
    }
}
