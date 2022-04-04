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

    hresult_t Engine::initializeImpl(std::shared_ptr<Robot>              robot,
                                     std::shared_ptr<AbstractController> controller,
                                     callbackFunctor_t                   callbackFct)
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
            EngineMultiRobot::removeSystem("");  // It cannot fail at this point
            robot_ = nullptr;
            controller_ = nullptr;
            isInitialized_ = false;
        }

        /* Add the system without associated name, since
           it is irrelevant for a single robot engine. */
        if (controller)
        {
            returnCode = EngineMultiRobot::addSystem(
                "", robot, controller, std::move(callbackFct));
        }
        else
        {
            returnCode = EngineMultiRobot::addSystem(
                "", robot, std::move(callbackFct));
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

    hresult_t Engine::initialize(std::shared_ptr<Robot>              robot,
                                 std::shared_ptr<AbstractController> controller,
                                 callbackFunctor_t                   callbackFct)
    {
        return initializeImpl(robot, controller, callbackFct);
    }

    hresult_t Engine::initialize(std::shared_ptr<Robot> robot,
                                 callbackFunctor_t      callbackFct)
    {
        return initializeImpl(robot, std::shared_ptr<AbstractController>(), callbackFct);
    }

    hresult_t Engine::setController(std::shared_ptr<AbstractController> controller)
    {
        return EngineMultiRobot::setController("", controller);
    }

    hresult_t Engine::addSystem(std::string const & /* systemName */,
                                std::shared_ptr<Robot> /* robot */,
                                std::shared_ptr<AbstractController> /* controller */)
    {
        PRINT_ERROR("This method is not supported by this class. Please call "
                    "`initialize` instead to set the model, or use `EngineMultiRobot` "
                    "class directly to simulate multiple robots simultaneously.");
        return hresult_t::ERROR_GENERIC;
    }

    hresult_t Engine::removeSystem(std::string const & /* systemName */)
    {
        PRINT_ERROR("This method is not supported by this class. Please call "
                    "`initialize` instead to set the model, or use `EngineMultiRobot` "
                    "class directly to simulate multiple robots simultaneously.");
        return hresult_t::ERROR_GENERIC;
    }

    hresult_t singleToMultipleSystemsInitialData(Robot const & robot,
                                                 bool_t const & isStateTheoretical,
                                                 vectorN_t const & qInit,
                                                 vectorN_t const & vInit,
                                                 boost::optional<vectorN_t> const & aInit,
                                                 std::map<std::string, vectorN_t> & qInitList,
                                                 std::map<std::string, vectorN_t> & vInitList,
                                                 boost::optional<std::map<std::string, vectorN_t> > & aInitList)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (isStateTheoretical && robot.mdlOptions_->dynamics.enableFlexibleModel)
        {
            vectorN_t q0;
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
                vectorN_t v0;
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
                    vectorN_t a0;
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

    hresult_t Engine::start(vectorN_t const & qInit,
                            vectorN_t const & vInit,
                            boost::optional<vectorN_t> const & aInit,
                            bool_t    const & isStateTheoretical)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (!isInitialized_)
        {
            PRINT_ERROR("The engine is not initialized.");
            returnCode = hresult_t::ERROR_INIT_FAILED;
        }

        std::map<std::string, vectorN_t> qInitList;
        std::map<std::string, vectorN_t> vInitList;
        boost::optional<std::map<std::string, vectorN_t> > aInitList = boost::none;
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

    hresult_t Engine::simulate(float64_t const & tEnd,
                               vectorN_t const & qInit,
                               vectorN_t const & vInit,
                               boost::optional<vectorN_t> const & aInit,
                               bool_t    const & isStateTheoretical)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (!isInitialized_)
        {
            PRINT_ERROR("The engine is not initialized.");
            returnCode = hresult_t::ERROR_INIT_FAILED;
        }

        std::map<std::string, vectorN_t> qInitList;
        std::map<std::string, vectorN_t> vInitList;
        boost::optional<std::map<std::string, vectorN_t> > aInitList = boost::none;
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

    hresult_t Engine::registerForceImpulse(std::string      const & frameName,
                                           float64_t        const & t,
                                           float64_t        const & dt,
                                           pinocchio::Force const & F)
    {
        return EngineMultiRobot::registerForceImpulse("", frameName, t, dt, F);
    }

    hresult_t Engine::registerForceProfile(std::string const & frameName,
                                           forceProfileFunctor_t const & forceFct,
                                           float64_t const & updatePeriod)
    {
        return EngineMultiRobot::registerForceProfile("", frameName, forceFct, updatePeriod);
    }

    hresult_t Engine::removeForcesImpulse(void)
    {
        return EngineMultiRobot::removeForcesImpulse("");
    }

    hresult_t Engine::removeForcesProfile(void)
    {
        return EngineMultiRobot::removeForcesProfile("");
    }

    forceImpulseRegister_t const & Engine::getForcesImpulse(void) const
    {
        forceImpulseRegister_t const * forcesImpulse;
        EngineMultiRobot::getForcesImpulse("", forcesImpulse);
        return *forcesImpulse;
    }

    forceProfileRegister_t const & Engine::getForcesProfile(void) const
    {
        forceProfileRegister_t const * forcesProfile;
        EngineMultiRobot::getForcesProfile("", forcesProfile);
        return *forcesProfile;
    }

    hresult_t Engine::registerForceCoupling(std::string const & frameName1,
                                            std::string const & frameName2,
                                            forceProfileFunctor_t forceFct)
    {
        auto forceCouplingFct = [forceFct](float64_t const & t,
                                           vectorN_t const & q1,
                                           vectorN_t const & v1,
                                           vectorN_t const & /* q2 */,
                                           vectorN_t const & /* v2 */)
                                {
                                    return forceFct(t, q1, v1);
                                };
        return EngineMultiRobot::registerForceCoupling(
            "", "", frameName1, frameName2, forceCouplingFct);
    }

    hresult_t Engine::registerViscoElasticForceCoupling(std::string const & frameName1,
                                                        std::string const & frameName2,
                                                        vectorN_t   const & stiffness,
                                                        vectorN_t   const & damping)
    {
        return EngineMultiRobot::registerViscoElasticForceCoupling(
            "", "", frameName1, frameName2, stiffness, damping);
    }

    hresult_t Engine::registerViscoElasticDirectionalForceCoupling(std::string const & frameName1,
                                                                   std::string const & frameName2,
                                                                   float64_t   const & stiffness,
                                                                   float64_t   const & damping)
    {
        return EngineMultiRobot::registerViscoElasticDirectionalForceCoupling(
            "", "", frameName1, frameName2, stiffness, damping);
    }

    hresult_t Engine::removeForcesCoupling(void)
    {
        return EngineMultiRobot::removeForcesCoupling("");
    }

    bool_t const & Engine::getIsInitialized(void) const
    {
        return isInitialized_;
    }

    hresult_t Engine::getSystem(systemHolder_t * & system)
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

    hresult_t Engine::getSystemState(systemState_t const * & systemState) const
    {
        static systemState_t const systemStateEmpty;

        hresult_t returnCode = hresult_t::SUCCESS;

        if (!isInitialized_)
        {
            PRINT_ERROR("The engine is not initialized.");
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
