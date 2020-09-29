#ifndef JIMINY_ENGINE_H
#define JIMINY_ENGINE_H

#include "jiminy/core/engine/EngineMultiRobot.h"


namespace jiminy
{
    class Engine : public EngineMultiRobot
    {
    public:
        // Disable the copy of the class
        Engine(Engine const & engine) = delete;
        Engine & operator = (Engine const & other) = delete;

    public:
        Engine(void);
        ~Engine(void) = default;

        hresult_t initialize(std::shared_ptr<Robot>              robot,
                             std::shared_ptr<AbstractController> controller,
                             callbackFunctor_t                   callbackFct);
        hresult_t initialize(std::shared_ptr<Robot> robot,
                             callbackFunctor_t      callbackFct);

        hresult_t setController(std::shared_ptr<AbstractController> controller);

        /// \brief Reset the engine and compute initial state.
        ///
        /// \details This function reset the engine, the robot and the controller, and update internal data
        ///          to match the given initial state.
        ///
        /// \param[in] xInit Initial state.
        /// \param[in] isStateTheoretical Specify if the initial state is associated with the current or theoretical model
        /// \param[in] resetRandomNumbers Whether or not to reset the random number generator.
        /// \param[in] resetDynamicForceRegister Whether or not to register the external force profiles applied
        ///                                      during the simulation.
        hresult_t start(vectorN_t const & xInit,
                        bool_t    const & isStateTheoretical = false,
                        bool_t    const & resetRandomNumbers = false,
                        bool_t    const & resetDynamicForceRegister = false);

        /// \brief Run a simulation of duration tEnd, starting at xInit.
        ///
        /// \param[in] tEnd End time, i.e. amount of time to simulate.
        /// \param[in] xInit Initial state, i.e. state at t=0.
        /// \param[in] isStateTheoretical Specify if the initial state is associated with the current or theoretical model
        hresult_t simulate(float64_t const & tEnd,
                           vectorN_t const & xInit,
                           bool_t    const & isStateTheoretical = false);

        hresult_t registerForceImpulse(std::string      const & frameName,
                                       float64_t        const & t,
                                       float64_t        const & dt,
                                       pinocchio::Force const & F);
        hresult_t registerForceProfile(std::string           const & frameName,
                                       forceProfileFunctor_t         forceFct);

        bool_t const & getIsInitialized(void) const;
        Robot const & getRobot(void) const;
        std::shared_ptr<Robot> getRobot(void);
        AbstractController const & getController(void) const;
        std::shared_ptr<AbstractController> getController(void);
        systemState_t const & getSystemState(void) const;

    private:
        // Make private some methods to deter their use
        using EngineMultiRobot::addSystem;
        using EngineMultiRobot::removeSystem;
        using EngineMultiRobot::setController;
        using EngineMultiRobot::addCouplingForce;
        using EngineMultiRobot::removeCouplingForces;
        using EngineMultiRobot::start;
        using EngineMultiRobot::simulate;
        using EngineMultiRobot::registerForceImpulse;
        using EngineMultiRobot::registerForceProfile;
        using EngineMultiRobot::getSystem;
        using EngineMultiRobot::getSystemState;

    protected:
        bool_t isInitialized_;
        Robot * robot_;
        AbstractController * controller_;
    };
}

#endif //end of JIMINY_ENGINE_H
