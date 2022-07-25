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
        Engine(void) = default;
        ~Engine(void) = default;

        hresult_t initialize(std::shared_ptr<Robot>              robot,
                             std::shared_ptr<AbstractController> controller,
                             callbackFunctor_t                   callbackFct);
        hresult_t initialize(std::shared_ptr<Robot> robot,
                             callbackFunctor_t      callbackFct);

        hresult_t setController(std::shared_ptr<AbstractController> controller);

        /* Forbid direct usage of these methods since it does not make sense for single
           robot engine (every overloads are affected at once). */
        hresult_t addSystem(std::string const & systemName,
                            std::shared_ptr<Robot> robot,
                            std::shared_ptr<AbstractController> controller);
        hresult_t removeSystem(std::string const & systemName);

        /// \brief Reset the engine and compute initial state.
        ///
        /// \details This function does NOT reset the engine, robot and controller.
        ///          It is up to the user to do so, by calling `reset` method first.
        ///
        /// \param[in] qInit Initial configuration.
        /// \param[in] vInit Initial velocity.
        /// \param[in] aInit Initial acceleration. Optional: Zero by default.
        /// \param[in] isStateTheoretical Specify if the initial state is associated with the current or theoretical model
        hresult_t start(vectorN_t const & qInit,
                        vectorN_t const & vInit,
                        std::optional<vectorN_t> const & aInit = std::nullopt,
                        bool_t    const & isStateTheoretical = false);

        /// \brief Run a simulation of duration tEnd, starting at xInit.
        ///
        /// \param[in] tEnd End time, i.e. amount of time to simulate.
        /// \param[in] qInit Initial configuration, i.e. state at t=0.
        /// \param[in] vInit Initial velocity, i.e. state at t=0.
        /// \param[in] aInit Initial acceleration, i.e. state at t=0.
        /// \param[in] isStateTheoretical Specify if the initial state is associated with the current or theoretical model
        hresult_t simulate(float64_t const & tEnd,
                           vectorN_t const & qInit,
                           vectorN_t const & vInit,
                           std::optional<vectorN_t> const & aInit = std::nullopt,
                           bool_t    const & isStateTheoretical = false);

        hresult_t registerForceImpulse(std::string      const & frameName,
                                       float64_t        const & t,
                                       float64_t        const & dt,
                                       pinocchio::Force const & F);
        hresult_t registerForceProfile(std::string const & frameName,
                                       forceProfileFunctor_t const & forceFct,
                                       float64_t const & updatePeriod = 0.0);

        // Redefined to take advantage of C++ name hiding of overloaded methods of base class in dervied class
        hresult_t removeForcesImpulse(void);
        hresult_t removeForcesProfile(void);

        forceImpulseRegister_t const & getForcesImpulse(void) const;
        forceProfileRegister_t const & getForcesProfile(void) const;

        hresult_t registerForceCoupling(std::string const & frameName1,
                                        std::string const & frameName2,
                                        forceProfileFunctor_t forceFct);
        hresult_t registerViscoelasticForceCoupling(std::string const & frameName1,
                                                    std::string const & frameName2,
                                                    vector6_t   const & stiffness,
                                                    vector6_t   const & damping);
        hresult_t registerViscoelasticDirectionalForceCoupling(std::string const & frameName1,
                                                               std::string const & frameName2,
                                                               float64_t   const & stiffness,
                                                               float64_t   const & damping,
                                                               float64_t   const & restLength = 0.0);

        hresult_t removeForcesCoupling(void);

        hresult_t removeAllForces(void);

        bool_t const & getIsInitialized(void) const;
        hresult_t getSystem(systemHolder_t * & system);
        hresult_t getRobot(std::shared_ptr<Robot> & robot);
        hresult_t getController(std::shared_ptr<AbstractController> & controller);
        hresult_t getSystemState(systemState_t const * & systemState) const;

    private:
        hresult_t initializeImpl(std::shared_ptr<Robot>              robot,
                                 std::shared_ptr<AbstractController> controller,
                                 callbackFunctor_t                   callbackFct);

    protected:
        bool_t isInitialized_;
        Robot * robot_;
        AbstractController * controller_;
    };
}

#endif //end of JIMINY_ENGINE_H
