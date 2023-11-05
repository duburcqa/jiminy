#ifndef JIMINY_ENGINE_H
#define JIMINY_ENGINE_H

#include "jiminy/core/engine/engine_multi_robot.h"


namespace jiminy
{
    class Engine : public EngineMultiRobot
    {
    public:
        // Disable the copy of the class
        Engine(const Engine & engine) = delete;
        Engine & operator=(const Engine & other) = delete;

    public:
        Engine() = default;
        virtual ~Engine() = default;

        hresult_t initialize(std::shared_ptr<Robot> robot,
                             std::shared_ptr<AbstractController> controller,
                             callbackFunctor_t callbackFct);
        hresult_t initialize(std::shared_ptr<Robot> robot, callbackFunctor_t callbackFct);

        hresult_t setController(std::shared_ptr<AbstractController> controller);

        /* Forbid direct usage of these methods since it does not make sense for single robot
           engine (every overloads are affected at once). */
        hresult_t addSystem(const std::string & systemName,
                            std::shared_ptr<Robot> robot,
                            std::shared_ptr<AbstractController> controller);
        hresult_t removeSystem(const std::string & systemName);

        /// \brief Reset the engine and compute initial state.
        ///
        /// \details This function does NOT reset the engine, robot and controller.
        ///          It is up to the user to do so, by calling `reset` method first.
        ///
        /// \param[in] qInit Initial configuration.
        /// \param[in] vInit Initial velocity.
        /// \param[in] aInit Initial acceleration.
        ///                  Optional: Zero by default.
        /// \param[in] isStateTheoretical Specify if the initial state is associated with the
        ///                               current or theoretical model.
        hresult_t start(const vectorN_t & qInit,
                        const vectorN_t & vInit,
                        const std::optional<vectorN_t> & aInit = std::nullopt,
                        const bool_t & isStateTheoretical = false);

        /// \brief Run a simulation of duration tEnd, starting at xInit.
        ///
        /// \param[in] tEnd End time, i.e. amount of time to simulate.
        /// \param[in] qInit Initial configuration, i.e. state at t=0.
        /// \param[in] vInit Initial velocity, i.e. state at t=0.
        /// \param[in] aInit Initial acceleration, i.e. state at t=0.
        /// \param[in] isStateTheoretical Specify if the initial state is associated with the
        ///                               current or theoretical model.
        hresult_t simulate(const float64_t & tEnd,
                           const vectorN_t & qInit,
                           const vectorN_t & vInit,
                           const std::optional<vectorN_t> & aInit = std::nullopt,
                           const bool_t & isStateTheoretical = false);

        hresult_t registerForceImpulse(const std::string & frameName,
                                       const float64_t & t,
                                       const float64_t & dt,
                                       const pinocchio::Force & F);
        hresult_t registerForceProfile(const std::string & frameName,
                                       const forceProfileFunctor_t & forceFct,
                                       const float64_t & updatePeriod = 0.0);

        // Redefined to leverage C++ name hiding of overloaded base methods in derived class
        hresult_t removeForcesImpulse();
        hresult_t removeForcesProfile();

        const forceImpulseRegister_t & getForcesImpulse() const;
        const forceProfileRegister_t & getForcesProfile() const;

        hresult_t registerForceCoupling(const std::string & frameName1,
                                        const std::string & frameName2,
                                        forceProfileFunctor_t forceFct);
        hresult_t registerViscoelasticForceCoupling(const std::string & frameName1,
                                                    const std::string & frameName2,
                                                    const vector6_t & stiffness,
                                                    const vector6_t & damping,
                                                    const float64_t & alpha = 0.5);
        hresult_t registerViscoelasticDirectionalForceCoupling(const std::string & frameName1,
                                                               const std::string & frameName2,
                                                               const float64_t & stiffness,
                                                               const float64_t & damping,
                                                               const float64_t & restLength = 0.0);

        hresult_t removeForcesCoupling();

        hresult_t removeAllForces();

        const bool_t & getIsInitialized() const;
        hresult_t getSystem(systemHolder_t *& system);
        hresult_t getRobot(std::shared_ptr<Robot> & robot);
        hresult_t getController(std::shared_ptr<AbstractController> & controller);
        hresult_t getSystemState(const systemState_t *& systemState) const;

    private:
        hresult_t initializeImpl(std::shared_ptr<Robot> robot,
                                 std::shared_ptr<AbstractController> controller,
                                 callbackFunctor_t callbackFct);

    protected:
        bool_t isInitialized_;
        Robot * robot_;
        AbstractController * controller_;
    };
}

#endif  // end of JIMINY_ENGINE_H
