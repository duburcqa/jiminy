#ifndef JIMINY_ENGINE_H
#define JIMINY_ENGINE_H

#include <optional>

#include "jiminy/core/fwd.h"
#include "jiminy/core/engine/engine_multi_robot.h"


namespace jiminy
{
    class JIMINY_DLLAPI Engine : public EngineMultiRobot
    {
    public:
        DISABLE_COPY(Engine)

    public:
        explicit Engine() = default;
        ~Engine() = default;

        hresult_t initialize(std::shared_ptr<Robot> robot,
                             std::shared_ptr<AbstractController> controller,
                             CallbackFunctor callbackFct);
        hresult_t initialize(std::shared_ptr<Robot> robot, CallbackFunctor callbackFct);

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
        hresult_t start(const Eigen::VectorXd & qInit,
                        const Eigen::VectorXd & vInit,
                        const std::optional<Eigen::VectorXd> & aInit = std::nullopt,
                        bool isStateTheoretical = false);

        /// \brief Run a simulation of duration tEnd, starting at xInit.
        ///
        /// \param[in] tEnd End time, i.e. amount of time to simulate.
        /// \param[in] qInit Initial configuration, i.e. state at t=0.
        /// \param[in] vInit Initial velocity, i.e. state at t=0.
        /// \param[in] aInit Initial acceleration, i.e. state at t=0.
        /// \param[in] isStateTheoretical Specify if the initial state is associated with the
        ///                               current or theoretical model.
        hresult_t simulate(double tEnd,
                           const Eigen::VectorXd & qInit,
                           const Eigen::VectorXd & vInit,
                           const std::optional<Eigen::VectorXd> & aInit = std::nullopt,
                           bool isStateTheoretical = false);

        hresult_t registerForceImpulse(
            const std::string & frameName, double t, double dt, const pinocchio::Force & F);
        hresult_t registerForceProfile(const std::string & frameName,
                                       const ForceProfileFunctor & forceFct,
                                       double updatePeriod = 0.0);

        // Redefined to leverage C++ name hiding of overloaded base methods in derived class
        hresult_t removeForcesImpulse();
        hresult_t removeForcesProfile();

        const ForceImpulseRegister & getForcesImpulse() const;
        const ForceProfileRegister & getForcesProfile() const;

        hresult_t registerForceCoupling(const std::string & frameName1,
                                        const std::string & frameName2,
                                        ForceProfileFunctor forceFct);
        hresult_t registerViscoelasticForceCoupling(const std::string & frameName1,
                                                    const std::string & frameName2,
                                                    const Vector6d & stiffness,
                                                    const Vector6d & damping,
                                                    double alpha = 0.5);
        hresult_t registerViscoelasticDirectionalForceCoupling(const std::string & frameName1,
                                                               const std::string & frameName2,
                                                               double stiffness,
                                                               double damping,
                                                               double restLength = 0.0);

        hresult_t removeForcesCoupling();

        hresult_t removeAllForces();

        bool getIsInitialized() const;
        hresult_t getSystem(systemHolder_t *& system);
        hresult_t getRobot(std::shared_ptr<Robot> & robot);
        hresult_t getController(std::shared_ptr<AbstractController> & controller);
        hresult_t getSystemState(const systemState_t *& systemState) const;

    private:
        hresult_t initializeImpl(std::shared_ptr<Robot> robot,
                                 std::shared_ptr<AbstractController> controller,
                                 CallbackFunctor callbackFct);

    protected:
        bool isInitialized_;
        Robot * robot_;
        AbstractController * controller_;
    };
}

#endif  // end of JIMINY_ENGINE_H
