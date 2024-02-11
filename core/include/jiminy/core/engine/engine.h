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

        void initialize(std::shared_ptr<Robot> robot,
                        std::shared_ptr<AbstractController> controller,
                        const AbortSimulationFunction & callback);
        void initialize(std::shared_ptr<Robot> robot, const AbortSimulationFunction & callback);

        void setController(std::shared_ptr<AbstractController> controller);

        /* Forbid direct usage of these methods since it does not make sense for single robot
           engine (every overloads are affected at once). */
        [[noreturn]] void addSystem(const std::string & systemName,
                                    std::shared_ptr<Robot> robot,
                                    std::shared_ptr<AbstractController> controller);
        [[noreturn]] void removeSystem(const std::string & systemName);

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
        void start(const Eigen::VectorXd & qInit,
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
        void simulate(double tEnd,
                      const Eigen::VectorXd & qInit,
                      const Eigen::VectorXd & vInit,
                      const std::optional<Eigen::VectorXd> & aInit = std::nullopt,
                      bool isStateTheoretical = false);

        void registerImpulseForce(
            const std::string & frameName, double t, double dt, const pinocchio::Force & force);
        void registerProfileForce(const std::string & frameName,
                                  const ProfileForceFunction & forceFunc,
                                  double updatePeriod = 0.0);

        // Redefined to leverage C++ name hiding of overloaded base methods in derived class
        void removeImpulseForces();
        void removeProfileForces();

        const ImpulseForceVector & getImpulseForces() const;
        const ProfileForceVector & getProfileForces() const;

        void registerCouplingForce(const std::string & frameName1,
                                   const std::string & frameName2,
                                   const ProfileForceFunction & forceFunc);
        void registerViscoelasticCouplingForce(const std::string & frameName1,
                                               const std::string & frameName2,
                                               const Vector6d & stiffness,
                                               const Vector6d & damping,
                                               double alpha = 0.5);
        void registerViscoelasticDirectionalCouplingForce(const std::string & frameName1,
                                                          const std::string & frameName2,
                                                          double stiffness,
                                                          double damping,
                                                          double restLength = 0.0);

        void removeCouplingForces();

        void removeAllForces();

        bool getIsInitialized() const;
        System & getSystem();
        std::shared_ptr<Robot> getRobot();
        std::shared_ptr<AbstractController> getController();
        const SystemState & getSystemState() const;

    private:
        void initializeImpl(std::shared_ptr<Robot> robot,
                            std::shared_ptr<AbstractController> controller,
                            const AbortSimulationFunction & callback);

    protected:
        bool isInitialized_;
        Robot * robot_;
        AbstractController * controller_;
    };
}

#endif  // end of JIMINY_ENGINE_H
