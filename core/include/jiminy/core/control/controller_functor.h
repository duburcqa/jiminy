/// \brief Controller wrapping any command and internal state methods.
///
/// \details This class wraps 'callables' computing the command and internal state so that it can
///          be used as a Jiminy Controller to initialize the Jiminy Engine. In practice, those
///          'callables' can be function pointers, functors or even lambda expressions.

#ifndef CONTROLLER_FUNCTOR_H
#define CONTROLLER_FUNCTOR_H

#include "jiminy/core/fwd.h"
#include "jiminy/core/control/abstract_controller.h"

namespace jiminy
{
    template<typename F>
    using callable_t = std::conditional_t<std::is_function_v<F>, std::add_pointer_t<F>, F>;

    using ControllerFunctorSignature = void(double /* t */,
                                            const Eigen::VectorXd & /* q */,
                                            const Eigen::VectorXd & /* v */,
                                            const SensorsDataMap & /* sensorMeasurements */,
                                            Eigen::VectorXd & /* command */);

    template<typename F1 = std::add_pointer_t<ControllerFunctorSignature>,
             typename F2 = std::add_pointer_t<ControllerFunctorSignature>>
    class ControllerFunctor : public AbstractController
    {
    public:
        DISABLE_COPY(ControllerFunctor)

    public:
        /// \remark A valid 'callable' is a function pointer, functor or lambda with signature:
        ///             void(double t,
        ///                  const Eigen::VectorXd & q,
        ///                  const Eigen::VectorXd & v,
        ///                  const SensorsDataMap & sensorsData,
        ///                  Eigen::VectorXd & command)
        ///         where I is range(n), with n the number of different type of sensor.
        ///
        /// \param[in] commandFct 'Callable' computing the command.
        /// \param[in] internalDynamicsFct 'Callable' computing the internal dynamics.
        explicit ControllerFunctor(F1 & commandFct, F2 & internalDynamicsFct) noexcept;
        explicit ControllerFunctor(F1 && commandFct, F2 && internalDynamicsFct) noexcept;

        virtual ~ControllerFunctor() = default;

        /// \brief Compute the command.
        ///
        /// \details It assumes that the robot internal state (including sensors) is consistent
        ///          with other input arguments. It fetches the sensor data automatically.
        ///
        /// \param[in] t Current time
        /// \param[in] q Current configuration vector
        /// \param[in] v Current velocity vector
        /// \param[out] command Output effort vector
        ///
        /// \return Return code to determine whether the execution of the method was successful.
        virtual hresult_t computeCommand(double t,
                                         const Eigen::VectorXd & q,
                                         const Eigen::VectorXd & v,
                                         Eigen::VectorXd & command) override;

        /// \brief Emulate custom phenomenon that are part of the internal dynamics of the system
        ///        but not included in the physics engine.
        ///
        /// \param[in] t Current time.
        /// \param[in] q Current configuration vector.
        /// \param[in] v Current velocity vector.
        /// \param[in] command Output effort vector.
        ///
        /// \return Return code to determine whether the execution of the method was successful.
        virtual hresult_t internalDynamics(double t,
                                           const Eigen::VectorXd & q,
                                           const Eigen::VectorXd & v,
                                           Eigen::VectorXd & uCustom) override;

    private:
        /// \brief 'Callable' computing the command.
        callable_t<F1> commandFct_;
        /// \brief 'Callable' computing the internal dynamics.
        callable_t<F2> internalDynamicsFct_;
    };
}

#include "jiminy/core/control/controller_functor.hxx"

#endif  // end of CONTROLLER_FUNCTOR_H
