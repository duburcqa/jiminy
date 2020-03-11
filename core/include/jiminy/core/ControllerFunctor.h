///////////////////////////////////////////////////////////////////////////////////////////////
///
/// \brief          Controller wrapping any command and internal state methods.
///
///                 This class wraps 'callables' computing the command and internal state so
///                 that it can be used as a Jiminy Controller to initialize the Jiminy Engine.
///
///                 In practice, those 'callables' can be function pointers, functors or even
///                 lambda expressions. Their signature must be the following:
///                     void(float64_t const & t,
///                          vectorN_t const & q,
///                          vectorN_t const & v,
///                          matrixN_t const & sensorsData[I]...,
///                          vectorN_t       & u)
///                 where I is range(n), with n the number of different type of sensor.
///
///////////////////////////////////////////////////////////////////////////////////////////////

#ifndef WDC_CONTROLLER_FUNCTOR_H
#define WDC_CONTROLLER_FUNCTOR_H

#include "jiminy/core/Model.h"
#include "jiminy/core/AbstractController.h"
#include "jiminy/core/Types.h"

namespace jiminy
{
    template<typename F1, typename F2>
    class ControllerFunctor : public AbstractController
    {
    public:
        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Forbid the copy of the class
        ///////////////////////////////////////////////////////////////////////////////////////////////
        ControllerFunctor(ControllerFunctor const & controller) = delete;
        ControllerFunctor & operator = (ControllerFunctor const & controller) = delete;

        ///////////////////////////////////////////////////////////////////////////////////////////////
        ///
        /// \brief      Constructor
        ///
        /// \remark     A valid 'callable' is a function pointer, functor or lambda with signature:
        ///                 void(float64_t const & t,
        ///                      vectorN_t const & q,
        ///                      vectorN_t const & v,
        ///                      matrixN_t const & sensorsData[I]...,
        ///                      vectorN_t       & u)
        ///             where I is range(n), with n the number of different type of sensor.
        ///
        /// \param[in]  commandFct              'Callable' computing the command
        /// \param[in]  internalDynamicsFct     'Callable' computing the internal dynamics
        ///
        ///////////////////////////////////////////////////////////////////////////////////////////////
        ControllerFunctor(F1 & commandFct,
                          F2 & internalDynamicsFct);
        ControllerFunctor(F1 && commandFct,
                          F2 && internalDynamicsFct);

        ~ControllerFunctor(void) = default;

        ///////////////////////////////////////////////////////////////////////////////////////////////
        ///
        /// \brief      Set the parameters of the controller.
        ///
        /// \param[in]  model   Model of the system
        ///
        /// \return     Return code to determine whether the execution of the method was successful.
        ///
        ///////////////////////////////////////////////////////////////////////////////////////////////
        virtual result_t initialize(std::shared_ptr<Model const> const & model) override;

        ///////////////////////////////////////////////////////////////////////////////////////////////
        ///
        /// \brief      Compute the command.
        ///
        /// \details    It assumes that the model internal state (including sensors) is consistent
        ///             with other input arguments. It fetches the sensor data automatically.
        ///
        /// \param[in]  t       Current time
        /// \param[in]  q       Current configuration vector
        /// \param[in]  v       Current velocity vector
        /// \param[out] u       Output torque vector
        ///
        /// \return     Return code to determine whether the execution of the method was successful.
        ///
        ///////////////////////////////////////////////////////////////////////////////////////////////
        result_t computeCommand(float64_t const & t,
                                vectorN_t const & q,
                                vectorN_t const & v,
                                vectorN_t       & u) override;

        ///////////////////////////////////////////////////////////////////////////////////////////////
        ///
        /// \brief      Emulate internal dynamics of the system at are not included in the
        ///             physics engine.
        ///
        /// \param[in]  t       Current time
        /// \param[in]  q       Current configuration vector
        /// \param[in]  v       Current velocity vector
        /// \param[in]  u       Output torque vector
        ///
        /// \return     Return code to determine whether the execution of the method was successful.
        ///
        ///////////////////////////////////////////////////////////////////////////////////////////////
        result_t internalDynamics(float64_t const & t,
                                  vectorN_t const & q,
                                  vectorN_t const & v,
                                  vectorN_t       & u) override;

    private:
        // std::conditional_t enables to use both functors and lambdas
        std::conditional_t<std::is_function<F1>::value,
                           std::add_pointer_t<F1>, F1> commandFct_;             // 'Callable' computing the command
        std::conditional_t<std::is_function<F2>::value,
                           std::add_pointer_t<F2>, F2> internalDynamicsFct_;    // 'Callable' computing the internal dynamics
        sensorsDataMap_t sensorsData_;                                          // Vector of the data associated with type of sensors
    };
}

#include "jiminy/core/ControllerFunctor.tpp"

#endif //end of WDC_CONTROLLER_FUNCTOR_H
