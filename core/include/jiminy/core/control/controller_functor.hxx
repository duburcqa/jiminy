#include <cassert>

#include "jiminy/core/robot/robot.h"


namespace jiminy
{
    template<typename F1, typename F2>
    ControllerFunctor<F1, F2>::ControllerFunctor(F1 & commandFct,
                                                 F2 & internalDynamicsFct) noexcept :
    AbstractController(),
    commandFct_{commandFct},
    internalDynamicsFct_{internalDynamicsFct}
    {
        static_assert(std::is_constructible_v<std::function<ControllerFunctorSignature>, F1> &&
                      std::is_constructible_v<std::function<ControllerFunctorSignature>, F2>);
    }

    template<typename F1, typename F2>
    ControllerFunctor<F1, F2>::ControllerFunctor(F1 && commandFct,
                                                 F2 && internalDynamicsFct) noexcept :
    AbstractController(),
    commandFct_(std::move(commandFct)),
    internalDynamicsFct_(std::move(internalDynamicsFct))
    {
    }

    template<typename F1, typename F2>
    hresult_t ControllerFunctor<F1, F2>::computeCommand(
        double t, const Eigen::VectorXd & q, const Eigen::VectorXd & v, Eigen::VectorXd & command)
    {
        if (!getIsInitialized())
        {
            PRINT_ERROR("The controller is not initialized.");
            return hresult_t::ERROR_INIT_FAILED;
        }

        commandFct_(t, q, v, sensorsData_, command);

        return hresult_t::SUCCESS;
    }

    template<typename F1, typename F2>
    hresult_t ControllerFunctor<F1, F2>::internalDynamics(
        double t, const Eigen::VectorXd & q, const Eigen::VectorXd & v, Eigen::VectorXd & uCustom)
    {
        if (!getIsInitialized())
        {
            PRINT_ERROR("The controller is not initialized.");
            return hresult_t::ERROR_INIT_FAILED;
        }

        // Sensor data are already up-to-date
        internalDynamicsFct_(t, q, v, sensorsData_, uCustom);

        return hresult_t::SUCCESS;
    }
}