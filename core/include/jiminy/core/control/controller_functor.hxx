#include <cassert>

#include "jiminy/core/robot/robot.h"


namespace jiminy
{
    template<typename F1, typename F2>
    FunctionalController<F1, F2>::FunctionalController(F1 & commandFun,
                                                       F2 & internalDynamicsFun) noexcept :
    AbstractController(),
    commandFun_{commandFun},
    internalDynamicsFun_{internalDynamicsFun}
    {
        static_assert(std::is_constructible_v<std::function<FunctionalControllerSignature>, F1> &&
                      std::is_constructible_v<std::function<FunctionalControllerSignature>, F2>);
    }

    template<typename F1, typename F2>
    FunctionalController<F1, F2>::FunctionalController(F1 && commandFun,
                                                       F2 && internalDynamicsFun) noexcept :
    AbstractController(),
    commandFun_(std::move(commandFun)),
    internalDynamicsFun_(std::move(internalDynamicsFun))
    {
    }

    template<typename F1, typename F2>
    hresult_t FunctionalController<F1, F2>::computeCommand(
        double t, const Eigen::VectorXd & q, const Eigen::VectorXd & v, Eigen::VectorXd & command)
    {
        if (!getIsInitialized())
        {
            PRINT_ERROR("The controller is not initialized.");
            return hresult_t::ERROR_INIT_FAILED;
        }

        commandFun_(t, q, v, sensorMeasurements_, command);

        return hresult_t::SUCCESS;
    }

    template<typename F1, typename F2>
    hresult_t FunctionalController<F1, F2>::internalDynamics(
        double t, const Eigen::VectorXd & q, const Eigen::VectorXd & v, Eigen::VectorXd & uCustom)
    {
        if (!getIsInitialized())
        {
            PRINT_ERROR("The controller is not initialized.");
            return hresult_t::ERROR_INIT_FAILED;
        }

        // Sensor data are already up-to-date
        internalDynamicsFun_(t, q, v, sensorMeasurements_, uCustom);

        return hresult_t::SUCCESS;
    }
}