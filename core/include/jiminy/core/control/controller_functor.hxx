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
    void FunctionalController<F1, F2>::computeCommand(
        double t, const Eigen::VectorXd & q, const Eigen::VectorXd & v, Eigen::VectorXd & command)
    {
        if (!getIsInitialized())
        {
            JIMINY_THROW(bad_control_flow, "Controller not initialized.");
        }

        commandFun_(t, q, v, sensorMeasurements_, command);
    }

    template<typename F1, typename F2>
    void FunctionalController<F1, F2>::internalDynamics(
        double t, const Eigen::VectorXd & q, const Eigen::VectorXd & v, Eigen::VectorXd & uCustom)
    {
        if (!getIsInitialized())
        {
            JIMINY_THROW(bad_control_flow, "Controller not initialized.");
        }

        // Sensor data are already up-to-date
        internalDynamicsFun_(t, q, v, sensorMeasurements_, uCustom);
    }
}
