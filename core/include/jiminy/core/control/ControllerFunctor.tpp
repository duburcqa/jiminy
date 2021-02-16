#include <cassert>

#include "jiminy/core/robot/Robot.h"


namespace jiminy
{
    template<typename F1, typename F2>
    ControllerFunctor<F1, F2>::ControllerFunctor(F1 & commandFct,
                                                 F2 & internalDynamicsFct) :
    AbstractController(),
    commandFct_(commandFct),
    internalDynamicsFct_(internalDynamicsFct)
    {
        // Empty.
    }

    template<typename F1, typename F2>
    ControllerFunctor<F1, F2>::ControllerFunctor(F1 && commandFct,
                                                 F2 && internalDynamicsFct) :
    AbstractController(),
    commandFct_(std::move(commandFct)),
    internalDynamicsFct_(std::move(internalDynamicsFct))
    {
        // Empty.
    }

    template<typename F1, typename F2>
    hresult_t ControllerFunctor<F1, F2>::computeCommand(float64_t const & t,
                                                        vectorN_t const & q,
                                                        vectorN_t const & v,
                                                        vectorN_t       & command)
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
    hresult_t ControllerFunctor<F1, F2>::internalDynamics(float64_t const & t,
                                                          vectorN_t const & q,
                                                          vectorN_t const & v,
                                                          vectorN_t       & uCustom)
    {
        if (!getIsInitialized())
        {
            PRINT_ERROR("The controller is not initialized.");
            return hresult_t::ERROR_INIT_FAILED;
        }

        internalDynamicsFct_(t, q, v, sensorsData_, uCustom);  // The sensor data are already up-to-date

        return hresult_t::SUCCESS;
    }
}