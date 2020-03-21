#include <cassert>

#include "jiminy/core/robot/Robot.h"


namespace jiminy
{
    template<typename F1, typename F2>
    ControllerFunctor<F1, F2>::ControllerFunctor(F1 & commandFct,
                                                 F2 & internalDynamicsFct) :
    AbstractController(),
    commandFct_(commandFct),
    internalDynamicsFct_(internalDynamicsFct),
    sensorsData_()
    {
        // Empty.
    }

    template<typename F1, typename F2>
    ControllerFunctor<F1, F2>::ControllerFunctor(F1 && commandFct,
                                                 F2 && internalDynamicsFct) :
    AbstractController(),
    commandFct_(std::move(commandFct)),
    internalDynamicsFct_(std::move(internalDynamicsFct)),
    sensorsData_()
    {
        // Empty.
    }

    template<typename F1, typename F2>
    hresult_t ControllerFunctor<F1, F2>::initialize(std::shared_ptr<Robot const> const & model)
    {
        sensorsData_ = model->getSensorsData(); // Only one copy is needed thanks to C++11 Copy Elision paradigm
        return AbstractController::initialize(model);
    }

    template<typename F1, typename F2>
    hresult_t ControllerFunctor<F1, F2>::computeCommand(float64_t const & t,
                                                        vectorN_t const & q,
                                                        vectorN_t const & v,
                                                        vectorN_t       & u)
    {
        if (!getIsInitialized())
        {
            std::cout << "Error - ControllerFunctor::computeCommand - The model is not initialized." << std::endl;
            return hresult_t::ERROR_INIT_FAILED;
        }

        commandFct_(t, q, v, sensorsData_, u);

        return hresult_t::SUCCESS;
    }

    template<typename F1, typename F2>
    hresult_t ControllerFunctor<F1, F2>::internalDynamics(float64_t const & t,
                                                          vectorN_t const & q,
                                                          vectorN_t const & v,
                                                          vectorN_t       & u)
    {
        if (!getIsInitialized())
        {
            std::cout << "Error - ControllerFunctor::internalDynamics - The model is not initialized." << std::endl;
            return hresult_t::ERROR_INIT_FAILED;
        }

        internalDynamicsFct_(t, q, v, sensorsData_, u); // The sensor data are already up-to-date

        return hresult_t::SUCCESS;
    }
}