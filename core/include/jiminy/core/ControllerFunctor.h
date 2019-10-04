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
        // Disable the copy of the class
        ControllerFunctor(ControllerFunctor const & controller) = delete;
        ControllerFunctor & operator = (ControllerFunctor const & controller) = delete;

    public:
        ControllerFunctor(F1 & commandFct,
                          F2 & internalDynamicsFct);
        ControllerFunctor(F1 && commandFct,
                          F2 && internalDynamicsFct);
        ~ControllerFunctor(void);

        result_t configureTelemetry(std::shared_ptr<TelemetryData> const & telemetryData);

        result_t computeCommand(float64_t const & t,
                                vectorN_t const & q,
                                vectorN_t const & v,
                                vectorN_t       & u) override;
        result_t internalDynamics(float64_t const & t,
                                  vectorN_t const & q,
                                  vectorN_t const & v,
                                  vectorN_t       & u) override;

    private:
        std::conditional_t<std::is_function<F1>::value, std::add_pointer_t<F1>, F1> commandFct_; // Trick to accept both functors and lambdas
        std::conditional_t<std::is_function<F2>::value, std::add_pointer_t<F2>, F2> internalDynamicsFct_;
        std::vector<matrixN_t> sensorsData_;
    };

    template<typename F1, typename F2>
    ControllerFunctor<F1, F2> createControllerFunctor(F1 & commandFct,
                                                      F2 & internalDynamicsFct)
    {
        return {commandFct, internalDynamicsFct};
    }
}

#include "jiminy/core/ControllerFunctor.tcc"

#endif //end of WDC_CONTROLLER_FUNCTOR_H
