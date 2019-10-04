#include <utility>
#include <cassert>

#include "jiminy/core/Model.h"


namespace jiminy
{
    // *********** Generic functor/lambda function caller utilities ************

    template <typename ReturnType, typename... Args>
    struct function_traits_defs {
        static constexpr size_t arity = sizeof...(Args) - 4;

        using result_type = ReturnType;

        template <size_t i>
        struct arg {
            using type = typename std::tuple_element<i, std::tuple<Args...>>::type;
        };
    };

    template <typename T>
    struct function_traits_impl;

    template <typename ReturnType, typename... Args>
    struct function_traits_impl<ReturnType(Args...)>
        : function_traits_defs<ReturnType, Args...> {};

    template <typename ReturnType, typename... Args>
    struct function_traits_impl<ReturnType(*)(Args...)>
        : function_traits_defs<ReturnType, Args...> {};

    template <typename ClassType, typename ReturnType, typename... Args>
    struct function_traits_impl<ReturnType(ClassType::*)(Args...)>
        : function_traits_defs<ReturnType, Args...> {};

    template <typename ClassType, typename ReturnType, typename... Args>
    struct function_traits_impl<ReturnType(ClassType::*)(Args...) const>
        : function_traits_defs<ReturnType, Args...> {};

    template <typename ClassType, typename ReturnType, typename... Args>
    struct function_traits_impl<ReturnType(ClassType::*)(Args...) const&>
        : function_traits_defs<ReturnType, Args...> {};

    template <typename ClassType, typename ReturnType, typename... Args>
    struct function_traits_impl<ReturnType(ClassType::*)(Args...) const&&>
        : function_traits_defs<ReturnType, Args...> {};

    template <typename ClassType, typename ReturnType, typename... Args>
    struct function_traits_impl<ReturnType(ClassType::*)(Args...) volatile>
        : function_traits_defs<ReturnType, Args...> {};

    template <typename ClassType, typename ReturnType, typename... Args>
    struct function_traits_impl<ReturnType(ClassType::*)(Args...) volatile&>
        : function_traits_defs<ReturnType, Args...> {};

    template <typename ClassType, typename ReturnType, typename... Args>
    struct function_traits_impl<ReturnType(ClassType::*)(Args...) volatile&&>
        : function_traits_defs<ReturnType, Args...> {};

    template <typename ClassType, typename ReturnType, typename... Args>
    struct function_traits_impl<ReturnType(ClassType::*)(Args...) const volatile>
        : function_traits_defs<ReturnType, Args...> {};

    template <typename ClassType, typename ReturnType, typename... Args>
    struct function_traits_impl<ReturnType(ClassType::*)(Args...) const volatile&>
        : function_traits_defs<ReturnType, Args...> {};

    template <typename ClassType, typename ReturnType, typename... Args>
    struct function_traits_impl<ReturnType(ClassType::*)(Args...) const volatile&&>
        : function_traits_defs<ReturnType, Args...> {};

    template <typename T, typename V = void>
    struct function_traits
        : function_traits_impl<T> {};

    template <typename T>
    struct function_traits<T, decltype((void)&T::operator())>
        : function_traits_impl<decltype(&T::operator())> {};

    template <size_t... Indices>
    struct indices
    {
        using next = indices<Indices..., sizeof...(Indices)>;
    };

    template <size_t N>
    struct build_indices
    {
        using type = typename build_indices<N - 1>::type::next;
    };

    template <>
    struct build_indices<0>
    {
        using type = indices<>;
    };

    template <size_t N>
    using BuildIndices = typename build_indices<N>::type;

    template <typename FuncType,
              typename VecType,
              size_t... I,
              typename Traits = function_traits<FuncType>,
              typename ReturnT = typename Traits::result_type>
    ReturnT do_call(FuncType        & func,
                    float64_t const & t,
                    vectorN_t const & q,
                    vectorN_t const & v,
                    VecType         & args,
                    vectorN_t       & u,
                    indices<I...>)
    {
        assert(args.size() == Traits::arity);
        return func(t, q, v, args[I]..., u);
    }

    template <typename FuncType,
              typename VecType,
              typename Traits = function_traits<FuncType>,
              typename ReturnT = typename Traits::result_type>
    ReturnT unpack_caller(FuncType        & func,
                          float64_t const & t,
                          vectorN_t const & q,
                          vectorN_t const & v,
                          VecType         & args,
                          vectorN_t       & u)
    {
        return do_call(func, t, q, v, args, u, BuildIndices<Traits::arity>());
    }

    // ************************************************************************

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
    ControllerFunctor<F1, F2>::~ControllerFunctor(void)
    {
        // Empty.
    }

    template<typename F1, typename F2>
    result_t ControllerFunctor<F1, F2>::computeCommand(float64_t const & t,
                                                       vectorN_t const & q,
                                                       vectorN_t const & v,
                                                       vectorN_t       & u)
    {
        result_t returnCode = result_t::SUCCESS;

        if (!getIsInitialized())
        {
            std::cout << "Error - ControllerFunctor::computeCommand - The model is not initialized." << std::endl;
            returnCode = result_t::ERROR_INIT_FAILED;
        }

        if (returnCode == result_t::SUCCESS)
        {
            returnCode = model_->getSensorsData(sensorsData_);
        }
        if (returnCode == result_t::SUCCESS)
        {
            unpack_caller(commandFct_, t, q, v, sensorsData_, u);
        }

        return returnCode;
    }

    template<typename F1, typename F2>
    result_t ControllerFunctor<F1, F2>::internalDynamics(float64_t const & t,
                                                         vectorN_t const & q,
                                                         vectorN_t const & v,
                                                         vectorN_t       & u)
    {
        result_t returnCode = result_t::SUCCESS;

        if (!getIsInitialized())
        {
            std::cout << "Error - ControllerFunctor::internalDynamics - The model is not initialized." << std::endl;
            returnCode = result_t::ERROR_INIT_FAILED;
        }

        if (returnCode == result_t::SUCCESS)
        {
            unpack_caller(internalDynamicsFct_, t, q, v, sensorsData_, u);
        }

        return returnCode;
    }
}