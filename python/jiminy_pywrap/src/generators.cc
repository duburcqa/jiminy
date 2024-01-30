#include "hpp/fcl/BVH/BVH_model.h"  // `hpp::fcl::BVHModel`, `hpp::fcl::OBBRSS`

#include "jiminy/core/utilities/random.h"
#include "jiminy/core/utilities/geometry.h"

#include "jiminy/python/utilities.h"
#include "jiminy/python/generators.h"


namespace jiminy::python
{
    namespace bp = boost::python;

    HeightmapFunctor sumHeightmaps(const bp::list & heightmapsPy)
    {
        auto heightmaps = convertFromPython<std::vector<HeightmapFunctor>>(heightmapsPy);
        return ::jiminy::sumHeightmaps(heightmaps);
    }

    HeightmapFunctor mergeHeightmaps(const bp::list & heightmapsPy)
    {
        auto heightmaps = convertFromPython<std::vector<HeightmapFunctor>>(heightmapsPy);
        return ::jiminy::mergeHeightmaps(heightmaps);
    }

    std::shared_ptr<PCG32> makePCG32FromSeedSed(bp::list & seedSeqPy)
    {
        std::vector<int> seedSeq = bp::extract<std::vector<int>>(seedSeqPy);
        return std::make_shared<PCG32>(std::seed_seq(seedSeq.cbegin(), seedSeq.cend()));
    }

    void seedPCG32FromSeedSed(PCG32 & generator, bp::list & seedSeqPy)
    {
        std::vector<int> seedSeq = bp::extract<std::vector<int>>(seedSeqPy);
        return generator.seed(std::seed_seq(seedSeq.cbegin(), seedSeq.cend()));
    }


#define GENERIC_DISTRIBUTION_WRAPPER(dist, arg1, arg2)                                            \
    Eigen::MatrixXd dist##FromStackedArgs(                                                        \
        const uniform_random_bit_generator_ref<uint32_t> & generator,                             \
        np::ndarray & arg1,                                                                       \
        np::ndarray & arg2)                                                                       \
    {                                                                                             \
        /* Extract Eigen::MatriXf from generic numpy arrays  */                                   \
        auto cast = [](np::ndarray & array) -> Eigen::Ref<const MatrixX<float>>                   \
        {                                                                                         \
            if (array.get_dtype() == np::dtype::get_builtin<float>())                             \
            {                                                                                     \
                return convertFromPython<Eigen::Ref<const MatrixX<float>>>(array);                \
            }                                                                                     \
            if (array.get_dtype() == np::dtype::get_builtin<double>())                            \
            {                                                                                     \
                return convertFromPython<Eigen::Ref<const MatrixX<double>>>(array).cast<float>(); \
            }                                                                                     \
            throw std::invalid_argument(                                                          \
                "Matrix arguments must have dtype 'np.float32' or 'np.float64'.");                \
        };                                                                                        \
        return dist(generator, cast(arg1), cast(arg2)).cast<double>();                            \
    }                                                                                             \
                                                                                                  \
    bp::object dist##FromSize(const uniform_random_bit_generator_ref<uint32_t> & generator,       \
                              float arg1,                                                         \
                              float arg2,                                                         \
                              bp::object & sizePy)                                                \
    {                                                                                             \
        /* Extract nrows, ncols from optional tuple with at most 2 elements  */                   \
        if (sizePy.is_none())                                                                     \
        {                                                                                         \
            return convertToPython(dist(generator, arg1, arg2), true);                            \
        }                                                                                         \
        auto size = convertFromPython<std::vector<int>>(sizePy);                                  \
        int nrows = 1;                                                                            \
        int ncols = 1;                                                                            \
        switch (size.size())                                                                      \
        {                                                                                         \
        case 2:                                                                                   \
            ncols = size[1];                                                                      \
            [[fallthrough]];                                                                      \
        case 1:                                                                                   \
            nrows = size[0];                                                                      \
            [[fallthrough]];                                                                      \
        case 0:                                                                                   \
            break;                                                                                \
        default:                                                                                  \
            throw std::invalid_argument("'size' must have at most 2 dimensions.");                \
        }                                                                                         \
        return convertToPython(dist(nrows, ncols, generator, arg1, arg2).cast<double>(), true);   \
    }

    GENERIC_DISTRIBUTION_WRAPPER(uniform, lo, hi)
    GENERIC_DISTRIBUTION_WRAPPER(normal, mean, stddev)

#undef GENERIC_DISTRIBUTION_WRAPPER

    template<typename R, typename F, typename... Args>
    R convertGeneratorToPythonAndInvoke(F callable, bp::object generatorPy, Args &&... args)
    {
        // First, check if the provided generator can be implicitly converted
        bp::extract<uniform_random_bit_generator_ref<uint32_t>> generatorPyGetter(generatorPy);
        if (generatorPyGetter.check())
        {
            return callable(generatorPyGetter(), std::forward<Args>(args)...);
        }

        // If not, assuming it is a numpy random generator and try to extract the raw function
        bp::object ctypes_addressof = bp::import("ctypes").attr("addressof");
        bp::object next_uint32_ctype = generatorPy.attr("ctypes").attr("next_uint32");
        uintptr_t next_uint32_addr = bp::extract<uintptr_t>(ctypes_addressof(next_uint32_ctype));
        auto next_uint32 = *reinterpret_cast<uint32_t (**)(void *)>(next_uint32_addr);
        bp::object state_ctype = generatorPy.attr("ctypes").attr("state_address");
        void * state_ptr = reinterpret_cast<void *>(bp::extract<uintptr_t>(state_ctype)());

        return callable([state_ptr, next_uint32]() -> uint32_t { return next_uint32(state_ptr); },
                        std::forward<Args>(args)...);
    }

    template<typename Signature, typename>
    class ConvertGeneratorToPythonAndInvoke;

    template<typename R, typename Generator, typename... Args>
    class ConvertGeneratorToPythonAndInvoke<R(Generator, Args...), void>
    {
    public:
        ConvertGeneratorToPythonAndInvoke(R (*fun)(Generator, Args...)) :
        fun_{fun}
        {
        }

        R operator()(bp::object generatorPy, Args... argsPy)
        {
            return convertGeneratorToPythonAndInvoke<R, R (*)(Generator, Args...), Args...>(
                fun_, generatorPy, std::forward<Args>(argsPy)...);
        }

    private:
        R (*fun_)(Generator, Args...);
    };

    template<typename R, typename... Args>
    ConvertGeneratorToPythonAndInvoke(R (*)(Args...))
        -> ConvertGeneratorToPythonAndInvoke<R(Args...), void>;

    template<typename T, typename R, typename Generator, typename... Args>
    class ConvertGeneratorToPythonAndInvoke<R(Generator, Args...), T>
    {
    public:
        ConvertGeneratorToPythonAndInvoke(R (T::*memFun)(Generator, Args...)) :
        memFun_{memFun}
        {
        }

        R operator()(T & obj, bp::object generatorPy, Args... argsPy)
        {
            auto callable = [&obj, memFun = memFun_](Generator generator, Args... args) -> R
            {
                return (obj.*memFun)(generator, args...);
            };
            return convertGeneratorToPythonAndInvoke<R, decltype(callable), Args...>(
                callable, generatorPy, std::forward<Args>(argsPy)...);
        }

    private:
        R (T::*memFun_)(Generator, Args...);
    };

    template<typename T, typename R, typename... Args>
    ConvertGeneratorToPythonAndInvoke(R (T::*)(Args...))
        -> ConvertGeneratorToPythonAndInvoke<R(Args...), T>;

    void exposeGenerators()
    {
        // clang-format off
        bp::class_<PCG32,
                   std::shared_ptr<PCG32>,
                   boost::noncopyable>("PCG32",
                   bp::init<uint64_t>((bp::arg("self"), "state")))
            .def(bp::init<>((bp::arg("self"))))
            .def("__init__", bp::make_constructor(&makePCG32FromSeedSed,
                             bp::default_call_policies(),
                             (bp::arg("seed_seq"))))
            .def("__call__", &PCG32::operator(), bp::args("self"))
            .def("seed", &seedPCG32FromSeedSed, (bp::arg("self"), "seed_seq"))
            .add_static_property(
                "min", &PCG32::min, getPropertySignaturesWithDoc(nullptr, &PCG32::min).c_str())
            .add_static_property(
                "max", &PCG32::max, getPropertySignaturesWithDoc(nullptr, &PCG32::max).c_str());

        bp::implicitly_convertible<PCG32, uniform_random_bit_generator_ref<uint32_t>>();

#define BIND_GENERIC_DISTRIBUTION(dist, arg1, arg2)                                          \
        bp::def(#dist, makeFunction(                                                         \
            ConvertGeneratorToPythonAndInvoke(&dist##FromStackedArgs),                       \
            bp::default_call_policies(),                                                     \
            (bp::arg("generator"), #arg1, #arg2)));                                          \
        bp::def(#dist, makeFunction(                                                         \
            ConvertGeneratorToPythonAndInvoke(&dist##FromSize),                              \
            bp::default_call_policies(),                                                     \
            (bp::arg("generator"), bp::arg(#arg1) = 0.0F, bp::arg(#arg2) = 1.0F,             \
             bp::arg("size") = bp::object())));

    BIND_GENERIC_DISTRIBUTION(uniform, lo, hi)
    BIND_GENERIC_DISTRIBUTION(normal, mean, stddev)

#undef BIND_GENERIC_DISTRIBUTION

        // Must be declared last to take precedence over generic declaration with default values
        bp::def("uniform", makeFunction(ConvertGeneratorToPythonAndInvoke(
            static_cast<
                float (*)(const uniform_random_bit_generator_ref<uint32_t> &)
            >(&uniform)),
            bp::default_call_policies(),
            (bp::arg("generator"))));

        bp::class_<PeriodicGaussianProcess,
                   std::shared_ptr<PeriodicGaussianProcess>,
                   boost::noncopyable>("PeriodicGaussianProcess",
                   bp::init<double, double>(
                   (bp::arg("self"), "wavelength", "period")))
            .def("__call__", &PeriodicGaussianProcess::operator(),
                             (bp::arg("self"), bp::arg("time")))
            .def("reset", makeFunction(
                          ConvertGeneratorToPythonAndInvoke(&PeriodicGaussianProcess::reset),
                          bp::default_call_policies(),
                          (bp::arg("self"), "generator")))
            .ADD_PROPERTY_GET_WITH_POLICY("wavelength",
                                          &PeriodicGaussianProcess::getWavelength,
                                          bp::return_value_policy<bp::return_by_value>())
            .ADD_PROPERTY_GET_WITH_POLICY("period",
                                          &PeriodicGaussianProcess::getPeriod,
                                          bp::return_value_policy<bp::return_by_value>());

        bp::class_<PeriodicFourierProcess,
                   std::shared_ptr<PeriodicFourierProcess>,
                   boost::noncopyable>("PeriodicFourierProcess",
                   bp::init<double, double>(
                   (bp::arg("self"), "wavelength", "period")))
            .def("__call__", &PeriodicFourierProcess::operator(),
                             (bp::arg("self"), bp::arg("time")))
            .def("reset", makeFunction(
                          ConvertGeneratorToPythonAndInvoke(&PeriodicFourierProcess::reset),
                          bp::default_call_policies(),
                          (bp::arg("self"), "generator")))
            .ADD_PROPERTY_GET_WITH_POLICY("wavelength",
                                          &PeriodicFourierProcess::getWavelength,
                                          bp::return_value_policy<bp::return_by_value>())
            .ADD_PROPERTY_GET_WITH_POLICY("period",
                                          &PeriodicFourierProcess::getPeriod,
                                          bp::return_value_policy<bp::return_by_value>());

        bp::class_<AbstractPerlinProcess,
                   std::shared_ptr<AbstractPerlinProcess>,
                   boost::noncopyable>("AbstractPerlinProcess", bp::no_init)
            .def("__call__", &AbstractPerlinProcess::operator(),
                             (bp::arg("self"), "time"))
            .def("reset", makeFunction(
                          ConvertGeneratorToPythonAndInvoke(&AbstractPerlinProcess::reset),
                          bp::default_call_policies(),
                          (bp::arg("self"), "generator")))
            .ADD_PROPERTY_GET_WITH_POLICY("wavelength",
                                          &AbstractPerlinProcess::getWavelength,
                                          bp::return_value_policy<bp::return_by_value>())
            .ADD_PROPERTY_GET_WITH_POLICY("num_octaves",
                                          &AbstractPerlinProcess::getNumOctaves,
                                          bp::return_value_policy<bp::return_by_value>());

        bp::class_<RandomPerlinProcess, bp::bases<AbstractPerlinProcess>,
                   std::shared_ptr<RandomPerlinProcess>,
                   boost::noncopyable>("RandomPerlinProcess",
                   bp::init<double, uint32_t>(
                   (bp::arg("self"), "wavelength", bp::arg("num_octaves") = 6U)));

        bp::class_<PeriodicPerlinProcess, bp::bases<AbstractPerlinProcess>,
                   std::shared_ptr<PeriodicPerlinProcess>,
                   boost::noncopyable>("PeriodicPerlinProcess",
                   bp::init<double, double, uint32_t>(
                   (bp::arg("self"), "wavelength", "period", bp::arg("num_octaves") = 6U)))
            .ADD_PROPERTY_GET_WITH_POLICY("period",
                                          &PeriodicPerlinProcess::getPeriod,
                                          bp::return_value_policy<bp::return_by_value>());

        bp::def("random_tile_ground", &tiles,
                                      (bp::arg("size"), "height_max", "interp_delta",
                                       "sparsity", "orientation", "seed"));
        bp::def("sum_heightmaps", &sumHeightmaps, (bp::arg("heightmaps")));
        bp::def("merge_heightmaps", &mergeHeightmaps, (bp::arg("heightmaps")));

        bp::def("discretize_heightmap", &discretizeHeightmap,
                                        (bp::arg("heightmap"), "x_min", "x_max", "x_unit", "y_min",
                                         "y_max", "y_unit", bp::arg("must_simplify") = false));
        // clang-format on
    }
}
