#include "hpp/fcl/BVH/BVH_model.h"  // `hpp::fcl::BVHModel`, `hpp::fcl::OBBRSS`

#include "jiminy/core/utilities/random.h"
#include "jiminy/core/utilities/geometry.h"

#include "jiminy/python/utilities.h"
#include "jiminy/python/generators.h"


namespace jiminy::python
{
    namespace bp = boost::python;

    HeightmapFunction sumHeightmaps(const bp::object & heightmapsPy)
    {
        auto heightmaps = convertFromPython<std::vector<HeightmapFunction>>(heightmapsPy);
        return ::jiminy::sumHeightmaps(heightmaps);
    }

    HeightmapFunction mergeHeightmaps(const bp::object & heightmapsPy)
    {
        auto heightmaps = convertFromPython<std::vector<HeightmapFunction>>(heightmapsPy);
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
            JIMINY_THROW(std::invalid_argument,                                                   \
                         "Matrix arguments must have dtype 'np.float32' or 'np.float64'.");       \
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
            JIMINY_THROW(std::invalid_argument, "'size' must have at most 2 dimensions.");        \
        }                                                                                         \
        return convertToPython(dist(nrows, ncols, generator, arg1, arg2).cast<double>(), true);   \
    }

    GENERIC_DISTRIBUTION_WRAPPER(uniform, lo, hi)
    GENERIC_DISTRIBUTION_WRAPPER(normal, mean, stddev)

#undef GENERIC_DISTRIBUTION_WRAPPER

    template<typename DerivedPerlinProcess, typename... Args>
    std::enable_if_t<std::conjunction_v<std::is_arithmetic<std::decay_t<Args>>...>, double>
    evaluatePerlinProcessUnpacked(DerivedPerlinProcess & fun, Args... args)
    {
        return fun(Eigen::Matrix<double, sizeof...(Args), 1>{args...});
    }

    template<typename DerivedPerlinProcess, typename... Args>
    std::enable_if_t<std::conjunction_v<std::is_arithmetic<std::decay_t<Args>>...>,
                     typename DerivedPerlinProcess::template VectorN<double>>
    gradPerlinProcessUnpacked(DerivedPerlinProcess & fun, Args... args)
    {
        return fun.grad(Eigen::Matrix<double, sizeof...(Args), 1>{args...});
    }

    template<typename T, size_t>
    using type_t = T;

    template<typename DerivedPerlinProcess, size_t... Is>
    auto evaluatePerlinProcessUnpackedSignature(
        std::index_sequence<Is...>) -> double (*)(DerivedPerlinProcess &, type_t<double, Is>...);

    template<typename DerivedPerlinProcess, size_t... Is>
    auto gradPerlinProcessUnpackedSignature(std::index_sequence<Is...>) ->
        typename DerivedPerlinProcess::template VectorN<double> (*)(DerivedPerlinProcess &,
                                                                    type_t<double, Is>...);

    template<unsigned int N>
    struct PyPerlinProcessVisitor : public bp::def_visitor<PyPerlinProcessVisitor<N>>
    {
    public:
        template<typename PyClass>
        static void visit(PyClass & cl)
        {
            using DerivedPerlinProcess = typename PyClass::wrapped_type;

            // clang-format off
            cl
                .def("__call__",
                    static_cast<decltype(evaluatePerlinProcessUnpackedSignature<DerivedPerlinProcess>(
                        std::make_index_sequence<N>{}))>(evaluatePerlinProcessUnpacked))
                .def("__call__", &DerivedPerlinProcess::operator(), (bp::arg("self"), "vec"))
                .def("grad",
                    static_cast<decltype(gradPerlinProcessUnpackedSignature<DerivedPerlinProcess>(
                        std::make_index_sequence<N>{}))>(gradPerlinProcessUnpacked))
                .def("grad", &DerivedPerlinProcess::grad, (bp::arg("self"), "vec"))
                .def(
                    "reset",
                    makeFunction(ConvertGeneratorFromPythonAndInvoke<
                        void(const uniform_random_bit_generator_ref<uint32_t> &), DerivedPerlinProcess
                        >(&DerivedPerlinProcess::reset),
                    bp::default_call_policies(),
                    (bp::arg("self"), "generator")))
                .ADD_PROPERTY_GET("wavelength", &DerivedPerlinProcess::getWavelength)
                .ADD_PROPERTY_GET("num_octaves", &DerivedPerlinProcess::getNumOctaves);
            // clang-format on
        }

        static void expose()
        {
            bp::class_<RandomPerlinProcess<N>,
                       // bp::bases<AbstractPerlinProcess<RandomPerlinNoiseOctave, N>>,
                       std::shared_ptr<RandomPerlinProcess<N>>,
                       boost::noncopyable>(
                toString("RandomPerlinProcess", N, "D").c_str(),
                bp::init<double, uint32_t>(
                    (bp::arg("self"), "wavelength", bp::arg("num_octaves") = 6U)))
                .def(PyPerlinProcessVisitor<N>());

            bp::class_<PeriodicPerlinProcess<N>,
                       // bp::bases<AbstractPerlinProcess<PeriodicPerlinNoiseOctave, N>>,
                       std::shared_ptr<PeriodicPerlinProcess<N>>,
                       boost::noncopyable>(
                toString("PeriodicPerlinProcess", N, "D").c_str(),
                bp::init<double, double, uint32_t>(
                    (bp::arg("self"), "wavelength", "period", bp::arg("num_octaves") = 6U)))
                .ADD_PROPERTY_GET("period", &PeriodicPerlinProcess<N>::getPeriod)
                .def(PyPerlinProcessVisitor<N>());
        }
    };

    void exposeGenerators()
    {
        bp::class_<PCG32, std::shared_ptr<PCG32>, boost::noncopyable>(
            "PCG32", bp::init<uint64_t>((bp::arg("self"), "state")))
            .def(bp::init<>((bp::arg("self"))))
            .def("__init__",
                 bp::make_constructor(
                     &makePCG32FromSeedSed, bp::default_call_policies(), (bp::arg("seed_seq"))))
            .def("__call__", &PCG32::operator(), (bp::arg("self")))
            .def("seed", &seedPCG32FromSeedSed, (bp::arg("self"), "seed_seq"))
            .add_static_property(
                "min", &PCG32::min, getPropertySignaturesWithDoc(nullptr, &PCG32::min).c_str())
            .add_static_property(
                "max", &PCG32::max, getPropertySignaturesWithDoc(nullptr, &PCG32::max).c_str());

        bp::implicitly_convertible<PCG32, uniform_random_bit_generator_ref<uint32_t>>();

#define BIND_GENERIC_DISTRIBUTION(dist, arg1, arg2)                                   \
    bp::def(#dist,                                                                    \
            makeFunction(ConvertGeneratorFromPythonAndInvoke(&dist##FromStackedArgs), \
                         bp::default_call_policies(),                                 \
                         (bp::arg("generator"), #arg1, #arg2)));                      \
    bp::def(#dist,                                                                    \
            makeFunction(ConvertGeneratorFromPythonAndInvoke(&dist##FromSize),        \
                         bp::default_call_policies(),                                 \
                         (bp::arg("generator"),                                       \
                          bp::arg(#arg1) = 0.0F,                                      \
                          bp::arg(#arg2) = 1.0F,                                      \
                          bp::arg("size") = bp::object())));

        BIND_GENERIC_DISTRIBUTION(uniform, lo, hi)
        BIND_GENERIC_DISTRIBUTION(normal, mean, stddev)

#undef BIND_GENERIC_DISTRIBUTION

        // Must be declared last to take precedence over generic declaration with default values
        bp::def("uniform",
                makeFunction(
                    ConvertGeneratorFromPythonAndInvoke(
                        static_cast<float (*)(const uniform_random_bit_generator_ref<uint32_t> &)>(
                            &uniform)),
                    bp::default_call_policies(),
                    (bp::arg("generator"))));

        bp::class_<PeriodicTabularProcess,
                   std::shared_ptr<PeriodicTabularProcess>,
                   boost::noncopyable>("PeriodicTabularProcess", bp::no_init)
            .def("__call__", &PeriodicTabularProcess::operator(), (bp::arg("self"), "time"))
            .def("grad", &PeriodicTabularProcess::grad, (bp::arg("self"), "time"))
            .def("reset",
                 makeFunction(ConvertGeneratorFromPythonAndInvoke(&PeriodicTabularProcess::reset),
                              bp::default_call_policies(),
                              (bp::arg("self"), "generator")))
            .ADD_PROPERTY_GET("wavelength", &PeriodicTabularProcess::getWavelength)
            .ADD_PROPERTY_GET("period", &PeriodicTabularProcess::getPeriod);

        bp::class_<PeriodicGaussianProcess,
                   bp::bases<PeriodicTabularProcess>,
                   std::shared_ptr<PeriodicGaussianProcess>,
                   boost::noncopyable>(
            "PeriodicGaussianProcess",
            bp::init<double, double>((bp::arg("self"), "wavelength", "period")));

        bp::class_<PeriodicFourierProcess,
                   bp::bases<PeriodicTabularProcess>,
                   std::shared_ptr<PeriodicFourierProcess>,
                   boost::noncopyable>(
            "PeriodicFourierProcess",
            bp::init<double, double>((bp::arg("self"), "wavelength", "period")));

        /* FIXME: Use template lambda and compile-time for-loop when moving to c++20.
           For reference: https://stackoverflow.com/a/76272348/4820605 */
        PyPerlinProcessVisitor<1>::expose();
        PyPerlinProcessVisitor<2>::expose();
        PyPerlinProcessVisitor<3>::expose();

        bp::def(
            "random_tile_ground",
            &tiles,
            (bp::arg("size"), "height_max", "interp_delta", "sparsity", "orientation", "seed"));
        bp::def("periodic_stairs_ground",
                &periodicStairs,
                (bp::arg("step_width"), "step_height", "step_number", "orientation"));
        bp::def("unidirectional_random_perlin_ground",
                &unidirectionalRandomPerlinGround,
                (bp::arg("wavelength"), "num_octaves", "orientation", "seed"));
        bp::def("unidirectional_periodic_perlin_ground",
                &unidirectionalPeriodicPerlinGround,
                (bp::arg("wavelength"), "period", "num_octaves", "orientation", "seed"));
        bp::def("random_perlin_ground",
                &randomPerlinGround,
                (bp::arg("wavelength"), "num_octaves", "seed"));
        bp::def("periodic_perlin_ground",
                &periodicPerlinGround,
                (bp::arg("wavelength"), "period", "num_octaves", "seed"));
        bp::def("sum_heightmaps", &sumHeightmaps, (bp::arg("heightmaps")));
        bp::def("merge_heightmaps", &mergeHeightmaps, (bp::arg("heightmaps")));

        bp::def("discretize_heightmap",
                &discretizeHeightmap,
                (bp::arg("heightmap"),
                 "x_min",
                 "x_max",
                 "x_unit",
                 "y_min",
                 "y_max",
                 "y_unit",
                 bp::arg("must_simplify") = false));
    }
}
