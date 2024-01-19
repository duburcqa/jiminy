#include "hpp/fcl/BVH/BVH_model.h"  // `hpp::fcl::BVHModel`, `hpp::fcl::OBBRSS`

#include "jiminy/core/utilities/random.h"
#include "jiminy/core/utilities/geometry.h"

#include "jiminy/python/utilities.h"
#include "jiminy/python/generators.h"


namespace jiminy::python
{
    namespace bp = boost::python;

    std::shared_ptr<PCG32> makePCG32FromSeedSed(bp::list & seedSeqPy)
    {
        std::vector<int> seedSeq = bp::extract<std::vector<int>>(seedSeqPy);
        return std::make_shared<PCG32>(std::seed_seq{seedSeq.cbegin(), seedSeq.cend()});
    }

    void seedPCG32FromSeedSed(PCG32 & generator, bp::list & seedSeqPy)
    {
        std::vector<int> seedSeq = bp::extract<std::vector<int>>(seedSeqPy);
        return generator.seed(std::seed_seq{seedSeq.cbegin(), seedSeq.cend()});
    }


#define GENERIC_DISTRIBUTION_WRAPPER(dist, arg1, arg2)                                            \
    Eigen::MatrixXd dist(const uniform_random_bit_generator_ref<uint32_t> & generator,            \
                         np::ndarray & arg1,                                                      \
                         np::ndarray & arg2)                                                      \
    {                                                                                             \
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
        return ::jiminy::dist(generator, cast(arg1), cast(arg2)).cast<double>();                  \
    }

    GENERIC_DISTRIBUTION_WRAPPER(uniform, lo, hi)
    GENERIC_DISTRIBUTION_WRAPPER(normal, mean, stddev)

#undef GENERIC_DISTRIBUTION_WRAPPER

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
            .add_static_property("min", &PCG32::min, getPropertySignaturesWithDoc(nullptr, &PCG32::min).c_str())
            .add_static_property("max", &PCG32::max, getPropertySignaturesWithDoc(nullptr, &PCG32::max).c_str());

        bp::implicitly_convertible<PCG32, uniform_random_bit_generator_ref<uint32_t>>();

#define BIND_GENERIC_DISTRIBUTION(dist, arg1, arg2)                                          \
        bp::def(#dist,                                                                       \
            static_cast<                                                                     \
                float (*)(const uniform_random_bit_generator_ref<uint32_t> &, float, float)  \
            >(&::jiminy::dist),                                                              \
            (bp::arg("generator"), bp::arg(#arg1) = 0.0F, bp::arg(#arg2) = 1.0F));           \
        bp::def(#dist, &dist, (bp::arg("generator"), #arg2, #arg2));                         \
        bp::def(#dist,                                                                       \
            +[](Eigen::Index nrows,                                                          \
                Eigen::Index ncols,                                                          \
                const uniform_random_bit_generator_ref<uint32_t> & generator,                \
                float arg1,                                                                  \
                float arg2                                                                   \
                ) -> Eigen::MatrixXd                                                         \
            {                                                                                \
                return ::jiminy::dist(nrows, ncols, generator, arg1, arg2).cast<double>();   \
            },                                                                               \
            (bp::arg("nrows"), "ncols", "generator",                                         \
             bp::arg(#arg1) = 0.0F, bp::arg(#arg2) = 1.0F));

    BIND_GENERIC_DISTRIBUTION(uniform, lo, hi)
    BIND_GENERIC_DISTRIBUTION(normal, mean, stddev)

#undef BIND_GENERIC_DISTRIBUTION

        // Must be declared last to take precedence over generic declaration with default values
        bp::def("uniform",
            static_cast<
                float (*)(const uniform_random_bit_generator_ref<uint32_t> &)
            >(&::jiminy::uniform),
            (bp::arg("generator")));

        bp::class_<AbstractPerlinProcess,
                   std::shared_ptr<AbstractPerlinProcess>,
                   boost::noncopyable>("AbstractPerlinProcess", bp::no_init)
            .def("__call__", &AbstractPerlinProcess::operator(),
                             (bp::arg("self"), "time"))
            .def("reset", &AbstractPerlinProcess::reset, (bp::arg("self"), "generator"))
            .ADD_PROPERTY_GET_WITH_POLICY("wavelength",
                                          &AbstractPerlinProcess::wavelength,
                                          bp::return_value_policy<bp::return_by_value>())
            .ADD_PROPERTY_GET_WITH_POLICY("num_octaves",
                                          &AbstractPerlinProcess::num_octaves,
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
                                          &PeriodicPerlinProcess::period,
                                          bp::return_value_policy<bp::return_by_value>());

        bp::class_<PeriodicGaussianProcess,
                   std::shared_ptr<PeriodicGaussianProcess>,
                   boost::noncopyable>("PeriodicGaussianProcess",
                   bp::init<double, double>(
                   (bp::arg("self"), "wavelength", "period")))
            .def("__call__", &PeriodicGaussianProcess::operator(),
                             (bp::arg("self"), bp::arg("time")))
            .def("reset", &PeriodicGaussianProcess::reset, (bp::arg("self"), "generator"))
            .ADD_PROPERTY_GET_WITH_POLICY("wavelength",
                                          &PeriodicGaussianProcess::wavelength,
                                          bp::return_value_policy<bp::return_by_value>())
            .ADD_PROPERTY_GET_WITH_POLICY("period",
                                          &PeriodicGaussianProcess::period,
                                          bp::return_value_policy<bp::return_by_value>());

        bp::class_<PeriodicFourierProcess,
                   std::shared_ptr<PeriodicFourierProcess>,
                   boost::noncopyable>("PeriodicFourierProcess",
                   bp::init<double, double>(
                   (bp::arg("self"), "wavelength", "period")))
            .def("__call__", &PeriodicFourierProcess::operator(),
                             (bp::arg("self"), bp::arg("time")))
            .def("reset", &PeriodicFourierProcess::reset, (bp::arg("self"), "generator"))
            .ADD_PROPERTY_GET_WITH_POLICY("wavelength",
                                          &PeriodicFourierProcess::wavelength,
                                          bp::return_value_policy<bp::return_by_value>())
            .ADD_PROPERTY_GET_WITH_POLICY("period",
                                          &PeriodicFourierProcess::period,
                                          bp::return_value_policy<bp::return_by_value>());

        bp::def("random_tile_ground", &tiles,
                                      (bp::arg("size"), "height_max", "interp_delta", "sparsity", "orientation", "seed"));
        bp::def("sum_heightmaps", &sumHeightmaps, (bp::arg("heightmaps")));
        bp::def("merge_heightmaps", &mergeHeightmaps, (bp::arg("heightmaps")));

        bp::def("discretize_heightmap", &discretizeHeightmap,
                                        (bp::arg("heightmap"), "x_min", "x_max", "x_unit", "y_min", "y_max", "y_unit",
                                         bp::arg("must_simplify") = false));
        // clang-format on
    }
}
