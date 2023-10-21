#include "jiminy/core/utilities/random.h"

#include "jiminy/python/utilities.h"
#include "jiminy/python/generators.h"


namespace jiminy::python
{
    namespace bp = boost::python;

    heightmapFunctor_t sumHeightmap(const bp::list & heightmapsPy)
    {
        auto heightmaps = convertFromPython<std::vector<heightmapFunctor_t>>(heightmapsPy);
        return ::jiminy::sumHeightmap(heightmaps);
    }

    heightmapFunctor_t mergeHeightmap(const bp::list & heightmapsPy)
    {
        auto heightmaps = convertFromPython<std::vector<heightmapFunctor_t>>(heightmapsPy);
        return ::jiminy::mergeHeightmap(heightmaps);
    }

    void resetRandomGenerators(const bp::object & seedPy)
    {
        std::optional<uint32_t> seed = std::nullopt;
        if (!seedPy.is_none())
        {
            seed = bp::extract<uint32_t>(seedPy);
        }
        ::jiminy::resetRandomGenerators(seed);
    }

    void exposeGenerators(void)
    {
        // clang-format off
        bp::def("reset_random_generator", &resetRandomGenerators, (bp::arg("seed") = bp::object()));

        bp::class_<AbstractPerlinProcess,
                   std::shared_ptr<AbstractPerlinProcess>,
                   boost::noncopyable>("AbstractPerlinProcess", bp::no_init)
            .def("__call__", &AbstractPerlinProcess::operator(),
                             (bp::arg("self"), bp::arg("time")))
            .def("reset", &AbstractPerlinProcess::reset)
            .ADD_PROPERTY_GET_WITH_POLICY("wavelength",
                                          &AbstractPerlinProcess::getWavelength,
                                          bp::return_value_policy<bp::return_by_value>())
            .ADD_PROPERTY_GET_WITH_POLICY("num_octaves",
                                          &AbstractPerlinProcess::getNumOctaves,
                                          bp::return_value_policy<bp::return_by_value>())
            .ADD_PROPERTY_GET_WITH_POLICY("scale",
                                          &AbstractPerlinProcess::getScale,
                                          bp::return_value_policy<bp::return_by_value>());

        bp::class_<RandomPerlinProcess, bp::bases<AbstractPerlinProcess>,
                   std::shared_ptr<RandomPerlinProcess>,
                   boost::noncopyable>("RandomPerlinProcess",
                   bp::init<const float64_t &, const float64_t &, const uint32_t &>(
                   (bp::arg("self"), "wavelength", bp::arg("scale") = 1.0, bp::arg("num_octaves") = 6U)));

        bp::class_<PeriodicPerlinProcess, bp::bases<AbstractPerlinProcess>,
                   std::shared_ptr<PeriodicPerlinProcess>,
                   boost::noncopyable>("PeriodicPerlinProcess",
                   bp::init<const float64_t &, const float64_t &, const float64_t &, const uint32_t &>(
                   (bp::arg("self"), "wavelength", "period", bp::arg("scale") = 1.0, bp::arg("num_octaves") = 6U)))
            .ADD_PROPERTY_GET_WITH_POLICY("period",
                                          &PeriodicPerlinProcess::getPeriod,
                                          bp::return_value_policy<bp::return_by_value>());

        bp::class_<PeriodicGaussianProcess,
                   std::shared_ptr<PeriodicGaussianProcess>,
                   boost::noncopyable>("PeriodicGaussianProcess",
                   bp::init<const float64_t &, const float64_t &, const float64_t &>(
                   (bp::arg("self"), "wavelength", "period", bp::arg("scale") = 1.0)))
            .def("__call__", &PeriodicGaussianProcess::operator(),
                             (bp::arg("self"), bp::arg("time")))
            .def("reset", &PeriodicGaussianProcess::reset)
            .ADD_PROPERTY_GET_WITH_POLICY("wavelength",
                                          &PeriodicGaussianProcess::getWavelength,
                                          bp::return_value_policy<bp::return_by_value>())
            .ADD_PROPERTY_GET_WITH_POLICY("period",
                                          &PeriodicGaussianProcess::getPeriod,
                                          bp::return_value_policy<bp::return_by_value>())
            .ADD_PROPERTY_GET_WITH_POLICY("dt",
                                          &PeriodicGaussianProcess::getDt,
                                          bp::return_value_policy<bp::return_by_value>());

        bp::class_<PeriodicFourierProcess,
                   std::shared_ptr<PeriodicFourierProcess>,
                   boost::noncopyable>("PeriodicFourierProcess",
                   bp::init<const float64_t &, const float64_t &, const float64_t &>(
                   (bp::arg("self"), "wavelength", "period", bp::arg("scale") = 1.0)))
            .def("__call__", &PeriodicFourierProcess::operator(),
                             (bp::arg("self"), bp::arg("time")))
            .def("reset", &PeriodicFourierProcess::reset)
            .ADD_PROPERTY_GET_WITH_POLICY("wavelength",
                                          &PeriodicFourierProcess::getWavelength,
                                          bp::return_value_policy<bp::return_by_value>())
            .ADD_PROPERTY_GET_WITH_POLICY("period",
                                          &PeriodicFourierProcess::getPeriod,
                                          bp::return_value_policy<bp::return_by_value>())
            .ADD_PROPERTY_GET_WITH_POLICY("dt",
                                          &PeriodicFourierProcess::getDt,
                                          bp::return_value_policy<bp::return_by_value>())
            .ADD_PROPERTY_GET_WITH_POLICY("num_harmonics",
                                          &PeriodicFourierProcess::getNumHarmonics,
                                          bp::return_value_policy<bp::return_by_value>());

        bp::def("random_tile_ground", &randomTileGround,
                                      (bp::arg("size"), "height_max", "interp_delta", "sparsity", "orientation", "seed"));
        bp::def("sum_heightmap", &sumHeightmap, (bp::arg("heightmaps")));
        bp::def("merge_heightmap", &mergeHeightmap, (bp::arg("heightmaps")));

        bp::def("discretize_heightmap", &discretizeHeightmap, (bp::arg("heightmap"), "grid_size", "grid_unit"));
        // clang-format on
    }
}
