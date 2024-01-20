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

    void exposeGenerators()
    {
        // clang-format off
        bp::class_<AbstractPerlinProcess,
                   std::shared_ptr<AbstractPerlinProcess>,
                   boost::noncopyable>("AbstractPerlinProcess", bp::no_init)
            .def("__call__", &AbstractPerlinProcess::operator(),
                             (bp::arg("self"), "time"))
            .def("reset", &AbstractPerlinProcess::reset)
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
            .def("reset", &PeriodicGaussianProcess::reset)
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
            .def("reset", &PeriodicFourierProcess::reset)
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
