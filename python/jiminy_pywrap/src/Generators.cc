#include "jiminy/core/utilities/Random.h"

#include "jiminy/python/Utilities.h"
#include "jiminy/python/Generators.h"


namespace jiminy
{
namespace python
{
    namespace bp = boost::python;

    heightMapFunctor_t sumHeightMap(bp::list const & heightMapsPy)
    {
        auto heightMaps = convertFromPython<std::vector<heightMapFunctor_t> >(heightMapsPy);
        return ::jiminy::sumHeightMap(heightMaps);
    }

    heightMapFunctor_t mergeHeightMap(bp::list const & heightMapsPy)
    {
        auto heightMaps = convertFromPython<std::vector<heightMapFunctor_t> >(heightMapsPy);
        return ::jiminy::mergeHeightMap(heightMaps);
    }

    void exposeGenerators(void)
    {
        bp::def("reset_random_generator", &resetRandomGenerators, (bp::arg("seed") = bp::object()));

        bp::class_<RandomPerlinProcess,
                   std::shared_ptr<RandomPerlinProcess>,
                   boost::noncopyable>("RandomPerlinProcess",
                   bp::init<float64_t const &, uint32_t const &>(
                   (bp::arg("self"), "wavelength", bp::arg("num_octaves") = 6U)))
            .def("__call__", &RandomPerlinProcess::operator(),
                             (bp::arg("self"), bp::arg("time")))
            .def("reset", &RandomPerlinProcess::reset)
            .add_property("wavelength", bp::make_function(&RandomPerlinProcess::getWavelength,
                                        bp::return_value_policy<bp::copy_const_reference>()))
            .add_property("num_octaves", bp::make_function(&RandomPerlinProcess::getNumOctaves,
                                         bp::return_value_policy<bp::copy_const_reference>()));

        bp::class_<PeriodicPerlinProcess,
                   std::shared_ptr<PeriodicPerlinProcess>,
                   boost::noncopyable>("PeriodicPerlinProcess",
                   bp::init<float64_t const &, float64_t const &, uint32_t const &>(
                   (bp::arg("self"), "wavelength", "period", bp::arg("num_octaves") = 6U)))
            .def("__call__", &PeriodicPerlinProcess::operator(),
                             (bp::arg("self"), bp::arg("time")))
            .def("reset", &PeriodicPerlinProcess::reset)
            .add_property("wavelength", bp::make_function(&PeriodicPerlinProcess::getWavelength,
                                        bp::return_value_policy<bp::copy_const_reference>()))
            .add_property("period", bp::make_function(&PeriodicPerlinProcess::getPeriod,
                                    bp::return_value_policy<bp::copy_const_reference>()))
            .add_property("num_octaves", bp::make_function(&PeriodicPerlinProcess::getNumOctaves,
                                         bp::return_value_policy<bp::copy_const_reference>()));

        bp::class_<PeriodicGaussianProcess,
                   std::shared_ptr<PeriodicGaussianProcess>,
                   boost::noncopyable>("PeriodicGaussianProcess",
                   bp::init<float64_t const &, float64_t const &>(
                   bp::args("self", "wavelength", "period")))
            .def("__call__", &PeriodicGaussianProcess::operator(),
                             (bp::arg("self"), bp::arg("time")))
            .def("reset", &PeriodicGaussianProcess::reset)
            .add_property("wavelength", bp::make_function(&PeriodicGaussianProcess::getWavelength,
                                        bp::return_value_policy<bp::copy_const_reference>()))
            .add_property("period", bp::make_function(&PeriodicGaussianProcess::getPeriod,
                                    bp::return_value_policy<bp::copy_const_reference>()))
            .add_property("dt", bp::make_function(&PeriodicGaussianProcess::getDt,
                                bp::return_value_policy<bp::copy_const_reference>()));

        bp::class_<PeriodicFourierProcess,
                   std::shared_ptr<PeriodicFourierProcess>,
                   boost::noncopyable>("PeriodicFourierProcess",
                   bp::init<float64_t const &, float64_t const &>(
                   bp::args("self", "wavelength", "period")))
            .def("__call__", &PeriodicFourierProcess::operator(),
                             (bp::arg("self"), bp::arg("time")))
            .def("reset", &PeriodicFourierProcess::reset)
            .add_property("wavelength", bp::make_function(&PeriodicFourierProcess::getWavelength,
                                        bp::return_value_policy<bp::copy_const_reference>()))
            .add_property("period", bp::make_function(&PeriodicFourierProcess::getPeriod,
                                    bp::return_value_policy<bp::copy_const_reference>()))
            .add_property("dt", bp::make_function(&PeriodicFourierProcess::getDt,
                                bp::return_value_policy<bp::copy_const_reference>()))
            .add_property("num_harmonics", bp::make_function(&PeriodicFourierProcess::getNumHarmonics,
                                           bp::return_value_policy<bp::copy_const_reference>()));

        bp::def("random_tile_ground", &randomTileGround,
                                      bp::args("tile_size", "sparsity", "tile_height_max", "tile_interp_delta", "seed"));

        bp::def("sum_heightmap", &sumHeightMap, bp::args("heightmaps"));
        bp::def("merge_heightmap", &mergeHeightMap, bp::args("heightmaps"));

        bp::def("discretize_heightmap", &discretizeHeightmap, bp::args("heightmap", "grid_size", "grid_unit"));
    }

}  // End of namespace python.
}  // End of namespace jiminy.



