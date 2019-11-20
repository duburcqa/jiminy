///////////////////////////////////////////////////////////////////////////////
///
/// \brief             Python module implementation for Jiminy.
///
////////////////////////////////////////////////////////////////////////////////

// Define the maximum number of sensor types that can accept the 'ControllerFunctor' Python bindings
#define  PYTHON_CONTROLLER_FUNCTOR_MAX_SENSOR_TYPES 6

#include "jiminy/python/Jiminy.h"
#include "jiminy/python/Utilities.h"
#include "jiminy/core/Types.h"

#include <eigenpy/eigenpy.hpp>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>


#if PY_VERSION_HEX >= 0x03000000
    static void* initNumpyC() {
        import_array();
        return NULL;
    }
#else
    static void initNumpyC() {
        import_array();
    }
#endif


namespace jiminy
{
namespace python
{
    namespace bp = boost::python;

    BOOST_PYTHON_MODULE(libjiminy_pywrap)
    {
        // Requirement to handle numpy::ndarray, and create PyArrays<->Eigen automatic converters
        eigenpy::enableEigenPy();
        bp::numpy::initialize();
        initNumpyC();

        // Interfaces for result_t enum
        bp::enum_<result_t>("result_t")
        .value("SUCCESS", result_t::SUCCESS)
        .value("ERROR_GENERIC", result_t::ERROR_GENERIC)
        .value("ERROR_BAD_INPUT", result_t::ERROR_BAD_INPUT)
        .value("ERROR_INIT_FAILED", result_t::ERROR_INIT_FAILED);

        // Interfaces for heatMapType_t enum
        bp::enum_<heatMapType_t>("heatMapType_t")
        .value("CONSTANT", heatMapType_t::CONSTANT)
        .value("STAIRS", heatMapType_t::STAIRS)
        .value("GENERIC", heatMapType_t::GENERIC);

        // Enable some automatic C++ to Python converters
        bp::to_python_converter<std::vector<std::string>, stdVectorToListPyConverter<std::string> >();
        bp::to_python_converter<std::vector<int32_t>, stdVectorToListPyConverter<int32_t> >();
        bp::to_python_converter<std::vector<vectorN_t>, stdVectorToListPyConverter<vectorN_t> >();
        bp::to_python_converter<std::vector<matrixN_t>, stdVectorToListPyConverter<matrixN_t> >();
        bp::to_python_converter<std::vector<matrixN_t>, stdVectorToListPyConverter<matrixN_t> >();

        // Expose classes
        jiminy::python::SensorsDataMapVisitor::expose();
        jiminy::python::PyModelVisitor::expose();
        jiminy::python::PySensorVisitor::expose();
        jiminy::python::PyAbstractControllerVisitor::expose();
        bp::def("ControllerFunctor",
                ControllerFunctorPyFactory,
                (bp::arg("command_handle"), "internal_dynamics_handle", bp::arg("nb_sensor_types")=-1),
                bp::return_value_policy<bp::manage_new_object>());
        jiminy::python::HeatMapFunctorVisitor::expose();
        jiminy::python::PyStepperVisitor::expose();
        jiminy::python::PyEngineVisitor::expose();
    }
}
}
