///////////////////////////////////////////////////////////////////////////////
///
/// \brief             Python module implementation for Jiminy.
///
////////////////////////////////////////////////////////////////////////////////

// Manually import the Python C API to avoid relying on eigenpy to do so, to be compatible with any version.
// The PY_ARRAY_UNIQUE_SYMBOL cannot be changed, since its value is enforced by boost::numpy without checking
// if already defined... Luckily, eigenpy is more clever and does the check on its side so that they can work together.
#define PY_ARRAY_UNIQUE_SYMBOL BOOST_NUMPY_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/ndarrayobject.h"
#define NO_IMPORT_ARRAY

#include "jiminy/python/Jiminy.h"
#include "jiminy/python/Utilities.h"
#include "jiminy/core/Types.h"

#include <eigenpy/eigenpy.hpp>
#include <boost/python.hpp>
#include <boost/python/scope.hpp>
#include <boost/python/numpy.hpp>


namespace jiminy
{
namespace python
{
    namespace bp = boost::python;

    template<typename T>
    using TimeStateFct = typename std::function<T (float64_t const&, vectorN_t const&, vectorN_t const&)>;

    #define TIME_STATE_FCT_EXPOSE(T) \
    bp::class_<TimeStateFct<T>, boost::noncopyable>("TimeStateFunctor_"#T, bp::no_init) \
        .def("__call__", &TimeStateFct<T>::operator(), \
                         bp::return_value_policy<bp::return_by_value>(), \
                         (bp::arg("self"), bp::arg("t"), bp::arg("q"), bp::arg("v")));

    template<typename T>
    struct converterToPython
    {
        static PyObject * convert(T const & data)
        {
            return bp::incref(convertToPython<T>(data).ptr());
        }
    };

    BOOST_PYTHON_MODULE(libjiminy_pywrap)
    {
        // Required to initialized Python C API
        Py_Initialize();
        // Required to handle numpy::ndarray object (it loads Python C API of Numpy) and ufunc
        bp::numpy::initialize();
        // Required and create PyArrays<->Eigen automatic converters.
        eigenpy::enableEigenPy();

        // Expose the version
        bp::scope().attr("__version__") = bp::str(JIMINY_VERSION);
        bp::scope().attr("__raw_version__") = bp::str(JIMINY_VERSION);

        // Interfaces for hresult_t enum
        bp::enum_<hresult_t>("hresult_t")
        .value("SUCCESS",           hresult_t::SUCCESS)
        .value("ERROR_GENERIC",     hresult_t::ERROR_GENERIC)
        .value("ERROR_BAD_INPUT",   hresult_t::ERROR_BAD_INPUT)
        .value("ERROR_INIT_FAILED", hresult_t::ERROR_INIT_FAILED);

        // Interfaces for heatMapType_t enum
        bp::enum_<heatMapType_t>("heatMapType_t")
        .value("CONSTANT", heatMapType_t::CONSTANT)
        .value("STAIRS",   heatMapType_t::STAIRS)
        .value("GENERIC",  heatMapType_t::GENERIC);

        // Enable some automatic C++ to Python converters
        bp::to_python_converter<std::vector<std::string>, converterToPython<std::vector<std::string> > >();
        bp::to_python_converter<std::vector<int32_t>,     converterToPython<std::vector<int32_t> > >();
        bp::to_python_converter<std::vector<vectorN_t>,   converterToPython<std::vector<vectorN_t> > >();
        bp::to_python_converter<std::vector<matrixN_t>,   converterToPython<std::vector<matrixN_t> > >();
        bp::to_python_converter<configHolder_t,           converterToPython<configHolder_t> >();

        // Expose classes
        TIME_STATE_FCT_EXPOSE(bool_t)
        TIME_STATE_FCT_EXPOSE(pinocchio::Force)
        HeatMapFunctorVisitor::expose();
        SensorsDataMapVisitor::expose();
        PyModelVisitor::expose();
        PyRobotVisitor::expose();
        PyMotorVisitor::expose();
        PySensorVisitor::expose();
        PyAbstractControllerVisitor::expose();
        PyControllerFunctorVisitor::expose();
        PyStepperStateVisitor::expose();
        PySystemStateVisitor::expose();
        PySystemDataVisitor::expose();
        PyEngineMultiRobotVisitor::expose();
        PyEngineVisitor::expose();
    }

    #undef TIME_STATE_FCT_EXPOSE
}
}
