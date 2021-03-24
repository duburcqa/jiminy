///////////////////////////////////////////////////////////////////////////////
///
/// \brief             Python module implementation for Jiminy.
///
////////////////////////////////////////////////////////////////////////////////

/* If defined the python type of __init__ method "self" parameters is properly generated,
   Undefined by default because it increases binary size by about 14%. */
#define BOOST_PYTHON_PY_SIGNATURES_PROPER_INIT_SELF_TYPE

// Manually import the Python C API to avoid relying on eigenpy and boost::numpy to do so.
#define PY_ARRAY_UNIQUE_SYMBOL JIMINY_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"
#define NO_IMPORT_ARRAY

#include "jiminy/python/Utilities.h"
#include "jiminy/python/Helpers.h"
#include "jiminy/python/Functors.h"
#include "jiminy/python/Engine.h"
#include "jiminy/python/Constraints.h"
#include "jiminy/python/Controllers.h"
#include "jiminy/python/Robot.h"
#include "jiminy/python/Motors.h"
#include "jiminy/python/Sensors.h"

#include "jiminy/core/utilities/Random.h"
#include "jiminy/core/Types.h"

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <eigenpy/eigenpy.hpp>


static void * initNumpy()
{
    import_array();
    return NULL;
}


namespace jiminy
{
namespace python
{
    namespace bp = boost::python;

    template<typename T>
    using TimeStateFct = typename std::function<T (float64_t const &, vectorN_t const &, vectorN_t const &)>;



    #define TIME_STATE_FCT_EXPOSE(Type, Name) \
    bp::class_<TimeStateFct<Type>, boost::noncopyable>("TimeStateFunctor"#Name, bp::no_init) \
        .def("__call__", &TimeStateFct<Type>::operator(), \
                         bp::return_value_policy<bp::return_by_value>(), \
                         (bp::arg("self"), bp::arg("t"), bp::arg("q"), bp::arg("v")));

    template<typename T>
    struct converterToPython
    {
        static PyObject * convert(T const & data)
        {
            return bp::incref(convertToPython<T>(data).ptr());
        }

        static PyTypeObject const * get_pytype()
        {
            std::type_info const * typeId(&typeid(bp::object));
            if constexpr (is_vector<T>::value)
            {
                typeId = &typeid(bp::list);
            }
            else if constexpr (std::is_same<T, configHolder_t>::value
                            || std::is_same<T, flexibleJointData_t>::value)
            {
                typeId = &typeid(bp::dict);
            }
            bp::converter::registration const * r = bp::converter::registry::query(*typeId);
            return r ? r->to_python_target_type(): 0;
        }
    };

    BOOST_PYTHON_MODULE(PYTHON_LIBRARY_NAME)
    {
        // Initialize Jiminy random number generator
        resetRandomGenerators(0);

        // Initialized C API of Python, required to handle raw Python native object
        Py_Initialize();
        // Initialized C API of Numpy, required to handle raw numpy::ndarray object
        initNumpy();
        // Initialized boost::python::numpy, required to handle boost::python::numpy::ndarray object
        bp::numpy::initialize();
        // Initialized EigenPy, enabling PyArrays<->Eigen automatic converters
        eigenpy::enableEigenPy();

        // Expose the version
        bp::scope().attr("__version__") = bp::str(JIMINY_VERSION);
        bp::scope().attr("__raw_version__") = bp::str(JIMINY_VERSION);

        // Interfaces for hresult_t enum
        bp::enum_<hresult_t>("hresult_t")
        .value("SUCCESS", hresult_t::SUCCESS)
        .value("ERROR_GENERIC", hresult_t::ERROR_GENERIC)
        .value("ERROR_BAD_INPUT", hresult_t::ERROR_BAD_INPUT)
        .value("ERROR_INIT_FAILED", hresult_t::ERROR_INIT_FAILED);

        // Interfaces for joint_t enum
        bp::enum_<joint_t>("joint_t")
        .value("NONE", joint_t::NONE)
        .value("LINEAR", joint_t::LINEAR)
        .value("ROTARY", joint_t::ROTARY)
        .value("ROTARY_UNBOUNDED", joint_t::ROTARY_UNBOUNDED)
        .value("PLANAR", joint_t::PLANAR)
        .value("SPHERICAL", joint_t::SPHERICAL)
        .value("FREE", joint_t::FREE);

        // Interfaces for heatMapType_t enum
        bp::enum_<heatMapType_t>("heatMapType_t")
        .value("CONSTANT", heatMapType_t::CONSTANT)
        .value("STAIRS", heatMapType_t::STAIRS)
        .value("GENERIC", heatMapType_t::GENERIC);

        // Enable some automatic C++ to Python converters
        bp::to_python_converter<std::vector<std::string>, converterToPython<std::vector<std::string> >, true>();
        bp::to_python_converter<std::vector<std::vector<int32_t> >, converterToPython<std::vector<std::vector<int32_t> > >, true>();
        bp::to_python_converter<std::vector<int32_t>, converterToPython<std::vector<int32_t> >, true>();
        bp::to_python_converter<std::vector<vectorN_t>, converterToPython<std::vector<vectorN_t> >, true>();
        bp::to_python_converter<std::vector<matrixN_t>, converterToPython<std::vector<matrixN_t> >, true>();
        bp::to_python_converter<configHolder_t, converterToPython<configHolder_t>, true>();

        // Disable CPP docstring
        bp::docstring_options doc_options;
        doc_options.disable_cpp_signatures();

        // Expose functors
        TIME_STATE_FCT_EXPOSE(bool_t, Bool)
        TIME_STATE_FCT_EXPOSE(pinocchio::Force, PinocchioForce)
        exposeHeatMapFunctor();

        // Expose helpers
        exposeHelpers();

        // Expose structs and classes
        exposeSensorsDataMap();
        exposeModel();
        exposeRobot();
        exposeConstraint();
        exposeConstraintsHolder();
        exposeAbstractMotor();
        exposeSimpleMotor();
        exposeAbstractSensor();
        exposeBasicSensors();
        exposeAbstractController();
        exposeControllerFunctor();
        exposeForces();
        exposeStepperState();
        exposeSystemState();
        exposeSystem();
        exposeEngineMultiRobot();
        exposeEngine();
    }

    #undef TIME_STATE_FCT_EXPOSE
}
}
