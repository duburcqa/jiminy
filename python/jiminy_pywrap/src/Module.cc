///////////////////////////////////////////////////////////////////////////////
///
/// \brief             Python module implementation for Jiminy.
///
////////////////////////////////////////////////////////////////////////////////

/* If defined the python type of __init__ method "self" parameters is properly generated,
   Undefined by default because it increases binary size by about 14%. */
#define BOOST_PYTHON_PY_SIGNATURES_PROPER_INIT_SELF_TYPE

#include "pinocchio/fwd.hpp"
#include "eigenpy/eigenpy.hpp"

#include "jiminy/core/utilities/Random.h"
#include "jiminy/core/Types.h"

#include "jiminy/python/Utilities.h"
#include "jiminy/python/Helpers.h"
#include "jiminy/python/Functors.h"
#include "jiminy/python/Engine.h"
#include "jiminy/python/Constraints.h"
#include "jiminy/python/Controllers.h"
#include "jiminy/python/Robot.h"
#include "jiminy/python/Motors.h"
#include "jiminy/python/Sensors.h"

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>


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

    template <typename T>
    class arrayScalarFromPython
    {
    public:
        static void * convertible(PyObject * obj)
        {
            PyTypeObject const * pytype = reinterpret_cast<PyArray_Descr*>(
                bp::numpy::dtype::get_builtin<T>().ptr())->typeobj;
            return (obj->ob_type == pytype) ? obj : 0;
        }

        static void convert(PyObject * obj, bp::converter::rvalue_from_python_stage1_data * data)
        {
            void * storage = reinterpret_cast<bp::converter::rvalue_from_python_storage<T>*>(data)->storage.bytes;
            PyArray_ScalarAsCtype(obj, reinterpret_cast<T*>(storage));
            data->convertible = storage;
        }

        static void declare()
        {
            // Note that no `get_pytype` is provided, so that the already existing one will be used
            bp::converter::registry::push_back(&convertible, &convert, bp::type_id<T>());
        }
    };

    uint32_t getRandomSeed(void)
    {
        uint32_t seed;
        ::jiminy::getRandomSeed(seed);  // Cannot fail since random number generators are initialized when imported
        return seed;
    }

    BOOST_PYTHON_MODULE(PYTHON_LIBRARY_NAME)
    {
        // Initialize Jiminy random number generator
        resetRandomGenerators(0U);

        /* Initialized boost::python::numpy.
           It is required to handle boost::python::numpy::ndarray object directly.
           Note that numpy scalar to native type automatic converters are disabled
           because they are messing up with the docstring. */
        bp::numpy::initialize(false);
        // Initialized EigenPy, enabling PyArrays<->Eigen automatic converters
        eigenpy::enableEigenPy();

        // Expose the version
        bp::scope().attr("__version__") = bp::str(JIMINY_VERSION);
        bp::scope().attr("__raw_version__") = bp::str(JIMINY_VERSION);

        bp::def("get_random_seed", bp::make_function(&getRandomSeed,
                                   bp::return_value_policy<bp::return_by_value>()));

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

        // Add some automatic C++ to Python converters for numpy array of scalars,
        // which is different from a 0-dimensional numpy array.
        arrayScalarFromPython<bool>::declare();
        arrayScalarFromPython<npy_uint8>::declare();
        arrayScalarFromPython<npy_uint32>::declare();
        arrayScalarFromPython<npy_float32>::declare();

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
