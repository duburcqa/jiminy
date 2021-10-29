///////////////////////////////////////////////////////////////////////////////
///
/// \brief             Python module implementation for Jiminy.
///
////////////////////////////////////////////////////////////////////////////////

/* If defined the python type of __init__ method "self" parameters is properly generated,
   Undefined by default because it increases binary size by about 14%. */
#define BOOST_PYTHON_PY_SIGNATURES_PROPER_INIT_SELF_TYPE

#include "pinocchio/spatial/force.hpp"  // `Pinocchio::Force`

#include "jiminy/core/utilities/Random.h"
#include "jiminy/core/Types.h"

/* Eigenpy must be imported first, since it sets pre-processor
   definitions used by Boost Python to configure Python C API. */
#include "eigenpy/eigenpy.hpp"
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

#include "jiminy/python/Compatibility.h"
#include "jiminy/python/Utilities.h"
#include "jiminy/python/Helpers.h"
#include "jiminy/python/Functors.h"
#include "jiminy/python/Generators.h"
#include "jiminy/python/Engine.h"
#include "jiminy/python/Constraints.h"
#include "jiminy/python/Controllers.h"
#include "jiminy/python/Robot.h"
#include "jiminy/python/Motors.h"
#include "jiminy/python/Sensors.h"


namespace jiminy
{
namespace python
{
    namespace bp = boost::python;
    namespace np = boost::python::numpy;

    template<typename T>
    using TimeStateFct = typename std::function<T (float64_t const &, vectorN_t const &, vectorN_t const &)>;

    #define TIME_STATE_FCT_EXPOSE(Type, Name) \
    bp::class_<TimeStateFct<Type>, boost::noncopyable>("TimeStateFunctor"#Name, bp::no_init) \
        .def("__call__", &TimeStateFct<Type>::operator(), \
                         bp::return_value_policy<bp::return_by_value>(), \
                         (bp::arg("self"), bp::arg("t"), bp::arg("q"), bp::arg("v")));

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
        np::initialize(false);
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

        // Interfaces for heightmapType_t enum
        bp::enum_<heightmapType_t>("heightmapType_t")
        .value("CONSTANT", heightmapType_t::CONSTANT)
        .value("STAIRS", heightmapType_t::STAIRS)
        .value("GENERIC", heightmapType_t::GENERIC);

        // Disable CPP docstring
        bp::docstring_options doc_options;
        doc_options.disable_cpp_signatures();

        // Enable some automatic C++ to Python converters
        bp::to_python_converter<std::vector<std::string>, converterToPython<std::vector<std::string> >, true>();
        bp::to_python_converter<std::vector<std::vector<int32_t> >, converterToPython<std::vector<std::vector<int32_t> > >, true>();
        bp::to_python_converter<std::vector<uint32_t>, converterToPython<std::vector<uint32_t> >, true>();
        bp::to_python_converter<std::vector<int32_t>, converterToPython<std::vector<int32_t> >, true>();
        bp::to_python_converter<std::vector<vectorN_t>, converterToPython<std::vector<vectorN_t> >, true>();
        bp::to_python_converter<std::vector<matrixN_t>, converterToPython<std::vector<matrixN_t> >, true>();
        bp::to_python_converter<configHolder_t, converterToPython<configHolder_t>, true>();

        // Expose functors
        TIME_STATE_FCT_EXPOSE(bool_t, Bool)
        TIME_STATE_FCT_EXPOSE(pinocchio::Force, PinocchioForce)
        exposeHeightmapFunctor();

        /* Expose compatibility layer, to support both new and old C++ ABI, and to
           restore automatic converters of numpy scalars without altering python
           docstring signature. */
        exposeCompatibility();

        // Expose helpers and generators
        exposeHelpers();
        exposeGenerators();

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
