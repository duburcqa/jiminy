/* If defined the python type of __init__ method "self" parameters is properly generated, Undefined
   by default because it increases binary size by about 14%. */
#define BOOST_PYTHON_PY_SIGNATURES_PROPER_INIT_SELF_TYPE

#include "pinocchio/spatial/force.hpp"  // `Pinocchio::Force`

#include "jiminy/core/utilities/random.h"
#include "jiminy/core/types.h"

/* Eigenpy must be imported first, since it sets pre-processor definitions used by Boost Python
   to configure Python C API. */
#include "pinocchio/bindings/python/fwd.hpp"
#include <boost/python/numpy.hpp>

#include "jiminy/python/compatibility.h"
#include "jiminy/python/utilities.h"
#include "jiminy/python/helpers.h"
#include "jiminy/python/functors.h"
#include "jiminy/python/generators.h"
#include "jiminy/python/engine.h"
#include "jiminy/python/constraints.h"
#include "jiminy/python/controllers.h"
#include "jiminy/python/robot.h"
#include "jiminy/python/motors.h"
#include "jiminy/python/sensors.h"


namespace jiminy::python
{
    namespace bp = boost::python;
    namespace np = boost::python::numpy;

    template<typename T>
    using TimeStateFct = typename std::function<T(
        const float64_t &, const Eigen::VectorXd &, const Eigen::VectorXd &)>;

#define TIME_STATE_FCT_EXPOSE(Name, Type)                                                     \
    bp::class_<TimeStateFct<Type>, boost::noncopyable>("TimeStateFunctor" #Name, bp::no_init) \
        .def("__call__",                                                                      \
             &TimeStateFct<Type>::operator(),                                                 \
             bp::return_value_policy<bp::return_by_value>(),                                  \
             (bp::arg("self"), "t", "q", "v"));

#define REGISTER_CONVERTER(Type, Copy)                                                  \
    {                                                                                   \
        bp::type_info info = bp::type_id<Type>();                                       \
        const bp::converter::registration * reg = bp::converter::registry::query(info); \
        if (reg == nullptr || *reg->m_to_python == nullptr)                             \
        {                                                                               \
            bp::to_python_converter<Type, converterToPython<Type, Copy>, true>();       \
        }                                                                               \
    }

    BOOST_PYTHON_MODULE(PYTHON_LIBRARY_NAME)
    {
        // Initialize Jiminy random number generator
        resetRandomGenerators(0U);

        /* Initialized boost::python::numpy.
           It is required to handle boost::python::numpy::ndarray object directly.

           Note that numpy scalar to native type automatic converters are disabled because they are
           messing up with the docstring. */
        np::initialize(false);
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

        // Interfaces for heightmapType_t enum
        bp::enum_<heightmapType_t>("heightmapType_t")
            .value("CONSTANT", heightmapType_t::CONSTANT)
            .value("STAIRS", heightmapType_t::STAIRS)
            .value("GENERIC", heightmapType_t::GENERIC);

        // Disable CPP docstring
        bp::docstring_options doc_options;
        doc_options.disable_cpp_signatures();

        /* Enable some automatic C++ to Python converters.
           By default, conversion is by-value unless specified explicitly via a call policy. */
        REGISTER_CONVERTER(std::vector<std::vector<int32_t>>, true);
        REGISTER_CONVERTER(std::vector<uint32_t>, true);
        REGISTER_CONVERTER(std::vector<int32_t>, true);
        REGISTER_CONVERTER(std::vector<Eigen::VectorXd>, true);
        REGISTER_CONVERTER(std::vector<Eigen::MatrixXd>, true);
        REGISTER_CONVERTER(configHolder_t, true);

        // Expose functors
        TIME_STATE_FCT_EXPOSE(Bool, bool_t)
        TIME_STATE_FCT_EXPOSE(PinocchioForce, pinocchio::Force)
        exposeHeightmapFunctor();

        /* Expose compatibility layer, to support both new and old C++ ABI, and to restore
           automatic converters of numpy scalars without altering python docstring signature. */
        exposeCompatibility();

        // Expose helpers and generators
        exposeHelpers();
        exposeGenerators();

        // Expose structs and classes
        exposeSensorsDataMap();
        exposeConstraint();
        exposeConstraintsHolder();
        exposeModel();
        exposeRobot();
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
