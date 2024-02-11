/* If defined the python type of __init__ method "self" parameters is properly generated, Undefined
   by default because it increases binary size by about 14%. */
#define BOOST_PYTHON_PY_SIGNATURES_PROPER_INIT_SELF_TYPE

#include "pinocchio/spatial/force.hpp"  // `Pinocchio::Force`

#include "jiminy/core/utilities/random.h"

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
    void exposeTimeStateFunc(const std::string_view & name)
    {
        using TimeStateFun =
            typename std::function<T(double, const Eigen::VectorXd &, const Eigen::VectorXd &)>;

        bp::class_<TimeStateFun, boost::noncopyable>(name.data(), bp::no_init)
            .def("__call__",
                 &TimeStateFun::operator(),
                 bp::return_value_policy<bp::return_by_value>(),
                 (bp::arg("self"), "t", "q", "v"));
    }

    template<typename T>
    void registerToPythonByValueConverter()
    {
        bp::type_info info = bp::type_id<T>();
        const bp::converter::registration * reg = bp::converter::registry::query(info);
        if (reg == nullptr || *reg->m_to_python == nullptr)
        {
            bp::to_python_converter<T, converterToPython<const T &, true>, true>();
        }
    }

    BOOST_PYTHON_MODULE(PYTHON_LIBRARY_NAME)
    {
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

        // Interfaces for JointModelType enum
        bp::enum_<JointModelType>("JointModelType")
            .value("NONE", JointModelType::UNSUPPORTED)
            .value("LINEAR", JointModelType::LINEAR)
            .value("ROTARY", JointModelType::ROTARY)
            .value("ROTARY_UNBOUNDED", JointModelType::ROTARY_UNBOUNDED)
            .value("PLANAR", JointModelType::PLANAR)
            .value("SPHERICAL", JointModelType::SPHERICAL)
            .value("FREE", JointModelType::FREE);

        // Interfaces for heightmapType_t enum
        bp::enum_<heightmapType_t>("heightmapType_t")
            .value("CONSTANT", heightmapType_t::CONSTANT)
            .value("STAIRS", heightmapType_t::STAIRS)
            .value("GENERIC", heightmapType_t::GENERIC);

        /* Expose some standard and Jiminy-specific exceptions.
           The tree of native Python exceptions and their corresponding C API are available here:
           - https://docs.python.org/3/library/exceptions.html#exception-hierarchy
           - https://docs.python.org/3/c-api/exceptions.html#standard-exceptions */
        PyObject * PyExc_LogicError = registerException<std::logic_error>("LogicError");
        registerException<std::ios_base::failure>("OSError", PyExc_OSError);
        registerException<not_implemented_error>("NotImplementedError", PyExc_LogicError);
        registerException<bad_control_flow>("BadControlFlow", PyExc_LogicError);
        registerException<lookup_error>("LookupError", PyExc_LookupError);

        // Disable CPP docstring
        bp::docstring_options doc_options;
        doc_options.disable_cpp_signatures();

        /* Enable some automatic C++ to Python converters.
           By default, conversion is by-value unless specified explicitly via a call policy. */
        registerToPythonByValueConverter<GenericConfig>();

        // Expose functors
        exposeTimeStateFunc<bool>("TimeStateBoolFunctor");
        exposeTimeStateFunc<pinocchio::Force>("TimeStateForceFunctor");
        exposeHeightmapFunction();

        /* Expose compatibility layer, to support both new and old C++ ABI, and to restore
           automatic converters of numpy scalars without altering python docstring signature. */
        exposeCompatibility();

        // Expose helpers and generators
        exposeHelpers();
        exposeGenerators();

        // Expose structs and classes
        exposeSensorMeasurementTree();
        exposeConstraint();
        exposeConstraintTree();
        exposeModel();
        exposeRobot();
        exposeAbstractMotor();
        exposeSimpleMotor();
        exposeAbstractSensor();
        exposeBasicSensors();
        exposeAbstractController();
        exposeFunctionalController();
        exposeForces();
        exposeStepperState();
        exposeSystemState();
        exposeSystem();
        exposeEngineMultiRobot();
        exposeEngine();
    }

#undef TIME_STATE_FCT_EXPOSE
}
