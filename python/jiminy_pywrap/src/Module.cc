///////////////////////////////////////////////////////////////////////////////
///
/// \brief             Python module implementation for Jiminy.
///
////////////////////////////////////////////////////////////////////////////////

// Manually import the Python C API to avoid relying on eigenpy and boost::numpy to do so.
#define PY_ARRAY_UNIQUE_SYMBOL JIMINY_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"
#define NO_IMPORT_ARRAY

#include "jiminy/python/Utilities.h"
#include "jiminy/python/Engine.h"
#include "jiminy/python/Constraints.h"
#include "jiminy/python/Controllers.h"
#include "jiminy/python/Robot.h"
#include "jiminy/python/Motors.h"
#include "jiminy/python/Sensors.h"
#include "jiminy/python/Functors.h"

#include "jiminy/core/Types.h"
#include "jiminy/core/Utilities.h"
#include "jiminy/core/engine/PinocchioOverloadAlgorithms.h"

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

    joint_t getJointTypeFromIdx(pinocchio::Model const & model,
                                int32_t          const & idIn)
    {
        joint_t jointType = joint_t::NONE;
        ::jiminy::getJointTypeFromIdx(model, idIn, jointType);
        return jointType;
    }

    bool_t isPositionValid(pinocchio::Model const & model,
                           vectorN_t        const & position)
    {
        bool_t isValid;
        hresult_t returnCode = ::jiminy::isPositionValid(
            model, position, isValid, Eigen::NumTraits<float64_t>::dummy_precision());
        if (returnCode != hresult_t::SUCCESS)
        {
            return false;
        }
        return isValid;
    }

    matrixN_t interpolate(pinocchio::Model const & modelIn,
                          vectorN_t        const & timesIn,
                          matrixN_t        const & positionsIn,
                          vectorN_t        const & timesOut)
    {
        matrixN_t positionOut;
        ::jiminy::interpolate(modelIn, timesIn, positionsIn, timesOut, positionOut);
        return positionOut;
    }

    BOOST_PYTHON_MODULE(PYTHON_LIBRARY_NAME)
    {
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
        bp::to_python_converter<std::vector<std::string>, converterToPython<std::vector<std::string> > >();
        bp::to_python_converter<std::vector<std::vector<int32_t> >, converterToPython<std::vector<std::vector<int32_t> > > >();
        bp::to_python_converter<std::vector<int32_t>, converterToPython<std::vector<int32_t> > >();
        bp::to_python_converter<std::vector<vectorN_t>, converterToPython<std::vector<vectorN_t> > >();
        bp::to_python_converter<std::vector<matrixN_t>, converterToPython<std::vector<matrixN_t> > >();
        bp::to_python_converter<configHolder_t, converterToPython<configHolder_t> >();

        // Disable CPP docstring
        bp::docstring_options doc_options;
        doc_options.disable_cpp_signatures();

        // Expose generic utilities
        bp::def("get_joint_type", &getJointTypeFromIdx,
                                  (bp::arg("pinocchio_model"), "joint_idx"));
        bp::def("is_position_valid", &isPositionValid,
                                     (bp::arg("pinocchio_model"), "position"));
        bp::def("interpolate", &interpolate,
                               (bp::arg("pinocchio_model"), "times_in", "positions_in", "times_out"));
        bp::def("aba",
                &pinocchio_overload::aba<
                    float64_t, 0, pinocchio::JointCollectionDefaultTpl, vectorN_t, vectorN_t, vectorN_t, pinocchio::Force>,
                bp::args("pinocchio_model", "pinocchio_data", "q", "v", "u", "fext"),
                "Compute ABA with external forces, store the result in Data::ddq and return it.",
                bp::return_value_policy<bp::return_by_value>());
        bp::def("rnea",
                &pinocchio_overload::aba<
                    float64_t, 0, pinocchio::JointCollectionDefaultTpl, vectorN_t, vectorN_t, vectorN_t, pinocchio::Force>,
                bp::args("pinocchio_model", "pinocchio_data", "q", "v", "a", "fext"),
                "Compute the RNEA with external forces, store the result in Data and return it.",
                bp::return_value_policy<bp::return_by_value>());
        bp::def("crba",
                &pinocchio_overload::crba<
                    float64_t, 0, pinocchio::JointCollectionDefaultTpl, vectorN_t>,
                bp::args("pinocchio_model", "pinocchio_data", "q"),
                "Computes CRBA, store the result in Data and return it.",
                bp::return_value_policy<bp::return_by_value>());
        bp::def("computeKineticEnergy",
                &pinocchio_overload::kineticEnergy<
                    float64_t, 0, pinocchio::JointCollectionDefaultTpl, vectorN_t, vectorN_t>,
                bp::args("pinocchio_model", "pinocchio_data", "q", "v", "update_kinematics"),
                "Computes the forward kinematics and the kinematic energy of the model for the "
                "given joint configuration and velocity given as input. "
                "The result is accessible through data.kinetic_energy.");

        // Expose functors
        TIME_STATE_FCT_EXPOSE(bool_t)
        TIME_STATE_FCT_EXPOSE(pinocchio::Force)
        exposeHeatMapFunctor();

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
        exposeStepperState();
        exposeSystemState();
        exposeSystem();
        exposeEngineMultiRobot();
        exposeEngine();
    }

    #undef TIME_STATE_FCT_EXPOSE
}
}
