#include "jiminy/core/telemetry/fwd.h"
#include "jiminy/core/hardware/abstract_sensor.h"
#include "jiminy/core/control/abstract_controller.h"
#include "jiminy/core/control/controller_functor.h"

#include "pinocchio/bindings/python/fwd.hpp"

#include "jiminy/python/utilities.h"
#include "jiminy/python/functors.h"
#include "jiminy/python/controllers.h"


namespace jiminy::python
{
    namespace bp = boost::python;

    // ***************************** PyAbstractControllerVisitor **********************************

    /* Using an intermediary class is a trick to enable defining `bp::base<...>` in conjunction
       with `bp::wrapper<...>`. */
    class AbstractControllerImpl : public AbstractController
    {
    };

    class AbstractControllerWrapper :
    public AbstractControllerImpl,
        public bp::wrapper<AbstractControllerImpl>
    {
    public:
        void reset(bool resetDynamicTelemetry)
        {
            bp::override func = this->get_override("reset");
            if (func)
            {
                func(resetDynamicTelemetry);
            }
            else
            {
                AbstractController::reset(resetDynamicTelemetry);
            }
        }

        void default_reset(bool resetDynamicTelemetry)
        {
            return this->AbstractController::reset(resetDynamicTelemetry);
        }

        void computeCommand(double t,
                            const Eigen::VectorXd & q,
                            const Eigen::VectorXd & v,
                            Eigen::VectorXd & command)
        {
            bp::override func = this->get_override("compute_command");
            if (func)
            {
                func(t,
                     FunPyWrapperArgToPython(q),
                     FunPyWrapperArgToPython(v),
                     FunPyWrapperArgToPython(command));
            }
        }

        void internalDynamics(double t,
                              const Eigen::VectorXd & q,
                              const Eigen::VectorXd & v,
                              Eigen::VectorXd & uCustom)
        {
            bp::override func = this->get_override("internal_dynamics");
            if (func)
            {
                func(t,
                     FunPyWrapperArgToPython(q),
                     FunPyWrapperArgToPython(v),
                     FunPyWrapperArgToPython(uCustom));
            }
        }
    };

    struct PyAbstractControllerVisitor : public bp::def_visitor<PyAbstractControllerVisitor>
    {
    public:
        template<class PyClass>
        void visit(PyClass & cl) const
        {
            // clang-format off
            cl
                .def("initialize", &PyAbstractControllerVisitor::initialize,
                                   (bp::arg("self"), "robot"))
                .def("reset", &AbstractController::reset,
                              (bp::arg("self"), bp::arg("reset_dynamic_telemetry") = false))
                .ADD_PROPERTY_GET_WITH_POLICY("is_initialized",
                                              &AbstractController::getIsInitialized,
                                              bp::return_value_policy<bp::return_by_value>())
                .def("register_variable", &PyAbstractControllerVisitor::registerVariable,
                                          (bp::arg("self"), "fieldname", "value"),
                                          "@copydoc AbstractController::registerVariable")
                .def("register_variables", &PyAbstractControllerVisitor::registerVariableArray,
                                           (bp::arg("self"), "fieldnames", "values"))
                .def("register_constants", &PyAbstractControllerVisitor::registerConstant,
                                           (bp::arg("self"), "fieldnames", "values"))
                .def("remove_entries", &AbstractController::removeEntries)
                .def("set_options", &PyAbstractControllerVisitor::setOptions,
                                    (bp::arg("self"), "options"))
                .def("get_options", &AbstractController::getOptions)
                .ADD_PROPERTY_GET("robot", &PyAbstractControllerVisitor::getRobot)
                .DEF_READONLY("sensor_measurements", &AbstractController::sensorMeasurements_)
                ;
            // clang-format on
        }

        static void initialize(AbstractController & self, const std::shared_ptr<Robot> & robot)
        {
            /* Cannot use input shared pointer because its reference counter is corrupted for some
               reason, making it impossible to use it in conjunction with weak_ptr. The only known
               workaround is using `enable_shared_from_this` trick:
               https://github.com/boostorg/python/issues/189 */
            return self.initialize(robot->shared_from_this());
        }

        static void registerVariable(
            AbstractController & self, const std::string & fieldname, PyObject * dataPy)
        {
            // Note that const qualifier is not supported by PyArray_DATA

            if (!PyArray_Check(dataPy))
            {
                THROW_ERROR(std::invalid_argument,
                            "'value' input must have type 'numpy.ndarray'.");
            }

            PyArrayObject * dataPyArray = reinterpret_cast<PyArrayObject *>(dataPy);
            if (PyArray_SIZE(dataPyArray) > 1U)
            {
                THROW_ERROR(std::invalid_argument,
                            "'value' input array must have a single element.");
            }

            if (PyArray_TYPE(dataPyArray) == NPY_FLOAT64)
            {
                auto data = static_cast<double *>(PyArray_DATA(dataPyArray));
                return self.registerVariable(fieldname, *data);
            }
            if (PyArray_TYPE(dataPyArray) == NPY_INT64)
            {
                auto data = static_cast<int64_t *>(PyArray_DATA(dataPyArray));
                return self.registerVariable(fieldname, *data);
            }
            THROW_ERROR(not_implemented_error,
                        "'value' input array must have dtype 'np.float64' or 'np.int64'.");
        }

        template<typename Scalar>
        static void registerVariableArrayImpl(
            AbstractController & self,
            const bp::list & fieldnamesPy,
            Eigen::Map<MatrixX<Scalar>, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> & data)
        {
            // Check if fieldnames are stored in one or two dimensional list
            if (bp::len(fieldnamesPy) > 0 && bp::extract<std::string>(fieldnamesPy[0]).check())
            {
                // Extract fieldnames
                auto fieldnames = convertFromPython<std::vector<std::string>>(fieldnamesPy);

                // Check fieldnames and array have same length
                if (static_cast<std::size_t>(data.size()) != fieldnames.size())
                {
                    THROW_ERROR(std::invalid_argument,
                                "'values' input array must have same length than 'fieldnames'.");
                }

                // Register all variables at once
                self.registerVariable(fieldnames, data.col(0));
            }
            else
            {
                // Extract fieldnames
                auto fieldnames =
                    convertFromPython<std::vector<std::vector<std::string>>>(fieldnamesPy);

                // Check fieldnames and array have same shape
                bool are_fieldnames_valid = static_cast<std::size_t>(data.rows()) ==
                                            fieldnames.size();
                for (const std::vector<std::string> & subfieldnames : fieldnames)
                {
                    if (static_cast<std::size_t>(data.cols()) != subfieldnames.size())
                    {
                        are_fieldnames_valid = false;
                        break;
                    }
                }
                if (!are_fieldnames_valid)
                {
                    THROW_ERROR(std::invalid_argument,
                                "'fieldnames' must be nested list with same shape than 'value'.");
                }

                // Register rows sequentially
                for (Eigen::Index i = 0; i < data.rows(); ++i)
                {
                    self.registerVariable(fieldnames[i], data.row(i));
                }
            }
        }

        static void registerVariableArray(
            AbstractController & self, const bp::list & fieldnamesPy, PyObject * dataPy)
        {
            return std::visit([&](auto && arg)
                              { return registerVariableArrayImpl(self, fieldnamesPy, arg); },
                              getEigenReference(dataPy));
        }

        static void registerConstant(
            AbstractController & self, const std::string & fieldname, PyObject * dataPy)
        {
            if (PyArray_Check(dataPy))
            {
                return std::visit([&](auto && arg)
                                  { return self.registerConstant(fieldname, arg); },
                                  getEigenReference(dataPy));
            }
            if (PyFloat_Check(dataPy))
            {
                return self.registerConstant(fieldname, PyFloat_AsDouble(dataPy));
            }
            if (PyLong_Check(dataPy))
            {
                return self.registerConstant(fieldname, PyLong_AsLong(dataPy));
            }
            if (PyBytes_Check(dataPy))
            {
                return self.registerConstant(fieldname, PyBytes_AsString(dataPy));
            }
            if (PyUnicode_Check(dataPy))
            {
                return self.registerConstant(fieldname, PyUnicode_AsUTF8(dataPy));
            }
            THROW_ERROR(not_implemented_error, "'value' type is unsupported.");
        }

        static void setOptions(AbstractController & self, const bp::dict & configPy)
        {
            GenericConfig config = self.getOptions();
            convertFromPython(configPy, config);
            return self.setOptions(config);
        }

        static std::shared_ptr<Robot> getRobot(AbstractController & self)
        {
            // It is not possible to keep constness
            return std::const_pointer_cast<Robot>(self.robot_.lock());
        }

        static void expose()
        {
            // clang-format off
            bp::class_<AbstractController,
                       std::shared_ptr<AbstractController>,
                       boost::noncopyable>("AbstractController", bp::no_init)
                .def(PyAbstractControllerVisitor());

            bp::class_<AbstractControllerWrapper, bp::bases<AbstractController>,
                       std::shared_ptr<AbstractControllerWrapper>,
                       boost::noncopyable>("BaseController")
                .def("reset", &AbstractController::reset, &AbstractControllerWrapper::default_reset,
                              (bp::arg("self"), bp::arg("reset_dynamic_telemetry") = false))
                .def("compute_command", bp::pure_virtual(&AbstractController::computeCommand),
                                        (bp::arg("self"), "t", "q", "v", "command"))
                .def("internal_dynamics", bp::pure_virtual(&AbstractController::internalDynamics),
                                          (bp::arg("self"), "t", "q", "v", "u_custom"));
            // clang-format on
        }
    };

    BOOST_PYTHON_VISITOR_EXPOSE(AbstractController)

    // ***************************** PyFunctionalControllerVisitor ***************************** //

    using FunctionalControllerPyBase =
        FunctionalController<ControllerFunPyWrapper, ControllerFunPyWrapper>;

    class FunctionalControllerPyInterface : public FunctionalControllerPyBase
    {
    };

    class FunctionalControllerPy :
    public FunctionalControllerPyInterface,
        public bp::wrapper<FunctionalControllerPyInterface>
    {
    public:
        void reset(bool resetDynamicTelemetry)
        {
            bp::override func = this->get_override("reset");
            if (func)
            {
                func(resetDynamicTelemetry);
            }
            else
            {
                return FunctionalControllerPyBase::reset(resetDynamicTelemetry);
            }
        }

        void default_reset(bool resetDynamicTelemetry)
        {
            return this->FunctionalControllerPyBase::reset(resetDynamicTelemetry);
        }
    };

    struct PyFunctionalControllerVisitor : public bp::def_visitor<PyFunctionalControllerVisitor>
    {
    public:
        /// \brief Expose C++ API through the visitor.
        template<class PyClass>
        void visit(PyClass & cl) const
        {
            // clang-format off
            cl
                .def("compute_command", &AbstractController::computeCommand,
                                        (bp::arg("self"), "t", "q", "v", "command"))
                .def("internal_dynamics", &AbstractController::internalDynamics,
                                          (bp::arg("self"), "t", "q", "v", "u_custom"));
                ;
            // clang-format on
        }

        static std::shared_ptr<FunctionalControllerPyBase> factory(bp::object & commandPy,
                                                                   bp::object & internalDynamicsPy)
        {
            return std::make_shared<FunctionalControllerPyBase>(
                ControllerFunPyWrapper(commandPy), ControllerFunPyWrapper(internalDynamicsPy));
        }

        static void expose()
        {
            // clang-format off
            bp::class_<FunctionalControllerPyBase, bp::bases<AbstractController>,
                       std::shared_ptr<FunctionalControllerPyBase>,
                       boost::noncopyable>("AbstractFunctionalController", bp::no_init)
                .def(PyFunctionalControllerVisitor());

            bp::class_<FunctionalControllerPy, bp::bases<FunctionalControllerPyBase>,
                       std::shared_ptr<FunctionalControllerPy>,
                       boost::noncopyable>("FunctionalController", bp::no_init)
                .def("__init__", bp::make_constructor(&PyFunctionalControllerVisitor::factory,
                                 bp::default_call_policies(),
                                (bp::arg("compute_command") = bp::object(),
                                 bp::arg("internal_dynamics") = bp::object())))
                .def("reset", &FunctionalControllerPyBase::reset, &FunctionalControllerPy::default_reset,
                              (bp::arg("self"), bp::arg("reset_dynamic_telemetry") = false));
            // clang-format on
        }
    };

    BOOST_PYTHON_VISITOR_EXPOSE(FunctionalController)
}
