#include "jiminy/core/telemetry/fwd.h"
#include "jiminy/core/hardware/abstract_sensor.h"
#include "jiminy/core/control/abstract_controller.h"
#include "jiminy/core/control/controller_functor.h"

#define NO_IMPORT_ARRAY
#include "jiminy/python/fwd.h"
#include "jiminy/python/utilities.h"
#include "jiminy/python/functors.h"
#include "jiminy/python/controllers.h"


namespace jiminy::python
{
    namespace bp = boost::python;

    // *********************************** AbstractController ********************************** //

    /* Using an intermediary class is a trick to enable defining `bp::base<...>` in conjunction
       with `bp::wrapper<...>`. */
    class AbstractControllerPyInterface : public AbstractController
    {
    };

    class AbstractControllerPyWrapper :
    public AbstractControllerPyInterface,
        public bp::wrapper<AbstractControllerPyInterface>
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

        void ResetDefault(bool resetDynamicTelemetry)
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

    namespace internal::abstract_controller
    {

        static void initialize(AbstractController & self, const std::shared_ptr<Robot> & robot)
        {
            /* Cannot use input shared pointer because its reference counter is corrupted for some
               reason, making it impossible to use it in conjunction with weak_ptr. The only known
               workaround is using `enable_shared_from_this` trick:
               https://github.com/boostorg/python/issues/189 */
            return self.initialize(robot->shared_from_this());
        }

        static void registerVariable(
            AbstractController & self, const std::string & name, PyObject * dataPy)
        {
            // Note that const qualifier is not supported by PyArray_DATA

            if (!PyArray_Check(dataPy))
            {
                JIMINY_THROW(std::invalid_argument,
                             "'value' input must have type 'numpy.ndarray'.");
            }

            PyArrayObject * dataPyArray = reinterpret_cast<PyArrayObject *>(dataPy);
            if (PyArray_SIZE(dataPyArray) > 1U)
            {
                JIMINY_THROW(std::invalid_argument,
                             "'value' input array must have a single element.");
            }

            if (PyArray_TYPE(dataPyArray) == NPY_FLOAT64)
            {
                auto data = static_cast<double *>(PyArray_DATA(dataPyArray));
                return self.registerVariable(name, *data);
            }
            if (PyArray_TYPE(dataPyArray) == NPY_INT64)
            {
                auto data = static_cast<int64_t *>(PyArray_DATA(dataPyArray));
                return self.registerVariable(name, *data);
            }
            JIMINY_THROW(not_implemented_error,
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
                    JIMINY_THROW(std::invalid_argument,
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
                    JIMINY_THROW(std::invalid_argument,
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
            AbstractController & self, const std::string & name, PyObject * dataPy)
        {
            if (PyBytes_Check(dataPy))
            {
                return self.registerConstant(name, PyBytes_AsString(dataPy));
            }
            if (PyBytes_Check(dataPy))
            {
                return self.registerConstant(name, PyBytes_AsString(dataPy));
            }
            JIMINY_THROW(not_implemented_error, "'value' must have type 'bytes' or 'str'.");
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
    }

    void exposeAbstractController()
    {
        bp::class_<AbstractController, std::shared_ptr<AbstractController>, boost::noncopyable>(
            "AbstractController", bp::no_init)
            .def("initialize",
                 &internal::abstract_controller::initialize,
                 (bp::arg("self"), "robot"))
            .def("reset",
                 &AbstractController::reset,
                 (bp::arg("self"), bp::arg("reset_dynamic_telemetry") = false))
            .def("compute_command",
                 &AbstractController::computeCommand,
                 (bp::arg("self"), "t", "q", "v", "command"))
            .def("internal_dynamics",
                 &AbstractController::internalDynamics,
                 (bp::arg("self"), "t", "q", "v", "u_custom"))
            .ADD_PROPERTY_GET("is_initialized", &AbstractController::getIsInitialized)
            .def("register_constant",
                 &internal::abstract_controller::registerConstant,
                 (bp::arg("self"), "name", "value"))
            .def("register_variable",
                 &internal::abstract_controller::registerVariable,
                 (bp::arg("self"), "name", "value"))
            .def("register_variables",
                 &internal::abstract_controller::registerVariableArray,
                 (bp::arg("self"), "fieldnames", "values"))
            .def("remove_entries", &AbstractController::removeEntries)
            .def("set_options",
                 &internal::abstract_controller::setOptions,
                 (bp::arg("self"), "options"))
            .def("get_options",
                 &AbstractController::getOptions,
                 bp::return_value_policy<bp::return_by_value>())
            .ADD_PROPERTY_GET("robot", &internal::abstract_controller::getRobot)
            .DEF_READONLY("sensor_measurements", &AbstractController::sensorMeasurements_);

        bp::class_<AbstractControllerPyWrapper,
                   bp::bases<AbstractController>,
                   std::shared_ptr<AbstractControllerPyWrapper>,
                   boost::noncopyable>("BaseController")
            .def("reset",
                 &AbstractController::reset,
                 &AbstractControllerPyWrapper::ResetDefault,
                 (bp::arg("self"), bp::arg("reset_dynamic_telemetry") = false))
            .def("compute_command",
                 bp::pure_virtual(&AbstractController::computeCommand),
                 (bp::arg("self"), "t", "q", "v", "command"))
            .def("internal_dynamics",
                 bp::pure_virtual(&AbstractController::internalDynamics),
                 (bp::arg("self"), "t", "q", "v", "u_custom"));
    }

    // ********************************** FunctionalController ********************************* //

    using BaseFunctionalControllerPy =
        FunctionalController<ControllerFunPyWrapper, ControllerFunPyWrapper>;

    class FunctionalControllerPyInterface : public BaseFunctionalControllerPy
    {
        using BaseFunctionalControllerPy::BaseFunctionalControllerPy;
    };

    class FunctionalControllerPyWrapper :
    public FunctionalControllerPyInterface,
        public bp::wrapper<FunctionalControllerPyInterface>
    {
    public:
        using FunctionalControllerPyInterface::FunctionalControllerPyInterface;

        void reset(bool resetDynamicTelemetry)
        {
            bp::override func = this->get_override("reset");
            if (func)
            {
                func(resetDynamicTelemetry);
            }
            else
            {
                return BaseFunctionalControllerPy::reset(resetDynamicTelemetry);
            }
        }

        void ResetDefault(bool resetDynamicTelemetry)
        {
            return this->BaseFunctionalControllerPy::reset(resetDynamicTelemetry);
        }
    };

    namespace internal::functional_controller
    {
        static std::shared_ptr<FunctionalControllerPyWrapper> factory(
            bp::object & commandPy, bp::object & internalDynamicsPy)
        {
            return std::make_shared<FunctionalControllerPyWrapper>(
                ControllerFunPyWrapper(commandPy), ControllerFunPyWrapper(internalDynamicsPy));
        }
    }

    void exposeFunctionalController()
    {
        bp::class_<BaseFunctionalControllerPy,
                   bp::bases<AbstractController>,
                   std::shared_ptr<BaseFunctionalControllerPy>,
                   boost::noncopyable>("BaseFunctionalController", bp::no_init);

        bp::class_<FunctionalControllerPyWrapper,
                   bp::bases<BaseFunctionalControllerPy>,
                   std::shared_ptr<FunctionalControllerPyWrapper>,
                   boost::noncopyable>("FunctionalController", bp::no_init)
            .def("__init__",
                 bp::make_constructor(&internal::functional_controller::factory,
                                      bp::default_call_policies(),
                                      (bp::arg("compute_command") = bp::object(),
                                       bp::arg("internal_dynamics") = bp::object())))
            .def("reset",
                 &BaseFunctionalControllerPy::reset,
                 &FunctionalControllerPyWrapper::ResetDefault,
                 (bp::arg("self"), bp::arg("reset_dynamic_telemetry") = false));
    }
}
