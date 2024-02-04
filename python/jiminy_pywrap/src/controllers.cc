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
        hresult_t reset(bool resetDynamicTelemetry)
        {
            bp::override func = this->get_override("reset");
            if (func)
            {
                func(resetDynamicTelemetry);
                return hresult_t::SUCCESS;
            }
            return AbstractController::reset(resetDynamicTelemetry);
        }

        hresult_t default_reset(bool resetDynamicTelemetry)
        {
            return this->AbstractController::reset(resetDynamicTelemetry);
        }

        hresult_t computeCommand(double t,
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
            return hresult_t::SUCCESS;
        }

        hresult_t internalDynamics(double t,
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
            return hresult_t::SUCCESS;
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

        static hresult_t initialize(AbstractController & self,
                                    const std::shared_ptr<Robot> & robot)
        {
            /* Cannot use input shared pointer because its reference counter is corrupted for some
               reason, making it impossible to use it in conjunction with weak_ptr. The only known
               workaround is using `enable_shared_from_this` trick:
               https://github.com/boostorg/python/issues/189 */
            return self.initialize(robot->shared_from_this());
        }

        static hresult_t registerVariable(
            AbstractController & self, const std::string & fieldname, PyObject * dataPy)
        {
            // Note that const qualifier is not supported by PyArray_DATA

            if (PyArray_Check(dataPy))
            {
                PyArrayObject * dataPyArray = reinterpret_cast<PyArrayObject *>(dataPy);
                if (PyArray_SIZE(dataPyArray) <= 1U)
                {
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
                    else
                    {
                        PRINT_ERROR(
                            "'value' input array must have dtype 'np.float64' or 'np.int64'.");
                        return hresult_t::ERROR_BAD_INPUT;
                    }
                }
                else
                {
                    PRINT_ERROR("'value' input array must have a single element.");
                    return hresult_t::ERROR_BAD_INPUT;
                }
            }
            else
            {
                PRINT_ERROR("'value' input must have type 'numpy.ndarray'.");
                return hresult_t::ERROR_BAD_INPUT;
            }
        }

        template<typename Scalar>
        static hresult_t registerVariableArrayImpl(
            AbstractController & self,
            const bp::list & fieldnamesPy,
            Eigen::Map<MatrixX<Scalar>, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> & data)
        {
            hresult_t returnCode = hresult_t::SUCCESS;

            // Check if fieldnames are stored in one or two dimensional list
            if (bp::len(fieldnamesPy) > 0 && bp::extract<std::string>(fieldnamesPy[0]).check())
            {
                // Extract fieldnames
                auto fieldnames = convertFromPython<std::vector<std::string>>(fieldnamesPy);

                // Check fieldnames and array have same length
                if (static_cast<std::size_t>(data.size()) != fieldnames.size())
                {
                    PRINT_ERROR("'values' input array must have same length than 'fieldnames'.");
                    returnCode = hresult_t::ERROR_BAD_INPUT;
                }

                // Register all variables at once
                if (returnCode == hresult_t::SUCCESS)
                {
                    returnCode = self.registerVariable(fieldnames, data.col(0));
                }
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
                    PRINT_ERROR("'fieldnames' must be nested list with same shape than 'value'.");
                    returnCode = hresult_t::ERROR_BAD_INPUT;
                }

                // Register rows sequentially
                for (Eigen::Index i = 0; i < data.rows(); ++i)
                {
                    if (returnCode == hresult_t::SUCCESS)
                    {
                        returnCode = self.registerVariable(fieldnames[i], data.row(i));
                    }
                }
            }

            return returnCode;
        }

        static hresult_t registerVariableArray(
            AbstractController & self, const bp::list & fieldnamesPy, PyObject * dataPy)
        {
            auto data = getEigenReference(dataPy);
            if (!data)
            {
                return hresult_t::ERROR_BAD_INPUT;
            }
            return std::visit([&](auto && arg)
                              { return registerVariableArrayImpl(self, fieldnamesPy, arg); },
                              data.value());
        }

        static hresult_t registerConstant(
            AbstractController & self, const std::string & fieldname, PyObject * dataPy)
        {
            if (PyArray_Check(dataPy))
            {
                auto data = getEigenReference(dataPy);
                if (!data)
                {
                    return hresult_t::ERROR_BAD_INPUT;
                }
                return std::visit([&](auto && arg)
                                  { return self.registerConstant(fieldname, arg); },
                                  data.value());
            }
            else if (PyFloat_Check(dataPy))
            {
                return self.registerConstant(fieldname, PyFloat_AsDouble(dataPy));
            }
            else if (PyLong_Check(dataPy))
            {
                return self.registerConstant(fieldname, PyLong_AsLong(dataPy));
            }
            else if (PyBytes_Check(dataPy))
            {
                return self.registerConstant(fieldname, PyBytes_AsString(dataPy));
            }
            else if (PyUnicode_Check(dataPy))
            {
                return self.registerConstant(fieldname, PyUnicode_AsUTF8(dataPy));
            }
            else
            {
                PRINT_ERROR("'value' type is unsupported.");
                return hresult_t::ERROR_BAD_INPUT;
            }
        }

        static hresult_t setOptions(AbstractController & self, const bp::dict & configPy)
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
        hresult_t reset(bool resetDynamicTelemetry)
        {
            bp::override func = this->get_override("reset");
            if (func)
            {
                func(resetDynamicTelemetry);
                return hresult_t::SUCCESS;
            }
            return FunctionalControllerPyBase::reset(resetDynamicTelemetry);
        }

        hresult_t default_reset(bool resetDynamicTelemetry)
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
