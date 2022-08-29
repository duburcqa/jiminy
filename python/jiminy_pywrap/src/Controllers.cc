#include "jiminy/core/control/AbstractController.h"
#include "jiminy/core/control/ControllerFunctor.h"

#include <boost/python.hpp>

#include "jiminy/python/Utilities.h"
#include "jiminy/python/Functors.h"
#include "jiminy/python/Controllers.h"


namespace jiminy
{
namespace python
{
    namespace bp = boost::python;

    // ***************************** PyAbstractControllerVisitor ***********************************

    // Using an intermediary class is a trick to enable defining bp::base<...> in conjunction with bp::wrapper<...>
    class AbstractControllerImpl: public AbstractController {};

    class AbstractControllerWrapper: public AbstractControllerImpl, public bp::wrapper<AbstractControllerImpl>
    {
    public:
        hresult_t reset(bool_t const & resetDynamicTelemetry)
        {
            bp::override func = this->get_override("reset");
            if (func)
            {
                func(resetDynamicTelemetry);
                return hresult_t::SUCCESS;
            }
            return AbstractController::reset(resetDynamicTelemetry);
        }

        hresult_t default_reset(bool_t const & resetDynamicTelemetry)
        {
            return this->AbstractController::reset(resetDynamicTelemetry);
        }

        hresult_t computeCommand(float64_t const & t,
                                 vectorN_t const & q,
                                 vectorN_t const & v,
                                 vectorN_t       & command)
        {
            bp::override func = this->get_override("compute_command");
            if (func)
            {
                func(t,
                     FctPyWrapperArgToPython(q),
                     FctPyWrapperArgToPython(v),
                     FctPyWrapperArgToPython(command));
            }
            return hresult_t::SUCCESS;
        }

        hresult_t internalDynamics(float64_t const & t,
                                   vectorN_t const & q,
                                   vectorN_t const & v,
                                   vectorN_t       & uCustom)
        {
            bp::override func = this->get_override("internal_dynamics");
            if (func)
            {
                func(t,
                     FctPyWrapperArgToPython(q),
                     FctPyWrapperArgToPython(v),
                     FctPyWrapperArgToPython(uCustom));
            }
            return hresult_t::SUCCESS;
        }
    };

    struct PyAbstractControllerVisitor
        : public bp::def_visitor<PyAbstractControllerVisitor>
    {
    public:
        ///////////////////////////////////////////////////////////////////////////////
        /// \brief Expose C++ API through the visitor.
        ///////////////////////////////////////////////////////////////////////////////
        template<class PyClass>
        void visit(PyClass & cl) const
        {
            cl
                .def("initialize", &PyAbstractControllerVisitor::initialize,
                                   (bp::arg("self"), "robot"))
                .def("reset", &AbstractController::reset,
                              (bp::arg("self"), bp::arg("reset_dynamic_telemetry") = false))
                .add_property("is_initialized", bp::make_function(&AbstractController::getIsInitialized,
                                                bp::return_value_policy<bp::copy_const_reference>()))
                .def("register_variable", &PyAbstractControllerVisitor::registerVariable,
                                          (bp::arg("self"), "fieldname", "value"),
                                          "@copydoc AbstractController::registerVariable")
                .def("register_variables", &PyAbstractControllerVisitor::registerVariableArray,
                                           (bp::arg("self"), "fieldnames", "values"))
                .def("register_constants", &PyAbstractControllerVisitor::registerConstant,
                                           (bp::arg("self"), "fieldnames", "values"))
                .def("remove_entries", &AbstractController::removeEntries)
                .def("set_options", &PyAbstractControllerVisitor::setOptions)
                .def("get_options", &AbstractController::getOptions)
                .add_property("robot", &PyAbstractControllerVisitor::getRobot)
                .add_property("sensors_data", bp::make_getter(&AbstractController::sensorsData_,
                                              bp::return_internal_reference<>()))
                ;
        }

        static hresult_t initialize(AbstractController           & self,
                                    std::shared_ptr<Robot> const & robot)
        {
            /* Cannot use input shared pointer because its reference counter is corrupted for some reason,
               making it impossible to use it in conjunction with weak_ptr. The only known workaround is
               using `enable_shared_from_this` trick: https://github.com/boostorg/python/issues/189 */
            return self.initialize(robot->shared_from_this());
        }

        static hresult_t registerVariable(AbstractController       & self,
                                          std::string        const & fieldName,
                                          PyObject                 * dataPy)
        {
            // Note that const qualifier is not supported by PyArray_DATA

            if (PyArray_Check(dataPy))
            {
                PyArrayObject * dataPyArray = reinterpret_cast<PyArrayObject *>(dataPy);
                if (PyArray_TYPE(dataPyArray) == NPY_FLOAT64 && PyArray_SIZE(dataPyArray) == 1U)
                {
                    float64_t const * data = static_cast<float64_t *>(PyArray_DATA(dataPyArray));
                    return self.registerVariable(fieldName, *data);
                }
                else
                {
                    PRINT_ERROR("'value' input array must have dtype 'np.float64' and a single element.");
                    return hresult_t::ERROR_BAD_INPUT;
                }
            }
            else
            {
                PRINT_ERROR("'value' input must have type 'numpy.ndarray'.");
                return hresult_t::ERROR_BAD_INPUT;
            }
        }

        static hresult_t registerVariableArray(AbstractController       & self,
                                               bp::list           const & fieldNamesPy,
                                               PyObject                 * dataPy)
        {
            // Note that const qualifier is not supported by PyArray_DATA

            // Get Eigen::Map from Numpy array
            auto pair = getEigenReference(dataPy);  // Structured bindings is not supported by gcc<7.3, and tie cannot be used since Eigen::Map has no default constructor
            auto returnCode = std::get<0>(pair); auto data = std::get<1>(pair);

            if (returnCode == hresult_t::SUCCESS)
            {
                // Check if fieldnames are stored in one or two dimensional list
                if (bp::len(fieldNamesPy) > 0 && bp::extract<std::string>(fieldNamesPy[0]).check())
                {
                    // Extract fieldnames
                    auto fieldnames = convertFromPython<std::vector<std::string> >(fieldNamesPy);

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
                    auto fieldnames = convertFromPython<std::vector<std::vector<std::string> > >(fieldNamesPy);

                    // Check fieldnames and array have same shape
                    bool_t are_fieldnames_valid = (static_cast<std::size_t>(data.rows()) == fieldnames.size());
                    for (std::vector<std::string> const & subfieldnames : fieldnames)
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
            }

            return returnCode;
        }

        static hresult_t registerConstant(AbstractController       & self,
                                          std::string        const & fieldName,
                                          PyObject                 * dataPy)
        {
            if (PyArray_Check(dataPy))
            {
                auto pair = getEigenReference(dataPy);
                auto returnCode = std::get<0>(pair); auto data = std::get<1>(pair);
                if (returnCode == hresult_t::SUCCESS)
                {
                    returnCode = self.registerConstant(fieldName, data);
                }
                return returnCode;
            }
            else if (PyFloat_Check(dataPy))
            {
                return self.registerConstant(fieldName, PyFloat_AsDouble(dataPy));
            }
            else if (PyLong_Check(dataPy))
            {
                return self.registerConstant(fieldName, PyLong_AsLong(dataPy));
            }
            else if (PyBytes_Check(dataPy))
            {
                return self.registerConstant(fieldName, PyBytes_AsString(dataPy));
            }
            else if (PyUnicode_Check(dataPy))
            {
                return self.registerConstant(fieldName, PyUnicode_AsUTF8(dataPy));
            }
            else
            {
                PRINT_ERROR("'value' type is unsupported.");
                return hresult_t::ERROR_BAD_INPUT;
            }
        }

        static void setOptions(AbstractController       & self,
                               bp::dict           const & configPy)
        {
            configHolder_t config = self.getOptions();
            convertFromPython(configPy, config);
            self.setOptions(config);
        }

        static std::shared_ptr<Robot> getRobot(AbstractController & self)
        {
            return std::const_pointer_cast<Robot>(self.robot_.lock());  // It is not possible to keep constness
        }

        ///////////////////////////////////////////////////////////////////////////////
        /// \brief Expose.
        ///////////////////////////////////////////////////////////////////////////////
        static void expose()
        {
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
        }
    };

    BOOST_PYTHON_VISITOR_EXPOSE(AbstractController)

    // ***************************** PyControllerFunctorVisitor ***********************************

    /* Take advantage of type erasure of std::function to support both
       lambda functions and python handle wrapper depending whether or not
       'compute_command' and 'internal_dynamics' has been specified.
       It is likely to cause a small overhead because the compiler will
       probably not be able to inline ControllerFctWrapper, as it would have
       been the case otherwise, but it is the price to pay for versatility. */
    using CtrlFunctor = ControllerFunctor<ControllerFct, ControllerFct>;

    class CtrlFunctorImpl: public CtrlFunctor {};

    class CtrlFunctorWrapper: public CtrlFunctorImpl, public bp::wrapper<CtrlFunctorImpl>
    {
    public:
        hresult_t reset(bool_t const & resetDynamicTelemetry)
        {
            bp::override func = this->get_override("reset");
            if (func)
            {
                func(resetDynamicTelemetry);
                return hresult_t::SUCCESS;
            }
            return CtrlFunctor::reset(resetDynamicTelemetry);
        }

        hresult_t default_reset(bool_t const & resetDynamicTelemetry)
        {
            return this->CtrlFunctor::reset(resetDynamicTelemetry);
        }
    };

    struct PyControllerFunctorVisitor
        : public bp::def_visitor<PyControllerFunctorVisitor>
    {
    public:
        ///////////////////////////////////////////////////////////////////////////////
        /// \brief Expose C++ API through the visitor.
        ///////////////////////////////////////////////////////////////////////////////
        template<class PyClass>
        void visit(PyClass & cl) const
        {
            cl
                .def("compute_command", &AbstractController::computeCommand,
                                        (bp::arg("self"), "t", "q", "v", "command"))
                .def("internal_dynamics", &AbstractController::internalDynamics,
                                          (bp::arg("self"), "t", "q", "v", "u_custom"));
                ;
        }

        static std::shared_ptr<CtrlFunctor> factory(bp::object & commandPy,
                                                    bp::object & internalDynamicsPy)
        {
            ControllerFct commandFct;
            if (!commandPy.is_none())
            {
                commandFct = ControllerFctWrapper(commandPy);
            }
            else
            {
                commandFct = [](float64_t        const & /* t */,
                                vectorN_t        const & /* q */,
                                vectorN_t        const & /* v */,
                                sensorsDataMap_t const & /* sensorsData */,
                                vectorN_t              & /* command */) {};
            }
            ControllerFct internalDynamicsFct;
            if (!internalDynamicsPy.is_none())
            {
                internalDynamicsFct = ControllerFctWrapper(internalDynamicsPy);
            }
            else
            {
                internalDynamicsFct = [](float64_t        const & /* t */,
                                         vectorN_t        const & /* q */,
                                         vectorN_t        const & /* v */,
                                         sensorsDataMap_t const & /* sensorsData */,
                                         vectorN_t              & /* command */) {};
            }
            return std::make_shared<CtrlFunctor>(std::move(commandFct),
                                                 std::move(internalDynamicsFct));
        }

        ///////////////////////////////////////////////////////////////////////////////
        /// \brief Expose.
        ///////////////////////////////////////////////////////////////////////////////
        static void expose()
        {
            bp::class_<CtrlFunctor, bp::bases<AbstractController>,
                       std::shared_ptr<CtrlFunctor>,
                       boost::noncopyable>("AbstractControllerFunctor", bp::no_init)
                .def(PyControllerFunctorVisitor());

            bp::class_<CtrlFunctorWrapper, bp::bases<CtrlFunctor>,
                       std::shared_ptr<CtrlFunctorWrapper>,
                       boost::noncopyable>("ControllerFunctor", bp::no_init)
                .def("__init__", bp::make_constructor(&PyControllerFunctorVisitor::factory,
                                 bp::default_call_policies(),
                                (bp::arg("compute_command") = bp::object(),
                                 bp::arg("internal_dynamics") = bp::object())))
                .def("reset", &CtrlFunctor::reset, &CtrlFunctorWrapper::default_reset,
                              (bp::arg("self"), bp::arg("reset_dynamic_telemetry") = false));
        }
    };

    BOOST_PYTHON_VISITOR_EXPOSE(ControllerFunctor)
}  // End of namespace python.
}  // End of namespace jiminy.
