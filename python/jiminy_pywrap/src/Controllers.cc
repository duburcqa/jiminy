#include "jiminy/core/control/AbstractController.h"
#include "jiminy/core/control/ControllerFunctor.h"

#include "jiminy/python/Functors.h"
#include "jiminy/python/Controllers.h"

#include <boost/python.hpp>


namespace jiminy
{
namespace python
{
    namespace bp = boost::python;

    // ***************************** PyAbstractControllerVisitor ***********************************

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
                .add_property("is_initialized", bp::make_function(&AbstractController::getIsInitialized,
                                                bp::return_value_policy<bp::copy_const_reference>()))
                .def("register_variable", &PyAbstractControllerVisitor::registerVariable,
                                          (bp::arg("self"), "fieldname", "value"),
                                          "@copydoc AbstractController::registerVariable")
                .def("register_variables", &PyAbstractControllerVisitor::registerVariableVector,
                                           (bp::arg("self"), "fieldnames", "values"))
                .def("register_constants", &PyAbstractControllerVisitor::registerConstant,
                                           (bp::arg("self"), "fieldnames", "values"))
                .def("remove_entries", &AbstractController::removeEntries)
                .def("set_options", &PyAbstractControllerVisitor::setOptions)
                .def("get_options", &AbstractController::getOptions)
                .add_property("robot", bp::make_function(&AbstractController::getRobot,
                                       bp::return_internal_reference<>()))
                ;
        }

        static void initialize(AbstractController           & self,
                               std::shared_ptr<Robot> const & robot)
        {
            self.initialize(robot.get());
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
                    float64_t const * data = (float64_t *) PyArray_DATA(dataPyArray);
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

        static hresult_t registerVariableVector(AbstractController       & self,
                                                bp::list           const & fieldNamesPy,
                                                PyObject                 * dataPy)
        {
            // Note that const qualifier is not supported by PyArray_DATA

            if (PyArray_Check(dataPy))
            {
                auto fieldnames = convertFromPython<std::vector<std::string> >(fieldNamesPy);
                PyArrayObject * dataPyArray = reinterpret_cast<PyArrayObject *>(dataPy);
                if (PyArray_TYPE(dataPyArray) == NPY_FLOAT64 && PyArray_SIZE(dataPyArray) == uint32_t(fieldnames.size()))
                {
                    Eigen::Map<vectorN_t> data((float64_t *) PyArray_DATA(dataPyArray), PyArray_SIZE(dataPyArray));
                    return self.registerVariable(fieldnames, data);
                }
                else
                {
                    PRINT_ERROR("'values' input array must have dtype 'np.float64' and the same length as 'fieldnames'.");
                    return hresult_t::ERROR_BAD_INPUT;
                }
            }
            else
            {
                PRINT_ERROR("'values' input must have type 'numpy.ndarray'.");
                return hresult_t::ERROR_BAD_INPUT;
            }
        }

        static hresult_t registerConstant(AbstractController       & self,
                                          std::string        const & fieldName,
                                          PyObject                 * dataPy)
        {
            if (PyArray_Check(dataPy))
            {
                PyArrayObject * dataPyArray = reinterpret_cast<PyArrayObject *>(dataPy);

                int dataPyArrayDtype = PyArray_TYPE(dataPyArray);
                if (dataPyArrayDtype != NPY_FLOAT64)
                {
                    PRINT_ERROR("The only dtype supported for 'numpy.ndarray' is float.");
                    return hresult_t::ERROR_BAD_INPUT;
                }
                float64_t * dataPyArrayData = (float64_t *) PyArray_DATA(dataPyArray);
                int dataPyArrayNdims = PyArray_NDIM(dataPyArray);
                npy_intp * dataPyArrayShape = PyArray_SHAPE(dataPyArray);
                if (dataPyArrayNdims == 0)
                {
                    return self.registerConstant(fieldName, *dataPyArrayData);
                }
                else if (dataPyArrayNdims == 1)
                {
                    Eigen::Map<vectorN_t> data(dataPyArrayData, dataPyArrayShape[0]);
                    return self.registerConstant(fieldName, data);
                }
                else if (dataPyArrayNdims == 2)
                {
                    Eigen::Map<matrixN_t> data(dataPyArrayData, dataPyArrayShape[0], dataPyArrayShape[1]);
                    return self.registerConstant(fieldName, data);
                }
                else
                {
                    PRINT_ERROR("The max number of dims supported for 'numpy.ndarray' is 2.");
                    return hresult_t::ERROR_BAD_INPUT;
                }
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

        ///////////////////////////////////////////////////////////////////////////////
        /// \brief Expose.
        ///////////////////////////////////////////////////////////////////////////////
        static void expose()
        {
            bp::class_<AbstractController,
                       std::shared_ptr<AbstractController>,
                       boost::noncopyable>("AbstractController", bp::no_init)
                .def(PyAbstractControllerVisitor());
        }
    };

    BOOST_PYTHON_VISITOR_EXPOSE(AbstractController)

    // ***************************** PyControllerFunctorVisitor ***********************************

    struct PyControllerFunctorVisitor
        : public bp::def_visitor<PyControllerFunctorVisitor>
    {
    public:
        /* Take advantage of type erasure of std::function to support both
           lambda functions and python handle wrapper depending whether or not
           'compute_command' and 'internal_dynamics' has been specified.
           It is likely to cause a small overhead because the compiler will
           probably not be able to inline ControllerFctWrapper, as it would have
           been the case otherwise, but it is the price to pay for versatility. */
        using CtrlFunctor = ControllerFunctor<ControllerFct, ControllerFct>;

    public:
        ///////////////////////////////////////////////////////////////////////////////
        /// \brief Expose C++ API through the visitor.
        ///////////////////////////////////////////////////////////////////////////////
        template<class PyClass>
        void visit(PyClass & cl) const
        {
            cl
                .def("__init__", bp::make_constructor(&PyControllerFunctorVisitor::factory,
                                 bp::default_call_policies(),
                                (bp::arg("compute_command") = bp::object(),  // bp::object() means 'None' in Python
                                 bp::arg("internal_dynamics") = bp::object())))
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
                commandFct = [](float64_t        const & t,
                                vectorN_t        const & q,
                                vectorN_t        const & v,
                                sensorsDataMap_t const & sensorsData,
                                vectorN_t              & uCommand) {};
            }
            ControllerFct internalDynamicsFct;
            if (!internalDynamicsPy.is_none())
            {
                internalDynamicsFct = ControllerFctWrapper(internalDynamicsPy);
            }
            else
            {
                internalDynamicsFct = [](float64_t        const & t,
                                         vectorN_t        const & q,
                                         vectorN_t        const & v,
                                         sensorsDataMap_t const & sensorsData,
                                         vectorN_t              & uCommand) {};
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
                       boost::noncopyable>("ControllerFunctor", bp::no_init)
                .def(PyControllerFunctorVisitor());
        }
    };

    BOOST_PYTHON_VISITOR_EXPOSE(ControllerFunctor)
}  // End of namespace python.
}  // End of namespace jiminy.
