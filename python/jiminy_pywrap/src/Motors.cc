#include "jiminy/core/robot/BasicMotors.h"

#include <boost/python.hpp>

#include "jiminy/python/Utilities.h"
#include "jiminy/python/Motors.h"


namespace jiminy
{
namespace python
{
    namespace bp = boost::python;

    // ***************************** PyAbstractMotorVisitor ***********************************

    struct PyAbstractMotorVisitor
        : public bp::def_visitor<PyAbstractMotorVisitor>
    {
    public:
        ///////////////////////////////////////////////////////////////////////////////
        /// \brief Expose C++ API through the visitor.
        ///////////////////////////////////////////////////////////////////////////////

        template<class PyClass>
        void visit(PyClass & cl) const
        {
            cl
                .add_property("is_initialized", bp::make_function(&AbstractMotorBase::getIsInitialized,
                                                bp::return_value_policy<bp::copy_const_reference>()))
                .add_property("name", bp::make_function(&AbstractMotorBase::getName,
                                        bp::return_value_policy<bp::copy_const_reference>()))
                .add_property("idx", bp::make_function(&AbstractMotorBase::getIdx,
                                        bp::return_value_policy<bp::copy_const_reference>()))
                .add_property("joint_name", bp::make_function(&AbstractMotorBase::getJointName,
                                            bp::return_value_policy<bp::copy_const_reference>()))
                .add_property("joint_idx", bp::make_function(&AbstractMotorBase::getJointModelIdx,
                                            bp::return_value_policy<bp::copy_const_reference>()))
                .add_property("joint_type", bp::make_function(&AbstractMotorBase::getJointType,
                                            bp::return_value_policy<bp::copy_const_reference>()))
                .add_property("joint_position_idx", bp::make_function(&AbstractMotorBase::getJointPositionIdx,
                                                    bp::return_value_policy<bp::copy_const_reference>()))
                .add_property("joint_velocity_idx", bp::make_function(&AbstractMotorBase::getJointVelocityIdx,
                                                    bp::return_value_policy<bp::copy_const_reference>()))
                .add_property("command_limit", bp::make_function(&AbstractMotorBase::getCommandLimit,
                                              bp::return_value_policy<bp::copy_const_reference>()))
                .add_property("armature", bp::make_function(&AbstractMotorBase::getArmature,
                                               bp::return_value_policy<bp::copy_const_reference>()))

                .def("set_options", &PyAbstractMotorVisitor::setOptions)
                .def("get_options", &AbstractMotorBase::getOptions)
                ;
        }

    public:
        static void setOptions(AbstractMotorBase       & self,
                               bp::dict          const & configPy)
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
            bp::class_<AbstractMotorBase,
                       std::shared_ptr<AbstractMotorBase>,
                       boost::noncopyable>("AbstractMotor", bp::no_init)
                .def(PyAbstractMotorVisitor());
        }
    };

    BOOST_PYTHON_VISITOR_EXPOSE(AbstractMotor)

    // ***************************** PySimpleMotorVisitor ***********************************

    struct PySimpleMotorVisitor
        : public bp::def_visitor<PySimpleMotorVisitor>
    {
    public:
        ///////////////////////////////////////////////////////////////////////////////
        /// \brief Expose C++ API through the visitor.
        ///////////////////////////////////////////////////////////////////////////////

        template<class PyClass>
        class PyMotorVisitorImpl
        {
        public:
            using TMotor = typename PyClass::wrapped_type;

            static void visit(PyClass & cl)
            {
                cl
                    .def("initialize", &TMotor::initialize)
                    ;
            }
        };

    public:
        template<class PyClass>
        void visit(PyClass & cl) const
        {
            PyMotorVisitorImpl<PyClass>::visit(cl);
        }

        ///////////////////////////////////////////////////////////////////////////////
        /// \brief Expose.
        ///////////////////////////////////////////////////////////////////////////////
        static void expose()
        {
            bp::class_<SimpleMotor, bp::bases<AbstractMotorBase>,
                       std::shared_ptr<SimpleMotor>,
                       boost::noncopyable>("SimpleMotor",
                       bp::init<std::string const &>(
                       bp::args("self", "motor_name")))
                .def(PySimpleMotorVisitor());
        }
    };

    BOOST_PYTHON_VISITOR_EXPOSE(SimpleMotor)
}  // End of namespace python.
}  // End of namespace jiminy.
