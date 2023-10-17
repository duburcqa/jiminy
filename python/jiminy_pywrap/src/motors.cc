#include "jiminy/core/hardware/basic_motors.h"

#include "pinocchio/bindings/python/fwd.hpp"

#include "jiminy/python/utilities.h"
#include "jiminy/python/motors.h"


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
                .ADD_PROPERTY_GET_WITH_POLICY("is_initialized",
                                              &AbstractMotorBase::getIsInitialized,
                                              bp::return_value_policy<bp::return_by_value>())
                .ADD_PROPERTY_GET_WITH_POLICY("name",
                                              &AbstractMotorBase::getName,
                                              bp::return_value_policy<bp::return_by_value>())
                .ADD_PROPERTY_GET_WITH_POLICY("idx",
                                              &AbstractMotorBase::getIdx,
                                              bp::return_value_policy<bp::return_by_value>())
                .ADD_PROPERTY_GET_WITH_POLICY("joint_name",
                                              &AbstractMotorBase::getJointName,
                                              bp::return_value_policy<bp::return_by_value>())
                .ADD_PROPERTY_GET_WITH_POLICY("joint_idx",
                                              &AbstractMotorBase::getJointModelIdx,
                                              bp::return_value_policy<bp::return_by_value>())
                .ADD_PROPERTY_GET_WITH_POLICY("joint_type",
                                              &AbstractMotorBase::getJointType,
                                              bp::return_value_policy<bp::return_by_value>())
                .ADD_PROPERTY_GET_WITH_POLICY("joint_position_idx",
                                              &AbstractMotorBase::getJointPositionIdx,
                                              bp::return_value_policy<bp::return_by_value>())
                .ADD_PROPERTY_GET_WITH_POLICY("joint_velocity_idx",
                                              &AbstractMotorBase::getJointVelocityIdx,
                                              bp::return_value_policy<bp::return_by_value>())
                .ADD_PROPERTY_GET_WITH_POLICY("command_limit",
                                              &AbstractMotorBase::getCommandLimit,
                                              bp::return_value_policy<bp::return_by_value>())
                .ADD_PROPERTY_GET_WITH_POLICY("armature",
                                              &AbstractMotorBase::getArmature,
                                              bp::return_value_policy<bp::return_by_value>())

                .def("set_options", &PyAbstractMotorVisitor::setOptions)
                .def("get_options", &AbstractMotorBase::getOptions)
                ;
        }

    public:
        static hresult_t setOptions(AbstractMotorBase       & self,
                               bp::dict          const & configPy)
        {
            configHolder_t config = self.getOptions();
            convertFromPython(configPy, config);
            return self.setOptions(config);
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
                       (bp::arg("self"), "motor_name")))
                .def(PySimpleMotorVisitor());
        }
    };

    BOOST_PYTHON_VISITOR_EXPOSE(SimpleMotor)
}  // End of namespace python.
}  // End of namespace jiminy.
