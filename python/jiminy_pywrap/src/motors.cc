#include "jiminy/core/hardware/basic_motors.h"

#include "pinocchio/bindings/python/fwd.hpp"

#include "jiminy/python/utilities.h"
#include "jiminy/python/motors.h"


namespace jiminy::python
{
    namespace bp = boost::python;

    // ***************************** PyAbstractMotorVisitor ***********************************

    struct PyAbstractMotorVisitor : public bp::def_visitor<PyAbstractMotorVisitor>
    {
    public:
        /// \brief Expose C++ API through the visitor.
        template<class PyClass>
        void visit(PyClass & cl) const
        {
            // clang-format off
            cl
                .ADD_PROPERTY_GET_WITH_POLICY("is_initialized",
                                              &AbstractMotorBase::getIsInitialized,
                                              bp::return_value_policy<bp::return_by_value>())
                .ADD_PROPERTY_GET_WITH_POLICY("name",
                                              &AbstractMotorBase::getName,
                                              bp::return_value_policy<bp::return_by_value>())
                .ADD_PROPERTY_GET_WITH_POLICY("index",
                                              &AbstractMotorBase::getIndex,
                                              bp::return_value_policy<bp::return_by_value>())
                .ADD_PROPERTY_GET_WITH_POLICY("joint_name",
                                              &AbstractMotorBase::getJointName,
                                              bp::return_value_policy<bp::return_by_value>())
                .ADD_PROPERTY_GET_WITH_POLICY("joint_index",
                                              &AbstractMotorBase::getJointIndex,
                                              bp::return_value_policy<bp::return_by_value>())
                .ADD_PROPERTY_GET_WITH_POLICY("joint_type",
                                              &AbstractMotorBase::getJointType,
                                              bp::return_value_policy<bp::return_by_value>())
                .ADD_PROPERTY_GET_WITH_POLICY("joint_position_index",
                                              &AbstractMotorBase::getJointPositionIndex,
                                              bp::return_value_policy<bp::return_by_value>())
                .ADD_PROPERTY_GET_WITH_POLICY("joint_velocity_index",
                                              &AbstractMotorBase::getJointVelocityIndex,
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
            // clang-format on
        }

    public:
        static hresult_t setOptions(AbstractMotorBase & self, const bp::dict & configPy)
        {
            GenericConfig config = self.getOptions();
            convertFromPython(configPy, config);
            return self.setOptions(config);
        }

        static void expose()
        {
            // clang-format off
            bp::class_<AbstractMotorBase,
                       std::shared_ptr<AbstractMotorBase>,
                       boost::noncopyable>("AbstractMotor", bp::no_init)
                .def(PyAbstractMotorVisitor());
            // clang-format on
        }
    };

    BOOST_PYTHON_VISITOR_EXPOSE(AbstractMotor)

    // ***************************** PySimpleMotorVisitor ***********************************

    struct PySimpleMotorVisitor : public bp::def_visitor<PySimpleMotorVisitor>
    {
    public:
        template<class PyClass>
        void visit(PyClass & cl) const
        {
            using TMotor = typename PyClass::wrapped_type;

            // clang-format off
            cl
                .def("initialize", &TMotor::initialize)
                ;
            // clang-format on
        }

        static void expose()
        {
            // clang-format off
            bp::class_<SimpleMotor, bp::bases<AbstractMotorBase>,
                       std::shared_ptr<SimpleMotor>,
                       boost::noncopyable>("SimpleMotor",
                       bp::init<const std::string &>(
                       (bp::arg("self"), "motor_name")))
                .def(PySimpleMotorVisitor());
            // clang-format on
        }
    };

    BOOST_PYTHON_VISITOR_EXPOSE(SimpleMotor)
}
