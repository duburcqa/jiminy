#include "jiminy/core/hardware/basic_motors.h"

#include "pinocchio/bindings/python/fwd.hpp"

#include "jiminy/python/utilities.h"
#include "jiminy/python/motors.h"


namespace jiminy::python
{
    namespace bp = boost::python;

    // ************************************* AbstractMotor ************************************* //

    namespace internal::abstract_motor
    {
        static void setOptions(AbstractMotorBase & self, const bp::dict & configPy)
        {
            GenericConfig config = self.getOptions();
            convertFromPython(configPy, config);
            return self.setOptions(config);
        }
    }

    void exposeAbstractMotor()
    {
        bp::class_<AbstractMotorBase, std::shared_ptr<AbstractMotorBase>, boost::noncopyable>(
            "AbstractMotor", bp::no_init)
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

            .def("set_options", &internal::abstract_motor::setOptions)
            .def("get_options",
                 &AbstractMotorBase::getOptions,
                 bp::return_value_policy<bp::return_by_value>());
    }

    // ************************************** BasicMotors ************************************** //

    void exposeBasicMotors()
    {
        bp::class_<SimpleMotor,
                   bp::bases<AbstractMotorBase>,
                   std::shared_ptr<SimpleMotor>,
                   boost::noncopyable>(
            "SimpleMotor", bp::init<const std::string &>((bp::arg("self"), "motor_name")))
            .def("initialize", &SimpleMotor::initialize);
    }
}
