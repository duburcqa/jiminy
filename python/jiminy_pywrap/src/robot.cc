#include "jiminy/core/telemetry/fwd.h"
#include "jiminy/core/hardware/fwd.h"
#include "jiminy/core/hardware/abstract_sensor.h"
#include "jiminy/core/hardware/abstract_motor.h"
#include "jiminy/core/constraints/abstract_constraint.h"
#include "jiminy/core/control/abstract_controller.h"
#include "jiminy/core/robot/robot.h"

#include "pinocchio/bindings/python/fwd.hpp"

#include "jiminy/python/utilities.h"
#include "jiminy/python/robot.h"


namespace jiminy::python
{
    namespace bp = boost::python;

    // ***************************** PyModelVisitor ***********************************

    struct PyModelVisitor : public bp::def_visitor<PyModelVisitor>
    {
    public:
        /// \brief Expose C++ API through the visitor.
        template<class PyClass>
        void visit(PyClass & cl) const
        {
            // clang-format off
            cl
                .def("add_frame",
                     static_cast<
                         void (Model::*)(const std::string &, const std::string &, const pinocchio::SE3 &)
                     >(&Model::addFrame),
                     (bp::arg("self"), "frame_name", "parent_body_name", "frame_placement"))
                .def("remove_frame", &Model::removeFrame,
                                     (bp::arg("self"), "frame_name"))
                .def("add_collision_bodies", &PyModelVisitor::addCollisionBodies,
                                             (bp::arg("self"),
                                              bp::arg("body_names") = bp::list(),
                                              bp::arg("ignore_meshes") = false))
                .def("remove_collision_bodies", &PyModelVisitor::removeCollisionBodies,
                                                (bp::arg("self"), "body_names"))
                .def("add_contact_points", &PyModelVisitor::addContactPoints,
                                           (bp::arg("self"),
                                            bp::arg("frame_names") = bp::list()))
                .def("remove_contact_points", &PyModelVisitor::removeContactPoints,
                                              (bp::arg("self"), "frame_names"))

                .def("add_constraint",
                     static_cast<
                         void (Model::*)(const std::string &, const std::shared_ptr<AbstractConstraintBase> &)
                     >(&Model::addConstraint),
                     (bp::arg("self"), "name", "constraint"))
                .def("remove_constraint",
                     static_cast<
                         void (Model::*)(const std::string &)
                     >(&Model::removeConstraint),
                     (bp::arg("self"), "name"))
                .def("get_constraint",
                     static_cast<
                         std::shared_ptr<AbstractConstraintBase> (Model::*)(const std::string &)
                     >(&Model::getConstraint),
                     (bp::arg("self"), "constraint_name"))
                .def("exist_constraint", &Model::existConstraint,
                                         (bp::arg("self"), "constraint_name"))
                .ADD_PROPERTY_GET("has_constraints", &Model::hasConstraints)
                .ADD_PROPERTY_GET("constraints", &PyModelVisitor::getConstraints)
                .def("get_constraints_jacobian_and_drift", &PyModelVisitor::getConstraintsJacobianAndDrift)
                .def("compute_constraints", &Model::computeConstraints,
                                            (bp::arg("self"), "q", "v"))

                .def("get_flexible_configuration_from_rigid", &PyModelVisitor::getFlexiblePositionFromRigid,
                                                              (bp::arg("self"), "rigid_position"))
                .def("get_flexible_velocity_from_rigid", &PyModelVisitor::getFlexibleVelocityFromRigid,
                                                         (bp::arg("self"), "rigid_velocity"))
                .def("get_rigid_configuration_from_flexible", &PyModelVisitor::getRigidPositionFromFlexible,
                                                              (bp::arg("self"), "flexible_position"))
                .def("get_rigid_velocity_from_flexible", &PyModelVisitor::getRigidVelocityFromFlexible,
                                                         (bp::arg("self"), "flexible_velocity"))

                // FIXME: Disable automatic typing because typename returned by 'py_type_str' is missing module
                // prefix, which makes it impossible to distinguish 'pinocchio.Model' from 'jiminy.Model' classes.
                .def_readonly("pinocchio_model_th", &Model::pinocchioModelOrig_, "fget( (Model)self) -> pinocchio.Model")
                .def_readonly("pinocchio_model", &Model::pinocchioModel_, "fget( (Model)self) -> pinocchio.Model")
                .DEF_READONLY("collision_model_th", &Model::collisionModelOrig_)
                .DEF_READONLY("collision_model", &Model::collisionModel_)
                .DEF_READONLY("visual_model_th", &Model::visualModelOrig_)
                .DEF_READONLY("visual_model", &Model::visualModel_)
                .DEF_READONLY("visual_data", &Model::visualData_)
                .DEF_READONLY("pinocchio_data_th", &Model::pinocchioDataOrig_)
                .DEF_READONLY("pinocchio_data", &Model::pinocchioData_)
                .DEF_READONLY("collision_data", &Model::collisionData_)

                .DEF_READONLY("contact_forces", &Model::contactForces_)

                .ADD_PROPERTY_GET_WITH_POLICY("is_initialized",
                                              &Model::getIsInitialized,
                                              bp::return_value_policy<bp::return_by_value>())
                .ADD_PROPERTY_GET_WITH_POLICY("mesh_package_dirs",
                                              &Model::getMeshPackageDirs,
                                              bp::return_value_policy<result_converter<true>>())
                .ADD_PROPERTY_GET_WITH_POLICY("urdf_path",
                                              &Model::getUrdfPath,
                                              bp::return_value_policy<bp::return_by_value>())
                .ADD_PROPERTY_GET_WITH_POLICY("has_freeflyer",
                                              &Model::getHasFreeflyer,
                                              bp::return_value_policy<bp::return_by_value>())
                .ADD_PROPERTY_GET("is_flexible", &PyModelVisitor::isFlexibleModelEnabled)
                .ADD_PROPERTY_GET_WITH_POLICY("nq",
                                              &Model::nq,
                                              bp::return_value_policy<bp::return_by_value>())
                .ADD_PROPERTY_GET_WITH_POLICY("nv",
                                              &Model::nv,
                                              bp::return_value_policy<bp::return_by_value>())
                .ADD_PROPERTY_GET_WITH_POLICY("nx",
                                              &Model::nx,
                                              bp::return_value_policy<bp::return_by_value>())

                .ADD_PROPERTY_GET_WITH_POLICY("collision_body_names",
                                              &Model::getCollisionBodyNames,
                                              bp::return_value_policy<result_converter<true>>())
                .ADD_PROPERTY_GET_WITH_POLICY("collision_body_indices",
                                              &Model::getCollisionBodyIndices,
                                              bp::return_value_policy<result_converter<true>>())
                .ADD_PROPERTY_GET_WITH_POLICY("collision_pair_indices",
                                              &Model::getCollisionPairIndices,
                                              bp::return_value_policy<result_converter<true>>())
                .ADD_PROPERTY_GET_WITH_POLICY("contact_frame_names",
                                              &Model::getContactFrameNames,
                                              bp::return_value_policy<result_converter<true>>())
                .ADD_PROPERTY_GET_WITH_POLICY("contact_frame_indices",
                                              &Model::getContactFrameIndices,
                                              bp::return_value_policy<result_converter<true>>())
                .ADD_PROPERTY_GET_WITH_POLICY("rigid_joint_names",
                                              &Model::getRigidJointNames,
                                              bp::return_value_policy<result_converter<true>>())
                .ADD_PROPERTY_GET_WITH_POLICY("rigid_joint_index",
                                              &Model::getRigidJointIndices,
                                              bp::return_value_policy<result_converter<true>>())
                .ADD_PROPERTY_GET_WITH_POLICY("rigid_joint_position_indices",
                                              &Model::getRigidJointPositionIndices,
                                              bp::return_value_policy<result_converter<true>>())
                .ADD_PROPERTY_GET_WITH_POLICY("rigid_joint_velocity_indices",
                                              &Model::getRigidJointVelocityIndices,
                                              bp::return_value_policy<result_converter<true>>())
                .ADD_PROPERTY_GET_WITH_POLICY("flexible_joint_names",
                                              &Model::getFlexibleJointNames,
                                              bp::return_value_policy<result_converter<true>>())
                .ADD_PROPERTY_GET_WITH_POLICY("flexible_joint_indices",
                                              &Model::getFlexibleJointIndices,
                                              bp::return_value_policy<result_converter<true>>())

                .ADD_PROPERTY_GET_WITH_POLICY("position_limit_lower",
                                              &Model::getPositionLimitMin,
                                              bp::return_value_policy<bp::return_by_value>())
                .ADD_PROPERTY_GET_WITH_POLICY("position_limit_upper",
                                              &Model::getPositionLimitMax,
                                              bp::return_value_policy<bp::return_by_value>())
                .ADD_PROPERTY_GET_WITH_POLICY("velocity_limit",
                                              &Model::getVelocityLimit,
                                              bp::return_value_policy<bp::return_by_value>())

                .ADD_PROPERTY_GET_WITH_POLICY("log_position_fieldnames",
                                              &Model::getLogPositionFieldnames,
                                              bp::return_value_policy<result_converter<true>>())
                .ADD_PROPERTY_GET_WITH_POLICY("log_velocity_fieldnames",
                                              &Model::getLogVelocityFieldnames,
                                              bp::return_value_policy<result_converter<true>>())
                .ADD_PROPERTY_GET_WITH_POLICY("log_acceleration_fieldnames",
                                              &Model::getLogAccelerationFieldnames,
                                              bp::return_value_policy<result_converter<true>>())
                .ADD_PROPERTY_GET_WITH_POLICY("log_f_external_fieldnames",
                                              &Model::getLogForceExternalFieldnames,
                                              bp::return_value_policy<result_converter<true>>())
                ;
            // clang-format on
        }

        static void addCollisionBodies(
            Model & self, const bp::object & linkNamesPy, bool ignoreMeshes)
        {
            auto linkNames = convertFromPython<std::vector<std::string>>(linkNamesPy);
            return self.addCollisionBodies(linkNames, ignoreMeshes);
        }

        static void removeCollisionBodies(Model & self, const bp::object & linkNamesPy)
        {
            auto linkNames = convertFromPython<std::vector<std::string>>(linkNamesPy);
            return self.removeCollisionBodies(linkNames);
        }

        static void addContactPoints(Model & self, const bp::object & frameNamesPy)
        {
            auto frameNames = convertFromPython<std::vector<std::string>>(frameNamesPy);
            return self.addContactPoints(frameNames);
        }

        static void removeContactPoints(Model & self, const bp::object & frameNamesPy)
        {
            auto frameNames = convertFromPython<std::vector<std::string>>(frameNamesPy);
            return self.removeContactPoints(frameNames);
        }

        static std::shared_ptr<ConstraintTree> getConstraints(Model & self)
        {
            return std::make_shared<ConstraintTree>(self.getConstraints());
        }

        static bp::tuple getConstraintsJacobianAndDrift(Model & self)
        {
            Eigen::Index constraintRow = 0;
            Eigen::Index constraintsRows = 0;
            ConstraintTree constraints = self.getConstraints();
            constraints.foreach(
                [&constraintsRows](const std::shared_ptr<AbstractConstraintBase> & constraint,
                                   ConstraintNodeType /* node */)
                {
                    if (!constraint->getIsEnabled())
                    {
                        return;
                    }
                    constraintsRows += static_cast<Eigen::Index>(constraint->getDim());
                });
            Eigen::MatrixXd J(constraintsRows, self.nv());
            Eigen::VectorXd gamma(constraintsRows);
            constraints.foreach(
                [&J, &gamma, &constraintRow](
                    const std::shared_ptr<AbstractConstraintBase> & constraint,
                    ConstraintNodeType /* node */)
                {
                    if (!constraint->getIsEnabled())
                    {
                        return;
                    }
                    const Eigen::Index constraintDim =
                        static_cast<Eigen::Index>(constraint->getDim());
                    J.middleRows(constraintRow, constraintDim) = constraint->getJacobian();
                    gamma.segment(constraintRow, constraintDim) = constraint->getDrift();
                    constraintRow += constraintDim;
                });
            return bp::make_tuple(J, gamma);
        }

        static Eigen::VectorXd getFlexiblePositionFromRigid(Model & self,
                                                            const Eigen::VectorXd & qRigid)
        {
            Eigen::VectorXd qFlexible;
            self.getFlexiblePositionFromRigid(qRigid, qFlexible);
            return qFlexible;
        }

        static Eigen::VectorXd getFlexibleVelocityFromRigid(Model & self,
                                                            const Eigen::VectorXd & vRigid)
        {
            Eigen::VectorXd vFlexible;
            self.getFlexibleVelocityFromRigid(vRigid, vFlexible);
            return vFlexible;
        }

        static Eigen::VectorXd getRigidPositionFromFlexible(Model & self,
                                                            const Eigen::VectorXd & qFlexible)
        {
            Eigen::VectorXd qRigid;
            self.getRigidPositionFromFlexible(qFlexible, qRigid);
            return qRigid;
        }

        static Eigen::VectorXd getRigidVelocityFromFlexible(Model & self,
                                                            const Eigen::VectorXd & vFlexible)
        {
            Eigen::VectorXd vRigid;
            self.getRigidVelocityFromFlexible(vFlexible, vRigid);
            return vRigid;
        }

        static bool isFlexibleModelEnabled(Model & self)
        {
            return self.modelOptions_->dynamics.enableFlexibleModel;
        }

        static void expose()
        {
            // clang-format off
            bp::class_<Model,
                       std::shared_ptr<Model>,
                       boost::noncopyable
                       >("Model", bp::no_init)
                .def(PyModelVisitor());
            // clang-format on
        }
    };

    BOOST_PYTHON_VISITOR_EXPOSE(Model)

    // ***************************** PyRobotVisitor ***********************************

    struct PyRobotVisitor : public bp::def_visitor<PyRobotVisitor>
    {
    public:
        /// \brief Expose C++ API through the visitor.
        template<class PyClass>
        void visit(PyClass & cl) const
        {
            // clang-format off
            cl
                .def("initialize", &PyRobotVisitor::initialize,
                                   (bp::arg("self"), "urdf_path",
                                    bp::arg("has_freeflyer") = false,
                                    bp::arg("mesh_package_dirs") = bp::list(),
                                    bp::arg("load_visual_meshes") = false))
                .def("initialize",
                     static_cast<
                         void (Robot::*)(const pinocchio::Model &, const pinocchio::GeometryModel &, const pinocchio::GeometryModel &)
                     >(&Robot::initialize),
                     (bp::arg("self"), "pinocchio_model", "collision_model", "visual_model"))

                .ADD_PROPERTY_GET_WITH_POLICY("is_locked",
                                              &Robot::getIsLocked,
                                              bp::return_value_policy<bp::return_by_value>())

                .ADD_PROPERTY_GET_WITH_POLICY("name",
                                              &Robot::getName,
                                              bp::return_value_policy<bp::return_by_value>())

                .def("dump_options", &Robot::dumpOptions,
                                     (bp::arg("self"), "json_filename"))
                .def("load_options", &Robot::loadOptions,
                                     (bp::arg("self"), "json_filename"))

                .def("attach_motor", &Robot::attachMotor,
                                     (bp::arg("self"), "motor"))
                .def("get_motor",
                     static_cast<
                         std::shared_ptr<AbstractMotorBase> (Robot::*)(const std::string &)
                     >(&Robot::getMotor),
                     (bp::arg("self"), "motor_name"))
                .def("detach_motor", &Robot::detachMotor,
                                     (bp::arg("self"), "joint_name"))
                .def("detach_motors", &PyRobotVisitor::detachMotors,
                                      (bp::arg("self"),
                                       bp::arg("joints_names") = bp::list()))
                .def("attach_sensor", &Robot::attachSensor,
                                      (bp::arg("self"), "sensor"))
                .def("detach_sensor", &Robot::detachSensor,
                                      (bp::arg("self"), "sensor_type", "sensor_name"))
                .def("detach_sensors", &Robot::detachSensors,
                                       (bp::arg("self"),
                                        bp::arg("sensor_type") = std::string()))
                .def("get_sensor",
                     static_cast<
                         std::shared_ptr<AbstractSensorBase> (Robot::*)(const std::string &, const std::string &)
                     >(&Robot::getSensor),
                     (bp::arg("self"), "sensor_type", "sensor_name"))

                .ADD_PROPERTY_GET_SET("controller",
                                      static_cast<
                                          std::shared_ptr<AbstractController> (Robot::*)()
                                      >(&Robot::getController),
                                      &Robot::setController)

                .ADD_PROPERTY_GET("sensor_measurements", &PyRobotVisitor::getSensorMeasurements)

                .def("set_options", &PyRobotVisitor::setOptions,
                                    (bp::arg("self"), "robot_options"))
                .def("get_options", &Robot::getOptions)
                .def("set_model_options", &PyRobotVisitor::setModelOptions,
                                          (bp::arg("self"), "model_options"))
                .def("get_model_options", &Robot::getModelOptions)
                .def("set_motors_options", &PyRobotVisitor::setMotorsOptions,
                                           (bp::arg("self"), "motors_options"))
                .def("get_motors_options", &Robot::getMotorsOptions)
                .def("set_sensors_options", &PyRobotVisitor::setSensorsOptions,
                                            (bp::arg("self"), "sensors_options"))
                .def("get_sensors_options", &Robot::getSensorsOptions)
                .def("set_controller_options", &PyRobotVisitor::setControllerOptions,
                                              (bp::arg("self"), "controller_options"))
                .def("get_controller_options", &Robot::getControllerOptions)
                .def("set_telemetry_options", &PyRobotVisitor::setTelemetryOptions,
                                              (bp::arg("self"), "telemetry_options"))
                .def("get_telemetry_options", &Robot::getTelemetryOptions)

                .ADD_PROPERTY_GET("nmotors", &Robot::nmotors)
                .ADD_PROPERTY_GET_WITH_POLICY("motor_names",
                                              &Robot::getMotorNames,
                                              bp::return_value_policy<result_converter<true>>())
                .ADD_PROPERTY_GET_WITH_POLICY("motor_position_indices",
                                              &Robot::getMotorsPositionIndices,
                                              bp::return_value_policy<result_converter<true>>())
                .ADD_PROPERTY_GET_WITH_POLICY("motor_velocity_indices",
                                              &Robot::getMotorVelocityIndices,
                                              bp::return_value_policy<result_converter<true>>())
                .ADD_PROPERTY_GET("sensor_names", &PyRobotVisitor::getSensorNames)

                .ADD_PROPERTY_GET_WITH_POLICY("command_limit",
                                              &Robot::getCommandLimit,
                                              bp::return_value_policy<bp::return_by_value>())

                .ADD_PROPERTY_GET_WITH_POLICY("log_command_fieldnames",
                                              &Robot::getLogCommandFieldnames,
                                              bp::return_value_policy<result_converter<true>>())
                .ADD_PROPERTY_GET_WITH_POLICY("log_motor_effort_fieldnames",
                                              &Robot::getLogMotorEffortFieldnames,
                                              bp::return_value_policy<result_converter<true>>())
                ;
            // clang-format on
        }

        static void initialize(Robot & self,
                               const std::string & urdfPath,
                               bool hasFreeflyer,
                               const bp::object & meshPackageDirsPy,
                               bool loadVisualMeshes)
        {
            auto meshPackageDirs = convertFromPython<std::vector<std::string>>(meshPackageDirsPy);
            return self.initialize(urdfPath, hasFreeflyer, meshPackageDirs, loadVisualMeshes);
        }

        static void detachMotors(Robot & self, const bp::object & motorNamesPy)
        {
            auto motorNames = convertFromPython<std::vector<std::string>>(motorNamesPy);
            return self.detachMotors(motorNames);
        }

        static std::shared_ptr<SensorMeasurementTree> getSensorMeasurements(Robot & self)
        {
            return std::make_shared<SensorMeasurementTree>(self.getSensorMeasurements());
        }

        static bp::dict getSensorNames(Robot & self)
        {
            bp::dict sensorsNamesPy;
            const auto & sensorsNames = self.getSensorNames();
            for (const auto & sensorTypeNames : sensorsNames)
            {
                sensorsNamesPy[sensorTypeNames.first] = convertToPython(sensorTypeNames.second);
            }
            return sensorsNamesPy;
        }

        static void setOptions(Robot & self, const bp::dict & configPy)
        {
            GenericConfig config = self.getOptions();
            convertFromPython(configPy, config);
            return self.setOptions(config);
        }

        static void setModelOptions(Robot & self, const bp::dict & configPy)
        {
            GenericConfig config = self.getModelOptions();
            convertFromPython(configPy, config);
            return self.setModelOptions(config);
        }

        static void setMotorsOptions(Robot & self, const bp::dict & configPy)
        {
            GenericConfig config = self.getMotorsOptions();
            convertFromPython(configPy, config);
            return self.setMotorsOptions(config);
        }

        static void setSensorsOptions(Robot & self, const bp::dict & configPy)
        {
            GenericConfig config = self.getSensorsOptions();
            convertFromPython(configPy, config);
            return self.setSensorsOptions(config);
        }

        static void setControllerOptions(Robot & self, const bp::dict & configPy)
        {
            GenericConfig config = self.getControllerOptions();
            convertFromPython(configPy, config);
            return self.setControllerOptions(config);
        }

        static void setTelemetryOptions(Robot & self, const bp::dict & configPy)
        {
            GenericConfig config = self.getTelemetryOptions();
            convertFromPython(configPy, config);
            return self.setTelemetryOptions(config);
        }

        static void expose()
        {
            // clang-format off
            bp::class_<Robot, bp::bases<Model>,
                       std::shared_ptr<Robot>,
                       boost::noncopyable
                       >("Robot", bp::init<const std::string &>(bp::arg("name") = ""))
                .def(PyRobotVisitor());
            // clang-format on
        }
    };

    BOOST_PYTHON_VISITOR_EXPOSE(Robot)
}
