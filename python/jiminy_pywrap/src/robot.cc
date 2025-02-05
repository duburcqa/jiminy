#include "jiminy/core/telemetry/fwd.h"
#include "jiminy/core/hardware/fwd.h"
#include "jiminy/core/hardware/abstract_sensor.h"
#include "jiminy/core/hardware/abstract_motor.h"
#include "jiminy/core/constraints/abstract_constraint.h"
#include "jiminy/core/control/abstract_controller.h"
#include "jiminy/core/robot/robot.h"

#define NO_IMPORT_ARRAY
#include "jiminy/python/fwd.h"
#include "jiminy/python/utilities.h"
#include "jiminy/python/robot.h"


namespace jiminy::python
{
    namespace bp = boost::python;

    // ***************************************** Model ***************************************** //

    template<typename T>
    static void initialize(T & self,
                           const std::string & urdfPath,
                           bool hasFreeflyer,
                           const bp::object & meshPackageDirsPy,
                           bool loadVisualMeshes)
    {
        auto meshPackageDirs = convertFromPython<std::vector<std::string>>(meshPackageDirsPy);
        return self.initialize(urdfPath, hasFreeflyer, meshPackageDirs, loadVisualMeshes);
    }

    template<typename T>
    static void initializeFromModels(T & self,
                                     const pinocchio::Model & pinocchioModel,
                                     const bp::object & collisionModelPy,
                                     const bp::object & visualModelPy)
    {
        std::optional<pinocchio::GeometryModel> collisionModel;
        if (!collisionModelPy.is_none())
        {
            collisionModel.emplace(
                bp::extract<const pinocchio::GeometryModel &>(collisionModelPy));
        }
        std::optional<pinocchio::GeometryModel> visualModel;
        if (!visualModelPy.is_none())
        {
            visualModel.emplace(bp::extract<const pinocchio::GeometryModel &>(visualModelPy));
        }
        return self.initialize(pinocchioModel, collisionModel, visualModel);
    }

    template<typename T>
    static void setOptions(T & self, const bp::dict & configPy)
    {
        GenericConfig config = self.getOptions();
        convertFromPython(configPy, config);
        return self.setOptions(config);
    }

    namespace internal::model
    {
        static void removeFrames(Model & self, const bp::object & frameNamesPy)
        {
            auto frameNames = convertFromPython<std::vector<std::string>>(frameNamesPy);
            return self.removeFrames(frameNames);
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

        static bp::tuple getConstraintsJacobianAndDrift(Model & self)
        {
            Eigen::Index constraintRow = 0;
            Eigen::Index constraintsRows = 0;
            ConstraintTree constraints = self.getConstraints();
            constraints.foreach(
                [&constraintsRows](const std::shared_ptr<AbstractConstraintBase> & constraint,
                                   ConstraintRegistryType /* type */)
                {
                    if (!constraint->getIsEnabled())
                    {
                        return;
                    }
                    constraintsRows += static_cast<Eigen::Index>(constraint->getSize());
                });
            Eigen::MatrixXd J(constraintsRows, self.nv());
            Eigen::VectorXd gamma(constraintsRows);
            constraints.foreach(
                [&J, &gamma, &constraintRow](
                    const std::shared_ptr<AbstractConstraintBase> & constraint,
                    ConstraintRegistryType /* type */)
                {
                    if (!constraint->getIsEnabled())
                    {
                        return;
                    }
                    const Eigen::Index constraintSize =
                        static_cast<Eigen::Index>(constraint->getSize());
                    J.middleRows(constraintRow, constraintSize) = constraint->getJacobian();
                    gamma.segment(constraintRow, constraintSize) = constraint->getDrift();
                    constraintRow += constraintSize;
                });
            return bp::make_tuple(J, gamma);
        }

        static Eigen::VectorXd getExtendedPositionFromTheoretical(
            Model & self, const Eigen::VectorXd & qTheoretical)
        {
            Eigen::VectorXd qExtended;
            self.getExtendedPositionFromTheoretical(qTheoretical, qExtended);
            return qExtended;
        }

        static Eigen::VectorXd getExtendedVelocityFromTheoretical(
            Model & self, const Eigen::VectorXd & vTheoretical)
        {
            Eigen::VectorXd vExtended;
            self.getExtendedVelocityFromTheoretical(vTheoretical, vExtended);
            return vExtended;
        }

        static Eigen::VectorXd getTheoreticalPositionFromExtended(
            Model & self, const Eigen::VectorXd & qExtended)
        {
            Eigen::VectorXd qTheoretical;
            self.getTheoreticalPositionFromExtended(qExtended, qTheoretical);
            return qTheoretical;
        }

        static Eigen::VectorXd getTheoreticalVelocityFromExtended(
            Model & self, const Eigen::VectorXd & vExtended)
        {
            Eigen::VectorXd vTheoretical;
            self.getTheoreticalVelocityFromExtended(vExtended, vTheoretical);
            return vTheoretical;
        }

        static bool isFlexibilityEnabled(Model & self)
        {
            return self.modelOptions_->dynamics.enableFlexibility;
        }
    }

    void exposeModel()
    {
        bp::class_<Model, std::shared_ptr<Model>, boost::noncopyable>("Model", bp::init<>())
            .def("copy", &std::make_shared<Model, const Model &>, bp::arg("self"))

            .def("initialize",
                 &initialize<Model>,
                 (bp::arg("self"),
                  "urdf_path",
                  bp::arg("has_freeflyer") = false,
                  bp::arg("mesh_package_dirs") = bp::list(),
                  bp::arg("load_visual_meshes") = false))
            .def("initialize",
                 &initializeFromModels<Model>,
                 (bp::arg("self"),
                  "pinocchio_model",
                  bp::arg("collision_model") = bp::object(),
                  bp::arg("visual_model") = bp::object()))

            .def("reset",
                 makeFunction(ConvertGeneratorFromPythonAndInvoke(&Model::reset),
                              bp::default_call_policies(),
                              (bp::arg("self"), "generator")))

            .def("set_options", &setOptions<Model>, (bp::arg("self"), "options"))
            .def("get_options", &Model::getOptions, bp::return_value_policy<bp::return_by_value>())

            .def("add_frame",
                 static_cast<void (Model::*)(
                     const std::string &, const std::string &, const pinocchio::SE3 &)>(
                     &Model::addFrame),
                 (bp::arg("self"), "frame_name", "parent_body_name", "frame_placement"))
            .def("remove_frames", &internal::model::removeFrames, (bp::arg("self"), "frame_names"))

            .def("add_collision_bodies",
                 &internal::model::addCollisionBodies,
                 (bp::arg("self"),
                  bp::arg("body_names") = bp::list(),
                  bp::arg("ignore_meshes") = false))
            .def("remove_collision_bodies",
                 &internal::model::removeCollisionBodies,
                 (bp::arg("self"), "body_names"))
            .def("add_contact_points",
                 &internal::model::addContactPoints,
                 (bp::arg("self"), bp::arg("frame_names") = bp::list()))
            .def("remove_contact_points",
                 &internal::model::removeContactPoints,
                 (bp::arg("self"), "frame_names"))

            .def("add_constraint",
                 static_cast<void (Model::*)(const std::string &,
                                             const std::shared_ptr<AbstractConstraintBase> &)>(
                     &Model::addConstraint),
                 (bp::arg("self"), "name", "constraint"))
            .def("remove_constraint",
                 static_cast<void (Model::*)(const std::string &)>(&Model::removeConstraint),
                 (bp::arg("self"), "name"))
            .ADD_PROPERTY_GET("has_constraints", &Model::hasConstraints)
            .ADD_PROPERTY_GET_WITH_POLICY("constraints",
                                          &Model::getConstraints,
                                          bp::return_value_policy<result_converter<false>>())
            .def("get_constraints_jacobian_and_drift",
                 &internal::model::getConstraintsJacobianAndDrift)
            .def("compute_constraints", &Model::computeConstraints, (bp::arg("self"), "q", "v"))

            .def("get_extended_position_from_theoretical",
                 &internal::model::getExtendedPositionFromTheoretical,
                 (bp::arg("self"), "mechanical_position"))
            .def("get_extended_velocity_from_theoretical",
                 &internal::model::getExtendedVelocityFromTheoretical,
                 (bp::arg("self"), "mechanical_velocity"))
            .def("get_theoretical_position_from_extended",
                 &internal::model::getTheoreticalPositionFromExtended,
                 (bp::arg("self"), "flexibility_position"))
            .def("get_theoretical_velocity_from_extended",
                 &internal::model::getTheoreticalVelocityFromExtended,
                 (bp::arg("self"), "flexibility_velocity"))

            .DEF_READONLY("pinocchio_model_th", &Model::pinocchioModelTh_)
            .DEF_READONLY("pinocchio_model", &Model::pinocchioModel_)
            .DEF_READONLY("collision_model_th", &Model::collisionModelTh_)
            .DEF_READONLY("collision_model", &Model::collisionModel_)
            .DEF_READONLY("visual_model_th", &Model::visualModelTh_)
            .DEF_READONLY("visual_model", &Model::visualModel_)
            .DEF_READONLY("visual_data", &Model::visualData_)
            .DEF_READONLY("pinocchio_data_th", &Model::pinocchioDataTh_)
            .DEF_READONLY("pinocchio_data", &Model::pinocchioData_)
            .DEF_READONLY("collision_data", &Model::collisionData_)

            .DEF_READONLY("contact_forces", &Model::contactForces_)

            .ADD_PROPERTY_GET("is_initialized", &Model::getIsInitialized)
            .ADD_PROPERTY_GET_WITH_POLICY("mesh_package_dirs",
                                          &Model::getMeshPackageDirs,
                                          bp::return_value_policy<result_converter<true>>())
            .ADD_PROPERTY_GET_WITH_POLICY(
                "urdf_path", &Model::getUrdfPath, bp::return_value_policy<bp::return_by_value>())
            .ADD_PROPERTY_GET("has_freeflyer", &Model::getHasFreeflyer)
            .ADD_PROPERTY_GET("is_flexibility_enabled", &internal::model::isFlexibilityEnabled)
            .ADD_PROPERTY_GET("nq", &Model::nq)
            .ADD_PROPERTY_GET("nv", &Model::nv)
            .ADD_PROPERTY_GET("nx", &Model::nx)

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
            .ADD_PROPERTY_GET_WITH_POLICY("mechanical_joint_names",
                                          &Model::getMechanicalJointNames,
                                          bp::return_value_policy<result_converter<true>>())
            .ADD_PROPERTY_GET_WITH_POLICY("mechanical_joint_indices",
                                          &Model::getMechanicalJointIndices,
                                          bp::return_value_policy<result_converter<true>>())
            .ADD_PROPERTY_GET_WITH_POLICY("mechanical_joint_position_indices",
                                          &Model::getMechanicalJointPositionIndices,
                                          bp::return_value_policy<result_converter<true>>())
            .ADD_PROPERTY_GET_WITH_POLICY("mechanical_joint_velocity_indices",
                                          &Model::getMechanicalJointVelocityIndices,
                                          bp::return_value_policy<result_converter<true>>())
            .ADD_PROPERTY_GET_WITH_POLICY("flexibility_joint_names",
                                          &Model::getFlexibilityJointNames,
                                          bp::return_value_policy<result_converter<true>>())
            .ADD_PROPERTY_GET_WITH_POLICY("flexibility_joint_indices",
                                          &Model::getFlexibilityJointIndices,
                                          bp::return_value_policy<result_converter<true>>())
            .ADD_PROPERTY_GET_WITH_POLICY("backlash_joint_names",
                                          &Model::getBacklashJointNames,
                                          bp::return_value_policy<result_converter<true>>())
            .ADD_PROPERTY_GET_WITH_POLICY("backlash_joint_indices",
                                          &Model::getBacklashJointIndices,
                                          bp::return_value_policy<result_converter<true>>())

            .ADD_PROPERTY_GET_WITH_POLICY("log_position_fieldnames",
                                          &Model::getLogPositionFieldnames,
                                          bp::return_value_policy<result_converter<true>>())
            .ADD_PROPERTY_GET_WITH_POLICY("log_velocity_fieldnames",
                                          &Model::getLogVelocityFieldnames,
                                          bp::return_value_policy<result_converter<true>>())
            .ADD_PROPERTY_GET_WITH_POLICY("log_acceleration_fieldnames",
                                          &Model::getLogAccelerationFieldnames,
                                          bp::return_value_policy<result_converter<true>>())
            .ADD_PROPERTY_GET_WITH_POLICY("log_effort_fieldnames",
                                          &Model::getLogEffortFieldnames,
                                          bp::return_value_policy<result_converter<true>>())
            .ADD_PROPERTY_GET_WITH_POLICY("log_f_external_fieldnames",
                                          &Model::getLogForceExternalFieldnames,
                                          bp::return_value_policy<result_converter<true>>())
            .ADD_PROPERTY_GET_WITH_POLICY("log_constraint_fieldnames",
                                          &Model::getLogConstraintFieldnames,
                                          bp::return_value_policy<result_converter<true>>());
    }

    // ***************************************** Robot ***************************************** //

    namespace internal::robot
    {
        static void detachMotors(Robot & self, const bp::object & motorNamesPy)
        {
            auto motorNames = convertFromPython<std::vector<std::string>>(motorNamesPy);
            return self.detachMotors(motorNames);
        }

        static std::shared_ptr<SensorMeasurementTree> getSensorMeasurements(Robot & self)
        {
            return std::make_shared<SensorMeasurementTree>(self.getSensorMeasurements());
        }

        static void setModelOptions(Robot & self, const bp::dict & configPy)
        {
            GenericConfig config = self.getModelOptions();
            convertFromPython(configPy, config);
            return self.setModelOptions(config);
        }
    }

    void exposeRobot()
    {
        bp::class_<Robot, bp::bases<Model>, std::shared_ptr<Robot>, boost::noncopyable>(
            "Robot", bp::init<const std::string &>(bp::arg("name") = ""))
            .def("copy", &std::make_shared<Robot, const Robot &>, bp::arg("self"))

            .def("initialize",
                 &initialize<Robot>,
                 (bp::arg("self"),
                  "urdf_path",
                  bp::arg("has_freeflyer") = false,
                  bp::arg("mesh_package_dirs") = bp::list(),
                  bp::arg("load_visual_meshes") = false))
            .def("initialize",
                 &initializeFromModels<Robot>,
                 (bp::arg("self"),
                  "pinocchio_model",
                  bp::arg("collision_model") = bp::object(),
                  bp::arg("visual_model") = bp::object()))

            .ADD_PROPERTY_GET("is_locked", &Robot::getIsLocked)

            .ADD_PROPERTY_GET_WITH_POLICY(
                "name", &Robot::getName, bp::return_value_policy<bp::return_by_value>())

            .def("attach_motor", &Robot::attachMotor, (bp::arg("self"), "motor"))
            .def("detach_motor",
                 static_cast<void (Robot::*)(const std::string &)>(&Robot::detachMotor),
                 (bp::arg("self"), "joint_name"))
            .def("detach_motors",
                 &internal::robot::detachMotors,
                 (bp::arg("self"), bp::arg("joints_names") = bp::list()))
            .def("attach_sensor", &Robot::attachSensor, (bp::arg("self"), "sensor"))
            .def("detach_sensor",
                 &Robot::detachSensor,
                 (bp::arg("self"), "sensor_type", "sensor_name"))
            .def("detach_sensors",
                 &Robot::detachSensors,
                 (bp::arg("self"), bp::arg("sensor_type") = std::string()))

            .def("get_motor",
                 static_cast<std::shared_ptr<AbstractMotorBase> (Robot::*)(const std::string &)>(
                     &Robot::getMotor),
                 (bp::arg("self"), "motor_name"))
            .ADD_PROPERTY_GET_WITH_POLICY(
                "motors",
                static_cast<const Robot::MotorVector & (Robot::*)()>(&Robot::getMotors),
                bp::return_value_policy<result_converter<true>>())
            .def("get_sensor",
                 static_cast<std::shared_ptr<AbstractSensorBase> (Robot::*)(
                     const std::string &, const std::string &)>(&Robot::getSensor),
                 (bp::arg("self"), "sensor_type", "sensor_name"))
            .ADD_PROPERTY_GET_WITH_POLICY(
                "sensors",
                static_cast<const Robot::SensorTree & (Robot::*)()>(&Robot::getSensors),
                bp::return_value_policy<result_converter<true>>())

            .ADD_PROPERTY_GET_SET("controller",
                                  static_cast<std::shared_ptr<AbstractController> (Robot::*)()>(
                                      &Robot::getController),
                                  &Robot::setController)

            .def("compute_sensor_measurements",
                 &Robot::computeSensorMeasurements,
                 (bp::arg("self"), "t", "q", "v", "a", "u_motor", "f_external"))
            .ADD_PROPERTY_GET("sensor_measurements", &internal::robot::getSensorMeasurements)

            .def("set_options", &setOptions<Robot>, (bp::arg("self"), "robot_options"))
            .def("get_options", &Robot::getOptions, bp::return_value_policy<bp::return_by_value>())
            .def("set_model_options",
                 &internal::robot::setModelOptions,
                 (bp::arg("self"), "model_options"))
            .def("get_model_options",
                 &Robot::getModelOptions,
                 bp::return_value_policy<bp::return_by_value>())

            .ADD_PROPERTY_GET_WITH_POLICY("log_command_fieldnames",
                                          &Robot::getLogCommandFieldnames,
                                          bp::return_value_policy<result_converter<true>>())

            .ADD_PROPERTY_GET("nmotors", &Robot::nmotors);
    }
}
