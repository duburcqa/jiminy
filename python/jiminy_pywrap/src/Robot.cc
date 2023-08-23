#include "jiminy/core/robot/AbstractSensor.h"
#include "jiminy/core/robot/AbstractMotor.h"
#include "jiminy/core/constraints/AbstractConstraint.h"
#include "jiminy/core/robot/Robot.h"

#include "pinocchio/bindings/python/fwd.hpp"

#include "jiminy/python/Utilities.h"
#include "jiminy/python/Robot.h"


namespace jiminy
{
namespace python
{
    namespace bp = boost::python;

    // ***************************** PyModelVisitor ***********************************

    struct PyModelVisitor : public bp::def_visitor<PyModelVisitor>
    {
    public:
        ///////////////////////////////////////////////////////////////////////////////
        /// \brief Expose C++ API through the visitor.
        ///////////////////////////////////////////////////////////////////////////////
        template<class PyClass>
        void visit(PyClass & cl) const
        {
            cl
                .def("add_frame",
                    static_cast<
                        hresult_t (Model::*)(std::string const &, std::string const &, pinocchio::SE3 const &)
                    >(&Model::addFrame),
                    (bp::arg("self"), "frame_name", "parent_body_name", "frame_placement"))
                .def("remove_frame", &Model::removeFrame,
                                     (bp::arg("self"), "frame_name"))
                .def("add_collision_bodies", &PyModelVisitor::addCollisionBodies,
                                             (bp::arg("self"),
                                              bp::arg("bodies_names") = bp::list(),
                                              bp::arg("ignore_meshes") = false))
                .def("remove_collision_bodies", &PyModelVisitor::removeCollisionBodies,
                                                (bp::arg("self"), "bodies_names"))
                .def("add_contact_points", &PyModelVisitor::addContactPoints,
                                           (bp::arg("self"),
                                            bp::arg("frame_names") = bp::list()))
                .def("remove_contact_points", &PyModelVisitor::removeContactPoints,
                                              (bp::arg("self"), "frame_names"))

                .def("add_constraint",
                    static_cast<
                        hresult_t (Model::*)(std::string const &, std::shared_ptr<AbstractConstraintBase> const &)
                    >(&Model::addConstraint),
                    (bp::arg("self"), "name", "constraint"))
                .def("remove_constraint",
                    static_cast<
                        hresult_t (Model::*)(std::string const &)
                    >(&Model::removeConstraint),
                    (bp::arg("self"), "name"))
                .def("get_constraint", &PyModelVisitor::getConstraint,
                                      (bp::arg("self"), "constraint_name"))
                .def("exist_constraint", &Model::existConstraint,
                                         (bp::arg("self"), "constraint_name"))
                .ADD_PROPERTY_GET("has_constraints", &Model::hasConstraints)
                .ADD_PROPERTY_GET("constraints", &PyModelVisitor::getConstraints)
                .def("get_constraints_jacobian_and_drift", &PyModelVisitor::getConstraintsJacobianAndDrift)
                .def("compute_constraints", &Model::computeConstraints,
                                            (bp::arg("self"), "q", "v"))

                .def("get_flexible_configuration_from_rigid", &PyModelVisitor::getFlexibleConfigurationFromRigid,
                                                              (bp::arg("self"), "rigid_position"))
                .def("get_flexible_velocity_from_rigid", &PyModelVisitor::getFlexibleVelocityFromRigid,
                                                         (bp::arg("self"), "rigid_velocity"))
                .def("get_rigid_configuration_from_flexible", &PyModelVisitor::getRigidConfigurationFromFlexible,
                                                              (bp::arg("self"), "flexible_position"))
                .def("get_rigid_velocity_from_flexible", &PyModelVisitor::getRigidVelocityFromFlexible,
                                                         (bp::arg("self"), "flexible_velocity"))

                // FIXME: Disable automatic typing because typename returned by 'py_type_str' is missing module
                // prefix, which makes it impossible to distinguish 'pinocchio.Model' from 'jiminy.Model' classes.
                .def_readonly("pinocchio_model_th", &Model::pncModelOrig_, "fget( (Model)self) -> pinocchio.Model")
                .def_readonly("pinocchio_model", &Model::pncModel_, "fget( (Model)self) -> pinocchio.Model")
                .DEF_READONLY("collision_model_th", &Model::collisionModelOrig_)
                .DEF_READONLY("collision_model", &Model::collisionModel_)
                .DEF_READONLY("visual_model_th", &Model::visualModelOrig_)
                .DEF_READONLY("visual_model", &Model::visualModel_)
                .DEF_READONLY("visual_data", &Model::visualData_)
                .DEF_READONLY("pinocchio_data_th", &Model::pncDataOrig_)
                .DEF_READONLY("pinocchio_data", &Model::pncData_)
                .DEF_READONLY("collision_data", &Model::collisionData_)

                .ADD_PROPERTY_GET_WITH_POLICY("is_initialized",
                                              &Model::getIsInitialized,
                                              bp::return_value_policy<bp::return_by_value>())
                .ADD_PROPERTY_GET_WITH_POLICY("mesh_package_dirs",
                                              &Model::getMeshPackageDirs,
                                              bp::return_value_policy<result_converter<true> >())
                .ADD_PROPERTY_GET_WITH_POLICY("name",
                                              &Model::getName,
                                              bp::return_value_policy<bp::return_by_value>())
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

                .ADD_PROPERTY_GET_WITH_POLICY("collision_bodies_names",
                                              &Model::getCollisionBodiesNames,
                                              bp::return_value_policy<result_converter<true> >())
                .ADD_PROPERTY_GET_WITH_POLICY("collision_bodies_idx",
                                              &Model::getCollisionBodiesIdx,
                                              bp::return_value_policy<result_converter<true> >())
                .ADD_PROPERTY_GET_WITH_POLICY("collision_pairs_idx_by_body",
                                              &Model::getCollisionPairsIdx,
                                              bp::return_value_policy<result_converter<true> >())
                .ADD_PROPERTY_GET_WITH_POLICY("contact_frames_names",
                                              &Model::getContactFramesNames,
                                              bp::return_value_policy<result_converter<true> >())
                .ADD_PROPERTY_GET_WITH_POLICY("contact_frames_idx",
                                              &Model::getContactFramesIdx,
                                              bp::return_value_policy<result_converter<true> >())
                .ADD_PROPERTY_GET_WITH_POLICY("rigid_joints_names",
                                              &Model::getRigidJointsNames,
                                              bp::return_value_policy<result_converter<true> >())
                .ADD_PROPERTY_GET_WITH_POLICY("rigid_joints_idx",
                                              &Model::getRigidJointsModelIdx,
                                              bp::return_value_policy<result_converter<true> >())
                .ADD_PROPERTY_GET_WITH_POLICY("rigid_joints_position_idx",
                                              &Model::getRigidJointsPositionIdx,
                                              bp::return_value_policy<result_converter<true> >())
                .ADD_PROPERTY_GET_WITH_POLICY("rigid_joints_velocity_idx",
                                              &Model::getRigidJointsVelocityIdx,
                                              bp::return_value_policy<result_converter<true> >())
                .ADD_PROPERTY_GET_WITH_POLICY("flexible_joints_names",
                                              &Model::getFlexibleJointsNames,
                                              bp::return_value_policy<result_converter<true> >())
                .ADD_PROPERTY_GET_WITH_POLICY("flexible_joints_idx",
                                              &Model::getFlexibleJointsModelIdx,
                                              bp::return_value_policy<result_converter<true> >())

                .ADD_PROPERTY_GET_WITH_POLICY("position_limit_lower",
                                              &Model::getPositionLimitMin,
                                              bp::return_value_policy<bp::copy_const_reference>())
                .ADD_PROPERTY_GET_WITH_POLICY("position_limit_upper",
                                              &Model::getPositionLimitMax,
                                              bp::return_value_policy<bp::copy_const_reference>())
                .ADD_PROPERTY_GET_WITH_POLICY("velocity_limit",
                                              &Model::getVelocityLimit,
                                              bp::return_value_policy<bp::copy_const_reference>())

                .ADD_PROPERTY_GET_WITH_POLICY("log_fieldnames_position",
                                              &Model::getLogFieldnamesPosition,
                                              bp::return_value_policy<result_converter<true> >())
                .ADD_PROPERTY_GET_WITH_POLICY("log_fieldnames_velocity",
                                              &Model::getLogFieldnamesVelocity,
                                              bp::return_value_policy<result_converter<true> >())
                .ADD_PROPERTY_GET_WITH_POLICY("log_fieldnames_acceleration",
                                              &Model::getLogFieldnamesAcceleration,
                                              bp::return_value_policy<result_converter<true> >())
                .ADD_PROPERTY_GET_WITH_POLICY("log_fieldnames_f_external",
                                              &Model::getLogFieldnamesForceExternal,
                                              bp::return_value_policy<result_converter<true> >())
                ;
        }

        static hresult_t addCollisionBodies(Model          & self,
                                            bp::list const & linkNamesPy,
                                            bool_t   const & ignoreMeshes)
        {
            auto linkNames = convertFromPython<std::vector<std::string> >(linkNamesPy);
            return self.addCollisionBodies(linkNames, ignoreMeshes);
        }

        static hresult_t removeCollisionBodies(Model          & self,
                                             bp::list const & linkNamesPy)
        {
            auto linkNames = convertFromPython<std::vector<std::string> >(linkNamesPy);
            return self.removeCollisionBodies(linkNames);
        }

        static hresult_t addContactPoints(Model          & self,
                                          bp::list const & frameNamesPy)
        {
            auto frameNames = convertFromPython<std::vector<std::string> >(frameNamesPy);
            return self.addContactPoints(frameNames);
        }

        static hresult_t removeContactPoints(Model          & self,
                                             bp::list const & frameNamesPy)
        {
            auto frameNames = convertFromPython<std::vector<std::string> >(frameNamesPy);
            return self.removeContactPoints(frameNames);
        }

        static std::shared_ptr<AbstractConstraintBase> getConstraint(Model             & self,
                                                                     std::string const & constraintName)
        {
            std::shared_ptr<AbstractConstraintBase> constraint;
            self.getConstraint(constraintName, constraint);
            return constraint;
        }

        static std::shared_ptr<constraintsHolder_t> getConstraints(Model & self)
        {
            return std::make_shared<constraintsHolder_t>(self.getConstraints());
        }

        static bp::tuple getConstraintsJacobianAndDrift(Model & self)
        {
            Eigen::Index constraintRow = 0;
            Eigen::Index constraintsRows = 0;
            constraintsHolder_t constraintsHolder = self.getConstraints();
            constraintsHolder.foreach(
                [&constraintsRows](
                    std::shared_ptr<AbstractConstraintBase> const & constraint,
                    constraintsHolderType_t const & /* holderType */)
                {
                    if (!constraint->getIsEnabled())
                    {
                        return;
                    }
                    constraintsRows += static_cast<Eigen::Index>(constraint->getDim());
                });
            matrixN_t J(constraintsRows, self.nv());
            vectorN_t gamma(constraintsRows);
            constraintsHolder.foreach(
                [&J, &gamma, &constraintRow](
                    std::shared_ptr<AbstractConstraintBase> const & constraint,
                    constraintsHolderType_t const & /* holderType */)
                {
                    if (!constraint->getIsEnabled())
                    {
                        return;
                    }
                    Eigen::Index const constraintDim = static_cast<Eigen::Index>(constraint->getDim());
                    J.middleRows(constraintRow, constraintDim) = constraint->getJacobian();
                    gamma.segment(constraintRow, constraintDim) = constraint->getDrift();
                    constraintRow += constraintDim;
                });
            return bp::make_tuple(J, gamma);
        }

        static vectorN_t getFlexibleConfigurationFromRigid(Model           & self,
                                                           vectorN_t const & qRigid)
        {
            vectorN_t qFlexible;
            self.getFlexibleConfigurationFromRigid(qRigid, qFlexible);
            return qFlexible;
        }

        static vectorN_t getFlexibleVelocityFromRigid(Model           & self,
                                                      vectorN_t const & vRigid)
        {
            vectorN_t vFlexible;
            self.getFlexibleVelocityFromRigid(vRigid, vFlexible);
            return vFlexible;
        }

        static vectorN_t getRigidConfigurationFromFlexible(Model           & self,
                                                           vectorN_t const & qFlexible)
        {
            vectorN_t qRigid;
            self.getRigidConfigurationFromFlexible(qFlexible, qRigid);
            return qRigid;
        }

        static vectorN_t getRigidVelocityFromFlexible(Model           & self,
                                                      vectorN_t const & vFlexible)
        {
            vectorN_t vRigid;
            self.getRigidVelocityFromFlexible(vFlexible, vRigid);
            return vRigid;
        }

        ///////////////////////////////////////////////////////////////////////////////
        /// \brief      Getters and Setters
        ///////////////////////////////////////////////////////////////////////////////

        static bool_t isFlexibleModelEnabled(Model & self)
        {
            return self.mdlOptions_->dynamics.enableFlexibleModel;
        }

        ///////////////////////////////////////////////////////////////////////////////
        /// \brief Expose.
        ///////////////////////////////////////////////////////////////////////////////
        static void expose()
        {
            bp::class_<Model,
                       std::shared_ptr<Model>,
                       boost::noncopyable>("Model", bp::no_init)
                .def(PyModelVisitor());
        }
    };

    BOOST_PYTHON_VISITOR_EXPOSE(Model)

    // ***************************** PyRobotVisitor ***********************************

    struct PyRobotVisitor : public bp::def_visitor<PyRobotVisitor>
    {
    public:
        ///////////////////////////////////////////////////////////////////////////////
        /// \brief Expose C++ API through the visitor.
        ///////////////////////////////////////////////////////////////////////////////
        template<class PyClass>
        void visit(PyClass & cl) const
        {
            cl
                .def("initialize", &PyRobotVisitor::initialize,
                                   (bp::arg("self"), "urdf_path",
                                    bp::arg("has_freeflyer") = false,
                                    bp::arg("mesh_package_dirs") = bp::list(),
                                    bp::arg("load_visual_meshes") = false))
                .def("initialize",
                    static_cast<
                        hresult_t (Robot::*)(pinocchio::Model const &, pinocchio::GeometryModel const &, pinocchio::GeometryModel const &)
                    >(&Robot::initialize),
                    (bp::arg("self"), "pinocchio_model", "collision_model", "visual_model"))

                .ADD_PROPERTY_GET_WITH_POLICY("is_locked",
                                              &Robot::getIsLocked,
                                              bp::return_value_policy<bp::return_by_value>())

                .def("dump_options", &Robot::dumpOptions,
                                     (bp::arg("self"), "json_filename"))
                .def("load_options", &Robot::loadOptions,
                                     (bp::arg("self"), "json_filename"))

                .def("attach_motor", &Robot::attachMotor,
                                     (bp::arg("self"), "motor"))
                .def("get_motor", &PyRobotVisitor::getMotor,
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
                .def("get_sensor", &PyRobotVisitor::getSensor,
                                   (bp::arg("self"), "sensor_type", "sensor_name"))

                .ADD_PROPERTY_GET("sensors_data", &PyRobotVisitor::getSensorsData)

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
                .def("get_sensors_options",
                    static_cast<
                        configHolder_t (Robot::*)(void) const
                    >(&Robot::getSensorsOptions))
                .def("set_telemetry_options", &PyRobotVisitor::setTelemetryOptions,
                                              (bp::arg("self"), "telemetry_options"))
                .def("get_telemetry_options", &Robot::getTelemetryOptions)

                .ADD_PROPERTY_GET_WITH_POLICY("nmotors",
                                              &Robot::nmotors,
                                              bp::return_value_policy<bp::return_by_value>())
                .ADD_PROPERTY_GET_WITH_POLICY("motors_names",
                                              &Robot::getMotorsNames,
                                              bp::return_value_policy<result_converter<true> >())
                .ADD_PROPERTY_GET("motors_position_idx", &Robot::getMotorsPositionIdx)
                .ADD_PROPERTY_GET("motors_velocity_idx", &Robot::getMotorsVelocityIdx)
                .ADD_PROPERTY_GET("sensors_names", &PyRobotVisitor::getSensorsNames)

                .ADD_PROPERTY_GET_WITH_POLICY("command_limit",
                                              &Robot::getCommandLimit,
                                              bp::return_value_policy<bp::return_by_value>())

                .ADD_PROPERTY_GET_WITH_POLICY("log_fieldnames_command",
                                              &Robot::getCommandFieldnames,
                                              bp::return_value_policy<result_converter<true> >())
                .ADD_PROPERTY_GET_WITH_POLICY("log_fieldnames_motor_effort",
                                              &Robot::getMotorEffortFieldnames,
                                              bp::return_value_policy<result_converter<true> >())
                ;
        }

        static hresult_t initialize(Robot             & self,
                                    std::string const & urdfPath,
                                    bool_t      const & hasFreeflyer,
                                    bp::list    const & meshPackageDirsPy,
                                    bool_t      const & loadVisualMeshes)
        {
            auto meshPackageDirs = convertFromPython<std::vector<std::string> >(meshPackageDirsPy);
            return self.initialize(urdfPath, hasFreeflyer, meshPackageDirs, loadVisualMeshes);
        }

        static hresult_t detachMotors(Robot          & self,
                                      bp::list const & jointNamesPy)
        {
            auto jointNames = convertFromPython<std::vector<std::string> >(jointNamesPy);
            return self.detachMotors(jointNames);
        }

        ///////////////////////////////////////////////////////////////////////////////
        /// \brief      Getters and Setters
        ///////////////////////////////////////////////////////////////////////////////

        static std::shared_ptr<AbstractMotorBase> getMotor(Robot             & self,
                                                           std::string const & motorName)
        {
            std::shared_ptr<AbstractMotorBase> motor;
            self.getMotor(motorName, motor);
            return motor;
        }

        static std::shared_ptr<AbstractSensorBase> getSensor(Robot             & self,
                                                             std::string const & sensorType,
                                                             std::string const & sensorName)
        {
            std::shared_ptr<AbstractSensorBase> sensor;
            self.getSensor(sensorType, sensorName, sensor);
            return sensor;
        }

        static std::shared_ptr<sensorsDataMap_t> getSensorsData(Robot & self)
        {
            return std::make_shared<sensorsDataMap_t>(self.getSensorsData());
        }

        static bp::dict getSensorsNames(Robot & self)
        {
            bp::dict sensorsNamesPy;
            auto const & sensorsNames = self.getSensorsNames();
            for (auto const & sensorTypeNames : sensorsNames)
            {
                sensorsNamesPy[sensorTypeNames.first] =
                    convertToPython(sensorTypeNames.second);
            }
            return sensorsNamesPy;
        }

        static hresult_t setOptions(Robot          & self,
                                    bp::dict const & configPy)
        {
            configHolder_t config = self.getOptions();
            convertFromPython(configPy, config);
            return self.setOptions(config);
        }

        static hresult_t setModelOptions(Robot          & self,
                                         bp::dict const & configPy)
        {
            configHolder_t config = self.getModelOptions();
            convertFromPython(configPy, config);
            return self.setModelOptions(config);
        }

        static hresult_t setMotorsOptions(Robot          & self,
                                          bp::dict const & configPy)
        {
            configHolder_t config = self.getMotorsOptions();
            convertFromPython(configPy, config);
            return self.setMotorsOptions(config);
        }

        static hresult_t setSensorsOptions(Robot          & self,
                                           bp::dict const & configPy)
        {
            configHolder_t config = self.getSensorsOptions();
            convertFromPython(configPy, config);
            return self.setSensorsOptions(config);
        }

        static hresult_t setTelemetryOptions(Robot          & self,
                                             bp::dict const & configPy)
        {
            configHolder_t config = self.getTelemetryOptions();
            convertFromPython(configPy, config);
            return self.setTelemetryOptions(config);
        }

        ///////////////////////////////////////////////////////////////////////////////
        /// \brief Expose.
        ///////////////////////////////////////////////////////////////////////////////
        static void expose()
        {
            bp::class_<Robot, bp::bases<Model>,
                       std::shared_ptr<Robot>,
                       boost::noncopyable>("Robot")
                .def(PyRobotVisitor());
        }
    };

    BOOST_PYTHON_VISITOR_EXPOSE(Robot)
}  // End of namespace python.
}  // End of namespace jiminy.
