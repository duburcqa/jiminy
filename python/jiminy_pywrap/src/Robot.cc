#include "jiminy/core/robot/AbstractSensor.h"
#include "jiminy/core/robot/AbstractMotor.h"
#include "jiminy/core/robot/AbstractConstraint.h"
#include "jiminy/core/robot/Robot.h"

#include "jiminy/python/Robot.h"

#include <boost/python.hpp>


namespace jiminy
{
namespace python
{
    namespace bp = boost::python;

    // ***************************** PyModelVisitor ***********************************

    struct PyModelVisitor
        : public bp::def_visitor<PyModelVisitor>
    {
    public:
        ///////////////////////////////////////////////////////////////////////////////
        /// \brief Expose C++ API through the visitor.
        ///////////////////////////////////////////////////////////////////////////////
        template<class PyClass>
        void visit(PyClass & cl) const
        {
            cl
                .def("add_frame", &Model::addFrame,
                                  (bp::arg("self"), "frame_name", "parent_body_name", "frame_placement"))
                .def("remove_frame", &Model::removeFrame,
                                     (bp::arg("self"), "frame_name"))
                .def("add_collision_bodies", &PyModelVisitor::addCollisionBodies,
                                             (bp::arg("self"),
                                              bp::arg("bodies_names") = std::vector<std::string>(),
                                              bp::arg("ignore_meshes") = false))
                .def("remove_collision_bodies", &PyModelVisitor::removeCollisionBodies,
                                                (bp::arg("self"), "bodies_names"))
                .def("add_contact_points", &PyModelVisitor::addContactPoints,
                                           (bp::arg("self"),
                                            bp::arg("frame_names") = std::vector<std::string>()))
                .def("remove_contact_points", &PyModelVisitor::removeContactPoints,
                                              (bp::arg("self"), "frame_names"))

                .def("get_flexible_configuration_from_rigid", &PyModelVisitor::getFlexibleConfigurationFromRigid,
                                                              (bp::arg("self"), "rigid_position"))
                .def("get_flexible_velocity_from_rigid", &PyModelVisitor::getFlexibleVelocityFromRigid,
                                                         (bp::arg("self"), "rigid_velocity"))
                .def("get_rigid_configuration_from_flexible", &PyModelVisitor::getRigidConfigurationFromFlexible,
                                                              (bp::arg("self"), "flexible_position"))
                .def("get_rigid_velocity_from_flexible", &PyModelVisitor::getRigidVelocityFromFlexible,
                                                         (bp::arg("self"), "flexible_velocity"))

                .add_property("pinocchio_model", bp::make_getter(&Model::pncModel_,
                                                 bp::return_internal_reference<>()))
                .add_property("pinocchio_data", bp::make_getter(&Model::pncData_,
                                                bp::return_internal_reference<>()))
                .add_property("pinocchio_model_th", bp::make_getter(&Model::pncModelRigidOrig_,
                                                    bp::return_internal_reference<>()))
                .add_property("pinocchio_data_th", bp::make_getter(&Model::pncDataRigidOrig_,
                                                   bp::return_internal_reference<>()))
                .add_property("collision_model", bp::make_getter(&Model::pncGeometryModel_,
                                                 bp::return_internal_reference<>()))
                .add_property("collision_data", bp::make_function(&PyModelVisitor::getGeometryData,
                                                bp::return_internal_reference<>()))

                .add_property("is_initialized", bp::make_function(&Model::getIsInitialized,
                                                bp::return_value_policy<bp::copy_const_reference>()))
                .add_property("mesh_package_dirs", bp::make_function(&Model::getMeshPackageDirs,
                                                   bp::return_value_policy<bp::copy_const_reference>()))
                .add_property("urdf_path", bp::make_function(&Model::getUrdfPath,
                                           bp::return_value_policy<bp::copy_const_reference>()))
                .add_property("has_freeflyer", bp::make_function(&Model::getHasFreeflyer,
                                               bp::return_value_policy<bp::copy_const_reference>()))
                .add_property("is_flexible", &PyModelVisitor::isFlexibleModelEnable)
                .add_property("nq", bp::make_function(&Model::nq,
                                    bp::return_value_policy<bp::copy_const_reference>()))
                .add_property("nv", bp::make_function(&Model::nv,
                                    bp::return_value_policy<bp::copy_const_reference>()))
                .add_property("nx", bp::make_function(&Model::nx,
                                    bp::return_value_policy<bp::copy_const_reference>()))

                .add_property("collision_bodies_names", bp::make_function(&Model::getCollisionBodiesNames,
                                                        bp::return_value_policy<bp::copy_const_reference>()))
                .add_property("collision_bodies_idx", bp::make_function(&Model::getCollisionBodiesIdx,
                                                      bp::return_value_policy<bp::copy_const_reference>()))
                .add_property("collision_pairs_idx_by_body", bp::make_function(&Model::getCollisionPairsIdx,
                                                             bp::return_value_policy<bp::copy_const_reference>()))
                .add_property("contact_frames_names", bp::make_function(&Model::getContactFramesNames,
                                                      bp::return_value_policy<bp::copy_const_reference>()))
                .add_property("contact_frames_idx", bp::make_function(&Model::getContactFramesIdx,
                                                    bp::return_value_policy<bp::copy_const_reference>()))
                .add_property("rigid_joints_names", bp::make_function(&Model::getRigidJointsNames,
                                                    bp::return_value_policy<bp::copy_const_reference>()))
                .add_property("rigid_joints_idx", bp::make_function(&Model::getRigidJointsModelIdx,
                                                  bp::return_value_policy<bp::copy_const_reference>()))
                .add_property("rigid_joints_position_idx", bp::make_function(&Model::getRigidJointsPositionIdx,
                                                           bp::return_value_policy<bp::copy_const_reference>()))
                .add_property("rigid_joints_velocity_idx", bp::make_function(&Model::getRigidJointsVelocityIdx,
                                                           bp::return_value_policy<bp::copy_const_reference>()))
                .add_property("flexible_joints_names", bp::make_function(&Model::getFlexibleJointsNames,
                                                       bp::return_value_policy<bp::copy_const_reference>()))
                .add_property("flexible_joints_idx", bp::make_function(&Model::getFlexibleJointsModelIdx,
                                                     bp::return_value_policy<bp::copy_const_reference>()))

                .add_property("position_limit_lower", bp::make_function(&Model::getPositionLimitMin,
                                                      bp::return_value_policy<bp::copy_const_reference>()))
                .add_property("position_limit_upper", bp::make_function(&Model::getPositionLimitMax,
                                                      bp::return_value_policy<bp::copy_const_reference>()))
                .add_property("velocity_limit", bp::make_function(&Model::getVelocityLimit,
                                                bp::return_value_policy<bp::copy_const_reference>()))

                .add_property("logfile_position_headers", bp::make_function(&Model::getPositionFieldnames,
                                                          bp::return_value_policy<bp::copy_const_reference>()))
                .add_property("logfile_velocity_headers", bp::make_function(&Model::getVelocityFieldnames,
                                                          bp::return_value_policy<bp::copy_const_reference>()))
                .add_property("logfile_acceleration_headers", bp::make_function(&Model::getAccelerationFieldnames,
                                                              bp::return_value_policy<bp::copy_const_reference>()))
                ;
        }

        static pinocchio::GeometryData & getGeometryData(Model & self)
        {
            return *(self.pncGeometryData_);
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

        static bool_t isFlexibleModelEnable(Model & self)
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

    struct PyRobotVisitor
        : public bp::def_visitor<PyRobotVisitor>
    {
    public:
        ///////////////////////////////////////////////////////////////////////////////
        /// \brief Expose C++ API through the visitor.
        ///////////////////////////////////////////////////////////////////////////////
        template<class PyClass>
        void visit(PyClass & cl) const
        {
            cl
                .def("initialize", &Robot::initialize,
                                   (bp::arg("self"), "urdf_path",
                                    bp::arg("has_freeflyer") = false,
                                    bp::arg("mesh_package_dirs") = std::vector<std::string>()))

                .def("attach_motor", &Robot::attachMotor,
                                     (bp::arg("self"), "motor"))
                .def("get_motor", &PyRobotVisitor::getMotor,
                                  (bp::arg("self"), "motor_name"))
                .def("detach_motor", &Robot::detachMotor,
                                     (bp::arg("self"), "joint_name"))
                .def("detach_motors", &PyRobotVisitor::detachMotors,
                                      (bp::arg("self"),
                                       bp::arg("joints_names") = std::vector<std::string>()))
                .def("attach_sensor", &Robot::attachSensor,
                                      (bp::arg("self"), "sensor"))
                .def("detach_sensor", &Robot::detachSensor,
                                      (bp::arg("self"), "sensor_type", "sensor_name"))
                .def("detach_sensors", &Robot::detachSensors,
                                       (bp::arg("self"),
                                        bp::arg("sensor_type") = std::string()))
                .def("get_sensor", &PyRobotVisitor::getSensor,
                                   (bp::arg("self"), "sensor_type", "sensor_name"))
                .def("add_constraint", &Robot::addConstraint,
                                       (bp::arg("self"), "name", "constraint"))
                .def("get_constraint", &PyRobotVisitor::getConstraint,
                                  (bp::arg("self"), "constraint_name"))
                .def("remove_constraint", &Robot::removeConstraint,
                                          (bp::arg("self"), "name"))

                .add_property("sensors_data", &PyRobotVisitor::getSensorsData)

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

                .add_property("nmotors", bp::make_function(&Robot::nmotors,
                                         bp::return_value_policy<bp::copy_const_reference>()))
                .add_property("motors_names", bp::make_function(&Robot::getMotorsNames,
                                              bp::return_value_policy<bp::copy_const_reference>()))
                .add_property("motors_position_idx", &Robot::getMotorsPositionIdx)
                .add_property("motors_velocity_idx", &Robot::getMotorsVelocityIdx)
                .add_property("sensors_names", &PyRobotVisitor::getSensorsNames)

                .add_property("effort_limit", &Robot::getEffortLimit)
                .add_property("motors_inertias", &Robot::getMotorsInertias)

                .add_property("logfile_command_headers", bp::make_function(&Robot::getCommandFieldnames,
                                                         bp::return_value_policy<bp::copy_const_reference>()))
                ;
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

        static std::shared_ptr<AbstractConstraint> getConstraint(Robot             & self,
                                                                 std::string const & constraintName)
        {
            std::shared_ptr<AbstractConstraint> constraint;
            self.getConstraint(constraintName, constraint);
            return constraint;
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

        static void setOptions(Robot          & self,
                               bp::dict const & configPy)
        {
            configHolder_t config = self.getOptions();
            convertFromPython(configPy, config);
            self.setOptions(config);
        }

        static void setModelOptions(Robot          & self,
                                    bp::dict const & configPy)
        {
            configHolder_t config = self.getModelOptions();
            convertFromPython(configPy, config);
            self.setModelOptions(config);
        }

        static void setMotorsOptions(Robot          & self,
                                     bp::dict const & configPy)
        {
            configHolder_t config = self.getMotorsOptions();
            convertFromPython(configPy, config);
            self.setMotorsOptions(config);
        }

        static void setSensorsOptions(Robot          & self,
                                      bp::dict const & configPy)
        {
            configHolder_t config = self.getSensorsOptions();
            convertFromPython(configPy, config);
            self.setSensorsOptions(config);
        }

        static void setTelemetryOptions(Robot          & self,
                                        bp::dict const & configPy)
        {
            configHolder_t config = self.getTelemetryOptions();
            convertFromPython(configPy, config);
            self.setTelemetryOptions(config);
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
