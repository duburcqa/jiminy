#include "jiminy/core/utilities/helpers.h"
#include "jiminy/core/utilities/json.h"
#include "jiminy/core/io/serialization.h"
#include "jiminy/core/io/abstract_io_device.h"
#include "jiminy/core/io/memory_device.h"
#include "jiminy/core/control/abstract_controller.h"
#include "jiminy/core/robot/robot.h"
#include "jiminy/core/engine/engine.h"
#include "jiminy/core/engine/engine_multi_robot.h"

#include "pinocchio/bindings/python/fwd.hpp"
#include <boost/python/raw_function.hpp>

#include "jiminy/python/utilities.h"
#include "jiminy/python/functors.h"
#include "jiminy/python/engine.h"


namespace jiminy::python
{
    namespace bp = boost::python;

    // ************* Expose impulse, profile, and coupling force registers **************

    static bp::object profileForceWrapper(const ProfileForce & self)
    {
        bp::object func = makeFunction(
            self.func, bp::return_value_policy<bp::return_by_value>(), (bp::arg("t"), "q", "v"));
        setFunctionWrapperModule<ProfileForce>(func);
        return func;
    }

    static bp::object couplingForceWrapper(const CouplingForce & self)
    {
        bp::object func = makeFunction(self.func,
                                       bp::return_value_policy<bp::return_by_value>(),
                                       (bp::arg("t"), "q1", "v1", "q2", "v2"));
        setFunctionWrapperModule<CouplingForce>(func);
        return func;
    }

    void exposeForces()
    {
        // clang-format off
        bp::class_<ProfileForce,
                   std::shared_ptr<ProfileForce>,
                   boost::noncopyable>("ProfileForce", bp::no_init)
            .DEF_READONLY("frame_name", &ProfileForce::frameName)
            .DEF_READONLY("frame_index", &ProfileForce::frameIndex)
            .DEF_READONLY("update_period", &ProfileForce::updatePeriod)
            .DEF_READONLY("force", &ProfileForce::force)
            .ADD_PROPERTY_GET("func", profileForceWrapper);

        /* Note that it will be impossible to slice the vector if `boost::noncopyable` is set for
           the stl container, or if the value type contained itself. In such a case, it raises a
           runtime error rather than a compile-time error. */
        bp::class_<ProfileForceVector>("ProfileForceVector", bp::no_init)
            .def(vector_indexing_suite_no_contains<ProfileForceVector>());

        bp::class_<ImpulseForce,
                   std::shared_ptr<ImpulseForce>,
                   boost::noncopyable>("ImpulseForce", bp::no_init)
            .DEF_READONLY("frame_name", &ImpulseForce::frameName)
            .DEF_READONLY("frame_index", &ImpulseForce::frameIndex)
            .DEF_READONLY("t", &ImpulseForce::t)
            .DEF_READONLY("dt", &ImpulseForce::dt)
            .DEF_READONLY("force", &ImpulseForce::force);

        bp::class_<ImpulseForceVector,
                   boost::noncopyable>("ImpulseForceVector", bp::no_init)
            .def(vector_indexing_suite_no_contains<ImpulseForceVector>());

        bp::class_<CouplingForce,
                   std::shared_ptr<CouplingForce>,
                   boost::noncopyable>("CouplingForce", bp::no_init)
            .DEF_READONLY("system_name_1", &CouplingForce::systemName1)
            .DEF_READONLY("system_index_1", &CouplingForce::systemIndex1)
            .DEF_READONLY("system_name_2", &CouplingForce::systemName2)
            .DEF_READONLY("system_index_2", &CouplingForce::systemIndex2)
            .ADD_PROPERTY_GET("func", couplingForceWrapper);

        bp::class_<CouplingForceVector,
                   boost::noncopyable>("CouplingForceVector", bp::no_init)
            .def(vector_indexing_suite_no_contains<CouplingForceVector>());
        // clang-format on
    }

    // ***************************** PyStepperStateVisitor ***********************************

    struct PyStepperStateVisitor : public bp::def_visitor<PyStepperStateVisitor>
    {
    public:
        /// \brief Expose C++ API through the visitor.
        template<class PyClass>
        void visit(PyClass & cl) const
        {
            // clang-format off
            cl
                .DEF_READONLY("iter", &StepperState::iter)
                .DEF_READONLY("iter_failed", &StepperState::iterFailed)
                .DEF_READONLY("t", &StepperState::t)
                .DEF_READONLY("dt", &StepperState::dt)
                .ADD_PROPERTY_GET("q", &PyStepperStateVisitor::getQ)
                .ADD_PROPERTY_GET("v", &PyStepperStateVisitor::getV)
                .ADD_PROPERTY_GET("a", &PyStepperStateVisitor::getA)
                .def("__repr__", &PyStepperStateVisitor::repr)
                ;
            // clang-format on
        }

        static bp::object getQ(const StepperState & self)
        {
            return convertToPython(self.qSplit, false);
        }

        static bp::object getV(const StepperState & self)
        {
            return convertToPython(self.vSplit, false);
        }

        static bp::object getA(const StepperState & self)
        {
            return convertToPython(self.aSplit, false);
        }

        static std::string repr(const StepperState & self)
        {
            std::stringstream s;
            Eigen::IOFormat HeavyFmt(5, 1, ", ", "", "", "", "[", "]\n");
            s << "iter:\n    " << self.iter;
            s << "\niter_failed:\n    " << self.iterFailed;
            s << "\nt:\n    " << self.t;
            s << "\ndt:\n    " << self.dt;
            s << "\nq:";
            for (std::size_t i = 0; i < self.qSplit.size(); ++i)
            {
                s << "\n    (" << i << "): " << self.qSplit[i].transpose().format(HeavyFmt);
            }
            s << "\nv:";
            for (std::size_t i = 0; i < self.vSplit.size(); ++i)
            {
                s << "\n    (" << i << "): " << self.vSplit[i].transpose().format(HeavyFmt);
            }
            s << "\na:";
            for (std::size_t i = 0; i < self.aSplit.size(); ++i)
            {
                s << "\n    (" << i << "): " << self.aSplit[i].transpose().format(HeavyFmt);
            }
            return s.str();
        }

        static void expose()
        {
            // clang-format off
            bp::class_<StepperState,
                       std::shared_ptr<StepperState>,
                       boost::noncopyable>("StepperState", bp::no_init)
                .def(PyStepperStateVisitor());
            // clang-format on
        }
    };

    BOOST_PYTHON_VISITOR_EXPOSE(StepperState)

    // ***************************** PySystemStateVisitor ***********************************

    struct PySystemStateVisitor : public bp::def_visitor<PySystemStateVisitor>
    {
    public:
        /// \brief Expose C++ API through the visitor.
        template<class PyClass>
        void visit(PyClass & cl) const
        {
            // clang-format off
            cl
                .DEF_READONLY("q", &SystemState::q)
                .DEF_READONLY("v", &SystemState::v)
                .DEF_READONLY("a", &SystemState::a)
                .DEF_READONLY("command", &SystemState::command)
                .DEF_READONLY("u", &SystemState::u)
                .DEF_READONLY("u_motor", &SystemState::uMotor)
                .DEF_READONLY("u_internal", &SystemState::uInternal)
                .DEF_READONLY("u_custom", &SystemState::uCustom)
                .DEF_READONLY("f_external", &SystemState::fExternal)
                .def("__repr__", &PySystemStateVisitor::repr)
                ;
            // clang-format on
        }

        static std::string repr(SystemState & self)
        {
            std::stringstream s;
            Eigen::IOFormat HeavyFmt(5, 1, ", ", "", "", "", "[", "]\n");
            s << "q:\n    " << self.q.transpose().format(HeavyFmt);
            s << "v:\n    " << self.v.transpose().format(HeavyFmt);
            s << "a:\n    " << self.a.transpose().format(HeavyFmt);
            s << "command:\n    " << self.command.transpose().format(HeavyFmt);
            s << "u:\n    " << self.u.transpose().format(HeavyFmt);
            s << "u_motor:\n    " << self.uMotor.transpose().format(HeavyFmt);
            s << "u_internal:\n    " << self.uInternal.transpose().format(HeavyFmt);
            s << "u_custom:\n    " << self.uCustom.transpose().format(HeavyFmt);
            s << "f_external:\n";
            for (std::size_t i = 0; i < self.fExternal.size(); ++i)
            {
                s << "    (" << i
                  << "): " << self.fExternal[i].toVector().transpose().format(HeavyFmt);
            }
            return s.str();
        }

        static void expose()
        {
            // clang-format off
            bp::class_<SystemState,
                       std::shared_ptr<SystemState>,
                       boost::noncopyable>("SystemState", bp::no_init)
                .def(PySystemStateVisitor());
            // clang-format on
        }
    };

    BOOST_PYTHON_VISITOR_EXPOSE(SystemState)

    // ***************************** PySystemVisitor ***********************************

    struct PySystemVisitor : public bp::def_visitor<PySystemVisitor>
    {
    public:
        /// \brief Expose C++ API through the visitor.
        template<class PyClass>
        void visit(PyClass & cl) const
        {
            // clang-format off
            cl
                .DEF_READONLY("name", &System::name)
                .DEF_READONLY("robot", &System::robot)
                .DEF_READONLY("controller", &System::controller)
                .DEF_READONLY("callbackFct", &System::callback)
                ;
            // clang-format on
        }

        static void expose()
        {
            // clang-format off
            bp::class_<System>("System", bp::no_init)
                .def(PySystemVisitor());

            bp::class_<std::vector<System>>("SystemVector", bp::no_init)
                .def(vector_indexing_suite_no_contains<std::vector<System>>());
            // clang-format on
        }
    };

    BOOST_PYTHON_VISITOR_EXPOSE(System)

    // ************************* PyEngineMultiRobotVisitor ****************************

    struct PyEngineMultiRobotVisitor : public bp::def_visitor<PyEngineMultiRobotVisitor>
    {
    public:
        template<class PyClass>
        void visit(PyClass & cl) const
        {
            // clang-format off
            cl
                .def("add_system", &PyEngineMultiRobotVisitor::addSystem,
                                   (bp::arg("self"), "system_name", "robot",
                                    bp::arg("controller") = bp::object(),
                                    bp::arg("callback_function") = bp::object()))
                .def("remove_system", &EngineMultiRobot::removeSystem,
                                      (bp::arg("self"), "system_name"))
                .def("set_controller", &EngineMultiRobot::setController,
                                      (bp::arg("self"), "system_name", "controller"))

                .def("reset",
                     static_cast<
                         void (EngineMultiRobot::*)(bool, bool)
                     >(&EngineMultiRobot::reset),
                     (bp::arg("self"),
                      bp::arg("reset_random_generator") = false,
                      bp::arg("remove_all_forces") = false))
                .def("start", &PyEngineMultiRobotVisitor::start,
                              (bp::arg("self"), "q_init_list", "v_init_list",
                               bp::arg("a_init_list") = bp::object()))  // bp::object() means 'None' in Python
                .def("step", &PyEngineMultiRobotVisitor::step,
                             (bp::arg("self"), bp::arg("dt_desired") = -1))
                .def("stop", &EngineMultiRobot::stop, (bp::arg("self")))
                .def("simulate", &PyEngineMultiRobotVisitor::simulate,
                                 (bp::arg("self"), "t_end", "q_init_list", "v_init_list",
                                  bp::arg("a_init_list") = bp::object()))
                .def("compute_forward_kinematics", &EngineMultiRobot::computeForwardKinematics,
                                                   (bp::arg("system"), "q", "v", "a"))
                .staticmethod("compute_forward_kinematics")
                .def("compute_systems_dynamics", &PyEngineMultiRobotVisitor::computeSystemsDynamics,
                                                 bp::return_value_policy<result_converter<true>>(),
                                                 (bp::arg("self"), "t_end", "q_list", "v_list"))

                .ADD_PROPERTY_GET("log_data", &PyEngineMultiRobotVisitor::getLog)
                .def("read_log", &PyEngineMultiRobotVisitor::readLog,
                                 (bp::arg("fullpath"), bp::arg("format") = bp::object()),
                                 "Read a logfile from jiminy.\n\n"
                                 ".. note::\n    This function supports both binary and hdf5 log.\n\n"
                                 ":param fullpath: Name of the file to load.\n"
                                 ":param format: Name of the file to load.\n\n"
                                 ":returns: Dictionary containing the logged constants and variables.")
                .staticmethod("read_log")
                .def("write_log", &EngineMultiRobot::writeLog, (bp::arg("self"), "fullpath", "format"))

                .def("register_impulse_force", &PyEngineMultiRobotVisitor::registerImpulseForce,
                                               (bp::arg("self"), "system_name",
                                                "frame_name", "t", "dt", "force"))
                .def("remove_impulse_forces",
                     static_cast<
                         void (EngineMultiRobot::*)(const std::string &)
                     >(&EngineMultiRobot::removeImpulseForces),
                     (bp::arg("self"), "system_name"))
                .def("remove_impulse_forces",
                     static_cast<
                         void (EngineMultiRobot::*)(void)
                     >(&EngineMultiRobot::removeImpulseForces),
                     (bp::arg("self")))
                .ADD_PROPERTY_GET("impulse_forces", &PyEngineMultiRobotVisitor::getImpulseForces)

                .def("register_profile_force", &PyEngineMultiRobotVisitor::registerProfileForce,
                                               (bp::arg("self"), "system_name",
                                                "frame_name", "force_function",
                                                bp::arg("update_period") = 0.0))
                .def("remove_profile_forces",
                     static_cast<
                         void (EngineMultiRobot::*)(const std::string &)
                     >(&EngineMultiRobot::removeProfileForces),
                     (bp::arg("self"), "system_name"))
                .def("remove_profile_forces",
                     static_cast<
                         void (EngineMultiRobot::*)(void)
                     >(&EngineMultiRobot::removeProfileForces),
                     (bp::arg("self")))
                .ADD_PROPERTY_GET("profile_forces", &PyEngineMultiRobotVisitor::getProfileForces)

                .def("register_coupling_force", &PyEngineMultiRobotVisitor::registerCouplingForce,
                                                (bp::arg("self"),
                                                 "system_name_1", "system_name_2",
                                                 "frame_name_1", "frame_name_2",
                                                 "force_function"))
                .def("register_viscoelastic_coupling_force",
                     static_cast<
                         void (EngineMultiRobot::*)(
                             const std::string &,
                             const std::string &,
                             const std::string &,
                             const std::string &,
                             const Vector6d &,
                             const Vector6d &,
                             double)
                     >(&EngineMultiRobot::registerViscoelasticCouplingForce),
                     (bp::arg("self"), "system_name_1", "system_name_2",
                      "frame_name_1", "frame_name_2", "stiffness", "damping", bp::arg("alpha") = 0.5))
                .def("register_viscoelastic_coupling_force",
                     static_cast<
                         void (EngineMultiRobot::*)(
                             const std::string &,
                             const std::string &,
                             const std::string &,
                             const Vector6d &,
                             const Vector6d &,
                             double)
                     >(&EngineMultiRobot::registerViscoelasticCouplingForce),
                     (bp::arg("self"), "system_name", "frame_name_1", "frame_name_2",
                      "stiffness", "damping", bp::arg("alpha") = 0.5))
                .def("register_viscoelastic_directional_coupling_force",
                     static_cast<
                         void (EngineMultiRobot::*)(
                             const std::string &,
                             const std::string &,
                             const std::string &,
                             const std::string &,
                             double,
                             double,
                             double)
                     >(&EngineMultiRobot::registerViscoelasticDirectionalCouplingForce),
                     (bp::arg("self"), "system_name_1", "system_name_2", "frame_name_1", "frame_name_2",
                      "stiffness", "damping", bp::arg("rest_length") = 0.0))
                .def("register_viscoelastic_directional_coupling_force",
                     static_cast<
                         void (EngineMultiRobot::*)(
                             const std::string &,
                             const std::string &,
                             const std::string &,
                             double,
                             double,
                             double)
                     >(&EngineMultiRobot::registerViscoelasticDirectionalCouplingForce),
                     (bp::arg("self"), "system_name", "frame_name_1", "frame_name_2",
                      "stiffness", "damping", bp::arg("rest_length") = 0.0))
                .def("remove_coupling_forces",
                     static_cast<
                         void (EngineMultiRobot::*)(const std::string &, const std::string &)
                     >(&EngineMultiRobot::removeCouplingForces),
                     (bp::arg("self"), "system_name_1", "system_name_2"))
                .def("remove_coupling_forces",
                     static_cast<
                         void (EngineMultiRobot::*)(const std::string &)
                     >(&EngineMultiRobot::removeCouplingForces),
                     (bp::arg("self"), "system_name"))
                .def("remove_coupling_forces",
                     static_cast<
                         void (EngineMultiRobot::*)(void)
                     >(&EngineMultiRobot::removeCouplingForces),
                     (bp::arg("self")))
                .ADD_PROPERTY_GET_WITH_POLICY("coupling_forces",
                                              &EngineMultiRobot::getCouplingForces,
                                              bp::return_value_policy<result_converter<false>>())

                .def("remove_all_forces", &EngineMultiRobot::removeAllForces)

                .def("set_options", &PyEngineMultiRobotVisitor::setOptions)
                .def("get_options", &EngineMultiRobot::getOptions)

                .DEF_READONLY("systems", &EngineMultiRobot::systems_)
                .ADD_PROPERTY_GET_WITH_POLICY("system_names",
                                              &EngineMultiRobot::getSystemNames,
                                              bp::return_value_policy<result_converter<true>>())
                .ADD_PROPERTY_GET("system_states", &PyEngineMultiRobotVisitor::getSystemStates)
                .ADD_PROPERTY_GET_WITH_POLICY("stepper_state",
                                              &EngineMultiRobot::getStepperState,
                                              bp::return_value_policy<result_converter<false>>())
                .ADD_PROPERTY_GET_WITH_POLICY("is_simulation_running",
                                              &EngineMultiRobot::getIsSimulationRunning,
                                              bp::return_value_policy<result_converter<false>>())
                .add_static_property("simulation_duration_max", &EngineMultiRobot::getSimulationDurationMax)
                .add_static_property("telemetry_time_unit", &EngineMultiRobot::getTelemetryTimeUnit)
                ;
            // clang-format on
        }

        static void addSystem(EngineMultiRobot & self,
                              const std::string & systemName,
                              const std::shared_ptr<Robot> & robot,
                              const bp::object & controllerPy,
                              const bp::object & callbackPy)
        {
            AbortSimulationFunction callback;
            if (callbackPy.is_none())
            {
                callback = [](double /* t */,
                              const Eigen::VectorXd & /* q */,
                              const Eigen::VectorXd & /* v */) -> bool
                {
                    return true;
                };
            }
            else
            {
                callback = TimeStateFunPyWrapper<bool>(callbackPy);
            }
            if (!controllerPy.is_none())
            {
                const std::shared_ptr<AbstractController> controller =
                    bp::extract<std::shared_ptr<AbstractController>>(controllerPy);
                return self.addSystem(systemName, robot, controller, callback);
            }
            return self.addSystem(systemName, robot, callback);
        }

        static bp::dict getImpulseForces(EngineMultiRobot & self)
        {
            bp::dict impulseForcesPy;
            for (const auto & systemName : self.getSystemNames())
            {
                const ImpulseForceVector & impulseForces = self.getImpulseForces(systemName);
                impulseForcesPy[systemName] = convertToPython(impulseForces, false);
            }
            return impulseForcesPy;
        }

        static bp::dict getProfileForces(EngineMultiRobot & self)
        {
            bp::dict profileForcessPy;
            for (const auto & systemName : self.getSystemNames())
            {
                const ProfileForceVector & profileForces = self.getProfileForces(systemName);
                profileForcessPy[systemName] = convertToPython(profileForces, false);
            }
            return profileForcessPy;
        }

        static bp::list getSystemStates(EngineMultiRobot & self)
        {
            bp::list systemStates;
            for (const std::string & systemName : self.getSystemNames())
            {
                const SystemState & systemState = self.getSystemState(systemName);
                systemStates.append(convertToPython(systemState, false));
            }
            return systemStates;
        }

        static void registerCouplingForce(EngineMultiRobot & self,
                                          const std::string & systemName1,
                                          const std::string & systemName2,
                                          const std::string & frameName1,
                                          const std::string & frameName2,
                                          const bp::object & forceFuncPy)
        {
            TimeBistateFunPyWrapper<pinocchio::Force> forceFunc(forceFuncPy);
            return self.registerCouplingForce(
                systemName1, systemName2, frameName1, frameName2, forceFunc);
        }

        static void start(EngineMultiRobot & self,
                          const bp::dict & qInitPy,
                          const bp::dict & vInitPy,
                          const bp::object & aInitPy)
        {
            std::optional<std::map<std::string, Eigen::VectorXd>> aInit = std::nullopt;
            if (!aInitPy.is_none())
            {
                aInit.emplace(convertFromPython<std::map<std::string, Eigen::VectorXd>>(aInitPy));
            }
            return self.start(convertFromPython<std::map<std::string, Eigen::VectorXd>>(qInitPy),
                              convertFromPython<std::map<std::string, Eigen::VectorXd>>(vInitPy),
                              aInit);
        }

        static void step(EngineMultiRobot & self, double dtDesired)
        {
            // Only way to handle C++ default values that are not accessible in Python
            return self.step(dtDesired);
        }

        static void simulate(EngineMultiRobot & self,
                             double endTime,
                             const bp::dict & qInitPy,
                             const bp::dict & vInitPy,
                             const bp::object & aInitPy)
        {
            std::optional<std::map<std::string, Eigen::VectorXd>> aInit = std::nullopt;
            if (!aInitPy.is_none())
            {
                aInit.emplace(convertFromPython<std::map<std::string, Eigen::VectorXd>>(aInitPy));
            }
            return self.simulate(
                endTime,
                convertFromPython<std::map<std::string, Eigen::VectorXd>>(qInitPy),
                convertFromPython<std::map<std::string, Eigen::VectorXd>>(vInitPy),
                aInit);
        }

        static std::vector<Eigen::VectorXd> computeSystemsDynamics(EngineMultiRobot & self,
                                                                   double endTime,
                                                                   const bp::object & qSplitPy,
                                                                   const bp::object & vSplitPy)
        {
            std::vector<Eigen::VectorXd> aSplit;
            self.computeSystemsDynamics(endTime,
                                        convertFromPython<std::vector<Eigen::VectorXd>>(qSplitPy),
                                        convertFromPython<std::vector<Eigen::VectorXd>>(vSplitPy),
                                        aSplit);
            return aSplit;
        }

        static void registerImpulseForce(EngineMultiRobot & self,
                                         const std::string & systemName,
                                         const std::string & frameName,
                                         double t,
                                         double dt,
                                         const Vector6d & force)
        {
            return self.registerImpulseForce(
                systemName, frameName, t, dt, pinocchio::Force{force});
        }

        static void registerProfileForce(EngineMultiRobot & self,
                                         const std::string & systemName,
                                         const std::string & frameName,
                                         const bp::object & forceFuncPy,
                                         double updatePeriod)
        {
            TimeStateFunPyWrapper<pinocchio::Force> forceFunc(forceFuncPy);
            return self.registerProfileForce(systemName, frameName, forceFunc, updatePeriod);
        }

        static bp::dict formatLogData(const LogData & logData)
        {
            // Early return if empty
            if (logData.constants.empty())
            {
                return {};
            }

            // Initialize buffers
            bp::dict variables, constants;

            // Temporary contiguous storage for variables
            VectorX<int64_t> intVector;
            VectorX<double> floatVector;

            // Get the number of integer and float variables
            const Eigen::Index numInt = logData.integerValues.rows();
            const Eigen::Index numFloat = logData.floatValues.rows();

            // Get constants
            for (const auto & [key, value] : logData.constants)
            {
                if (endsWith(key, ".options"))
                {
                    std::vector<uint8_t> jsonStringVec(value.begin(), value.end());
                    std::shared_ptr<AbstractIODevice> device =
                        std::make_shared<MemoryDevice>(std::move(jsonStringVec));
                    GenericConfig robotOptions;
                    jsonLoad(robotOptions, device);
                    constants[key] = robotOptions;
                }
                else if (endsWith(key, ".pinocchio_model"))
                {
                    try
                    {
                        pinocchio::Model model;
                        ::jiminy::loadFromBinary<pinocchio::Model>(model, value);
                        constants[key] = model;
                    }
                    catch (const std::exception & e)
                    {
                        THROW_ERROR(std::ios_base::failure,
                                    "Failed to load pinocchio model from log: ",
                                    e.what());
                    }
                }
                else if (endsWith(key, ".visual_model") || endsWith(key, ".collision_model"))
                {
                    try
                    {
                        pinocchio::GeometryModel geometryModel;
                        ::jiminy::loadFromBinary<pinocchio::GeometryModel>(geometryModel, value);
                        constants[key] = geometryModel;
                    }
                    catch (const std::exception & e)
                    {
                        THROW_ERROR(std::ios_base::failure,
                                    "Failed to load collision and/or visual model from log: ",
                                    e.what());
                    }
                }
                else if (endsWith(key, ".mesh_package_dirs"))
                {
                    bp::list meshPackageDirs;
                    std::stringstream ss(value);
                    std::string item;
                    while (getline(ss, item, ';'))
                    {
                        meshPackageDirs.append(item);
                    }
                    constants[key] = meshPackageDirs;
                }
                else if (key == NUM_INTS || key == NUM_FLOATS)
                {
                    constants[key] = std::stol(value);
                }
                else if (key == TIME_UNIT)
                {
                    constants[key] = std::stod(value);
                }
                else
                {
                    constants[key] = value;  // convertToPython(value, false);
                }
            }

            // Get Global.Time
            bp::object timePy;
            if (logData.times.size() > 0)
            {
                const Eigen::VectorXd timeBuffer = logData.times.cast<double>() * logData.timeUnit;
                timePy = convertToPython(timeBuffer, true);
                PyArray_CLEARFLAGS(reinterpret_cast<PyArrayObject *>(timePy.ptr()),
                                   NPY_ARRAY_WRITEABLE);
            }
            else
            {
                npy_intp dims[1] = {npy_intp(0)};
                timePy = bp::object(bp::handle<>(PyArray_SimpleNew(1, dims, NPY_FLOAT64)));
            }
            variables[logData.variableNames[0]] = timePy;

            // Get integers
            if (numInt > 0)
            {
                for (Eigen::Index i = 0; i < numInt; ++i)
                {
                    const std::string & header_i = logData.variableNames[i + 1];
                    intVector = logData.integerValues.row(i);
                    bp::object array = convertToPython(intVector, true);
                    PyArray_CLEARFLAGS(reinterpret_cast<PyArrayObject *>(array.ptr()),
                                       NPY_ARRAY_WRITEABLE);
                    variables[header_i] = array;
                }
            }
            else
            {
                npy_intp dims[1] = {npy_intp(0)};
                for (Eigen::Index i = 0; i < numInt; ++i)
                {
                    const std::string & header_i = logData.variableNames[i + 1];
                    variables[header_i] =
                        bp::object(bp::handle<>(PyArray_SimpleNew(1, dims, NPY_INT64)));
                }
            }

            // Get floats
            if (numFloat > 0)
            {
                for (Eigen::Index i = 0; i < numFloat; ++i)
                {
                    const std::string & header_i = logData.variableNames[i + 1 + numInt];
                    floatVector = logData.floatValues.row(i);
                    bp::object array = convertToPython(floatVector, true);
                    PyArray_CLEARFLAGS(reinterpret_cast<PyArrayObject *>(array.ptr()),
                                       NPY_ARRAY_WRITEABLE);
                    variables[header_i] = array;
                }
            }
            else
            {
                npy_intp dims[1] = {npy_intp(0)};
                for (Eigen::Index i = 0; i < numFloat; ++i)
                {
                    const std::string & header_i = logData.variableNames[i + 1 + numInt];
                    variables[header_i] =
                        bp::object(bp::handle<>(PyArray_SimpleNew(1, dims, NPY_FLOAT64)));
                }
            }

            // Return aggregated data
            bp::dict logDataPy;
            logDataPy["constants"] = constants;
            logDataPy["variables"] = variables;
            return logDataPy;
        }

        static bp::dict getLog(EngineMultiRobot & self)
        {
            /* It is impossible to use static boost::python variables. Indeed, the global/static
               destructor is called after finalization of Python runtime, the later being required
               to call the destructor of Python objects. The easiest way to circumvent this
               limitation is to avoid them entirely. Alternatively, one can increase the reference
               counter to avoid calling the destructor at exit. Here some reference for more
               details and more robust solutions:
               - https://stackoverflow.com/a/24156996/4820605
               - https://stackoverflow.com/a/31444751/4820605 */

            static std::unique_ptr<bp::dict> logDataPy(nullptr);
            static std::shared_ptr<const LogData> logDataOld;
            std::shared_ptr<const LogData> logData = self.getLog();
            if (logData.use_count() == 2)
            {
                // Decrement the reference counter of old Python log data
                if (logDataPy)
                {
                    bp::decref(logDataPy->ptr());
                }

                /* The shared pointer is new, because otherwise the use count should larger than 2.
                   Indeed, both the engine and this method holds a single reference at this point.
                   If it was old, this method would holds at least 2 references, one for the old
                   reference and one for the new. */
                logDataPy = std::make_unique<bp::dict>(formatLogData(*logData));

                /* Reference counter must be incremented to avoid calling deleter by Boost Python
                   after runtime finalization. */
                bp::incref(logDataPy->ptr());

                // Update log data backup
                logDataOld = logData;
            }

            // Avoid potential null pointer dereference although it should never happen in practice
            if (logDataPy)
            {
                return *logDataPy;
            }
            return {};
        }

        static bp::dict readLog(const std::string & filename, const bp::object & formatPy)
        {
            std::string format;
            if (!formatPy.is_none())
            {
                format = convertFromPython<std::string>(formatPy);
            }
            else
            {
                const std::array<std::string, 3> extHDF5{{".h5", ".hdf5", ".tlmc"}};
                if (endsWith(filename, ".data"))
                {
                    format = "binary";
                }
                else if (std::any_of(extHDF5.begin(),
                                     extHDF5.end(),
                                     std::bind(endsWith, filename, std::placeholders::_1)))
                {
                    format = "hdf5";
                }
                else
                {
                    THROW_ERROR(std::runtime_error,
                                "Impossible to determine the file format "
                                "automatically. Please specify it manually.");
                }
            }
            const LogData logData = EngineMultiRobot::readLog(filename, format);
            return formatLogData(logData);
        }

        static void setOptions(EngineMultiRobot & self, const bp::dict & configPy)
        {
            GenericConfig config = self.getOptions();
            convertFromPython(configPy, config);
            return self.setOptions(config);
        }

        static void expose()
        {
            // clang-format off
            bp::class_<EngineMultiRobot,
                       std::shared_ptr<EngineMultiRobot>,
                       boost::noncopyable>("EngineMultiRobot")
                .def(PyEngineMultiRobotVisitor());
            // clang-format on
        }
    };

    BOOST_PYTHON_VISITOR_EXPOSE(EngineMultiRobot)

    // ***************************** PyEngineVisitor ***********************************

    struct PyEngineVisitor : public bp::def_visitor<PyEngineVisitor>
    {
    public:
        /// \brief Expose C++ API through the visitor.
        template<class PyClass>
        void visit(PyClass & cl) const
        {
            // clang-format off
            cl
                .def("add_system", &Engine::addSystem)
                .def("remove_system", &Engine::removeSystem,
                                      (bp::arg("self"), "system_name"))

                .def("initialize", &PyEngineVisitor::initialize,
                                   (bp::arg("self"), "robot",
                                    bp::arg("controller") = std::shared_ptr<AbstractController>(),
                                    bp::arg("callback_function") = bp::object()))
                .def("set_controller",
                     static_cast<
                         void (Engine::*)(std::shared_ptr<AbstractController>)
                     >(&Engine::setController),
                     (bp::arg("self"), "controller"))

                .def("start",
                    &PyEngineVisitor::start,
                    (bp::arg("self"), "q_init", "v_init",
                     bp::arg("a_init") = bp::object(),
                     bp::arg("is_state_theoretical") = false))
                .def("simulate",
                    &PyEngineVisitor::simulate,
                    (bp::arg("self"), "t_end", "q_init", "v_init",
                     bp::arg("a_init") = bp::object(),
                     bp::arg("is_state_theoretical") = false))

                .def("register_impulse_force", &PyEngineVisitor::registerImpulseForce,
                                               (bp::arg("self"), "frame_name", "t", "dt", "force"))
                .ADD_PROPERTY_GET_WITH_POLICY("impulse_forces",
                                              static_cast<
                                                  const ImpulseForceVector & (Engine::*)(void) const
                                              >(&Engine::getImpulseForces),
                                              bp::return_value_policy<result_converter<false>>())

                .def("register_profile_force", &PyEngineVisitor::registerProfileForce,
                                               (bp::arg("self"), "frame_name", "force_function",
                                                bp::arg("update_period") = 0.0))
                .ADD_PROPERTY_GET_WITH_POLICY("profile_forces",
                                              static_cast<
                                                  const ProfileForceVector & (Engine::*)(void) const
                                              >(&Engine::getProfileForces),
                                              bp::return_value_policy<result_converter<false>>())

                .def("register_coupling_force", &PyEngineVisitor::registerCouplingForce,
                                                (bp::arg("self"), "frame_name_1", "frame_name_2", "force_function"))
                .def("register_viscoelastic_coupling_force",
                     static_cast<
                         void (Engine::*)(
                             const std::string &,
                             const std::string &,
                             const Vector6d &,
                             const Vector6d &,
                             double)
                     >(&Engine::registerViscoelasticCouplingForce),
                     (bp::arg("self"), "frame_name_1", "frame_name_2", "stiffness", "damping", bp::arg("alpha") = 0.5))
                .def("register_viscoelastic_directional_coupling_force",
                     static_cast<
                         void (Engine::*)(
                             const std::string &,
                             const std::string &,
                             double,
                             double,
                             double)
                     >(&Engine::registerViscoelasticDirectionalCouplingForce),
                     (bp::arg("self"), "frame_name_1", "frame_name_2", "stiffness", "damping",
		              bp::arg("rest_length") = 0.0))

                .ADD_PROPERTY_GET_WITH_POLICY("is_initialized",
                                              &Engine::getIsInitialized,
                                              bp::return_value_policy<bp::return_by_value>())
                .ADD_PROPERTY_GET_WITH_POLICY("system",
                                              &Engine::getSystem,
                                              bp::return_value_policy<result_converter<false>>())
                .ADD_PROPERTY_GET("robot",  &Engine::getRobot)
                .ADD_PROPERTY_GET("controller", &Engine::getController)
                .ADD_PROPERTY_GET_WITH_POLICY("stepper_state",
                                              &Engine::getStepperState,
                                              bp::return_value_policy<result_converter<false>>())
                .ADD_PROPERTY_GET_WITH_POLICY("system_state",
                                              &Engine::getSystemState,
                                              bp::return_value_policy<result_converter<false>>())
                ;
            // clang-format on
        }

        static void initialize(Engine & self,
                               const std::shared_ptr<Robot> & robot,
                               const std::shared_ptr<AbstractController> & controller,
                               const bp::object & callbackPy)
        {
            if (callbackPy.is_none())
            {
                AbortSimulationFunction callback = [](double /* t */,
                                                      const Eigen::VectorXd & /* q */,
                                                      const Eigen::VectorXd & /* v */) -> bool
                {
                    return true;
                };
                if (controller)
                {
                    return self.initialize(robot, controller, callback);
                }
                return self.initialize(robot, callback);
            }
            else
            {
                TimeStateFunPyWrapper<bool> callback(callbackPy);
                if (controller)
                {
                    return self.initialize(robot, controller, callback);
                }
                return self.initialize(robot, callback);
            }
        }

        static void registerImpulseForce(Engine & self,
                                         const std::string & frameName,
                                         double t,
                                         double dt,
                                         const Vector6d & force)
        {
            return self.registerImpulseForce(frameName, t, dt, pinocchio::Force{force});
        }

        static void registerProfileForce(Engine & self,
                                         const std::string & frameName,
                                         const bp::object & forceFuncPy,
                                         double updatePeriod)
        {
            TimeStateFunPyWrapper<pinocchio::Force> forceFunc(forceFuncPy);
            return self.registerProfileForce(frameName, forceFunc, updatePeriod);
        }

        static void registerCouplingForce(Engine & self,
                                          const std::string & frameName1,
                                          const std::string & frameName2,
                                          const bp::object & forceFuncPy)
        {
            TimeStateFunPyWrapper<pinocchio::Force> forceFunc(forceFuncPy);
            return self.registerCouplingForce(frameName1, frameName2, forceFunc);
        }

        static void start(Engine & self,
                          const Eigen::VectorXd & qInit,
                          const Eigen::VectorXd & vInit,
                          const bp::object & aInitPy,
                          bool isStateTheoretical)
        {
            std::optional<Eigen::VectorXd> aInit = std::nullopt;
            if (!aInitPy.is_none())
            {
                aInit.emplace(convertFromPython<Eigen::VectorXd>(aInitPy));
            }
            return self.start(qInit, vInit, aInit, isStateTheoretical);
        }

        static void simulate(Engine & self,
                             double endTime,
                             const Eigen::VectorXd & qInit,
                             const Eigen::VectorXd & vInit,
                             const bp::object & aInitPy,
                             bool isStateTheoretical)
        {
            std::optional<Eigen::VectorXd> aInit = std::nullopt;
            if (!aInitPy.is_none())
            {
                aInit.emplace(convertFromPython<Eigen::VectorXd>(aInitPy));
            }
            return self.simulate(endTime, qInit, vInit, aInit, isStateTheoretical);
        }

        static void expose()
        {
            // clang-format off
            bp::class_<Engine, bp::bases<EngineMultiRobot>,
                       std::shared_ptr<Engine>,
                       boost::noncopyable>("Engine")
                .def(PyEngineVisitor());
            // clang-format on
        }
    };

    BOOST_PYTHON_VISITOR_EXPOSE(Engine)
}
