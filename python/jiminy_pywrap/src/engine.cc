#include "jiminy/core/exceptions.h"
#include "jiminy/core/io/serialization.h"
#include "jiminy/core/control/abstract_controller.h"
#include "jiminy/core/robot/robot.h"
#include "jiminy/core/engine/engine.h"
#include "jiminy/core/engine/engine_multi_robot.h"
#include "jiminy/core/telemetry/telemetry_data.h"
#include "jiminy/core/telemetry/telemetry_recorder.h"
#include "jiminy/core/utilities/json.h"
#include "jiminy/core/utilities/helpers.h"

#include <boost/optional.hpp>

#include "pinocchio/bindings/python/fwd.hpp"
#include <boost/python/raw_function.hpp>

#include "jiminy/python/utilities.h"
#include "jiminy/python/functors.h"
#include "jiminy/python/engine.h"


namespace jiminy::python
{
    namespace bp = boost::python;

    // ************* Expose impulse, profile, and coupling force registers **************

    static bp::object forceProfileWrapper(const ForceProfile & self)
    {
        bp::object func = bp::make_function(self.forceFct,
                                            bp::return_value_policy<bp::return_by_value>(),
                                            (bp::arg("t"), "q", "v"),
                                            functionToMLP(self.forceFct));
        setFunctionWrapperModule<ForceProfile>(func);
        return func;
    }

    static bp::object forceCouplingWrapper(const ForceCoupling & self)
    {
        bp::object func = bp::make_function(self.forceFct,
                                            bp::return_value_policy<bp::return_by_value>(),
                                            (bp::arg("t"), "q_1", "v_1", "q_2", "v_2"),
                                            functionToMLP(self.forceFct));
        setFunctionWrapperModule<ForceCoupling>(func);
        return func;
    }

    void exposeForces()
    {
        // clang-format off
        bp::class_<ForceProfile,
                   std::shared_ptr<ForceProfile>,
                   boost::noncopyable>("ForceProfile", bp::no_init)
            .DEF_READONLY("frame_name", &ForceProfile::frameName)
            .DEF_READONLY("frame_idx", &ForceProfile::frameIdx)
            .DEF_READONLY("update_period", &ForceProfile::updatePeriod)
            .DEF_READONLY("force_prev", &ForceProfile::forcePrev)
            .ADD_PROPERTY_GET("force_func", forceProfileWrapper);

        /* Note that it will be impossible to slice the vector if `boost::noncopyable` is set for
           the stl container, or if the value type contained itself. In such a case, it raises a
           runtime error rather than a compile-time error. */
        bp::class_<ForceProfileRegister>("ForceProfileVector", bp::no_init)
            .def(vector_indexing_suite_no_contains<ForceProfileRegister>());

        bp::class_<ForceImpulse,
                   std::shared_ptr<ForceImpulse>,
                   boost::noncopyable>("ForceImpulse", bp::no_init)
            .DEF_READONLY("frame_name", &ForceImpulse::frameName)
            .DEF_READONLY("frame_idx", &ForceImpulse::frameIdx)
            .DEF_READONLY("t", &ForceImpulse::t)
            .DEF_READONLY("dt", &ForceImpulse::dt)
            .DEF_READONLY("F", &ForceImpulse::F);

        bp::class_<ForceImpulseRegister,
                   boost::noncopyable>("ForceImpulseVector", bp::no_init)
            .def(vector_indexing_suite_no_contains<ForceImpulseRegister>());

        bp::class_<ForceCoupling,
                   std::shared_ptr<ForceCoupling>,
                   boost::noncopyable>("ForceCoupling", bp::no_init)
            .DEF_READONLY("system_name_1", &ForceCoupling::systemName1)
            .DEF_READONLY("system_idx_1", &ForceCoupling::systemIdx1)
            .DEF_READONLY("system_name_2", &ForceCoupling::systemName2)
            .DEF_READONLY("system_idx_2", &ForceCoupling::systemIdx2)
            .ADD_PROPERTY_GET("force_func", forceCouplingWrapper);

        bp::class_<ForceCouplingRegister,
                   boost::noncopyable>("ForceCouplingVector", bp::no_init)
            .def(vector_indexing_suite_no_contains<ForceCouplingRegister>());
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
                .DEF_READONLY("q", &systemState_t::q)
                .DEF_READONLY("v", &systemState_t::v)
                .DEF_READONLY("a", &systemState_t::a)
                .DEF_READONLY("command", &systemState_t::command)
                .DEF_READONLY("u", &systemState_t::u)
                .DEF_READONLY("u_motor", &systemState_t::uMotor)
                .DEF_READONLY("u_internal", &systemState_t::uInternal)
                .DEF_READONLY("u_custom", &systemState_t::uCustom)
                .DEF_READONLY("f_external", &systemState_t::fExternal)
                .def("__repr__", &PySystemStateVisitor::repr)
                ;
            // clang-format on
        }

        static std::string repr(systemState_t & self)
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
            bp::class_<systemState_t,
                       std::shared_ptr<systemState_t>,
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
                .DEF_READONLY("name", &systemHolder_t::name)
                .DEF_READONLY("robot", &systemHolder_t::robot)
                .DEF_READONLY("controller", &systemHolder_t::controller)
                .DEF_READONLY("callbackFct", &systemHolder_t::callbackFct)
                ;
            // clang-format on
        }

        static void expose()
        {
            // clang-format off
            bp::class_<systemHolder_t>("system", bp::no_init)
                .def(PySystemVisitor());

            bp::class_<std::vector<systemHolder_t>>("systemVector", bp::no_init)
                .def(vector_indexing_suite_no_contains<std::vector<systemHolder_t>>());
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
                        void (EngineMultiRobot::*)(bool_t, bool_t)
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

                .def("register_force_impulse", &PyEngineMultiRobotVisitor::registerForceImpulse,
                                               (bp::arg("self"), "system_name",
                                                "frame_name", "t", "dt", "F"))
                .def("remove_forces_impulse",
                    static_cast<
                        hresult_t (EngineMultiRobot::*)(const std::string &)
                    >(&EngineMultiRobot::removeForcesImpulse),
                    (bp::arg("self"), "system_name"))
                .def("remove_forces_impulse",
                    static_cast<
                        hresult_t (EngineMultiRobot::*)(void)
                    >(&EngineMultiRobot::removeForcesImpulse),
                    (bp::arg("self")))
                .ADD_PROPERTY_GET("forces_impulse", &PyEngineMultiRobotVisitor::getForcesImpulse)

                .def("register_force_profile", &PyEngineMultiRobotVisitor::registerForceProfile,
                                               (bp::arg("self"), "system_name",
                                                "frame_name", "force_function",
                                                bp::arg("update_period") = 0.0))
                .def("remove_forces_profile",
                    static_cast<
                        hresult_t (EngineMultiRobot::*)(const std::string &)
                    >(&EngineMultiRobot::removeForcesProfile),
                    (bp::arg("self"), "system_name"))
                .def("remove_forces_profile",
                    static_cast<
                        hresult_t (EngineMultiRobot::*)(void)
                    >(&EngineMultiRobot::removeForcesProfile),
                    (bp::arg("self")))
                .ADD_PROPERTY_GET("forces_profile", &PyEngineMultiRobotVisitor::getForcesProfile)

                .def("register_force_coupling", &PyEngineMultiRobotVisitor::registerForceCoupling,
                                                (bp::arg("self"),
                                                 "system_name_1", "system_name_2",
                                                 "frame_name_1", "frame_name_2",
                                                 "force_function"))
                .def("register_viscoelastic_force_coupling",
                    static_cast<
                        hresult_t (EngineMultiRobot::*)(
                            const std::string &,
                            const std::string &,
                            const std::string &,
                            const std::string &,
                            const Vector6d &,
                            const Vector6d &,
                            float64_t)
                    >(&EngineMultiRobot::registerViscoelasticForceCoupling),
                    (bp::arg("self"), "system_name_1", "system_name_2",
                     "frame_name_1", "frame_name_2", "stiffness", "damping", bp::arg("alpha") = 0.5))
                .def("register_viscoelastic_force_coupling",
                    static_cast<
                        hresult_t (EngineMultiRobot::*)(
                            const std::string &,
                            const std::string &,
                            const std::string &,
                            const Vector6d &,
                            const Vector6d &,
                            float64_t)
                    >(&EngineMultiRobot::registerViscoelasticForceCoupling),
                    (bp::arg("self"), "system_name", "frame_name_1", "frame_name_2",
                     "stiffness", "damping", bp::arg("alpha") = 0.5))
                .def("register_viscoelastic_directional_force_coupling",
                    static_cast<
                        hresult_t (EngineMultiRobot::*)(
                            const std::string &,
                            const std::string &,
                            const std::string &,
                            const std::string &,
                            float64_t,
                            float64_t,
                            float64_t)
                    >(&EngineMultiRobot::registerViscoelasticDirectionalForceCoupling),
                    (bp::arg("self"), "system_name_1", "system_name_2", "frame_name_1", "frame_name_2",
                     "stiffness", "damping", bp::arg("rest_length") = 0.0))
                .def("register_viscoelastic_directional_force_coupling",
                    static_cast<
                        hresult_t (EngineMultiRobot::*)(
                            const std::string &,
                            const std::string &,
                            const std::string &,
                            float64_t,
                            float64_t,
                            float64_t)
                    >(&EngineMultiRobot::registerViscoelasticDirectionalForceCoupling),
                    (bp::arg("self"), "system_name", "frame_name_1", "frame_name_2",
                     "stiffness", "damping", bp::arg("rest_length") = 0.0))
                .def("remove_forces_coupling",
                    static_cast<
                        hresult_t (EngineMultiRobot::*)(const std::string &, const std::string &)
                    >(&EngineMultiRobot::removeForcesCoupling),
                    (bp::arg("self"), "system_name_1", "system_name_2"))
                .def("remove_forces_coupling",
                    static_cast<
                        hresult_t (EngineMultiRobot::*)(const std::string &)
                    >(&EngineMultiRobot::removeForcesCoupling),
                    (bp::arg("self"), "system_name"))
                .def("remove_forces_coupling",
                    static_cast<
                        hresult_t (EngineMultiRobot::*)(void)
                    >(&EngineMultiRobot::removeForcesCoupling),
                    (bp::arg("self")))
                .ADD_PROPERTY_GET_WITH_POLICY("forces_coupling",
                                              &EngineMultiRobot::getForcesCoupling,
                                              bp::return_value_policy<result_converter<false>>())

                .def("remove_all_forces", &EngineMultiRobot::removeAllForces)

                .def("set_options", &PyEngineMultiRobotVisitor::setOptions)
                .def("get_options", &EngineMultiRobot::getOptions)

                .DEF_READONLY("systems", &EngineMultiRobot::systems_)
                .ADD_PROPERTY_GET_WITH_POLICY("systems_names",
                                              &EngineMultiRobot::getSystemsNames,
                                              bp::return_value_policy<result_converter<true>>())
                .ADD_PROPERTY_GET("systems_states", &PyEngineMultiRobotVisitor::getSystemState)
                .ADD_PROPERTY_GET_WITH_POLICY("stepper_state",
                                              &EngineMultiRobot::getStepperState,
                                              bp::return_value_policy<result_converter<false>>())
                .ADD_PROPERTY_GET_WITH_POLICY("is_simulation_running",
                                              &EngineMultiRobot::getIsSimulationRunning,
                                              bp::return_value_policy<result_converter<false>>())
                .add_static_property("simulation_duration_max", &EngineMultiRobot::getMaxSimulationDuration)
                .add_static_property("telemetry_time_unit", &EngineMultiRobot::getTelemetryTimeUnit)
                ;
            // clang-format on
        }

        static hresult_t addSystem(EngineMultiRobot & self,
                                   const std::string & systemName,
                                   const std::shared_ptr<Robot> & robot,
                                   const bp::object & controllerPy,
                                   const bp::object & callbackPy)
        {
            CallbackFunctor callbackFct;
            if (callbackPy.is_none())
            {
                callbackFct = [](float64_t /* t */,
                                 const Eigen::VectorXd & /* q */,
                                 const Eigen::VectorXd & /* v */) -> bool_t
                {
                    return true;
                };
            }
            else
            {
                callbackFct = TimeStateFctPyWrapper<bool_t>(callbackPy);
            }
            if (!controllerPy.is_none())
            {
                const std::shared_ptr<AbstractController> controller =
                    bp::extract<std::shared_ptr<AbstractController>>(controllerPy);
                return self.addSystem(systemName, robot, controller, std::move(callbackFct));
            }
            return self.addSystem(systemName, robot, std::move(callbackFct));
        }

        static systemHolder_t & getSystem(EngineMultiRobot & self, const std::string & systemName)
        {
            systemHolder_t * system;
            // It makes sure that system is always assigned to a well-defined systemHolder_t
            self.getSystem(systemName, system);
            return *system;
        }

        static bp::dict getForcesImpulse(EngineMultiRobot & self)
        {
            bp::dict forceImpulsesPy;
            for (const auto & systemName : self.getSystemsNames())
            {
                const ForceImpulseRegister * forcesImpulse;
                self.getForcesImpulse(systemName, forcesImpulse);
                forceImpulsesPy[systemName] = convertToPython(forcesImpulse, false);
            }
            return forceImpulsesPy;
        }

        static bp::dict getForcesProfile(EngineMultiRobot & self)
        {
            bp::dict forcesProfilesPy;
            for (const auto & systemName : self.getSystemsNames())
            {
                const ForceProfileRegister * forcesProfile;
                self.getForcesProfile(systemName, forcesProfile);
                forcesProfilesPy[systemName] = convertToPython(forcesProfile, false);
            }
            return forcesProfilesPy;
        }

        static bp::dict getSystemState(EngineMultiRobot & self)
        {
            bp::dict systemStates;
            for (const std::string & systemName : self.getSystemsNames())
            {
                /* Cannot fail, but `getSystemState` is making sure that systemState is assigned to
                   a well-defined object anyway. */
                const systemState_t * systemState;
                self.getSystemState(systemName, systemState);
                systemStates[systemName] = convertToPython(systemState, false);
            }
            return systemStates;
        }

        static hresult_t registerForceCoupling(EngineMultiRobot & self,
                                               const std::string & systemName1,
                                               const std::string & systemName2,
                                               const std::string & frameName1,
                                               const std::string & frameName2,
                                               const bp::object & forcePy)
        {
            TimeBistateFctPyWrapper<pinocchio::Force> forceFct(forcePy);
            return self.registerForceCoupling(
                systemName1, systemName2, frameName1, frameName2, std::move(forceFct));
        }

        static hresult_t start(EngineMultiRobot & self,
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

        static hresult_t step(EngineMultiRobot & self, float64_t dtDesired)
        {
            // Only way to handle C++ default values that are not accessible in Python
            return self.step(dtDesired);
        }

        static hresult_t simulate(EngineMultiRobot & self,
                                  float64_t endTime,
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
                                                                   float64_t endTime,
                                                                   const bp::list & qSplitPy,
                                                                   const bp::list & vSplitPy)
        {
            std::vector<Eigen::VectorXd> aSplit;
            self.computeSystemsDynamics(endTime,
                                        convertFromPython<std::vector<Eigen::VectorXd>>(qSplitPy),
                                        convertFromPython<std::vector<Eigen::VectorXd>>(vSplitPy),
                                        aSplit);
            return aSplit;
        }

        static hresult_t registerForceImpulse(EngineMultiRobot & self,
                                              const std::string & systemName,
                                              const std::string & frameName,
                                              float64_t t,
                                              float64_t dt,
                                              const Vector6d & F)
        {
            return self.registerForceImpulse(systemName, frameName, t, dt, pinocchio::Force(F));
        }

        static hresult_t registerForceProfile(EngineMultiRobot & self,
                                              const std::string & systemName,
                                              const std::string & frameName,
                                              const bp::object & forcePy,
                                              float64_t updatePeriod)
        {
            TimeStateFctPyWrapper<pinocchio::Force> forceFct(forcePy);
            return self.registerForceProfile(
                systemName, frameName, std::move(forceFct), updatePeriod);
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
            Eigen::Matrix<int64_t, Eigen::Dynamic, 1> intVector;
            Eigen::Matrix<float64_t, Eigen::Dynamic, 1> floatVector;

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
                        PRINT_ERROR("Failed to load pinocchio model from log: ", e.what());
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
                        PRINT_ERROR("Failed to load collision and/or visual model from log: ",
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
                else if (key == NUM_INTS.substr(0, key.size()) ||
                         key == NUM_FLOATS.substr(0, key.size()))
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
                const Eigen::VectorXd timeBuffer =
                    logData.times.cast<float64_t>() * logData.timeUnit;
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
            std::shared_ptr<const LogData> logData;
            self.getLog(logData);
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
                    throw std::runtime_error("Impossible to determine the file format "
                                             "automatically. Please specify it manually.");
                }
            }
            LogData logData;
            hresult_t returnCode = EngineMultiRobot::readLog(filename, format, logData);
            if (returnCode == hresult_t::SUCCESS)
            {
                return formatLogData(logData);
            }
            return {};
        }

        static hresult_t setOptions(EngineMultiRobot & self, const bp::dict & configPy)
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
                .def("add_system", raw_function(&PyEngineVisitor::addSystem, 1))
                .def("remove_system", &Engine::removeSystem,
                                      (bp::arg("self"), "system_name"))

                .def("initialize", &PyEngineVisitor::initialize,
                                   (bp::arg("self"), "robot",
                                    bp::arg("controller") = std::shared_ptr<AbstractController>(),
                                    bp::arg("callback_function") = bp::object()))
                .def("set_controller", static_cast<
                        hresult_t (Engine::*)(std::shared_ptr<AbstractController>)
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

                .def("register_force_impulse", &PyEngineVisitor::registerForceImpulse,
                                               (bp::arg("self"), "frame_name", "t", "dt", "F"))
                .ADD_PROPERTY_GET_WITH_POLICY("forces_impulse",
                                              static_cast<
                                                  const ForceImpulseRegister & (Engine::*)(void) const
                                              >(&Engine::getForcesImpulse),
                                              bp::return_value_policy<result_converter<false>>())

                .def("register_force_profile", &PyEngineVisitor::registerForceProfile,
                                               (bp::arg("self"), "frame_name", "force_function",
                                                bp::arg("update_period") = 0.0))
                .ADD_PROPERTY_GET_WITH_POLICY("forces_profile",
                                              static_cast<
                                                  const ForceProfileRegister & (Engine::*)(void) const
                                              >(&Engine::getForcesProfile),
                                              bp::return_value_policy<result_converter<false>>())

                .def("register_force_coupling", &PyEngineVisitor::registerForceCoupling,
                                                (bp::arg("self"), "frame_name_1", "frame_name_2", "force_function"))
                .def("register_viscoelastic_force_coupling",
                    static_cast<
                        hresult_t (Engine::*)(
                            const std::string &,
                            const std::string &,
                            const Vector6d &,
                            const Vector6d &,
                            float64_t)
                    >(&Engine::registerViscoelasticForceCoupling),
                    (bp::arg("self"), "frame_name_1", "frame_name_2", "stiffness", "damping", bp::arg("alpha") = 0.5))
                .def("register_viscoelastic_directional_force_coupling",
                    static_cast<
                        hresult_t (Engine::*)(
                            const std::string &,
                            const std::string &,
                            float64_t,
                            float64_t,
                            float64_t)
                    >(&Engine::registerViscoelasticDirectionalForceCoupling),
                    (bp::arg("self"), "frame_name_1", "frame_name_2", "stiffness", "damping",
		             bp::arg("rest_length") = 0.0))

                .ADD_PROPERTY_GET_WITH_POLICY("is_initialized",
                                              &Engine::getIsInitialized,
                                              bp::return_value_policy<bp::return_by_value>())
                .ADD_PROPERTY_GET_WITH_POLICY("system",
                                              &PyEngineVisitor::getSystem,
                                              bp::return_value_policy<result_converter<false>>())
                .ADD_PROPERTY_GET("robot", &PyEngineVisitor::getRobot)
                .ADD_PROPERTY_GET("controller", &PyEngineVisitor::getController)
                .ADD_PROPERTY_GET_WITH_POLICY("stepper_state",
                                              &Engine::getStepperState,
                                              bp::return_value_policy<result_converter<false>>())
                .ADD_PROPERTY_GET_WITH_POLICY("system_state",
                                              &PyEngineVisitor::getSystemState,
                                              bp::return_value_policy<result_converter<false>>())
                ;
            // clang-format on
        }

        static hresult_t addSystem(bp::tuple /* args */, bp::dict /* kwargs */)
        {
            // Hide all EngineMultiRobot `addSystem` overloads at once
            return Engine().addSystem(
                "", std::shared_ptr<Robot>(), std::shared_ptr<AbstractController>());
        }

        static hresult_t initialize(Engine & self,
                                    const std::shared_ptr<Robot> & robot,
                                    const std::shared_ptr<AbstractController> & controller,
                                    const bp::object & callbackPy)
        {
            if (callbackPy.is_none())
            {
                CallbackFunctor callbackFct = [](float64_t /* t */,
                                                 const Eigen::VectorXd & /* q */,
                                                 const Eigen::VectorXd & /* v */) -> bool_t
                {
                    return true;
                };
                if (controller)
                {
                    return self.initialize(robot, controller, std::move(callbackFct));
                }
                return self.initialize(robot, std::move(callbackFct));
            }
            else
            {
                TimeStateFctPyWrapper<bool_t> callbackFct(callbackPy);
                if (controller)
                {
                    return self.initialize(robot, controller, std::move(callbackFct));
                }
                return self.initialize(robot, std::move(callbackFct));
            }
        }

        static hresult_t registerForceImpulse(Engine & self,
                                              const std::string & frameName,
                                              float64_t t,
                                              float64_t dt,
                                              const Vector6d & F)
        {
            return self.registerForceImpulse(frameName, t, dt, pinocchio::Force(F));
        }

        static hresult_t registerForceProfile(Engine & self,
                                              const std::string & frameName,
                                              const bp::object & forcePy,
                                              float64_t updatePeriod)
        {
            TimeStateFctPyWrapper<pinocchio::Force> forceFct(forcePy);
            return self.registerForceProfile(frameName, std::move(forceFct), updatePeriod);
        }

        static hresult_t registerForceCoupling(Engine & self,
                                               const std::string & frameName1,
                                               const std::string & frameName2,
                                               const bp::object & forcePy)
        {
            TimeStateFctPyWrapper<pinocchio::Force> forceFct(forcePy);
            return self.registerForceCoupling(frameName1, frameName2, std::move(forceFct));
        }

        static systemHolder_t & getSystem(Engine & self)
        {
            systemHolder_t * system;
            self.getSystem(system);
            return *system;
        }

        static std::shared_ptr<Robot> getRobot(Engine & self)
        {
            std::shared_ptr<Robot> robot;
            self.getRobot(robot);
            return robot;
        }

        static std::shared_ptr<AbstractController> getController(Engine & self)
        {
            std::shared_ptr<AbstractController> controller;
            self.getController(controller);
            return controller;
        }

        static const systemState_t & getSystemState(Engine & self)
        {
            const systemState_t * systemState;
            // It makes sure that systemState is always assigned to a well-defined systemState_t
            self.getSystemState(systemState);
            return *systemState;
        }

        static hresult_t start(Engine & self,
                               const Eigen::VectorXd & qInit,
                               const Eigen::VectorXd & vInit,
                               const bp::object & aInitPy,
                               bool_t isStateTheoretical)
        {
            std::optional<Eigen::VectorXd> aInit = std::nullopt;
            if (!aInitPy.is_none())
            {
                aInit.emplace(convertFromPython<Eigen::VectorXd>(aInitPy));
            }
            return self.start(qInit, vInit, aInit, isStateTheoretical);
        }

        static hresult_t simulate(Engine & self,
                                  float64_t endTime,
                                  const Eigen::VectorXd & qInit,
                                  const Eigen::VectorXd & vInit,
                                  const bp::object & aInitPy,
                                  bool_t isStateTheoretical)
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
