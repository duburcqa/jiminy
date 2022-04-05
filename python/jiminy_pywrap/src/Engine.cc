#include "jiminy/core/control/AbstractController.h"
#include "jiminy/core/robot/Robot.h"
#include "jiminy/core/engine/Engine.h"
#include "jiminy/core/engine/EngineMultiRobot.h"
#include "jiminy/core/telemetry/TelemetryData.h"
#include "jiminy/core/telemetry/TelemetryRecorder.h"

#include <boost/optional.hpp>

#include <boost/python.hpp>
#include <boost/python/raw_function.hpp>

#include "jiminy/python/Utilities.h"
#include "jiminy/python/Functors.h"
#include "jiminy/python/Engine.h"


namespace jiminy
{
namespace python
{
    namespace bp = boost::python;

    // ************* Expose impulse, profile, and coupling force registers **************

    static bp::object forceProfileWrapper(forceProfile_t const & self)
    {
        bp::object func = bp::make_function(self.forceFct,
                                            bp::return_value_policy<bp::return_by_value>(),
                                            (bp::args("t", "q", "v")),
                                            functionToMLP(self.forceFct));
        setFunctionWrapperModule<forceProfile_t>(func);
        return func;
    }

    static bp::object forceCouplingWrapper(forceCoupling_t const & self)
    {
        bp::object func = bp::make_function(self.forceFct,
                                            bp::return_value_policy<bp::return_by_value>(),
                                            (bp::args("t", "q_1", "v_1", "q_2", "v_2")),
                                            functionToMLP(self.forceFct));
        setFunctionWrapperModule<forceCoupling_t>(func);
        return func;
    }

    void exposeForces(void)
    {
        bp::class_<forceProfile_t,
                   std::shared_ptr<forceProfile_t>,
                   boost::noncopyable>("ForceProfile", bp::no_init)
            .add_property("frame_name", bp::make_getter(&forceProfile_t::frameName,
                                        bp::return_value_policy<bp::return_by_value>()))
            .add_property("frame_idx", bp::make_getter(&forceProfile_t::frameIdx,
                                       bp::return_value_policy<bp::return_by_value>()))
            .add_property("update_period", bp::make_getter(&forceProfile_t::updatePeriod,
                                           bp::return_value_policy<bp::return_by_value>()))
            .add_property("force_prev", bp::make_getter(&forceProfile_t::forcePrev,
                                        bp::return_internal_reference<>()))
            .add_property("force_func", forceProfileWrapper);

        /* Note that it will be impossible to slice the vector if `boost::noncopyable` is set
           for the stl container, or if the value type contained itself. In such a case, it
           raises a runtime error rather than a compile-time error. */
        bp::class_<forceProfileRegister_t>("ForceProfileVector", bp::no_init)
            .def(vector_indexing_suite_no_contains<forceProfileRegister_t>());

        bp::class_<forceImpulse_t,
                   std::shared_ptr<forceImpulse_t>,
                   boost::noncopyable>("ForceImpulse", bp::no_init)
            .add_property("frame_name", bp::make_getter(&forceImpulse_t::frameName,
                                        bp::return_value_policy<bp::return_by_value>()))
            .add_property("frame_idx", bp::make_getter(&forceImpulse_t::frameIdx,
                                       bp::return_value_policy<bp::return_by_value>()))
            .add_property("t", bp::make_getter(&forceImpulse_t::t,
                               bp::return_value_policy<bp::return_by_value>()))
            .add_property("dt", bp::make_getter(&forceImpulse_t::dt,
                                bp::return_value_policy<bp::return_by_value>()))
            .add_property("F", bp::make_getter(&forceImpulse_t::F,
                               bp::return_internal_reference<>()));

        bp::class_<forceImpulseRegister_t,
                   boost::noncopyable>("ForceImpulseVector", bp::no_init)
            .def(vector_indexing_suite_no_contains<forceImpulseRegister_t>());

        bp::class_<forceCoupling_t,
                   std::shared_ptr<forceCoupling_t>,
                   boost::noncopyable>("ForceProfile", bp::no_init)
            .add_property("system_name_1", bp::make_getter(&forceCoupling_t::systemName1,
                                           bp::return_value_policy<bp::return_by_value>()))
            .add_property("system_idx_2", bp::make_getter(&forceCoupling_t::systemIdx1,
                                          bp::return_value_policy<bp::return_by_value>()))
            .add_property("system_name_2", bp::make_getter(&forceCoupling_t::systemName2,
                                           bp::return_value_policy<bp::return_by_value>()))
            .add_property("system_idx_2", bp::make_getter(&forceCoupling_t::systemIdx2,
                                          bp::return_value_policy<bp::return_by_value>()))
            .add_property("force_func", forceCouplingWrapper);

        bp::class_<forceCouplingRegister_t,
                   boost::noncopyable>("ForceCouplingVector", bp::no_init)
            .def(vector_indexing_suite_no_contains<forceCouplingRegister_t>());
    }

    // ***************************** PyStepperStateVisitor ***********************************

    struct PyStepperStateVisitor
        : public bp::def_visitor<PyStepperStateVisitor>
    {
    public:
        ///////////////////////////////////////////////////////////////////////////////
        /// \brief Expose C++ API through the visitor.
        ///////////////////////////////////////////////////////////////////////////////
        template<class PyClass>
        void visit(PyClass & cl) const
        {
            cl
                .def_readonly("iter", &stepperState_t::iter)
                .def_readonly("iter_failed", &stepperState_t::iterFailed)
                .def_readonly("t", &stepperState_t::t)
                .def_readonly("dt", &stepperState_t::dt)
                .add_property("q", bp::make_getter(&stepperState_t::qSplit,
                                   bp::return_value_policy<result_converter<false> >()))
                .add_property("v", bp::make_getter(&stepperState_t::vSplit,
                                   bp::return_value_policy<result_converter<false> >()))
                .add_property("a", bp::make_getter(&stepperState_t::aSplit,
                                   bp::return_value_policy<result_converter<false> >()))
                .def("__repr__", &PyStepperStateVisitor::repr)
                ;
        }

        static std::string repr(stepperState_t const & self)
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

        ///////////////////////////////////////////////////////////////////////////////
        /// \brief Expose.
        ///////////////////////////////////////////////////////////////////////////////
        static void expose()
        {
            bp::class_<stepperState_t,
                       std::shared_ptr<stepperState_t>,
                       boost::noncopyable>("StepperState", bp::no_init)
                .def(PyStepperStateVisitor());
        }
    };

    BOOST_PYTHON_VISITOR_EXPOSE(StepperState)

    // ***************************** PySystemStateVisitor ***********************************

    struct PySystemStateVisitor
        : public bp::def_visitor<PySystemStateVisitor>
    {
    public:
        ///////////////////////////////////////////////////////////////////////////////
        /// \brief Expose C++ API through the visitor.
        ///////////////////////////////////////////////////////////////////////////////
        template<class PyClass>
        void visit(PyClass & cl) const
        {
            cl
                .add_property("q", bp::make_getter(&systemState_t::q,
                                   bp::return_value_policy<result_converter<false> >()))
                .add_property("v", bp::make_getter(&systemState_t::v,
                                   bp::return_value_policy<result_converter<false> >()))
                .add_property("a", bp::make_getter(&systemState_t::a,
                                   bp::return_value_policy<result_converter<false> >()))
                .add_property("command", bp::make_getter(&systemState_t::command,
                                         bp::return_value_policy<result_converter<false> >()))
                .add_property("u", bp::make_getter(&systemState_t::u,
                                   bp::return_value_policy<result_converter<false> >()))
                .add_property("u_motor", bp::make_getter(&systemState_t::uMotor,
                                         bp::return_value_policy<result_converter<false> >()))
                .add_property("u_internal", bp::make_getter(&systemState_t::uInternal,
                                            bp::return_value_policy<result_converter<false> >()))
                .add_property("u_custom", bp::make_getter(&systemState_t::uCustom,
                                          bp::return_value_policy<result_converter<false> >()))
                .add_property("f_external", bp::make_getter(&systemState_t::fExternal,
                                            bp::return_internal_reference<>()))
                .def("__repr__", &PySystemStateVisitor::repr)
                ;
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
                s << "    (" << i << "): "
                  << self.fExternal[i].toVector().transpose().format(HeavyFmt);
            }
            return s.str();
        }

        ///////////////////////////////////////////////////////////////////////////////
        /// \brief Expose.
        ///////////////////////////////////////////////////////////////////////////////
        static void expose()
        {
            bp::class_<systemState_t,
                       std::shared_ptr<systemState_t>,
                       boost::noncopyable>("SystemState", bp::no_init)
                .def(PySystemStateVisitor());
        }
    };

    BOOST_PYTHON_VISITOR_EXPOSE(SystemState)

    // ***************************** PySystemVisitor ***********************************

    struct PySystemVisitor
        : public bp::def_visitor<PySystemVisitor>
    {
    public:
        ///////////////////////////////////////////////////////////////////////////////
        /// \brief Expose C++ API through the visitor.
        ///////////////////////////////////////////////////////////////////////////////
        template<class PyClass>
        void visit(PyClass & cl) const
        {
            cl
                .add_property("name", bp::make_getter(&systemHolder_t::name,
                                      bp::return_value_policy<bp::copy_non_const_reference>()))
                .add_property("robot", &systemHolder_t::robot)
                .add_property("controller", &systemHolder_t::controller)
                .add_property("callbackFct", bp::make_getter(&systemHolder_t::callbackFct,
                                             bp::return_internal_reference<>()))
                ;
        }

        ///////////////////////////////////////////////////////////////////////////////
        /// \brief Expose.
        ///////////////////////////////////////////////////////////////////////////////
        static void expose()
        {
            bp::class_<systemHolder_t>("system", bp::no_init)
                .def(PySystemVisitor());

            bp::class_<std::vector<systemHolder_t> >("systemVector", bp::no_init)
                .def(vector_indexing_suite_no_contains<std::vector<systemHolder_t> >());
        }
    };

    BOOST_PYTHON_VISITOR_EXPOSE(System)

    // ************************* PyEngineMultiRobotVisitor ****************************

    struct PyEngineMultiRobotVisitor
        : public bp::def_visitor<PyEngineMultiRobotVisitor>
    {
    public:
        ///////////////////////////////////////////////////////////////////////////////
        /// \brief Expose C++ API through the visitor.
        ///////////////////////////////////////////////////////////////////////////////
        template<class PyClass>
        void visit(PyClass & cl) const
        {
            cl
                .def("add_system", &PyEngineMultiRobotVisitor::addSystemWithoutController,
                                   (bp::arg("self"), "system_name", "robot"))
                .def("add_system", &PyEngineMultiRobotVisitor::addSystem,
                                   (bp::arg("self"), "system_name",
                                    "robot", "controller"))
                .def("add_system", &PyEngineMultiRobotVisitor::addSystemWithCallback,
                                   (bp::arg("self"), "system_name",
                                    "robot", "controller", "callback_function"))
                .def("remove_system", &EngineMultiRobot::removeSystem,
                                      (bp::arg("self"), "system_name"))
                .def("set_controller", &EngineMultiRobot::setController,
                                      (bp::arg("self"), "system_name", "controller"))

                .def("register_force_coupling", &PyEngineMultiRobotVisitor::registerForceCoupling,
                                                (bp::arg("self"),
                                                 "system_name_1", "system_name_2",
                                                 "frame_name_1", "frame_name_2",
                                                 "force_function"))
                .def("register_viscoelastic_force_coupling",
                    static_cast<
                        hresult_t (EngineMultiRobot::*)(
                            std::string const &,
                            std::string const &,
                            std::string const &,
                            std::string const &,
                            vectorN_t   const &,
                            vectorN_t   const &)
                    >(&EngineMultiRobot::registerViscoElasticForceCoupling),
                    (bp::arg("self"), "system_name_1", "system_name_2",
                     "frame_name_1", "frame_name_2", "stiffness", "damping"))
                .def("register_viscoelastic_force_coupling",
                    static_cast<
                        hresult_t (EngineMultiRobot::*)(
                            std::string const &,
                            std::string const &,
                            std::string const &,
                            vectorN_t   const &,
                            vectorN_t   const &)
                    >(&EngineMultiRobot::registerViscoElasticForceCoupling),
                    (bp::arg("self"), "system_name", "frame_name_1", "frame_name_2",
                     "stiffness", "damping"))
                .def("register_viscoelastic_directional_force_coupling",
                    static_cast<
                        hresult_t (EngineMultiRobot::*)(
                            std::string const &,
                            std::string const &,
                            std::string const &,
                            std::string const &,
                            float64_t   const &,
                            float64_t   const &)
                    >(&EngineMultiRobot::registerViscoElasticDirectionalForceCoupling),
                    (bp::arg("self"), "system_name_1", "system_name_2",
                     "frame_name_1", "frame_name_2", "stiffness", "damping"))
                .def("register_viscoelastic_directional_force_coupling",
                    static_cast<
                        hresult_t (EngineMultiRobot::*)(
                            std::string const &,
                            std::string const &,
                            std::string const &,
                            float64_t   const &,
                            float64_t   const &)
                    >(&EngineMultiRobot::registerViscoElasticDirectionalForceCoupling),
                    (bp::arg("self"), "system_name", "frame_name_1", "frame_name_2",
                     "stiffness", "damping"))
                .def("remove_forces_coupling",
                    static_cast<
                        hresult_t (EngineMultiRobot::*)(std::string const &, std::string const &)
                    >(&EngineMultiRobot::removeForcesCoupling),
                    (bp::arg("self"), "system_name_1", "system_name_2"))
                .def("remove_forces_coupling",
                    static_cast<
                        hresult_t (EngineMultiRobot::*)(std::string const &)
                    >(&EngineMultiRobot::removeForcesCoupling),
                    (bp::arg("self"), "system_name"))
                .def("remove_forces_coupling",
                    static_cast<
                        hresult_t (EngineMultiRobot::*)(void)
                    >(&EngineMultiRobot::removeForcesCoupling),
                    (bp::arg("self")))
                .add_property("forces_coupling", bp::make_function(&EngineMultiRobot::getForcesCoupling,
                                                 bp::return_internal_reference<>()))

                .def("reset",
                    static_cast<
                        void (EngineMultiRobot::*)(bool_t const &, bool_t const &)
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
                                                 (bp::arg("self"), "t_end", "q_list", "v_list"))

                .def("get_log", &PyEngineMultiRobotVisitor::getLog)
                .def("write_log", &EngineMultiRobot::writeLog,
                                  (bp::arg("self"), "filename",
                                   bp::arg("format") = "hdf5"))
                .def("read_log_binary", &PyEngineMultiRobotVisitor::parseLogBinary, (bp::arg("filename")))
                .staticmethod("read_log_binary")

                .def("register_force_impulse", &PyEngineMultiRobotVisitor::registerForceImpulse,
                                               (bp::arg("self"), "system_name",
                                                "frame_name", "t", "dt", "F"))
                .def("remove_forces_impulse",
                    static_cast<
                        hresult_t (EngineMultiRobot::*)(std::string const &)
                    >(&EngineMultiRobot::removeForcesImpulse),
                    (bp::arg("self"), "system_name"))
                .def("remove_forces_impulse",
                    static_cast<
                        hresult_t (EngineMultiRobot::*)(void)
                    >(&EngineMultiRobot::removeForcesImpulse),
                    (bp::arg("self")))
                .def("register_force_profile", &PyEngineMultiRobotVisitor::registerForceProfile,
                                               (bp::arg("self"), "system_name",
                                                "frame_name", "force_function",
                                                bp::arg("update_period") = 0.0))
                .def("remove_forces_profile",
                    static_cast<
                        hresult_t (EngineMultiRobot::*)(std::string const &)
                    >(&EngineMultiRobot::removeForcesProfile),
                    (bp::arg("self"), "system_name"))
                .def("remove_forces_profile",
                    static_cast<
                        hresult_t (EngineMultiRobot::*)(void)
                    >(&EngineMultiRobot::removeForcesProfile),
                    (bp::arg("self")))

                .add_property("forces_coupling", bp::make_function(&EngineMultiRobot::getForcesCoupling,
                                                 bp::return_internal_reference<>()))

                .add_property("forces_impulse", bp::make_function(&PyEngineMultiRobotVisitor::getForcesImpulse,
                                                bp::return_internal_reference<>()))
                .add_property("forces_profile", bp::make_function(&PyEngineMultiRobotVisitor::getForcesProfile,
                                                bp::return_internal_reference<>()))

                .def("remove_all_forces", &EngineMultiRobot::removeAllForces)

                .def("get_options", &EngineMultiRobot::getOptions)
                .def("set_options", &PyEngineMultiRobotVisitor::setOptions)

                .def("get_system", bp::make_function(&PyEngineMultiRobotVisitor::getSystem,
                                   bp::return_internal_reference<>(),
                                   (bp::arg("self"), "system_name")))
                .def("get_system_state", bp::make_function(&PyEngineMultiRobotVisitor::getSystemState,
                                         bp::return_internal_reference<>(),
                                         (bp::arg("self"), "system_name")))

                .add_property("systems", bp::make_getter(&EngineMultiRobot::systems_,
                                         bp::return_internal_reference<>()))
                .add_property("systems_names", bp::make_function(&EngineMultiRobot::getSystemsNames,
                                               bp::return_value_policy<result_converter<true> >()))
                .add_property("stepper_state", bp::make_function(&EngineMultiRobot::getStepperState,
                                               bp::return_internal_reference<>()))
                .add_property("is_simulation_running", &PyEngineMultiRobotVisitor::getIsSimulationRunning)
                .add_property("simulation_duration_max", &EngineMultiRobot::getMaxSimulationDuration)
                ;
        }

        static hresult_t addSystemWithCallback(EngineMultiRobot                          & self,
                                               std::string                         const & systemName,
                                               std::shared_ptr<Robot>              const & robot,
                                               std::shared_ptr<AbstractController> const & controller,
                                               bp::object                          const & callbackPy)
        {
            TimeStateFctPyWrapper<bool_t> callbackFct(callbackPy);
            return self.addSystem(systemName, robot, controller, std::move(callbackFct));
        }

        static hresult_t addSystem(EngineMultiRobot                          & self,
                                   std::string                         const & systemName,
                                   std::shared_ptr<Robot>              const & robot,
                                   std::shared_ptr<AbstractController> const & controller)
        {
            callbackFunctor_t callbackFct = [](float64_t const & /* t */,
                                               vectorN_t const & /* q */,
                                               vectorN_t const & /* v */) -> bool_t
                                            {
                                                return true;
                                            };
            return self.addSystem(systemName, robot, controller, std::move(callbackFct));
        }

        static hresult_t addSystemWithoutController(EngineMultiRobot             & self,
                                                    std::string            const & systemName,
                                                    std::shared_ptr<Robot> const & robot)
        {
            callbackFunctor_t callbackFct = [](float64_t const & /* t */,
                                               vectorN_t const & /* q */,
                                               vectorN_t const & /* v */) -> bool_t
                                            {
                                                return true;
                                            };
            return self.addSystem(systemName, robot, std::move(callbackFct));
        }

        static systemHolder_t & getSystem(EngineMultiRobot  & self,
                                          std::string const & systemName)
        {
            systemHolder_t * system;
            self.getSystem(systemName, system);  // getSystem is making sure that system is always assigned to a well-defined systemHolder_t
            return *system;
        }

        static forceImpulseRegister_t const & getForcesImpulse(EngineMultiRobot  & self,
                                                               std::string const & systemName)
        {
            forceImpulseRegister_t const * forcesImpulse;
            self.getForcesImpulse(systemName, forcesImpulse);
            return *forcesImpulse;
        }

        static forceProfileRegister_t const & getForcesProfile(EngineMultiRobot  & self,
                                                               std::string const & systemName)
        {
            forceProfileRegister_t const * forcesProfile;
            self.getForcesProfile(systemName, forcesProfile);
            return *forcesProfile;
        }

        static systemState_t const & getSystemState(EngineMultiRobot  & self,
                                                    std::string const & systemName)
        {
            systemState_t const * systemState;
            self.getSystemState(systemName, systemState);  // getSystemState is making sure that systemState is always assigned to a well-defined systemState_t
            return *systemState;
        }

        static hresult_t registerForceCoupling(EngineMultiRobot       & self,
                                               std::string      const & systemName1,
                                               std::string      const & systemName2,
                                               std::string      const & frameName1,
                                               std::string      const & frameName2,
                                               bp::object       const & forcePy)
        {
            TimeBistateFctPyWrapper<pinocchio::Force> forceFct(forcePy);
            return self.registerForceCoupling(
                systemName1, systemName2, frameName1, frameName2, std::move(forceFct));
        }

        static hresult_t start(EngineMultiRobot       & self,
                               bp::object       const & qInitPy,
                               bp::object       const & vInitPy,
                               bp::object       const & aInitPy)
        {
            boost::optional<std::map<std::string, vectorN_t> > aInit = boost::none;
            if (!aInitPy.is_none())
            {
                aInit.emplace(convertFromPython<std::map<std::string, vectorN_t> >(aInitPy));
            }
            return self.start(convertFromPython<std::map<std::string, vectorN_t> >(qInitPy),
                              convertFromPython<std::map<std::string, vectorN_t> >(vInitPy),
                              aInit);
        }

        static hresult_t step(EngineMultiRobot       & self,
                              float64_t        const & dtDesired)
        {
            // Only way to handle C++ default values that are not accessible in Python
            return self.step(dtDesired);
        }

        static hresult_t simulate(EngineMultiRobot       & self,
                                  float64_t        const & endTime,
                                  bp::object       const & qInitPy,
                                  bp::object       const & vInitPy,
                                  bp::object       const & aInitPy)
        {
            boost::optional<std::map<std::string, vectorN_t> > aInit = boost::none;
            if (!aInitPy.is_none())
            {
                aInit.emplace(convertFromPython<std::map<std::string, vectorN_t> >(aInitPy));
            }
            return self.simulate(endTime,
                                 convertFromPython<std::map<std::string, vectorN_t> >(qInitPy),
                                 convertFromPython<std::map<std::string, vectorN_t> >(vInitPy),
                                 aInit);
        }

        static bp::object computeSystemsDynamics(EngineMultiRobot       & self,
                                                 float64_t        const & endTime,
                                                 bp::object       const & qSplitPy,
                                                 bp::object       const & vSplitPy)
        {
            static std::vector<vectorN_t> aSplit;
            self.computeSystemsDynamics(
                endTime,
                convertFromPython<std::vector<vectorN_t> >(qSplitPy),
                convertFromPython<std::vector<vectorN_t> >(vSplitPy),
                aSplit
            );
            return convertToPython<std::vector<vectorN_t> >(aSplit, true);
        }

        static void registerForceImpulse(EngineMultiRobot       & self,
                                         std::string      const & systemName,
                                         std::string      const & frameName,
                                         float64_t        const & t,
                                         float64_t        const & dt,
                                         vector6_t        const & F)
        {
            self.registerForceImpulse(systemName, frameName, t, dt, pinocchio::Force(F));
        }

        static void registerForceProfile(EngineMultiRobot       & self,
                                         std::string      const & systemName,
                                         std::string      const & frameName,
                                         bp::object       const & forcePy,
                                         float64_t        const & updatePeriod)
        {
            TimeStateFctPyWrapper<pinocchio::Force> forceFct(forcePy);
            self.registerForceProfile(systemName, frameName, std::move(forceFct), updatePeriod);
        }

        ///////////////////////////////////////////////////////////////////////////////
        /// \brief      Getters and Setters
        ///////////////////////////////////////////////////////////////////////////////

        static bp::tuple formatLogData(logData_t const & logData)
        {
            bp::dict variables, constants;

            // Early return if empty
            if (logData.header.empty())
            {
                return bp::make_tuple(variables, constants);
            }

            // Get constants
            std::ptrdiff_t const lastConstantIdx = std::distance(
                logData.header.begin(), std::find(logData.header.begin(), logData.header.end(), START_COLUMNS));
            for (std::ptrdiff_t i = 1; i < lastConstantIdx; ++i)
            {
                std::size_t const delimiter = logData.header[i].find(TELEMETRY_CONSTANT_DELIMITER);
                constants[logData.header[i].substr(0, delimiter)] = logData.header[i].substr(delimiter + 1);
            }

            // Get Global.Time
            bp::object timePy;
            if (!logData.timestamps.empty())
            {
                vectorN_t timeBuffer = Eigen::Matrix<int64_t, 1, Eigen::Dynamic>::Map(
                    logData.timestamps.data(), logData.timestamps.size()).cast<float64_t>() / logData.timeUnit;
                timePy = convertToPython(timeBuffer, true);
                PyArray_CLEARFLAGS(reinterpret_cast<PyArrayObject *>(timePy.ptr()), NPY_ARRAY_WRITEABLE);
            }
            else
            {
                npy_intp dims[1] = {npy_intp(0)};
                timePy = bp::object(bp::handle<>(PyArray_SimpleNew(1, dims, NPY_FLOAT64)));
            }
            variables[logData.header[lastConstantIdx + 1]] = timePy;

            // Get intergers
            if (!logData.intData.empty())
            {
                Eigen::Matrix<int64_t, Eigen::Dynamic, 1> intVector;
                intVector.resize(logData.timestamps.size());

                for (std::size_t i = 0; i < logData.numInt; ++i)
                {
                    std::string const & header_i = logData.header[i + (lastConstantIdx + 1) + 1];
                    for (std::size_t j = 0; j < logData.intData.size(); ++j)
                    {
                        intVector[j] = logData.intData[j][i];
                    }
                    bp::object array = convertToPython(intVector, true);
                    PyArray_CLEARFLAGS(reinterpret_cast<PyArrayObject *>(array.ptr()), NPY_ARRAY_WRITEABLE);
                    variables[header_i] = array;
                }
            }
            else
            {
                npy_intp dims[1] = {npy_intp(0)};
                for (std::size_t i = 0; i < logData.numInt; ++i)
                {
                    std::string const & header_i = logData.header[i + (lastConstantIdx + 1) + 1];
                    variables[header_i] = bp::object(bp::handle<>(
                        PyArray_SimpleNew(1, dims, NPY_INT64)));
                }
            }

            // Get floats
            if (!logData.floatData.empty())
            {
                Eigen::Matrix<float64_t, Eigen::Dynamic, 1> floatVector;
                floatVector.resize(logData.timestamps.size());

                for (std::size_t i = 0; i < logData.numFloat; ++i)
                {
                    std::string const & header_i =
                        logData.header[i + (lastConstantIdx + 1) + 1 + logData.numInt];
                    for (std::size_t j = 0; j < logData.floatData.size(); ++j)
                    {
                        floatVector[j] = logData.floatData[j][i];
                    }
                    bp::object array = convertToPython(floatVector, true);
                    PyArray_CLEARFLAGS(reinterpret_cast<PyArrayObject *>(array.ptr()), NPY_ARRAY_WRITEABLE);
                    variables[header_i] = array;
                }
            }
            else
            {
                npy_intp dims[1] = {npy_intp(0)};
                for (std::size_t i = 0; i < logData.numFloat; ++i)
                {
                    std::string const & header_i =
                        logData.header[i + (lastConstantIdx + 1) + 1 + logData.numInt];
                    variables[header_i] = bp::object(bp::handle<>(
                        PyArray_SimpleNew(1, dims, NPY_FLOAT64)));
                }
            }

            return bp::make_tuple(variables, constants);
        }

        static bp::tuple getLog(EngineMultiRobot & self)
        {
            /* It is impossible to use static boost::python variables. Indeed,
               the global/static destructor is called after finalization of
               Python runtime, the later being required to call the destructor
               of Python objects. The easiest way to circumvent this limitation
               the easiest solution is to avoid them. Alternatively, increasing
               the reference counter to avoid calling the destructor at exit is
               another way to fix the issue. Here some reference for more details
               and more robust solutions:
               - https://stackoverflow.com/a/24156996/4820605
               - https://stackoverflow.com/a/31444751/4820605 */

            static std::unique_ptr<bp::tuple> logDataPy(nullptr);
            static std::shared_ptr<logData_t const> logDataOld;
            std::shared_ptr<logData_t const> logData;
            self.getLogDataRaw(logData);
            if (logData.use_count() == 2)
            {
                // Decrement the reference counter of old Python log data
                if (logDataPy)
                {
                    bp::decref(logDataPy->ptr());
                }

                /* The shared pointer is new, because otherwise the use count should larger
                   than 2. Indeed, both the engine and this method holds a single reference
                   at this point. If it was old, this method would holds at least 2
                   references, one for the old reference and one for the new. */
                logDataPy = std::make_unique<bp::tuple>(formatLogData(*logData));

                /* Reference counter must be incremented to avoid calling deleter by Boost
                   Python after runtime finalization. */
                bp::incref(logDataPy->ptr());

                // Update log data backup
                logDataOld = logData;
            }

            // Avoid potential null pointer dereference, although should never happen in practice
            if (logDataPy)
            {
                return *logDataPy;
            }
            else
            {
                return bp::make_tuple(bp::dict(), bp::dict());
            }
        }

        static bp::tuple parseLogBinary(std::string const & filename)
        {
            logData_t logData;
            hresult_t returnCode = EngineMultiRobot::parseLogBinaryRaw(filename, logData);
            if (returnCode == hresult_t::SUCCESS)
            {
                return formatLogData(logData);
            }
            else
            {
                return bp::make_tuple(bp::dict(), bp::dict());
            }
        }

        static hresult_t setOptions(EngineMultiRobot & self,
                                    bp::dict const   & configPy)
        {
            configHolder_t config = self.getOptions();
            convertFromPython(configPy, config);
            return self.setOptions(config);
        }

        static bp::object getIsSimulationRunning(EngineMultiRobot & self)
        {
            return bp::object(bp::handle<>(getNumpyReferenceFromScalar(self.getIsSimulationRunning())));
        }

        ///////////////////////////////////////////////////////////////////////////////
        /// \brief Expose.
        ///////////////////////////////////////////////////////////////////////////////
        static void expose()
        {
            bp::class_<EngineMultiRobot,
                       std::shared_ptr<EngineMultiRobot>,
                       boost::noncopyable>("EngineMultiRobot")
                .def(PyEngineMultiRobotVisitor());
        }
    };

    BOOST_PYTHON_VISITOR_EXPOSE(EngineMultiRobot)

    // ***************************** PyEngineVisitor ***********************************

    struct PyEngineVisitor
        : public bp::def_visitor<PyEngineVisitor>
    {
    public:
        ///////////////////////////////////////////////////////////////////////////////
        /// \brief Expose C++ API through the visitor.
        ///////////////////////////////////////////////////////////////////////////////
        template<class PyClass>
        void visit(PyClass & cl) const
        {
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
                .def("register_force_profile", &PyEngineVisitor::registerForceProfile,
                                               (bp::arg("self"), "frame_name", "force_function",
                                                bp::arg("update_period") = 0.0))
                .def("register_force_coupling", &PyEngineVisitor::registerForceCoupling,
                                                (bp::arg("self"), "frame_name_1", "frame_name_2", "force_function"))
                .def("register_viscoelastic_force_coupling",
                    static_cast<
                        hresult_t (Engine::*)(
                            std::string const &, std::string const &, vectorN_t const &, vectorN_t const &)
                    >(&Engine::registerViscoElasticForceCoupling),
                    (bp::arg("self"), "frame_name_1", "frame_name_2", "stiffness", "damping"))
                .def("register_viscoelastic_directional_force_coupling",
                    static_cast<
                        hresult_t (Engine::*)(
                            std::string const &, std::string const &, float64_t const &, float64_t const &)
                    >(&Engine::registerViscoElasticDirectionalForceCoupling),
                    (bp::arg("self"), "frame_name_1", "frame_name_2", "stiffness", "damping"))

                .add_property("forces_impulse", bp::make_function(
                                                static_cast<
                                                    forceImpulseRegister_t const & (Engine::*)(void) const
                                                >(&Engine::getForcesImpulse),
                                                bp::return_internal_reference<>()))
                .add_property("forces_profile", bp::make_function(
                                                static_cast<
                                                    forceProfileRegister_t const & (Engine::*)(void) const
                                                >(&Engine::getForcesProfile),
                                                bp::return_internal_reference<>()))

                .add_property("is_initialized", bp::make_function(&Engine::getIsInitialized,
                                                bp::return_value_policy<bp::copy_const_reference>()))
                .add_property("system", bp::make_function(&PyEngineVisitor::getSystem,
                                        bp::return_internal_reference<>()))
                .add_property("robot", &PyEngineVisitor::getRobot)
                .add_property("controller", &PyEngineVisitor::getController)
                .add_property("stepper_state", bp::make_function(&Engine::getStepperState,
                                               bp::return_internal_reference<>()))
                .add_property("system_state", bp::make_function(&PyEngineVisitor::getSystemState,
                                                                bp::return_internal_reference<>()))
                ;
        }

        static hresult_t addSystem(bp::tuple /* args */, bp::dict /* kwargs */)
        {
            // Hide all EngineMultiRobot `addSystem` overloads at once
            return Engine().addSystem("", std::shared_ptr<Robot>(), std::shared_ptr<AbstractController>());
        }

        static hresult_t initialize(Engine                                    & self,
                                    std::shared_ptr<Robot>              const & robot,
                                    std::shared_ptr<AbstractController> const & controller,
                                    bp::object                          const & callbackPy)
        {
            if (callbackPy.is_none())
            {
                callbackFunctor_t callbackFct = [](float64_t const & /* t */,
                                                   vectorN_t const & /* q */,
                                                   vectorN_t const & /* v */) -> bool_t
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

        static void registerForceImpulse(Engine            & self,
                                         std::string const & frameName,
                                         float64_t   const & t,
                                         float64_t   const & dt,
                                         vector6_t   const & F)
        {
            self.registerForceImpulse(frameName, t, dt, pinocchio::Force(F));
        }

        static void registerForceProfile(Engine            & self,
                                         std::string const & frameName,
                                         bp::object  const & forcePy,
                                         float64_t   const & updatePeriod)
        {
            TimeStateFctPyWrapper<pinocchio::Force> forceFct(forcePy);
            self.registerForceProfile(frameName, std::move(forceFct), updatePeriod);
        }

        static hresult_t registerForceCoupling(Engine            & self,
                                               std::string const & frameName1,
                                               std::string const & frameName2,
                                               bp::object  const & forcePy)
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

        static systemState_t const & getSystemState(Engine & self)
        {
            systemState_t const * systemState;
            self.getSystemState(systemState);  // getSystemState is making sure that systemState is always assigned to a well-defined systemState_t
            return *systemState;
        }

        static hresult_t start(Engine           & self,
                               vectorN_t  const & qInit,
                               vectorN_t  const & vInit,
                               bp::object const & aInitPy,
                               bool_t     const & isStateTheoretical)
        {
            boost::optional<vectorN_t> aInit = boost::none;
            if (!aInitPy.is_none())
            {
                aInit.emplace(convertFromPython<vectorN_t>(aInitPy));
            }
            return self.start(qInit, vInit, aInit, isStateTheoretical);
        }

        static hresult_t simulate(Engine           & self,
                                  float64_t  const & endTime,
                                  vectorN_t  const & qInit,
                                  vectorN_t  const & vInit,
                                  bp::object const & aInitPy,
                                  bool_t     const & isStateTheoretical)
        {
            boost::optional<vectorN_t> aInit = boost::none;
            if (!aInitPy.is_none())
            {
                aInit.emplace(convertFromPython<vectorN_t>(aInitPy));
            }
            return self.simulate(endTime, qInit, vInit, aInit, isStateTheoretical);
        }

        ///////////////////////////////////////////////////////////////////////////////
        /// \brief Expose.
        ///////////////////////////////////////////////////////////////////////////////
        static void expose()
        {
            bp::class_<Engine, bp::bases<EngineMultiRobot>,
                       std::shared_ptr<Engine>,
                       boost::noncopyable>("Engine")
                .def(PyEngineVisitor());
        }
    };

    BOOST_PYTHON_VISITOR_EXPOSE(Engine)
}  // End of namespace python.
}  // End of namespace jiminy.
