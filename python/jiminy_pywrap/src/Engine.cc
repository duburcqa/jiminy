#include "jiminy/core/io/Serialization.h"
#include "jiminy/core/control/AbstractController.h"
#include "jiminy/core/robot/Robot.h"
#include "jiminy/core/engine/Engine.h"
#include "jiminy/core/engine/EngineMultiRobot.h"
#include "jiminy/core/telemetry/TelemetryData.h"
#include "jiminy/core/telemetry/TelemetryRecorder.h"
#include "jiminy/core/utilities/Json.h"
#include "jiminy/core/utilities/Helpers.h"

#include <boost/optional.hpp>

#include "pinocchio/bindings/python/fwd.hpp"
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
                                            (bp::arg("t"), "q", "v"),
                                            functionToMLP(self.forceFct));
        setFunctionWrapperModule<forceProfile_t>(func);
        return func;
    }

    static bp::object forceCouplingWrapper(forceCoupling_t const & self)
    {
        bp::object func = bp::make_function(self.forceFct,
                                            bp::return_value_policy<bp::return_by_value>(),
                                            (bp::arg("t"), "q_1", "v_1", "q_2", "v_2"),
                                            functionToMLP(self.forceFct));
        setFunctionWrapperModule<forceCoupling_t>(func);
        return func;
    }

    void exposeForces(void)
    {
        bp::class_<forceProfile_t,
                   std::shared_ptr<forceProfile_t>,
                   boost::noncopyable>("ForceProfile", bp::no_init)
            .DEF_READONLY("frame_name", &forceProfile_t::frameName)
            .DEF_READONLY("frame_idx", &forceProfile_t::frameIdx)
            .DEF_READONLY("update_period", &forceProfile_t::updatePeriod)
            .DEF_READONLY("force_prev", &forceProfile_t::forcePrev)
            .ADD_PROPERTY_GET("force_func", forceProfileWrapper);

        /* Note that it will be impossible to slice the vector if `boost::noncopyable` is set
           for the stl container, or if the value type contained itself. In such a case, it
           raises a runtime error rather than a compile-time error. */
        bp::class_<forceProfileRegister_t>("ForceProfileVector", bp::no_init)
            .def(vector_indexing_suite_no_contains<forceProfileRegister_t>());

        bp::class_<forceImpulse_t,
                   std::shared_ptr<forceImpulse_t>,
                   boost::noncopyable>("ForceImpulse", bp::no_init)
            .DEF_READONLY("frame_name", &forceImpulse_t::frameName)
            .DEF_READONLY("frame_idx", &forceImpulse_t::frameIdx)
            .DEF_READONLY("t", &forceImpulse_t::t)
            .DEF_READONLY("dt", &forceImpulse_t::dt)
            .DEF_READONLY("F", &forceImpulse_t::F);

        bp::class_<forceImpulseRegister_t,
                   boost::noncopyable>("ForceImpulseVector", bp::no_init)
            .def(vector_indexing_suite_no_contains<forceImpulseRegister_t>());

        bp::class_<forceCoupling_t,
                   std::shared_ptr<forceCoupling_t>,
                   boost::noncopyable>("ForceCoupling", bp::no_init)
            .DEF_READONLY("system_name_1", &forceCoupling_t::systemName1)
            .DEF_READONLY("system_idx_1", &forceCoupling_t::systemIdx1)
            .DEF_READONLY("system_name_2", &forceCoupling_t::systemName2)
            .DEF_READONLY("system_idx_2", &forceCoupling_t::systemIdx2)
            .ADD_PROPERTY_GET("force_func", forceCouplingWrapper);

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
                .DEF_READONLY("iter", &stepperState_t::iter)
                .DEF_READONLY("iter_failed", &stepperState_t::iterFailed)
                .DEF_READONLY("t", &stepperState_t::t)
                .DEF_READONLY("dt", &stepperState_t::dt)
                .ADD_PROPERTY_GET("q", &PyStepperStateVisitor::getQ)
                .ADD_PROPERTY_GET("v", &PyStepperStateVisitor::getV)
                .ADD_PROPERTY_GET("a", &PyStepperStateVisitor::getA)
                .def("__repr__", &PyStepperStateVisitor::repr)
                ;
        }

        static bp::object getQ(stepperState_t const & self)
        {
            return convertToPython(self.qSplit, false);
        }

        static bp::object getV(stepperState_t const & self)
        {
            return convertToPython(self.vSplit, false);
        }

        static bp::object getA(stepperState_t const & self)
        {
            return convertToPython(self.aSplit, false);
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
                .DEF_READONLY("name", &systemHolder_t::name)
                .DEF_READONLY("robot", &systemHolder_t::robot)
                .DEF_READONLY("controller", &systemHolder_t::controller)
                .DEF_READONLY("callbackFct", &systemHolder_t::callbackFct)
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
                                                 bp::return_value_policy<result_converter<true> >(),
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
                        hresult_t (EngineMultiRobot::*)(std::string const &)
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
                        hresult_t (EngineMultiRobot::*)(std::string const &)
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
                            std::string const &,
                            std::string const &,
                            std::string const &,
                            std::string const &,
                            vector6_t   const &,
                            vector6_t   const &,
                            float64_t   const &)
                    >(&EngineMultiRobot::registerViscoelasticForceCoupling),
                    (bp::arg("self"), "system_name_1", "system_name_2",
                     "frame_name_1", "frame_name_2", "stiffness", "damping", bp::arg("alpha") = 0.5))
                .def("register_viscoelastic_force_coupling",
                    static_cast<
                        hresult_t (EngineMultiRobot::*)(
                            std::string const &,
                            std::string const &,
                            std::string const &,
                            vector6_t   const &,
                            vector6_t   const &,
                            float64_t   const &)
                    >(&EngineMultiRobot::registerViscoelasticForceCoupling),
                    (bp::arg("self"), "system_name", "frame_name_1", "frame_name_2",
                     "stiffness", "damping", bp::arg("alpha") = 0.5))
                .def("register_viscoelastic_directional_force_coupling",
                    static_cast<
                        hresult_t (EngineMultiRobot::*)(
                            std::string const &,
                            std::string const &,
                            std::string const &,
                            std::string const &,
                            float64_t   const &,
                            float64_t   const &,
                            float64_t   const &)
                    >(&EngineMultiRobot::registerViscoelasticDirectionalForceCoupling),
                    (bp::arg("self"), "system_name_1", "system_name_2", "frame_name_1", "frame_name_2",
                     "stiffness", "damping", bp::arg("rest_length") = 0.0))
                .def("register_viscoelastic_directional_force_coupling",
                    static_cast<
                        hresult_t (EngineMultiRobot::*)(
                            std::string const &,
                            std::string const &,
                            std::string const &,
                            float64_t   const &,
                            float64_t   const &,
                            float64_t   const &)
                    >(&EngineMultiRobot::registerViscoelasticDirectionalForceCoupling),
                    (bp::arg("self"), "system_name", "frame_name_1", "frame_name_2",
                     "stiffness", "damping", bp::arg("rest_length") = 0.0))
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
                .ADD_PROPERTY_GET_WITH_POLICY("forces_coupling",
                                              &EngineMultiRobot::getForcesCoupling,
                                              bp::return_value_policy<result_converter<false> >())

                .def("remove_all_forces", &EngineMultiRobot::removeAllForces)

                .def("set_options", &PyEngineMultiRobotVisitor::setOptions)
                .def("get_options", &EngineMultiRobot::getOptions)

                .DEF_READONLY("systems", &EngineMultiRobot::systems_)
                .ADD_PROPERTY_GET_WITH_POLICY("systems_names",
                                              &EngineMultiRobot::getSystemsNames,
                                              bp::return_value_policy<result_converter<true> >())
                .ADD_PROPERTY_GET("systems_states", &PyEngineMultiRobotVisitor::getSystemState)
                .ADD_PROPERTY_GET_WITH_POLICY("stepper_state",
                                              &EngineMultiRobot::getStepperState,
                                              bp::return_value_policy<result_converter<false> >())
                .ADD_PROPERTY_GET_WITH_POLICY("is_simulation_running",
                                              &EngineMultiRobot::getIsSimulationRunning,
                                              bp::return_value_policy<result_converter<false> >())
                .add_static_property("simulation_duration_max", &EngineMultiRobot::getMaxSimulationDuration)
                .add_static_property("telemetry_time_unit", &EngineMultiRobot::getTelemetryTimeUnit)
                ;
        }

        static hresult_t addSystem(EngineMultiRobot             & self,
                                   std::string            const & systemName,
                                   std::shared_ptr<Robot> const & robot,
                                   bp::object             const & controllerPy,
                                   bp::object             const & callbackPy)
        {
            callbackFunctor_t callbackFct;
            if (callbackPy.is_none())
            {
                callbackFct = [](float64_t const & /* t */,
                                 vectorN_t const & /* q */,
                                 vectorN_t const & /* v */) -> bool_t
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
                std::shared_ptr<AbstractController> const controller = bp::extract<std::shared_ptr<AbstractController> >(controllerPy);
                return self.addSystem(systemName, robot, controller, std::move(callbackFct));
            }
            return self.addSystem(systemName, robot, std::move(callbackFct));
        }

        static systemHolder_t & getSystem(EngineMultiRobot  & self,
                                          std::string const & systemName)
        {
            systemHolder_t * system;
            self.getSystem(systemName, system);  // getSystem is making sure that system is always assigned to a well-defined systemHolder_t
            return *system;
        }

        static bp::dict getForcesImpulse(EngineMultiRobot & self)
        {
            bp::dict forceImpulsesPy;
            for (auto const & systemName : self.getSystemsNames())
            {
                forceImpulseRegister_t const * forcesImpulse;
                self.getForcesImpulse(systemName, forcesImpulse);
                forceImpulsesPy[systemName] = convertToPython(forcesImpulse, false);
            }
            return forceImpulsesPy;
        }

        static bp::dict getForcesProfile(EngineMultiRobot  & self)
        {
            bp::dict forcesProfilesPy;
            for (auto const & systemName : self.getSystemsNames())
            {
                forceProfileRegister_t const * forcesProfile;
                self.getForcesProfile(systemName, forcesProfile);
                forcesProfilesPy[systemName] = convertToPython(forcesProfile, false);
            }
            return forcesProfilesPy;
        }

        static bp::dict getSystemState(EngineMultiRobot  & self)
        {
            bp::dict systemStates;
            for (std::string const & systemName : self.getSystemsNames())
            {
                /* Cannot fail, but `getSystemState` is making sure that systemState
                   is assigned to a well-defined object anyway. */
                systemState_t const * systemState;
                self.getSystemState(systemName, systemState);
                systemStates[systemName] = convertToPython(systemState, false);
            }
            return systemStates;
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
                               bp::dict         const & qInitPy,
                               bp::dict         const & vInitPy,
                               bp::object       const & aInitPy)
        {
            std::optional<std::map<std::string, vectorN_t> > aInit = std::nullopt;
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
                                  bp::dict         const & qInitPy,
                                  bp::dict         const & vInitPy,
                                  bp::object       const & aInitPy)
        {
            std::optional<std::map<std::string, vectorN_t> > aInit = std::nullopt;
            if (!aInitPy.is_none())
            {
                aInit.emplace(convertFromPython<std::map<std::string, vectorN_t> >(aInitPy));
            }
            return self.simulate(endTime,
                                 convertFromPython<std::map<std::string, vectorN_t> >(qInitPy),
                                 convertFromPython<std::map<std::string, vectorN_t> >(vInitPy),
                                 aInit);
        }

        static std::vector<vectorN_t> computeSystemsDynamics(EngineMultiRobot       & self,
                                                             float64_t        const & endTime,
                                                             bp::list         const & qSplitPy,
                                                             bp::list         const & vSplitPy)
        {
            std::vector<vectorN_t> aSplit;
            self.computeSystemsDynamics(
                endTime,
                convertFromPython<std::vector<vectorN_t> >(qSplitPy),
                convertFromPython<std::vector<vectorN_t> >(vSplitPy),
                aSplit
            );
            return aSplit;
        }

        static hresult_t registerForceImpulse(EngineMultiRobot       & self,
                                              std::string      const & systemName,
                                              std::string      const & frameName,
                                              float64_t        const & t,
                                              float64_t        const & dt,
                                              vector6_t        const & F)
        {
            return self.registerForceImpulse(systemName, frameName, t, dt, pinocchio::Force(F));
        }

        static hresult_t registerForceProfile(EngineMultiRobot       & self,
                                              std::string      const & systemName,
                                              std::string      const & frameName,
                                              bp::object       const & forcePy,
                                              float64_t        const & updatePeriod)
        {
            TimeStateFctPyWrapper<pinocchio::Force> forceFct(forcePy);
            return self.registerForceProfile(systemName, frameName, std::move(forceFct), updatePeriod);
        }

        ///////////////////////////////////////////////////////////////////////////////
        /// \brief      Getters and Setters
        ///////////////////////////////////////////////////////////////////////////////

        static bp::dict formatLogData(logData_t const & logData)
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
            Eigen::Index const numInt = logData.intData.rows();
            Eigen::Index const numFloat = logData.floatData.rows();

            // Get constants
            for (auto const & [key, value] : logData.constants)
            {
                if (endsWith(key, ".options"))
                {
                    std::vector<uint8_t> jsonStringVec(value.begin(), value.end());
                    std::shared_ptr<AbstractIODevice> device =
                        std::make_shared<MemoryDevice>(std::move(jsonStringVec));
                    configHolder_t robotOptions;
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
                    catch(const std::exception& e)
                    {
                        PRINT_ERROR("Failed to load pinocchio model from log.");
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
                    catch(const std::exception& e)
                    {
                        PRINT_ERROR("Failed to load collision and/or visual model from log.");
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
                else if (key == NUM_INTS.substr(0, key.size())
                      || key == NUM_FLOATS.substr(0, key.size()))
                {
                    constants[key] = std::stol(value);
                }
                else if (key == TIME_UNIT)
                {
                    constants[key] = std::stod(value);
                }
                else
                {
                    constants[key] = value; // convertToPython(value, false);
                }
            }

            // Get Global.Time
            bp::object timePy;
            if (logData.timestamps.size() > 0)
            {
                vectorN_t const timeBuffer = logData.timestamps.cast<float64_t>() * logData.timeUnit;
                timePy = convertToPython(timeBuffer, true);
                PyArray_CLEARFLAGS(reinterpret_cast<PyArrayObject *>(timePy.ptr()), NPY_ARRAY_WRITEABLE);
            }
            else
            {
                npy_intp dims[1] = {npy_intp(0)};
                timePy = bp::object(bp::handle<>(PyArray_SimpleNew(1, dims, NPY_FLOAT64)));
            }
            variables[logData.fieldnames[0]] = timePy;

            // Get integers
            if (numInt > 0)
            {
                for (Eigen::Index i = 0; i < numInt; ++i)
                {
                    std::string const & header_i = logData.fieldnames[i + 1];
                    intVector = logData.intData.row(i);
                    bp::object array = convertToPython(intVector, true);
                    PyArray_CLEARFLAGS(reinterpret_cast<PyArrayObject *>(array.ptr()), NPY_ARRAY_WRITEABLE);
                    variables[header_i] = array;
                }
            }
            else
            {
                npy_intp dims[1] = {npy_intp(0)};
                for (Eigen::Index i = 0; i < numInt; ++i)
                {
                    std::string const & header_i = logData.fieldnames[i + 1];
                    variables[header_i] = bp::object(bp::handle<>(
                        PyArray_SimpleNew(1, dims, NPY_INT64)));
                }
            }

            // Get floats
            if (numFloat > 0)
            {
                for (Eigen::Index i = 0; i < numFloat; ++i)
                {
                    std::string const & header_i = logData.fieldnames[i + 1 + numInt];
                    floatVector = logData.floatData.row(i);
                    bp::object array = convertToPython(floatVector, true);
                    PyArray_CLEARFLAGS(reinterpret_cast<PyArrayObject *>(array.ptr()), NPY_ARRAY_WRITEABLE);
                    variables[header_i] = array;
                }
            }
            else
            {
                npy_intp dims[1] = {npy_intp(0)};
                for (Eigen::Index i = 0; i < numFloat; ++i)
                {
                    std::string const & header_i = logData.fieldnames[i + 1 + numInt];
                    variables[header_i] = bp::object(bp::handle<>(
                        PyArray_SimpleNew(1, dims, NPY_FLOAT64)));
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

            static std::unique_ptr<bp::dict> logDataPy(nullptr);
            static std::shared_ptr<logData_t const> logDataOld;
            std::shared_ptr<logData_t const> logData;
            self.getLog(logData);
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
                logDataPy = std::make_unique<bp::dict>(formatLogData(*logData));

                /* Reference counter must be incremented to avoid calling deleter by Boost
                   Python after runtime finalization. */
                bp::incref(logDataPy->ptr());

                // Update log data backup
                logDataOld = logData;
            }

            // Avoid potential null pointer dereference, although it should never happen in practice
            if (logDataPy)
            {
                return *logDataPy;
            }
            return {};
        }

        static bp::dict readLog(std::string const & filename,
                                bp::object  const & formatPy)
        {
            std::string format;
            if (!formatPy.is_none())
            {
                format = convertFromPython<std::string>(formatPy);
            }
            else
            {
                std::array<std::string, 3> const extHdf5 {{".h5", ".hdf5", ".tlmc"}};
                if (endsWith(filename, ".data"))
                {
                    format = "binary";
                }
                else if (std::any_of(extHdf5.begin(), extHdf5.end(), std::bind(
                    endsWith, filename, std::placeholders::_1)))
                {
                    format = "hdf5";
                }
                else
                {
                    throw std::runtime_error(
                        "Impossible to determine the file format automatically. "
                        "Please specify it manually.");
                }
            }
            logData_t logData;
            hresult_t returnCode = EngineMultiRobot::readLog(filename, format, logData);
            if (returnCode == hresult_t::SUCCESS)
            {
                return formatLogData(logData);
            }
            return {};
        }

        static hresult_t setOptions(EngineMultiRobot & self,
                                    bp::dict const   & configPy)
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
                .ADD_PROPERTY_GET_WITH_POLICY("forces_impulse",
                                              static_cast<
                                                  forceImpulseRegister_t const & (Engine::*)(void) const
                                              >(&Engine::getForcesImpulse),
                                              bp::return_value_policy<result_converter<false> >())

                .def("register_force_profile", &PyEngineVisitor::registerForceProfile,
                                               (bp::arg("self"), "frame_name", "force_function",
                                                bp::arg("update_period") = 0.0))
                .ADD_PROPERTY_GET_WITH_POLICY("forces_profile",
                                              static_cast<
                                                  forceProfileRegister_t const & (Engine::*)(void) const
                                              >(&Engine::getForcesProfile),
                                              bp::return_value_policy<result_converter<false> >())

                .def("register_force_coupling", &PyEngineVisitor::registerForceCoupling,
                                                (bp::arg("self"), "frame_name_1", "frame_name_2", "force_function"))
                .def("register_viscoelastic_force_coupling",
                    static_cast<
                        hresult_t (Engine::*)(
                            std::string const &,
                            std::string const &,
                            vector6_t   const &,
                            vector6_t   const &,
                            float64_t   const &)
                    >(&Engine::registerViscoelasticForceCoupling),
                    (bp::arg("self"), "frame_name_1", "frame_name_2", "stiffness", "damping", bp::arg("alpha") = 0.5))
                .def("register_viscoelastic_directional_force_coupling",
                    static_cast<
                        hresult_t (Engine::*)(
                            std::string const &,
                            std::string const &,
                            float64_t   const &,
                            float64_t   const &,
                            float64_t   const &)
                    >(&Engine::registerViscoelasticDirectionalForceCoupling),
                    (bp::arg("self"), "frame_name_1", "frame_name_2", "stiffness", "damping",
		             bp::arg("rest_length") = 0.0))

                .ADD_PROPERTY_GET_WITH_POLICY("is_initialized",
                                              &Engine::getIsInitialized,
                                              bp::return_value_policy<bp::return_by_value>())
                .ADD_PROPERTY_GET_WITH_POLICY("system",
                                              &PyEngineVisitor::getSystem,
                                              bp::return_value_policy<result_converter<false> >())
                .ADD_PROPERTY_GET("robot", &PyEngineVisitor::getRobot)
                .ADD_PROPERTY_GET("controller", &PyEngineVisitor::getController)
                .ADD_PROPERTY_GET_WITH_POLICY("stepper_state",
                                              &Engine::getStepperState,
                                              bp::return_value_policy<result_converter<false> >())
                .ADD_PROPERTY_GET_WITH_POLICY("system_state",
                                              &PyEngineVisitor::getSystemState,
                                              bp::return_value_policy<result_converter<false> >())
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

        static hresult_t registerForceImpulse(Engine            & self,
                                              std::string const & frameName,
                                              float64_t   const & t,
                                              float64_t   const & dt,
                                              vector6_t   const & F)
        {
            return self.registerForceImpulse(frameName, t, dt, pinocchio::Force(F));
        }

        static hresult_t registerForceProfile(Engine            & self,
                                              std::string const & frameName,
                                              bp::object  const & forcePy,
                                              float64_t   const & updatePeriod)
        {
            TimeStateFctPyWrapper<pinocchio::Force> forceFct(forcePy);
            return self.registerForceProfile(frameName, std::move(forceFct), updatePeriod);
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
            std::optional<vectorN_t> aInit = std::nullopt;
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
            std::optional<vectorN_t> aInit = std::nullopt;
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
