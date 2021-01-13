#include "jiminy/core/control/AbstractController.h"
#include "jiminy/core/robot/Robot.h"
#include "jiminy/core/engine/Engine.h"
#include "jiminy/core/engine/EngineMultiRobot.h"
#include "jiminy/core/telemetry/TelemetryData.h"
#include "jiminy/core/telemetry/TelemetryRecorder.h"

#include "jiminy/python/Functors.h"
#include "jiminy/python/Engine.h"

#include <boost/python.hpp>


namespace jiminy
{
namespace python
{
    namespace bp = boost::python;

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
                .add_property("q", &PyStepperStateVisitor::getPosition)
                .add_property("v", &PyStepperStateVisitor::getVelocity)
                .add_property("a", &PyStepperStateVisitor::getAcceleration)
                .def("__repr__", &PyStepperStateVisitor::repr)
                ;
        }

        static bp::object getPosition(stepperState_t const & self)
        {
            return convertToPython<std::vector<vectorN_t> >(self.qSplit, false);
        }

        static bp::object getVelocity(stepperState_t const & self)
        {
            return convertToPython<std::vector<vectorN_t> >(self.vSplit, false);
        }

        static bp::object getAcceleration(stepperState_t const & self)
        {
            return convertToPython<std::vector<vectorN_t> >(self.aSplit, false);
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
            for (uint32_t i=0; i < self.qSplit.size(); ++i)
            {
                s << "\n    (" << i << "): " << self.qSplit[i].transpose().format(HeavyFmt);
            }
            s << "\nv:";
            for (uint32_t i=0; i < self.vSplit.size(); ++i)
            {
                s << "\n    (" << i << "): " << self.vSplit[i].transpose().format(HeavyFmt);
            }
            s << "\na:";
            for (uint32_t i=0; i < self.aSplit.size(); ++i)
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
                .add_property("q", &PySystemStateVisitor::getPosition)
                .add_property("v", &PySystemStateVisitor::getVelocity)
                .add_property("a", &PySystemStateVisitor::getAcceleration)
                .add_property("u", &PySystemStateVisitor::getTotalEffort)
                .add_property("u_motor", &PySystemStateVisitor::getMotorEffort)
                .add_property("u_command", &PySystemStateVisitor::getCommandEffort)
                .add_property("u_internal", &PySystemStateVisitor::getInternalEffort)
                .add_property("f_external", bp::make_getter(&systemState_t::fExternal,
                                            bp::return_internal_reference<>()))
                .def("__repr__", &PySystemStateVisitor::repr)
                ;
        }

        static bp::object getPosition(systemState_t const & self)
        {
            // Do not use automatic converter for efficiency
            return convertToPython<vectorN_t>(self.q, false);
        }

        static bp::object getVelocity(systemState_t const & self)
        {
            // Do not use automatic converter for efficiency
            return convertToPython<vectorN_t>(self.v, false);
        }

        static bp::object getAcceleration(systemState_t const & self)
        {
            // Do not use automatic converter for efficiency
            return convertToPython<vectorN_t>(self.a, false);
        }

        static bp::object getTotalEffort(systemState_t const & self)
        {
            // Do not use automatic converter for efficiency
            return convertToPython<vectorN_t>(self.u, false);
        }

        static bp::object getMotorEffort(systemState_t const & self)
        {
            // Do not use automatic converter for efficiency
            return convertToPython<vectorN_t>(self.uMotor, false);
        }

        static bp::object getCommandEffort(systemState_t const & self)
        {
            // Do not use automatic converter for efficiency
            return convertToPython<vectorN_t>(self.uCommand, false);
        }

        static bp::object getInternalEffort(systemState_t const & self)
        {
            // Do not use automatic converter for efficiency
            return convertToPython<vectorN_t>(self.uInternal, false);
        }

        static std::string repr(systemState_t & self)
        {
            std::stringstream s;
            Eigen::IOFormat HeavyFmt(5, 1, ", ", "", "", "", "[", "]\n");
            s << "q:\n    " << self.q.transpose().format(HeavyFmt);
            s << "v:\n    " << self.v.transpose().format(HeavyFmt);
            s << "a:\n    " << self.a.transpose().format(HeavyFmt);
            s << "u:\n    " << self.u.transpose().format(HeavyFmt);
            s << "u_motor:\n    " << self.uMotor.transpose().format(HeavyFmt);
            s << "u_command:\n    " << self.uCommand.transpose().format(HeavyFmt);
            s << "u_internal:\n    " << self.uInternal.transpose().format(HeavyFmt);
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

        static uint32_t getLength(std::vector<systemHolder_t> & self)
        {
            return self.size();
        }

        static systemHolder_t & getItem(std::vector<systemHolder_t>       & self,
                                            int32_t                         const & idx)
        {
            return self[idx];
        }

        ///////////////////////////////////////////////////////////////////////////////
        /// \brief Expose.
        ///////////////////////////////////////////////////////////////////////////////
        static void expose()
        {
            bp::class_<systemHolder_t,
                       boost::noncopyable>("system", bp::no_init)
                .def(PySystemVisitor());

            bp::class_<std::vector<systemHolder_t>,
                       boost::noncopyable>("systemVector", bp::no_init)
                .def("__len__", bp::make_function(&PySystemVisitor::getLength,
                                bp::return_value_policy<bp::return_by_value>()))
                .def("__iter__", bp::iterator<std::vector<systemHolder_t>,
                                 bp::return_internal_reference<> >())
                .def("__getitem__", bp::make_function(&PySystemVisitor::getItem,
                                    bp::return_internal_reference<>(),
                                    (bp::arg("self"), "idx")));
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
                .def("add_coupling_force", &PyEngineMultiRobotVisitor::addCouplingForce,
                                           (bp::arg("self"),
                                            "system_name_1", "system_name_2",
                                            "frame_name_1", "frame_name_2",
                                            "force_function"))
                .def("remove_coupling_forces",
                    static_cast<
                        hresult_t (EngineMultiRobot::*)(std::string const &, std::string const &)
                    >(&EngineMultiRobot::removeCouplingForces),
                    (bp::arg("self"), "system_name_1", "system_name_2"))
                .def("remove_coupling_forces",
                    static_cast<
                        hresult_t (EngineMultiRobot::*)(std::string const &)
                    >(&EngineMultiRobot::removeCouplingForces),
                    (bp::arg("self"), "system_name"))

                .def("reset",
                    static_cast<
                        void (EngineMultiRobot::*)(bool_t const &)
                    >(&EngineMultiRobot::reset),
                    (bp::arg("self"), bp::arg("remove_forces") = false))
                .def("start", &PyEngineMultiRobotVisitor::start,
                              (bp::arg("self"), "q_init_list", "v_init_list",
                               bp::arg("a_init_list") = bp::object(),  // bp::object() means 'None' in Python
                               bp::arg("reset_random_generator") = false,
                               bp::arg("remove_forces") = false))
                .def("step", &PyEngineMultiRobotVisitor::step,
                             (bp::arg("self"), bp::arg("dt_desired") = -1))
                .def("stop", &EngineMultiRobot::stop, (bp::arg("self")))
                .def("simulate", &PyEngineMultiRobotVisitor::simulate,
                                 (bp::arg("self"), "t_end", "q_init_list", "v_init_list",
                                  bp::arg("a_init_list") = bp::object()))
                .def("compute_system_dynamics", &PyEngineMultiRobotVisitor::computeSystemDynamics,
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
                .def("register_force_profile", &PyEngineMultiRobotVisitor::registerForceProfile,
                                               (bp::arg("self"), "system_name",
                                                "frame_name", "force_function"))
                .def("remove_forces", &PyEngineMultiRobotVisitor::removeForces)

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
                                               bp::return_value_policy<bp::return_by_value>()))
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
            callbackFunctor_t callbackFct = [](float64_t const & t,
                                               vectorN_t const & q,
                                               vectorN_t const & v) -> bool_t
                                            {
                                                return true;
                                            };
            return self.addSystem(systemName, robot, controller, std::move(callbackFct));
        }

        static hresult_t addSystemWithoutController(EngineMultiRobot             & self,
                                                    std::string            const & systemName,
                                                    std::shared_ptr<Robot> const & robot)
        {
            callbackFunctor_t callbackFct = [](float64_t const & t,
                                               vectorN_t const & q,
                                               vectorN_t const & v) -> bool_t
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

        static systemState_t const & getSystemState(EngineMultiRobot  & self,
                                                    std::string const & systemName)
        {
            systemState_t const * systemState;
            self.getSystemState(systemName, systemState);  // getSystemState is making sure that systemState is always assigned to a well-defined systemState_t
            return *systemState;
        }

        static hresult_t addCouplingForce(EngineMultiRobot       & self,
                                          std::string      const & systemName1,
                                          std::string      const & systemName2,
                                          std::string      const & frameName1,
                                          std::string      const & frameName2,
                                          bp::object       const & forcePy)
        {
            TimeBistateFctPyWrapper<pinocchio::Force> forceFct(forcePy);
            return self.addCouplingForce(
                systemName1, systemName2, frameName1, frameName2, std::move(forceFct));
        }

        static hresult_t start(EngineMultiRobot       & self,
                               bp::object       const & qInitPy,
                               bp::object       const & vInitPy,
                               bp::object       const & aInitPy,
                               bool             const & resetRandomGenerator,
                               bool             const & removeForces)
        {
            std::optional<std::map<std::string, vectorN_t> > aInit = std::nullopt;
            if (!aInitPy.is_none())
            {
                aInit.emplace(convertFromPython<std::map<std::string, vectorN_t> >(aInitPy));
            }
            return self.start(convertFromPython<std::map<std::string, vectorN_t> >(qInitPy),
                              convertFromPython<std::map<std::string, vectorN_t> >(vInitPy),
                              aInit,
                              resetRandomGenerator,
                              removeForces);
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

        static bp::object computeSystemDynamics(EngineMultiRobot       & self,
                                                float64_t        const & endTime,
                                                bp::object       const & qSplitPy,
                                                bp::object       const & vSplitPy)
        {
            std::vector<vectorN_t> aSplit;
            self.computeSystemDynamics(
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
                                         bp::object       const & forcePy)
        {
            TimeStateFctPyWrapper<pinocchio::Force> forceFct(forcePy);
            self.registerForceProfile(systemName, frameName, std::move(forceFct));
        }

        static void removeForces(Engine & self)
        {
            self.reset(true);
        }

        ///////////////////////////////////////////////////////////////////////////////
        /// \brief      Getters and Setters
        ///////////////////////////////////////////////////////////////////////////////

        static bp::tuple formatLogData(logData_t & logData)
        {
            bp::dict variables;
            bp::dict constants;

            // Early return if empty
            if (logData.header.empty())
            {
                return bp::make_tuple(variables, constants);
            }

            // Get constants
            int32_t const lastConstantIdx = std::distance(
                logData.header.begin(), std::find(logData.header.begin(), logData.header.end(), START_COLUMNS));
            for (int32_t i = 1; i < lastConstantIdx; ++i)
            {
                int32_t const delimiter = logData.header[i].find(TELEMETRY_CONSTANT_DELIMITER);
                constants[logData.header[i].substr(0, delimiter)] = logData.header[i].substr(delimiter + 1);
            }

            // Get Global.Time
            bp::object timePy;
            if (!logData.timestamps.empty())
            {
                vectorN_t timeBuffer = Eigen::Matrix<int64_t, 1, Eigen::Dynamic>::Map(
                    logData.timestamps.data(), logData.timestamps.size()).cast<float64_t>() / logData.timeUnit;
                timePy = convertToPython(timeBuffer);
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

                for (uint32_t i=0; i<logData.numInt; ++i)
                {
                    std::string const & header_i = logData.header[i + (lastConstantIdx + 1) + 1];
                    for (uint32_t j=0; j < logData.intData.size(); ++j)
                    {
                        intVector[j] = logData.intData[j][i];
                    }
                    variables[header_i] = convertToPython(intVector);
                }
            }
            else
            {
                npy_intp dims[1] = {npy_intp(0)};
                for (uint32_t i=0; i<logData.numInt; ++i)
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

                for (uint32_t i=0; i<logData.numFloat; ++i)
                {
                    std::string const & header_i =
                        logData.header[i + (lastConstantIdx + 1) + 1 + logData.numInt];
                    for (uint32_t j=0; j < logData.floatData.size(); ++j)
                    {
                        floatVector[j] = logData.floatData[j][i];
                    }
                    variables[header_i] = convertToPython(floatVector);
                }
            }
            else
            {
                npy_intp dims[1] = {npy_intp(0)};
                for (uint32_t i=0; i<logData.numFloat; ++i)
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
            logData_t logData;
            self.getLogDataRaw(logData);
            return formatLogData(logData);
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
                .def("initialize", &PyEngineVisitor::initializeWithoutController,
                                   (bp::arg("self"), "robot"))
                .def("initialize", &PyEngineVisitor::initialize,
                                   (bp::arg("self"), "robot", "controller"))
                .def("initialize", &PyEngineVisitor::initializeWithCallback,
                                   (bp::arg("self"), "robot", "controller", "callback_function"))
                .def("set_controller", static_cast<
                        hresult_t (Engine::*)(std::shared_ptr<AbstractController>)
                    >(&Engine::setController),
                    (bp::arg("self"), "controller"))

                .def("start",
                    &PyEngineVisitor::start,
                    (bp::arg("self"), "q_init", "v_init",
                     bp::arg("a_init") = bp::object(),
                     bp::arg("is_state_theoretical") = false,
                     bp::arg("reset_random_generator") = false,
                     bp::arg("remove_forces") = false))
                .def("simulate",
                    &PyEngineVisitor::simulate,
                    (bp::arg("self"), "t_end", "q_init", "v_init",
                     bp::arg("a_init") = bp::object(),
                     bp::arg("is_state_theoretical") = false))

                .def("register_force_impulse", &PyEngineVisitor::registerForceImpulse,
                                               (bp::arg("self"), "frame_name", "t", "dt", "F"))
                .def("register_force_profile", &PyEngineVisitor::registerForceProfile,
                                               (bp::arg("self"), "frame_name", "force_function"))
                .def("add_coupling_force", &PyEngineVisitor::addCouplingForce,
                                           (bp::arg("self"), "frame_name_1", "frame_name_2", "force_function"))

                .add_property("is_initialized", bp::make_function(&Engine::getIsInitialized,
                                                bp::return_value_policy<bp::copy_const_reference>()))
                .add_property("robot",  &PyEngineVisitor::getRobot)
                .add_property("controller", &PyEngineVisitor::getController)
                .add_property("stepper_state", bp::make_function(&Engine::getStepperState,
                                               bp::return_internal_reference<>()))
                .add_property("system_state", bp::make_function(&PyEngineVisitor::getSystemState,
                                                                bp::return_internal_reference<>()))
                ;
        }

        static hresult_t initializeWithCallback(Engine                                    & self,
                                                std::shared_ptr<Robot>              const & robot,
                                                std::shared_ptr<AbstractController> const & controller,
                                                bp::object                          const & callbackPy)
        {
            TimeStateFctPyWrapper<bool_t> callbackFct(callbackPy);
            return self.initialize(robot, controller, std::move(callbackFct));
        }

        static hresult_t initialize(Engine                                    & self,
                                    std::shared_ptr<Robot>              const & robot,
                                    std::shared_ptr<AbstractController> const & controller)
        {
            callbackFunctor_t callbackFct = [](float64_t const & t,
                                               vectorN_t const & q,
                                               vectorN_t const & v) -> bool_t
                                            {
                                                return true;
                                            };
            return self.initialize(robot, controller, std::move(callbackFct));
        }

        static hresult_t initializeWithoutController(Engine                       & self,
                                                     std::shared_ptr<Robot> const & robot)
        {
            callbackFunctor_t callbackFct = [](float64_t const & t,
                                               vectorN_t const & q,
                                               vectorN_t const & v) -> bool_t
                                            {
                                                return true;
                                            };
            return self.initialize(robot, std::move(callbackFct));
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
                                         bp::object  const & forcePy)
        {
            TimeStateFctPyWrapper<pinocchio::Force> forceFct(forcePy);
            self.registerForceProfile(frameName, std::move(forceFct));
        }

        static hresult_t addCouplingForce(Engine            & self,
                                          std::string const & frameName1,
                                          std::string const & frameName2,
                                          bp::object  const & forcePy)
        {
            TimeStateFctPyWrapper<pinocchio::Force> forceFct(forcePy);
            return self.addCouplingForce(frameName1, frameName2, std::move(forceFct));
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
                               bool       const & isStateTheoretical,
                               bool       const & resetRandomGenerator,
                               bool       const & removeForces)
        {
            std::optional<vectorN_t> aInit = std::nullopt;
            if (!aInitPy.is_none())
            {
                aInit.emplace(convertFromPython<vectorN_t>(aInitPy));
            }
            return self.start(qInit, vInit, aInit, isStateTheoretical, resetRandomGenerator, removeForces);
        }

        static hresult_t simulate(Engine           & self,
                                  float64_t  const & endTime,
                                  vectorN_t  const & qInit,
                                  vectorN_t  const & vInit,
                                  bp::object const & aInitPy,
                                  bool       const & isStateTheoretical)
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
