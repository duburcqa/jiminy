#include "jiminy/core/utilities/helpers.h"
#include "jiminy/core/utilities/json.h"
#include "jiminy/core/io/serialization.h"
#include "jiminy/core/io/abstract_io_device.h"
#include "jiminy/core/io/memory_device.h"
#include "jiminy/core/control/abstract_controller.h"
#include "jiminy/core/robot/robot.h"
#include "jiminy/core/engine/engine.h"
#include "jiminy/core/stepper/abstract_stepper.h"

#include "pinocchio/bindings/python/fwd.hpp"
#include <boost/python/raw_function.hpp>

#include "jiminy/python/utilities.h"
#include "jiminy/python/functors.h"
#include "jiminy/python/engine.h"


namespace jiminy::python
{
    namespace bp = boost::python;

    // ***************************************** Forces **************************************** //

    namespace internal::forces
    {
        static bp::object profileForceWrapper(const ProfileForce & self)
        {
            bp::object func = makeFunction(self.func,
                                           bp::return_value_policy<bp::return_by_value>(),
                                           (bp::arg("t"), "q", "v"));
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
    }

    void exposeForces()
    {
        bp::class_<ProfileForce, std::shared_ptr<ProfileForce>, boost::noncopyable>("ProfileForce",
                                                                                    bp::no_init)
            .DEF_READONLY("frame_name", &ProfileForce::frameName)
            .DEF_READONLY("frame_index", &ProfileForce::frameIndex)
            .DEF_READONLY("update_period", &ProfileForce::updatePeriod)
            .DEF_READONLY("force", &ProfileForce::force)
            .ADD_PROPERTY_GET("func", internal::forces::profileForceWrapper);

        /* Note that it will be impossible to slice the vector if `boost::noncopyable` is set for
           the stl container, or if the value type contained itself. In such a case, it raises a
           runtime error rather than a compile-time error. */
        bp::class_<ProfileForceVector>("ProfileForceVector", bp::no_init)
            .def(vector_indexing_suite_no_contains<ProfileForceVector>());

        bp::class_<ImpulseForce, std::shared_ptr<ImpulseForce>, boost::noncopyable>("ImpulseForce",
                                                                                    bp::no_init)
            .DEF_READONLY("frame_name", &ImpulseForce::frameName)
            .DEF_READONLY("frame_index", &ImpulseForce::frameIndex)
            .DEF_READONLY("t", &ImpulseForce::t)
            .DEF_READONLY("dt", &ImpulseForce::dt)
            .DEF_READONLY("force", &ImpulseForce::force);

        bp::class_<ImpulseForceVector, boost::noncopyable>("ImpulseForceVector", bp::no_init)
            .def(vector_indexing_suite_no_contains<ImpulseForceVector>());

        bp::class_<CouplingForce, std::shared_ptr<CouplingForce>, boost::noncopyable>(
            "CouplingForce", bp::no_init)
            .DEF_READONLY("robot_name_1", &CouplingForce::robotName1)
            .DEF_READONLY("robot_index_1", &CouplingForce::robotIndex1)
            .DEF_READONLY("robot_name_2", &CouplingForce::robotName2)
            .DEF_READONLY("robot_index_2", &CouplingForce::robotIndex2)
            .ADD_PROPERTY_GET("func", internal::forces::couplingForceWrapper);

        bp::class_<CouplingForceVector, boost::noncopyable>("CouplingForceVector", bp::no_init)
            .def(vector_indexing_suite_no_contains<CouplingForceVector>());
    }

    // ************************************** StepperState ************************************* //

    namespace internal::stepper_state
    {
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
    }

    void exposeStepperState()
    {
        bp::class_<StepperState, std::shared_ptr<StepperState>, boost::noncopyable>("StepperState",
                                                                                    bp::no_init)
            .DEF_READONLY("iter", &StepperState::iter)
            .DEF_READONLY("iter_failed", &StepperState::iterFailed)
            .DEF_READONLY("t", &StepperState::t)
            .DEF_READONLY("dt", &StepperState::dt)
            .ADD_PROPERTY_GET("q", &internal::stepper_state::getQ)
            .ADD_PROPERTY_GET("v", &internal::stepper_state::getV)
            .ADD_PROPERTY_GET("a", &internal::stepper_state::getA)
            .def("__repr__", &internal::stepper_state::repr);
    }

    // *************************************** RobotState ************************************** //

    namespace internal::robot_state
    {
        static std::string repr(RobotState & self)
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
    }

    void exposeRobotState()
    {
        bp::class_<RobotState, std::shared_ptr<RobotState>, boost::noncopyable>("RobotState",
                                                                                bp::no_init)
            .DEF_READONLY("q", &RobotState::q)
            .DEF_READONLY("v", &RobotState::v)
            .DEF_READONLY("a", &RobotState::a)
            .DEF_READONLY("command", &RobotState::command)
            .DEF_READONLY("u", &RobotState::u)
            .DEF_READONLY("u_motor", &RobotState::uMotor)
            .DEF_READONLY("u_internal", &RobotState::uInternal)
            .DEF_READONLY("u_custom", &RobotState::uCustom)
            .DEF_READONLY("f_external", &RobotState::fExternal)
            .def("__repr__", &internal::robot_state::repr);
    }

    // ***************************************** Engine **************************************** //

    namespace internal::engine
    {
        static bp::dict getImpulseForces(Engine & self)
        {
            bp::dict impulseForcesPy;
            for (const auto & robot : self.robots_)
            {
                const std::string & robotName = robot->getName();
                const ImpulseForceVector & impulseForces = self.getImpulseForces(robotName);
                impulseForcesPy[robotName] = convertToPython(impulseForces, false);
            }
            return impulseForcesPy;
        }

        static bp::dict getProfileForces(Engine & self)
        {
            bp::dict profileForcessPy;
            for (const auto & robot : self.robots_)
            {
                const std::string & robotName = robot->getName();
                const ProfileForceVector & profileForces = self.getProfileForces(robotName);
                profileForcessPy[robotName] = convertToPython(profileForces, false);
            }
            return profileForcessPy;
        }

        static bp::list getRobotStates(Engine & self)
        {
            bp::list systemStates;
            for (const auto & robot : self.robots_)
            {
                const std::string & robotName = robot->getName();
                const RobotState & systemState = self.getRobotState(robotName);
                systemStates.append(convertToPython(systemState, false));
            }
            return systemStates;
        }

        static void registerCouplingForce(Engine & self,
                                          const std::string & robotName1,
                                          const std::string & robotName2,
                                          const std::string & frameName1,
                                          const std::string & frameName2,
                                          const bp::object & forceFuncPy)
        {
            TimeBistateFunPyWrapper<pinocchio::Force> forceFunc(forceFuncPy);
            return self.registerCouplingForce(
                robotName1, robotName2, frameName1, frameName2, forceFunc);
        }

        static void startFromDict(Engine & self,
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

        static void simulateFromDict(Engine & self,
                                     double endTime,
                                     const bp::dict & qInitPy,
                                     const bp::dict & vInitPy,
                                     const bp::object & aInitPy,
                                     const bp::object & callbackPy)
        {
            std::optional<std::map<std::string, Eigen::VectorXd>> aInit = std::nullopt;
            if (!aInitPy.is_none())
            {
                aInit.emplace(convertFromPython<std::map<std::string, Eigen::VectorXd>>(aInitPy));
            }
            AbortSimulationFunction callback;
            if (!callbackPy.is_none())
            {
                callback = callbackPy;
            }
            else
            {
                callback = []()
                {
                    return true;
                };
            }
            return self.simulate(
                endTime,
                convertFromPython<std::map<std::string, Eigen::VectorXd>>(qInitPy),
                convertFromPython<std::map<std::string, Eigen::VectorXd>>(vInitPy),
                aInit,
                callback);
        }

        static void simulate(Engine & self,
                             double endTime,
                             const Eigen::VectorXd & qInit,
                             const Eigen::VectorXd & vInit,
                             const bp::object & aInitPy,
                             bool isStateTheoretical,
                             const bp::object & callbackPy)
        {
            std::optional<Eigen::VectorXd> aInit = std::nullopt;
            if (!aInitPy.is_none())
            {
                aInit.emplace(convertFromPython<Eigen::VectorXd>(aInitPy));
            }
            AbortSimulationFunction callback;
            if (!callbackPy.is_none())
            {
                callback = callbackPy;
            }
            else
            {
                callback = []()
                {
                    return true;
                };
            }
            return self.simulate(endTime, qInit, vInit, aInit, isStateTheoretical, callback);
        }

        static std::vector<Eigen::VectorXd> computeRobotsDynamics(Engine & self,
                                                                  double endTime,
                                                                  const bp::object & qSplitPy,
                                                                  const bp::object & vSplitPy)
        {
            std::vector<Eigen::VectorXd> aSplit;
            self.computeRobotsDynamics(endTime,
                                       convertFromPython<std::vector<Eigen::VectorXd>>(qSplitPy),
                                       convertFromPython<std::vector<Eigen::VectorXd>>(vSplitPy),
                                       aSplit);
            return aSplit;
        }

        static void registerImpulseForce(Engine & self,
                                         const std::string & robotName,
                                         const std::string & frameName,
                                         double t,
                                         double dt,
                                         const Vector6d & force)
        {
            return self.registerImpulseForce(robotName, frameName, t, dt, pinocchio::Force{force});
        }

        static void registerProfileForce(Engine & self,
                                         const std::string & robotName,
                                         const std::string & frameName,
                                         const bp::object & forceFuncPy,
                                         double updatePeriod)
        {
            TimeStateFunPyWrapper<pinocchio::Force> forceFunc(forceFuncPy);
            return self.registerProfileForce(robotName, frameName, forceFunc, updatePeriod);
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
                else if (key.find(".pinocchio_model") != std::string::npos)
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

        static bp::dict getLog(Engine & self)
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
            const LogData logData = Engine::readLog(filename, format);
            return formatLogData(logData);
        }

        static void setOptions(Engine & self, const bp::dict & configPy)
        {
            GenericConfig config = self.getOptions();
            convertFromPython(configPy, config);
            return self.setOptions(config);
        }
    }

    void exposeEngine()
    {
        bp::class_<Engine, std::shared_ptr<Engine>, boost::noncopyable>("Engine")
            .def("add_robot", &Engine::addRobot, (bp::arg("self"), "robot"))
            .def("remove_robot", &Engine::removeRobot, (bp::arg("self"), "robot_name"))

            .def("reset",
                 static_cast<void (Engine::*)(bool, bool)>(&Engine::reset),
                 (bp::arg("self"),
                  bp::arg("reset_random_generator") = false,
                  bp::arg("remove_all_forces") = false))
            .def("start",
                 &internal::engine::startFromDict,
                 (bp::arg("self"),
                  "q_init_dict",
                  "v_init_dict",
                  bp::arg("a_init_dict") = bp::object()))  // bp::object() means 'None' in Python
            .def("start",
                 &internal::engine::start,
                 (bp::arg("self"),
                  "q_init",
                  "v_init",
                  bp::arg("a_init") = bp::object(),
                  bp::arg("is_state_theoretical") = false))
            .def("step", &Engine::step, (bp::arg("self"), bp::arg("step_dt") = -1))
            .def("stop", &Engine::stop, (bp::arg("self")))
            .def("simulate",
                 &internal::engine::simulateFromDict,
                 (bp::arg("self"),
                  "t_end",
                  "q_init_dict",
                  "v_init_dict",
                  bp::arg("a_init_dict") = bp::object(),
                  bp::arg("callback") = bp::object()))
            .def("simulate",
                 &internal::engine::simulate,
                 (bp::arg("self"),
                  "t_end",
                  "q_init",
                  "v_init",
                  bp::arg("a_init") = bp::object(),
                  bp::arg("is_state_theoretical") = false,
                  bp::arg("callback") = bp::object()))
            .def("compute_forward_kinematics",
                 &Engine::computeForwardKinematics,
                 (bp::arg("robot"), "q", "v", "a"))
            .staticmethod("compute_forward_kinematics")
            .def("compute_robots_dynamics",
                 &internal::engine::computeRobotsDynamics,
                 bp::return_value_policy<result_converter<true>>(),
                 (bp::arg("self"), "t_end", "q_list", "v_list"))

            .ADD_PROPERTY_GET("log_data", &internal::engine::getLog)
            .def("read_log",
                 &internal::engine::readLog,
                 (bp::arg("fullpath"), bp::arg("format") = bp::object()),
                 "Read a logfile from jiminy.\n\n"
                 ".. note::\n    This function supports both binary and hdf5 log.\n\n"
                 ":param fullpath: Name of the file to load.\n"
                 ":param format: Name of the file to load.\n\n"
                 ":returns: Dictionary containing the logged constants and variables.")
            .staticmethod("read_log")
            .def("write_log", &Engine::writeLog, (bp::arg("self"), "fullpath", "format"))

            .def("register_impulse_force",
                 &internal::engine::registerImpulseForce,
                 (bp::arg("self"), "robot_name", "frame_name", "t", "dt", "force"))
            .def("remove_impulse_forces",
                 static_cast<void (Engine::*)(const std::string &)>(&Engine::removeImpulseForces),
                 (bp::arg("self"), "robot_name"))
            .def("remove_impulse_forces",
                 static_cast<void (Engine::*)(void)>(&Engine::removeImpulseForces),
                 (bp::arg("self")))
            .ADD_PROPERTY_GET("impulse_forces", &internal::engine::getImpulseForces)

            .def("register_profile_force",
                 &internal::engine::registerProfileForce,
                 (bp::arg("self"),
                  "robot_name",
                  "frame_name",
                  "force_func",
                  bp::arg("update_period") = 0.0))
            .def("remove_profile_forces",
                 static_cast<void (Engine::*)(const std::string &)>(&Engine::removeProfileForces),
                 (bp::arg("self"), "robot_name"))
            .def("remove_profile_forces",
                 static_cast<void (Engine::*)(void)>(&Engine::removeProfileForces),
                 (bp::arg("self")))
            .ADD_PROPERTY_GET("profile_forces", &internal::engine::getProfileForces)

            .def("register_coupling_force",
                 &internal::engine::registerCouplingForce,
                 (bp::arg("self"),
                  "robot_name_1",
                  "robot_name_2",
                  "frame_name_1",
                  "frame_name_2",
                  "force_func"))
            .def("register_viscoelastic_coupling_force",
                 static_cast<void (Engine::*)(const std::string &,
                                              const std::string &,
                                              const std::string &,
                                              const std::string &,
                                              const Vector6d &,
                                              const Vector6d &,
                                              double)>(&Engine::registerViscoelasticCouplingForce),
                 (bp::arg("self"),
                  "robot_name_1",
                  "robot_name_2",
                  "frame_name_1",
                  "frame_name_2",
                  "stiffness",
                  "damping",
                  bp::arg("alpha") = 0.5))
            .def("register_viscoelastic_coupling_force",
                 static_cast<void (Engine::*)(const std::string &,
                                              const std::string &,
                                              const std::string &,
                                              const Vector6d &,
                                              const Vector6d &,
                                              double)>(&Engine::registerViscoelasticCouplingForce),
                 (bp::arg("self"),
                  "robot_name",
                  "frame_name_1",
                  "frame_name_2",
                  "stiffness",
                  "damping",
                  bp::arg("alpha") = 0.5))
            .def("register_viscoelastic_directional_coupling_force",
                 static_cast<void (Engine::*)(const std::string &,
                                              const std::string &,
                                              const std::string &,
                                              const std::string &,
                                              double,
                                              double,
                                              double)>(
                     &Engine::registerViscoelasticDirectionalCouplingForce),
                 (bp::arg("self"),
                  "robot_name_1",
                  "robot_name_2",
                  "frame_name_1",
                  "frame_name_2",
                  "stiffness",
                  "damping",
                  bp::arg("rest_length") = 0.0))
            .def("register_viscoelastic_directional_coupling_force",
                 static_cast<void (Engine::*)(const std::string &,
                                              const std::string &,
                                              const std::string &,
                                              double,
                                              double,
                                              double)>(
                     &Engine::registerViscoelasticDirectionalCouplingForce),
                 (bp::arg("self"),
                  "robot_name",
                  "frame_name_1",
                  "frame_name_2",
                  "stiffness",
                  "damping",
                  bp::arg("rest_length") = 0.0))
            .def("remove_coupling_forces",
                 static_cast<void (Engine::*)(const std::string &, const std::string &)>(
                     &Engine::removeCouplingForces),
                 (bp::arg("self"), "robot_name_1", "robot_name_2"))
            .def("remove_coupling_forces",
                 static_cast<void (Engine::*)(const std::string &)>(&Engine::removeCouplingForces),
                 (bp::arg("self"), "robot_name"))
            .def("remove_coupling_forces",
                 static_cast<void (Engine::*)(void)>(&Engine::removeCouplingForces),
                 (bp::arg("self")))
            .ADD_PROPERTY_GET_WITH_POLICY("coupling_forces",
                                          &Engine::getCouplingForces,
                                          bp::return_value_policy<result_converter<false>>())

            .def("remove_all_forces", &Engine::removeAllForces)

            .def("set_options", &internal::engine::setOptions)
            .def("get_options", &Engine::getOptions)

            .DEF_READONLY_WITH_POLICY(
                "robots", &Engine::robots_, bp::return_value_policy<result_converter<true>>())
            .def("get_robot", &Engine::getRobot, (bp::arg("self"), "robot_name"))
            .def("get_robot_index", &Engine::getRobotIndex, (bp::arg("self"), "robot_name"))
            .def("get_robot_state",
                 &Engine::getRobotState,
                 (bp::arg("self"), "robot_name"),
                 bp::return_value_policy<result_converter<false>>())

            .ADD_PROPERTY_GET("robot_states", &internal::engine::getRobotStates)
            .ADD_PROPERTY_GET_WITH_POLICY("stepper_state",
                                          &Engine::getStepperState,
                                          bp::return_value_policy<result_converter<false>>())
            .ADD_PROPERTY_GET_WITH_POLICY("is_simulation_running",
                                          &Engine::getIsSimulationRunning,
                                          bp::return_value_policy<result_converter<false>>())
            .add_static_property("simulation_duration_max", &Engine::getSimulationDurationMax)
            .add_static_property("telemetry_time_unit", &Engine::getTelemetryTimeUnit);
    }
}
