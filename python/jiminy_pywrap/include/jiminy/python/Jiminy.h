///////////////////////////////////////////////////////////////////////////////
/// \brief             Python exposition functions for Jiminy project.
////////////////////////////////////////////////////////////////////////////////

#ifndef SIMULATOR_PYTHON_H
#define SIMULATOR_PYTHON_H

#include <cassert>

#include "jiminy/core/engine/EngineMultiRobot.h"
#include "jiminy/core/engine/Engine.h"
#include "jiminy/core/robot/Robot.h"
#include "jiminy/core/robot/BasicMotors.h"
#include "jiminy/core/robot/BasicSensors.h"
#include "jiminy/core/robot/FixedFrameConstraint.h"
#include "jiminy/core/robot/WheelConstraint.h"
#include "jiminy/core/control/ControllerFunctor.h"
#include "jiminy/core/telemetry/TelemetryData.h"
#include "jiminy/core/telemetry/TelemetryRecorder.h"
#include "jiminy/core/Types.h"

#include "jiminy/python/Utilities.h"

#include <boost/preprocessor.hpp>

#include <boost/python.hpp>
#include <boost/python/def.hpp>
#include <boost/python/dict.hpp>


namespace jiminy
{
namespace python
{
    namespace bp = boost::python;

    // ************************** FctPyWrapper ******************************

    template<typename T>
    struct DataInternalBufferType
    {
        using type = typename std::add_lvalue_reference<T>::type;
    };

    template<>
    struct DataInternalBufferType<pinocchio::Force> {
        using type = typename Eigen::Ref<vector6_t>;
    };

    template<typename T>
    typename DataInternalBufferType<T>::type
    setDataInternalBuffer(T * arg)
    {
        return *arg;
    }

    template<>
    typename DataInternalBufferType<pinocchio::Force>::type
    setDataInternalBuffer<pinocchio::Force>(pinocchio::Force * arg)
    {
        return arg->toVector();
    }

    template<typename T>
    T * createInternalBuffer(void)
    {
        return (new T());
    }

    template<>
    pinocchio::Force * createInternalBuffer<pinocchio::Force>(void)
    {
        return (new pinocchio::Force(vector6_t::Zero()));
    }

    template<typename T>
    std::enable_if_t<std::is_arithmetic<T>::value, T>
    FctPyWrapperArgToPython(T const & arg)
    {
        return arg;
    }

    template<typename T>
    std::enable_if_t<is_eigen<T>::value, bp::handle<> >
    FctPyWrapperArgToPython(T const & arg)
    {
        // Pass the arguments by reference (be careful const qualifiers are lost)
        return bp::handle<>(getNumpyReference(const_cast<T &>(arg)));
    }

    template<typename T>
    std::enable_if_t<std::is_same<T, sensorsDataMap_t>::value, boost::reference_wrapper<sensorsDataMap_t const> >
    FctPyWrapperArgToPython(T const & arg)
    {
        return boost::ref(arg);
    }

    template<typename OutputArg, typename ... InputArgs>
    struct FctPyWrapper
    {
    public:
        using OutputBufferType = typename DataInternalBufferType<OutputArg>::type;
    public:
        // Disable the copy of the class
        FctPyWrapper & operator = (FctPyWrapper const & other) = delete;

    public:
        FctPyWrapper(bp::object const & objPy) :
        funcPyPtr_(objPy),
        outPtr_(createInternalBuffer<OutputArg>()),
        outData_(setDataInternalBuffer(outPtr_)),
        outPyPtr_(nullptr)
        {
            outPyPtr_ = getNumpyReference(outData_);
        }

        // Copy constructor, same as the normal constructor
        FctPyWrapper(FctPyWrapper const & other) :
        funcPyPtr_(other.funcPyPtr_),
        outPtr_(createInternalBuffer<OutputArg>()),
        outData_(setDataInternalBuffer(outPtr_)),
        outPyPtr_(nullptr)
        {
            *outPtr_ = *(other.outPtr_);
            outPyPtr_ = getNumpyReference(outData_);
        }

        // Move constructor, takes a rvalue reference &&
        FctPyWrapper(FctPyWrapper&& other) :
        funcPyPtr_(other.funcPyPtr_),
        outPtr_(nullptr),
        outData_(other.outData_),
        outPyPtr_(nullptr)
        {
            // Steal the resource from "other"
            outPtr_ = other.outPtr_;
            outPyPtr_ = other.outPyPtr_;

            /* "other" will soon be destroyed and its destructor will
               do nothing because we null out its resource here */
            other.outPtr_ = nullptr;
            other.outPyPtr_ = nullptr;
        }

        // Destructor
        ~FctPyWrapper()
        {
            Py_XDECREF(outPyPtr_);
            delete outPtr_;
        }

        // Move assignment, takes a rvalue reference &&
        FctPyWrapper& operator = (FctPyWrapper&& other)
        {
            /* "other" is soon going to be destroyed, so we let it destroy our current resource
               instead and we take "other"'s current resource via swapping */
            std::swap(funcPyPtr_, other.funcPyPtr_);
            std::swap(outPtr_, other.outPtr_);
            std::swap(outData_, other.outData_);
            std::swap(outPyPtr_, other.outPyPtr_);
            return *this;
        }

        OutputArg const & operator() (InputArgs const & ... args)
        {
            bp::handle<> outPy(bp::borrowed(outPyPtr_));
            funcPyPtr_(FctPyWrapperArgToPython(args)..., outPy);
            return *outPtr_;
        }

    private:
        bp::object funcPyPtr_;
        OutputArg * outPtr_;
        OutputBufferType outData_;
        PyObject * outPyPtr_;
    };

    template<typename T>
    using TimeStateRefFctPyWrapper = FctPyWrapper<T /* OutputType */,
                                                  float64_t /* t */,
                                                  vectorN_t /* q */,
                                                  vectorN_t /* v */>;

    template<typename T>
    using TimeStateFctPyWrapper = FctPyWrapper<T /* OutputType */,
                                               float64_t /* t */,
                                               vectorN_t /* q */,
                                               vectorN_t /* v */>;

    template<typename T>
    using TimeBistateRefFctPyWrapper = FctPyWrapper<T /* OutputType */,
                                                    float64_t /* t */,
                                                    vectorN_t /* q1 */,
                                                    vectorN_t /* v1 */,
                                                    vectorN_t /* q2 */,
                                                    vectorN_t /* v2 */ >;

    // **************************** FctInOutPyWrapper *******************************

    template<typename OutputArg, typename ... InputArgs>
    struct FctInOutPyWrapper
    {
    public:
        FctInOutPyWrapper(bp::object const & objPy) : funcPyPtr_(objPy) {}
        void operator() (InputArgs const & ... argsIn,
                         vectorN_t       &     argOut)
        {
            funcPyPtr_(FctPyWrapperArgToPython(argsIn)...,
                       FctPyWrapperArgToPython(argOut));
        }
    private:
        bp::object funcPyPtr_;
    };

    using ControllerFctWrapper = FctInOutPyWrapper<vectorN_t /* OutputType */,
                                                   float64_t /* t */,
                                                   vectorN_t /* q */,
                                                   vectorN_t /* v */,
                                                   sensorsDataMap_t /* sensorsData*/>;

    using ControllerFct = std::function<void(float64_t        const & /* t */,
                                             vectorN_t        const & /* q */,
                                             vectorN_t        const & /* v */,
                                             sensorsDataMap_t const & /* sensorsData */,
                                             vectorN_t              & /* u */)>;

    // ************************** HeatMapFunctorPyWrapper ******************************

    enum class heatMapType_t : uint8_t
    {
        CONSTANT = 0x01,
        STAIRS   = 0x02,
        GENERIC  = 0x03,
    };

    struct HeatMapFunctorPyWrapper {
    public:
        // Disable the copy of the class
        HeatMapFunctorPyWrapper & operator = (HeatMapFunctorPyWrapper const & other) = delete;

    public:
        HeatMapFunctorPyWrapper(bp::object    const & objPy,
                                heatMapType_t const & objType) :
        heatMapType_(objType),
        handlePyPtr_(objPy),
        out1Ptr_(new float64_t),
        out2Ptr_(new vector3_t),
        out1PyPtr_(),
        out2PyPtr_()
        {
            if (heatMapType_ == heatMapType_t::CONSTANT)
            {
                *out1Ptr_ = bp::extract<float64_t>(handlePyPtr_);
                *out2Ptr_ = (vector3_t() << 0.0, 0.0, 1.0).finished();
            }
            else if (heatMapType_ == heatMapType_t::STAIRS)
            {
                out1PyPtr_ = getNumpyReference(*out1Ptr_);
                *out2Ptr_ = (vector3_t() << 0.0, 0.0, 1.0).finished();
            }
            else if (heatMapType_ == heatMapType_t::GENERIC)
            {
                out1PyPtr_ = getNumpyReference(*out1Ptr_);
                out2PyPtr_ = getNumpyReference(*out2Ptr_);
            }
        }

        // Copy constructor, same as the normal constructor
        HeatMapFunctorPyWrapper(HeatMapFunctorPyWrapper const & other) :
        heatMapType_(other.heatMapType_),
        handlePyPtr_(other.handlePyPtr_),
        out1Ptr_(new float64_t),
        out2Ptr_(new vector3_t),
        out1PyPtr_(),
        out2PyPtr_()
        {
            *out1Ptr_ = *(other.out1Ptr_);
            *out2Ptr_ = *(other.out2Ptr_);
            out1PyPtr_ = getNumpyReference(*out1Ptr_);
            out2PyPtr_ = getNumpyReference(*out2Ptr_);
        }

        // Move constructor, takes a rvalue reference &&
        HeatMapFunctorPyWrapper(HeatMapFunctorPyWrapper && other) :
        heatMapType_(other.heatMapType_),
        handlePyPtr_(other.handlePyPtr_),
        out1Ptr_(nullptr),
        out2Ptr_(nullptr),
        out1PyPtr_(nullptr),
        out2PyPtr_(nullptr)
        {
            // Steal the resource from "other"
            out1Ptr_ = other.out1Ptr_;
            out2Ptr_ = other.out2Ptr_;
            out1PyPtr_ = other.out1PyPtr_;
            out2PyPtr_ = other.out2PyPtr_;

            /* "other" will soon be destroyed and its destructor will
               do nothing because we null out its resource here */
            other.out1Ptr_ = nullptr;
            other.out2Ptr_ = nullptr;
            other.out1PyPtr_ = nullptr;
            other.out2PyPtr_ = nullptr;
        }

        // Destructor
        ~HeatMapFunctorPyWrapper()
        {
            Py_XDECREF(out1PyPtr_);
            Py_XDECREF(out2PyPtr_);
            delete out1Ptr_;
            delete out2Ptr_;
        }

        // Move assignment, takes a rvalue reference &&
        HeatMapFunctorPyWrapper& operator = (HeatMapFunctorPyWrapper&& other)
        {
            /* "other" is soon going to be destroyed, so we let it destroy our current resource
               instead and we take "other"'s current resource via swapping */
            std::swap(heatMapType_, other.heatMapType_);
            std::swap(handlePyPtr_, other.handlePyPtr_);
            std::swap(out1Ptr_, other.out1Ptr_);
            std::swap(out2Ptr_, other.out2Ptr_);
            std::swap(out1PyPtr_, other.out1PyPtr_);
            std::swap(out2PyPtr_, other.out2PyPtr_);
            return *this;
        }

        std::pair<float64_t, vector3_t> operator() (vector3_t const & posFrame)
        {
            // Pass the arguments by reference (be careful const qualifiers are lost)

            if (heatMapType_ == heatMapType_t::STAIRS)
            {
                bp::handle<> out1Py(bp::borrowed(out1PyPtr_));
                handlePyPtr_(posFrame[0], posFrame[1], out1Py);
            }
            else if (heatMapType_ == heatMapType_t::GENERIC)
            {
                bp::handle<> out1Py(bp::borrowed(out1PyPtr_));
                bp::handle<> out2Py(bp::borrowed(out2PyPtr_));
                handlePyPtr_(posFrame[0], posFrame[1], out1Py, out2Py);
            }

            return {*out1Ptr_, *out2Ptr_};
        }

    private:
        heatMapType_t heatMapType_;
        bp::object handlePyPtr_;
        float64_t * out1Ptr_;
        vector3_t * out2Ptr_;
        PyObject * out1PyPtr_;
        PyObject * out2PyPtr_;
    };

    // **************************** HeatMapFunctorVisitor *****************************

    struct HeatMapFunctorVisitor
        : public bp::def_visitor<HeatMapFunctorVisitor>
    {
    public:
        ///////////////////////////////////////////////////////////////////////////////
        /// \brief Expose C++ API through the visitor.
        ///////////////////////////////////////////////////////////////////////////////
        template<class PyClass>
        void visit(PyClass& cl) const
        {
            cl
                .def("__init__", bp::make_constructor(&HeatMapFunctorVisitor::factory,
                                 bp::default_call_policies(),
                                (bp::args("heatmap_function", "heatmap_type"))))
                .def("__call__", &HeatMapFunctorVisitor::eval,
                                 (bp::arg("self"), bp::arg("position")))
                ;
        }

        static bp::tuple eval(heatMapFunctor_t       & self,
                              vector3_t        const & posFrame)
        {
            std::pair<float64_t, vector3_t> ground = self(posFrame);
            return bp::make_tuple(std::move(std::get<0>(ground)), std::move(std::get<1>(ground)));
        }

        static std::shared_ptr<heatMapFunctor_t> factory(bp::object          & objPy,
                                                         heatMapType_t const & objType)
        {
            return std::make_shared<heatMapFunctor_t>(HeatMapFunctorPyWrapper(std::move(objPy), objType));
        }

        ///////////////////////////////////////////////////////////////////////////////
        /// \brief Expose.
        ///////////////////////////////////////////////////////////////////////////////
        static void expose()
        {
            bp::class_<heatMapFunctor_t,
                       std::shared_ptr<heatMapFunctor_t> >("HeatMapFunctor", bp::no_init)
                .def(HeatMapFunctorVisitor());
        }
    };

    // ******************************* sensorsDataMap_t ********************************

    struct SensorsDataMapVisitor
        : public bp::def_visitor<SensorsDataMapVisitor>
    {
    public:
        ///////////////////////////////////////////////////////////////////////////////
        /// \brief Expose C++ API through the visitor.
        ///////////////////////////////////////////////////////////////////////////////
        template<class PyClass>
        void visit(PyClass& cl) const
        {
            cl
                .def("__init__", &SensorsDataMapVisitor::factoryWrapper,
                                 bp::with_custodian_and_ward_postcall<1, 2>(),
                                 (bp::arg("self"), "sensors_data_dict")) // with_custodian_and_ward_postcall is used to tie the lifetime of the Python object with the one of the C++ reference, so that the Python object does not get deleted while the C++ object is not
                .def("__len__", &SensorsDataMapVisitor::len,
                                (bp::arg("self")))
                .def("__getitem__", &SensorsDataMapVisitor::getItem,
                                    (bp::arg("self"), "(sensor_type, sensor_name)"))
                .def("__getitem__", &SensorsDataMapVisitor::getItemSplit,
                                    (bp::arg("self"), "sensor_type", "sensor_name"))
                .def("__getitem__", &SensorsDataMapVisitor::getSub,
                                    (bp::arg("self"), "sensor_type"))
                .def("__iter__", bp::iterator<sensorsDataMap_t>())
                .def("__contains__", &SensorsDataMapVisitor::contains,
                                     (bp::arg("self"), "key"))
                .def("__repr__", &SensorsDataMapVisitor::repr)
                .def("keys", &SensorsDataMapVisitor::keys,
                             (bp::arg("self")))
                .def("keys", &SensorsDataMapVisitor::keysSensorType,
                             (bp::arg("self"), "sensor_type"))
                .def("values", &SensorsDataMapVisitor::values,
                               (bp::arg("self")))
                .def("items", &SensorsDataMapVisitor::items,
                              (bp::arg("self")))
                ;
        }

        static uint32_t len(sensorsDataMap_t & self)
        {
            return self.size();
        }

        static bp::object getItem(sensorsDataMap_t        & self,
                                  bp::tuple         const & sensorInfo)
        {
            std::string const sensorType = bp::extract<std::string>(sensorInfo[0]);
            std::string const sensorName = bp::extract<std::string>(sensorInfo[1]);
            return SensorsDataMapVisitor::getItemSplit(self, sensorType, sensorName);
        }

        static bp::object getItemSplit(sensorsDataMap_t       & self,
                                       std::string      const & sensorType,
                                       std::string      const & sensorName)
        {
            try
            {
                auto & sensorsDataTypeByName = self.at(sensorType).get<IndexByName>();
                auto sensorDataIt = sensorsDataTypeByName.find(sensorName);
                Eigen::Ref<vectorN_t const> const & sensorDataValue = sensorDataIt->value;
                bp::handle<> valuePy(getNumpyReference(sensorDataValue));
                return bp::object(valuePy);
            }
            catch (...)
            {
                PyErr_SetString(PyExc_KeyError, "This combination of keys does not exist.");
                return bp::object();  // Return None
            }
        }

        static bp::object getSub(sensorsDataMap_t       & self,
                                 std::string      const & sensorType)
        {
            try
            {
                auto & sensorsDataType = self.at(sensorType);
                bp::handle<> valuePy(getNumpyReference(sensorsDataType.getAll()));
                return bp::object(valuePy);
            }
            catch (...)
            {
                PyErr_SetString(PyExc_KeyError, "This key does not exist.");
                return bp::object();  // Return None
            }
        }

        static bool_t contains(sensorsDataMap_t       & self,
                               bp::tuple        const & sensorInfo)
        {
            std::string const sensorType = bp::extract<std::string>(sensorInfo[0]);
            std::string const sensorName = bp::extract<std::string>(sensorInfo[1]);
            auto const & sensorsDataType = self.find(sensorType);
            if (sensorsDataType != self.end())
            {
                auto & sensorsDataTypeByName = sensorsDataType->second.get<IndexByName>();
                auto sensorDataIt = sensorsDataTypeByName.find(sensorName);
                if (sensorDataIt != sensorsDataTypeByName.end())
                {
                    return true;
                }
            }
            return false;
        }

        static bp::list keys(sensorsDataMap_t & self)
        {
            bp::list sensorsInfo;
            for (auto & sensorData : self)
            {
                sensorsInfo.append(sensorData.first);
            }
            return sensorsInfo;
        }

        static bp::list keysSensorType(sensorsDataMap_t & self,
                                       std::string const& sensorType)
        {
            bp::list sensorsInfo;
            for (auto & sensorData : self.at(sensorType))
            {
                sensorsInfo.append(sensorData.name);
            }
            return sensorsInfo;
        }

        static bp::list values(sensorsDataMap_t & self)
        {
            bp::list sensorsValue;
            for (auto const & sensorsDataType : self)
            {
                bp::handle<> valuePy(getNumpyReference(sensorsDataType.second.getAll()));
                sensorsValue.append(bp::object(valuePy));
            }
            return sensorsValue;
        }

        static bp::list items(sensorsDataMap_t & self)
        {
            bp::list sensorsDataPy;
            for (auto const & sensorsDataType : self)
            {
                bp::handle<> valuePy(getNumpyReference(sensorsDataType.second.getAll()));
                sensorsDataPy.append(bp::make_tuple(sensorsDataType.first, bp::object(valuePy)));
            }
            return sensorsDataPy;
        }

        static std::string repr(sensorsDataMap_t & self)
        {
            std::stringstream s;
            Eigen::IOFormat HeavyFmt(5, 1, ", ", "", "", "", "[", "]\n");

            for (auto const & sensorsDataType : self)
            {
                std::string const & sensorTypeName = sensorsDataType.first;
                s << sensorTypeName << ":\n";
                for (auto const & sensorData : sensorsDataType.second)
                {
                    std::string const & sensorName = sensorData.name;
                    int32_t const & sensorIdx = sensorData.idx;
                    Eigen::Ref<vectorN_t const> const & sensorDataValue = sensorData.value;
                    s << "    (" << sensorIdx << ") " <<  sensorName << ": "
                      << sensorDataValue.transpose().format(HeavyFmt);
                }
            }
            return s.str();
        }

        static std::shared_ptr<sensorsDataMap_t> factory(bp::object & sensorDataPy)
        {
            auto sensorData = convertFromPython<sensorsDataMap_t>(sensorDataPy);
            return std::make_shared<sensorsDataMap_t>(std::move(sensorData));
        }

        static void factoryWrapper(bp::object & self, bp::object & sensorDataPy)
        {
            auto constructor = bp::make_constructor(&SensorsDataMapVisitor::factory);
            constructor(self, sensorDataPy);
        }

        ///////////////////////////////////////////////////////////////////////////////
        /// \brief Expose.
        ///////////////////////////////////////////////////////////////////////////////
        static void expose()
        {
            bp::class_<sensorsDataMap_t,
                       std::shared_ptr<sensorsDataMap_t>,
                       boost::noncopyable>("sensorsData", bp::no_init)
                .def(SensorsDataMapVisitor());
        }
    };

    // ***************************** PyMotorVisitor ***********************************

    struct PyMotorVisitor
        : public bp::def_visitor<PyMotorVisitor>
    {
    public:
        ///////////////////////////////////////////////////////////////////////////////
        /// \brief Expose C++ API through the visitor.
        ///////////////////////////////////////////////////////////////////////////////

        template<class PyClass>
        class PyMotorVisit
        {
        public:
            using TMotor = typename PyClass::wrapped_type;

            static void visitAbstract(PyClass& cl)
            {
                cl
                    .def("set_options", &PyMotorVisitor::setOptions<TMotor>)
                    .def("get_options", &AbstractMotorBase::getOptions,
                                        bp::return_value_policy<bp::return_by_value>())

                    .add_property("is_initialized", bp::make_function(&AbstractMotorBase::getIsInitialized,
                                                    bp::return_value_policy<bp::copy_const_reference>()))
                    .add_property("name", bp::make_function(&AbstractMotorBase::getName,
                                          bp::return_value_policy<bp::copy_const_reference>()))
                    .add_property("idx", bp::make_function(&AbstractMotorBase::getIdx,
                                         bp::return_value_policy<bp::copy_const_reference>()))
                    .add_property("joint_name", bp::make_function(&AbstractMotorBase::getJointName,
                                                bp::return_value_policy<bp::copy_const_reference>()))
                    .add_property("joint_idx", bp::make_function(&AbstractMotorBase::getJointModelIdx,
                                               bp::return_value_policy<bp::copy_const_reference>()))
                    .add_property("joint_type", bp::make_function(&AbstractMotorBase::getJointType,
                                                bp::return_value_policy<bp::copy_const_reference>()))
                    .add_property("joint_position_idx", bp::make_function(&AbstractMotorBase::getJointPositionIdx,
                                                        bp::return_value_policy<bp::copy_const_reference>()))
                    .add_property("joint_velocity_idx", bp::make_function(&AbstractMotorBase::getJointVelocityIdx,
                                                        bp::return_value_policy<bp::copy_const_reference>()))
                    .add_property("effort_limit", bp::make_function(&AbstractMotorBase::getEffortLimit,
                                                  bp::return_value_policy<bp::copy_const_reference>()))
                    .add_property("rotor_inertia", bp::make_function(&AbstractMotorBase::getRotorInertia,
                                                   bp::return_value_policy<bp::copy_const_reference>()))
                    ;
            }

            template<class Q = TMotor>
            static std::enable_if_t<!std::is_same<Q, AbstractMotorBase>::value, void>
            visit(PyClass& cl)
            {
                visitAbstract(cl);

                cl
                    .def("initialize", &TMotor::initialize)
                    ;
            }

            template<class Q = TMotor>
            static std::enable_if_t<std::is_same<Q, AbstractMotorBase>::value, void>
            visit(PyClass& cl)
            {
                visitAbstract(cl);
            }
        };

    public:
        template<class PyClass>
        void visit(PyClass& cl) const
        {
            PyMotorVisit<PyClass>::visit(cl);
        }

        ///////////////////////////////////////////////////////////////////////////////
        /// \brief      Getters and Setters
        ///////////////////////////////////////////////////////////////////////////////

        template<typename TMotor>
        static void setOptions(TMotor         & self,
                               bp::dict const & configPy)
        {
            configHolder_t config = self.getOptions();
            convertFromPython(configPy, config);
            self.setOptions(config);
        }

        ///////////////////////////////////////////////////////////////////////////////
        /// \brief Expose.
        ///////////////////////////////////////////////////////////////////////////////
        static void expose()
        {
            bp::class_<AbstractMotorBase,
                       std::shared_ptr<AbstractMotorBase>,
                       boost::noncopyable>("AbstractMotor", bp::no_init)
                .def(PyMotorVisitor());

            bp::class_<SimpleMotor, bp::bases<AbstractMotorBase>,
                       std::shared_ptr<SimpleMotor>,
                       boost::noncopyable>("SimpleMotor", bp::init<std::string>())
                .def(PyMotorVisitor());
        }
    };

    // ***************************** PyConstraintVisitor ***********************************

    struct PyConstraintVisitor
        : public bp::def_visitor<PyConstraintVisitor>
    {
    public:
        ///////////////////////////////////////////////////////////////////////////////
        /// \brief Expose C++ API through the visitor.
        ///////////////////////////////////////////////////////////////////////////////

        template<class PyClass>
        class PyConstraintVisit
        {
        public:
            using TConstraint = typename PyClass::wrapped_type;

            static void visit(PyClass& cl)
            {
                cl
                    .add_property("jacobian", bp::make_function(&AbstractConstraint::getJacobian,
                                              bp::return_internal_reference<>()))
                    .add_property("drift", bp::make_function(&AbstractConstraint::getDrift,
                                           bp::return_internal_reference<>()))
                    ;
            }
        };

    public:
        template<class PyClass>
        void visit(PyClass& cl) const
        {
            PyConstraintVisit<PyClass>::visit(cl);
        }

        ///////////////////////////////////////////////////////////////////////////////
        /// \brief Expose.
        ///////////////////////////////////////////////////////////////////////////////
        static void expose()
        {
            bp::class_<AbstractConstraint,
                       std::shared_ptr<AbstractConstraint>,
                       boost::noncopyable>("AbstractConstraint", bp::no_init)
                .def(PyConstraintVisitor());

            bp::class_<FixedFrameConstraint, bp::bases<AbstractConstraint>,
                       std::shared_ptr<FixedFrameConstraint>,
                       boost::noncopyable>("FixedFrameConstraint", bp::init<std::string>())
                .def(PyConstraintVisitor());

            bp::class_<WheelConstraint, bp::bases<AbstractConstraint>,
                       std::shared_ptr<WheelConstraint>,
                       boost::noncopyable>("WheelConstraint", bp::init<std::string, float64_t, vector3_t, vector3_t>())
                .def(PyConstraintVisitor());
        }
    };

    // ***************************** PySensorVisitor ***********************************

    struct PySensorVisitor
        : public bp::def_visitor<PySensorVisitor>
    {
    public:
        ///////////////////////////////////////////////////////////////////////////////
        /// \brief Expose C++ API through the visitor.
        ///////////////////////////////////////////////////////////////////////////////

        template<class PyClass>
        class PySensorVisit
        {
        public:
            using TSensor = typename PyClass::wrapped_type;

            static void visitAbstract(PyClass& cl)
            {
                cl
                    .def("set_options", &PySensorVisitor::setOptions<TSensor>)
                    .def("get_options", &AbstractSensorBase::getOptions,
                                        bp::return_value_policy<bp::return_by_value>())

                    .add_property("is_initialized", bp::make_function(&AbstractSensorBase::getIsInitialized,
                                                    bp::return_value_policy<bp::copy_const_reference>()))
                    .add_property("name", bp::make_function(&AbstractSensorBase::getName,
                                          bp::return_value_policy<bp::copy_const_reference>()))
                    .add_property("idx", bp::make_function(&AbstractSensorBase::getIdx,
                                        bp::return_value_policy<bp::copy_const_reference>()))
                    .add_property("data", &PySensorVisit::getData)
                    .def("__repr__", &PySensorVisit::repr)
                    ;
            }

            static bp::object getData(AbstractSensorBase & self)
            {
                // Be careful, it removes the const qualifier, so that the data can be modified from Python
                Eigen::Ref<vectorN_t const> const & sensorDataValue = const_cast<AbstractSensorBase const &>(self).get();
                bp::handle<> valuePy(getNumpyReference(sensorDataValue));
                return bp::object(valuePy);
            }

            static std::string repr(AbstractSensorBase & self)
            {
                std::stringstream s;
                s << "type: " << self.getType() << "\n";
                s << "name: " << self.getName() << "\n";
                s << "idx: " << self.getIdx() << "\n";
                s << "data:\n    ";
                std::vector<std::string> const & fieldnames = self.getFieldnames();
                Eigen::Ref<vectorN_t const> const & sensorDataValue = const_cast<AbstractSensorBase const &>(self).get();
                for (uint32_t i=0; i<fieldnames.size(); ++i)
                {
                    std::string const & field = fieldnames[i];
                    float64_t const & value = sensorDataValue[i];
                    if (i > 0)
                    {
                       s << ", ";
                    }
                    s << field << ": " << value;
                }
                return s.str();
            }

            template<class Q = TSensor>
            static std::enable_if_t<std::is_same<Q, AbstractSensorBase>::value, void>
            visit(PyClass& cl)
            {
                visitAbstract(cl);

                cl
                    .add_property("type", bp::make_function(&AbstractSensorBase::getType,
                                          bp::return_value_policy<bp::copy_const_reference>()))
                    .add_property("fieldnames", bp::make_function(&AbstractSensorBase::getFieldnames,
                                                bp::return_value_policy<bp::return_by_value>()))
                    ;
            }

            template<class Q = TSensor>
            static std::enable_if_t<!std::is_same<Q, AbstractSensorBase>::value, void>
            visitBasicSensors(PyClass& cl)
            {
                visitAbstract(cl);

                cl
                    .def("initialize", &TSensor::initialize)
                    .def_readonly("type", &TSensor::type_)
                    .def_readonly("has_prefix", &TSensor::areFieldnamesGrouped_)
                    .add_static_property("fieldnames", bp::make_getter(&TSensor::fieldNames_,
                                                       bp::return_value_policy<bp::return_by_value>()))
                    ;
            }

            template<class Q = TSensor>
            static std::enable_if_t<std::is_same<Q, ImuSensor>::value
                                 || std::is_same<Q, ContactSensor>::value, void>
            visit(PyClass& cl)
            {
                visitAbstract(cl);
                visitBasicSensors(cl);

                cl
                    .add_property("frame_name", bp::make_function(&TSensor::getFrameName,
                                                bp::return_value_policy<bp::copy_const_reference>()))
                    .add_property("frame_idx", bp::make_function(&TSensor::getFrameIdx,
                                               bp::return_value_policy<bp::copy_const_reference>()))
                    ;
            }

            template<class Q = TSensor>
            static std::enable_if_t<std::is_same<Q, ForceSensor>::value, void>
            visit(PyClass& cl)
            {
                visitAbstract(cl);
                visitBasicSensors(cl);

                cl
                    .add_property("frame_name", bp::make_function(&TSensor::getFrameName,
                                                bp::return_value_policy<bp::copy_const_reference>()))
                    .add_property("frame_idx", bp::make_function(&TSensor::getFrameIdx,
                                               bp::return_value_policy<bp::copy_const_reference>()))
                    .add_property("joint_idx", bp::make_function(&TSensor::getJointIdx,
                                               bp::return_value_policy<bp::return_by_value>()))
                    ;
            }

            template<class Q = TSensor>
            static std::enable_if_t<std::is_same<Q, EncoderSensor>::value, void>
            visit(PyClass& cl)
            {
                visitAbstract(cl);
                visitBasicSensors(cl);

                cl
                    .add_property("joint_name", bp::make_function(&EncoderSensor::getJointName,
                                                bp::return_value_policy<bp::copy_const_reference>()))
                    .add_property("joint_idx", bp::make_function(&EncoderSensor::getJointIdx,
                                               bp::return_value_policy<bp::copy_const_reference>()))
                    .add_property("joint_type", bp::make_function(&EncoderSensor::getJointType,
                                                bp::return_value_policy<bp::copy_const_reference>()))
                    ;
            }

            template<class Q = TSensor>
            static std::enable_if_t<std::is_same<Q, EffortSensor>::value, void>
            visit(PyClass& cl)
            {
                visitAbstract(cl);
                visitBasicSensors(cl);

                cl
                    .add_property("motor_name", bp::make_function(&EffortSensor::getMotorName,
                                                bp::return_value_policy<bp::copy_const_reference>()))
                    .add_property("motor_idx", bp::make_function(&EffortSensor::getMotorIdx,
                                               bp::return_value_policy<bp::copy_const_reference>()))
                    ;
            }
        };

    public:
        template<class PyClass>
        void visit(PyClass& cl) const
        {
            PySensorVisit<PyClass>::visit(cl);
        }

        ///////////////////////////////////////////////////////////////////////////////
        /// \brief      Getters and Setters
        ///////////////////////////////////////////////////////////////////////////////

        template<typename TSensor>
        static void setOptions(TSensor        & self,
                               bp::dict const & configPy)
        {
            configHolder_t config = self.getOptions();
            convertFromPython(configPy, config);
            self.setOptions(config);
        }

        ///////////////////////////////////////////////////////////////////////////////
        /// \brief Expose.
        ///////////////////////////////////////////////////////////////////////////////
        static void expose()
        {
            bp::class_<AbstractSensorBase,
                       std::shared_ptr<AbstractSensorBase>,
                       boost::noncopyable>("AbstractSensor", bp::no_init)
                .def(PySensorVisitor());

            bp::class_<ImuSensor, bp::bases<AbstractSensorBase>,
                       std::shared_ptr<ImuSensor>,
                       boost::noncopyable>("ImuSensor", bp::init<std::string>())
                .def(PySensorVisitor());

            bp::class_<ContactSensor, bp::bases<AbstractSensorBase>,
                       std::shared_ptr<ContactSensor>,
                       boost::noncopyable>("ContactSensor", bp::init<std::string>())
                .def(PySensorVisitor());

            bp::class_<ForceSensor, bp::bases<AbstractSensorBase>,
                       std::shared_ptr<ForceSensor>,
                       boost::noncopyable>("ForceSensor", bp::init<std::string>())
                .def(PySensorVisitor());

            bp::class_<EncoderSensor, bp::bases<AbstractSensorBase>,
                       std::shared_ptr<EncoderSensor>,
                       boost::noncopyable>("EncoderSensor", bp::init<std::string>())
                .def(PySensorVisitor());

            bp::class_<EffortSensor, bp::bases<AbstractSensorBase>,
                       std::shared_ptr<EffortSensor>,
                       boost::noncopyable>("EffortSensor", bp::init<std::string>())
                .def(PySensorVisitor());
        }
    };

    // ***************************** PyModelVisitor ***********************************

    struct PyModelVisitor
        : public bp::def_visitor<PyModelVisitor>
    {
    public:
        ///////////////////////////////////////////////////////////////////////////////
        /// \brief Expose C++ API through the visitor.
        ///////////////////////////////////////////////////////////////////////////////
        template<class PyClass>
        void visit(PyClass& cl) const
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

    struct PyRobotVisitor
        : public bp::def_visitor<PyRobotVisitor>
    {
    public:
        ///////////////////////////////////////////////////////////////////////////////
        /// \brief Expose C++ API through the visitor.
        ///////////////////////////////////////////////////////////////////////////////
        template<class PyClass>
        void visit(PyClass& cl) const
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

    // ***************************** PyAbstractControllerVisitor ***********************************

    struct PyAbstractControllerVisitor
        : public bp::def_visitor<PyAbstractControllerVisitor>
    {
    public:
        ///////////////////////////////////////////////////////////////////////////////
        /// \brief Expose C++ API through the visitor.
        ///////////////////////////////////////////////////////////////////////////////
        template<class PyClass>
        void visit(PyClass& cl) const
        {
            cl
                .def("initialize", &PyAbstractControllerVisitor::initialize,
                                   (bp::arg("self"), "robot"))
                .add_property("is_initialized", bp::make_function(&AbstractController::getIsInitialized,
                                                bp::return_value_policy<bp::copy_const_reference>()))
                .def("register_variable", &PyAbstractControllerVisitor::registerVariable,
                                          (bp::arg("self"), "fieldname", "value"),
                                          "@copydoc AbstractController::registerVariable")
                .def("register_variables", &PyAbstractControllerVisitor::registerVariableVector,
                                           (bp::arg("self"), "fieldnames", "values"))
                .def("register_constants", &PyAbstractControllerVisitor::registerConstant,
                                           (bp::arg("self"), "fieldnames", "values"))
                .def("remove_entries", &AbstractController::removeEntries)
                .def("set_options", &PyAbstractControllerVisitor::setOptions)
                .def("get_options", &AbstractController::getOptions,
                                    bp::return_value_policy<bp::return_by_value>())
                ;
        }

        static void initialize(AbstractController           & self,
                               std::shared_ptr<Robot> const & robot)
        {
            self.initialize(robot.get());
        }

        static hresult_t registerVariable(AbstractController       & self,
                                          std::string        const & fieldName,
                                          PyObject                 * dataPy)
        {
            // Note that const qualifier is not supported by PyArray_DATA

            if (PyArray_Check(dataPy))
            {
                PyArrayObject * dataPyArray = reinterpret_cast<PyArrayObject *>(dataPy);
                if (PyArray_TYPE(dataPyArray) == NPY_FLOAT64 && PyArray_SIZE(dataPyArray) == 1U)
                {
                    float64_t const * data = (float64_t *) PyArray_DATA(dataPyArray);
                    return self.registerVariable(fieldName, *data);
                }
                else
                {
                    PRINT_ERROR("'value' input array must have dtype 'np.float64' and a single element.");
                    return hresult_t::ERROR_BAD_INPUT;
                }
            }
            else
            {
                PRINT_ERROR("'value' input must have type 'numpy.ndarray'.");
                return hresult_t::ERROR_BAD_INPUT;
            }
        }

        static hresult_t registerVariableVector(AbstractController       & self,
                                                bp::list           const & fieldNamesPy,
                                                PyObject                 * dataPy)
        {
            // Note that const qualifier is not supported by PyArray_DATA

            if (PyArray_Check(dataPy))
            {
                auto fieldnames = convertFromPython<std::vector<std::string> >(fieldNamesPy);
                PyArrayObject * dataPyArray = reinterpret_cast<PyArrayObject *>(dataPy);
                if (PyArray_TYPE(dataPyArray) == NPY_FLOAT64 && PyArray_SIZE(dataPyArray) == uint32_t(fieldnames.size()))
                {
                    Eigen::Map<vectorN_t> data((float64_t *) PyArray_DATA(dataPyArray), PyArray_SIZE(dataPyArray));
                    return self.registerVariable(fieldnames, data);
                }
                else
                {
                    PRINT_ERROR("'values' input array must have dtype 'np.float64' and the same length as 'fieldnames'.");
                    return hresult_t::ERROR_BAD_INPUT;
                }
            }
            else
            {
                PRINT_ERROR("'values' input must have type 'numpy.ndarray'.");
                return hresult_t::ERROR_BAD_INPUT;
            }
        }

        static hresult_t registerConstant(AbstractController       & self,
                                          std::string        const & fieldName,
                                          PyObject                 * dataPy)
        {
            if (PyArray_Check(dataPy))
            {
                PyArrayObject * dataPyArray = reinterpret_cast<PyArrayObject *>(dataPy);

                int dataPyArrayDtype = PyArray_TYPE(dataPyArray);
                if (dataPyArrayDtype != NPY_FLOAT64)
                {
                    PRINT_ERROR("The only dtype supported for 'numpy.ndarray' is float.");
                    return hresult_t::ERROR_BAD_INPUT;
                }
                float64_t * dataPyArrayData = (float64_t *) PyArray_DATA(dataPyArray);
                int dataPyArrayNdims = PyArray_NDIM(dataPyArray);
                npy_intp * dataPyArrayShape = PyArray_SHAPE(dataPyArray);
                if (dataPyArrayNdims == 0)
                {
                    return self.registerConstant(fieldName, *dataPyArrayData);
                }
                else if (dataPyArrayNdims == 1)
                {
                    Eigen::Map<vectorN_t> data(dataPyArrayData, dataPyArrayShape[0]);
                    return self.registerConstant(fieldName, data);
                }
                else if (dataPyArrayNdims == 2)
                {
                    Eigen::Map<matrixN_t> data(dataPyArrayData, dataPyArrayShape[0], dataPyArrayShape[1]);
                    return self.registerConstant(fieldName, data);
                }
                else
                {
                    PRINT_ERROR("The max number of dims supported for 'numpy.ndarray' is 2.");
                    return hresult_t::ERROR_BAD_INPUT;
                }
            }
            else if (PyFloat_Check(dataPy))
            {
                return self.registerConstant(fieldName, PyFloat_AsDouble(dataPy));
            }
            else if (PyLong_Check(dataPy))
            {
                return self.registerConstant(fieldName, PyLong_AsLong(dataPy));
            }
            else if (PyBytes_Check(dataPy))
            {
                return self.registerConstant(fieldName, PyBytes_AsString(dataPy));
            }
            else if (PyUnicode_Check(dataPy))
            {
                return self.registerConstant(fieldName, PyUnicode_AsUTF8(dataPy));
            }
            else
            {
                PRINT_ERROR("'value' type is unsupported.");
                return hresult_t::ERROR_BAD_INPUT;
            }
        }

        static void setOptions(AbstractController       & self,
                               bp::dict           const & configPy)
        {
            configHolder_t config = self.getOptions();
            convertFromPython(configPy, config);
            self.setOptions(config);
        }

        ///////////////////////////////////////////////////////////////////////////////
        /// \brief Expose.
        ///////////////////////////////////////////////////////////////////////////////
        static void expose()
        {
            bp::class_<AbstractController,
                       std::shared_ptr<AbstractController>,
                       boost::noncopyable>("AbstractController", bp::no_init)
                .def(PyAbstractControllerVisitor());
        }
    };

    // ***************************** PyControllerFunctorVisitor ***********************************

    struct PyControllerFunctorVisitor
        : public bp::def_visitor<PyControllerFunctorVisitor>
    {
    public:
        /* Take advantage of type erasure of std::function to support both
           lambda functions and python handle wrapper depending whether or not
           'compute_command' and 'internal_dynamics' has been specified.
           It is likely to cause a small overhead because the compiler will
           probably not be able to inline ControllerFctWrapper, as it would have
           been the case otherwise, but it is the price to pay for versatility. */
        using CtrlFunctor = ControllerFunctor<ControllerFct, ControllerFct>;

    public:
        ///////////////////////////////////////////////////////////////////////////////
        /// \brief Expose C++ API through the visitor.
        ///////////////////////////////////////////////////////////////////////////////
        template<class PyClass>
        void visit(PyClass& cl) const
        {
            cl
                .def("__init__", bp::make_constructor(&PyControllerFunctorVisitor::factory,
                                 bp::default_call_policies(),
                                (bp::arg("compute_command") = bp::object(),  // bp::object() means 'None' in Python
                                 bp::arg("internal_dynamics") = bp::object())))
                ;
        }

        static std::shared_ptr<CtrlFunctor> factory(bp::object & commandPy,
                                                    bp::object & internalDynamicsPy)
        {
            ControllerFct commandFct;
            if (!commandPy.is_none())
            {
                commandFct = ControllerFctWrapper(commandPy);
            }
            else
            {
                commandFct = [](float64_t        const & t,
                                vectorN_t        const & q,
                                vectorN_t        const & v,
                                sensorsDataMap_t const & sensorsData,
                                vectorN_t              & uCommand) {};
            }
            ControllerFct internalDynamicsFct;
            if (!internalDynamicsPy.is_none())
            {
                internalDynamicsFct = ControllerFctWrapper(internalDynamicsPy);
            }
            else
            {
                internalDynamicsFct = [](float64_t        const & t,
                                         vectorN_t        const & q,
                                         vectorN_t        const & v,
                                         sensorsDataMap_t const & sensorsData,
                                         vectorN_t              & uCommand) {};
            }
            return std::make_shared<CtrlFunctor>(std::move(commandFct),
                                                 std::move(internalDynamicsFct));
        }

        ///////////////////////////////////////////////////////////////////////////////
        /// \brief Expose.
        ///////////////////////////////////////////////////////////////////////////////
        static void expose()
        {
            bp::class_<CtrlFunctor, bp::bases<AbstractController>,
                       std::shared_ptr<CtrlFunctor>,
                       boost::noncopyable>("ControllerFunctor", bp::no_init)
                .def(PyControllerFunctorVisitor());
        }
    };

    // ***************************** PyStepperStateVisitor ***********************************

    struct PyStepperStateVisitor
        : public bp::def_visitor<PyStepperStateVisitor>
    {
    public:
        ///////////////////////////////////////////////////////////////////////////////
        /// \brief Expose C++ API through the visitor.
        ///////////////////////////////////////////////////////////////////////////////
        template<class PyClass>
        void visit(PyClass& cl) const
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

        static bp::object getPosition(stepperState_t & self)
        {
            return convertToPython<std::vector<vectorN_t> >(self.qSplit);
        }

        static bp::object getVelocity(stepperState_t & self)
        {
            return convertToPython<std::vector<vectorN_t> >(self.vSplit);
        }

        static bp::object getAcceleration(stepperState_t & self)
        {
            return convertToPython<std::vector<vectorN_t> >(self.aSplit);
        }

        static std::string repr(stepperState_t & self)
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

    // ***************************** PySystemStateVisitor ***********************************

    struct PySystemStateVisitor
        : public bp::def_visitor<PySystemStateVisitor>
    {
    public:
        ///////////////////////////////////////////////////////////////////////////////
        /// \brief Expose C++ API through the visitor.
        ///////////////////////////////////////////////////////////////////////////////
        template<class PyClass>
        void visit(PyClass& cl) const
        {
            cl
                .add_property("q", bp::make_getter(&systemState_t::q,
                                   bp::return_internal_reference<>()))
                .add_property("v", bp::make_getter(&systemState_t::v,
                                   bp::return_internal_reference<>()))
                .add_property("a", bp::make_getter(&systemState_t::a,
                                   bp::return_internal_reference<>()))
                .add_property("u", bp::make_getter(&systemState_t::u,
                                   bp::return_internal_reference<>()))
                .add_property("u_motor", bp::make_getter(&systemState_t::uMotor,
                                         bp::return_internal_reference<>()))
                .add_property("u_command", bp::make_getter(&systemState_t::uCommand,
                                           bp::return_internal_reference<>()))
                .add_property("u_internal", bp::make_getter(&systemState_t::uInternal,
                                            bp::return_internal_reference<>()))
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

    // ***************************** PySystemVisitor ***********************************

    struct PySystemVisitor
        : public bp::def_visitor<PySystemVisitor>
    {
    public:
        ///////////////////////////////////////////////////////////////////////////////
        /// \brief Expose C++ API through the visitor.
        ///////////////////////////////////////////////////////////////////////////////
        template<class PyClass>
        void visit(PyClass& cl) const
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

    // ************************* PyEngineMultiRobotVisitor ****************************

    struct PyEngineMultiRobotVisitor
        : public bp::def_visitor<PyEngineMultiRobotVisitor>
    {
    public:
        ///////////////////////////////////////////////////////////////////////////////
        /// \brief Expose C++ API through the visitor.
        ///////////////////////////////////////////////////////////////////////////////
        template<class PyClass>
        void visit(PyClass& cl) const
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
                .def("computeSystemDynamics", &PyEngineMultiRobotVisitor::computeSystemDynamics,
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

                .def("get_options", &EngineMultiRobot::getOptions,
                                    bp::return_value_policy<bp::return_by_value>())
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
                .add_property("is_simulation_running", bp::make_function(&EngineMultiRobot::getIsSimulationRunning,
                                                       bp::return_value_policy<bp::copy_const_reference>()))
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
            TimeBistateRefFctPyWrapper<pinocchio::Force> forceFct(forcePy);
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
            return convertToPython<std::vector<vectorN_t> >(aSplit);
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
            TimeStateRefFctPyWrapper<pinocchio::Force> forceFct(forcePy);
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

            // Get constants
            int32_t const lastConstantIdx = std::distance(
                logData.header.begin(), std::find(logData.header.begin(), logData.header.end(), START_COLUMNS));
            for (int32_t i = 1; i < lastConstantIdx; ++i)
            {
                int32_t const delimiter = logData.header[i].find("=");
                constants[logData.header[i].substr(0, delimiter)] = logData.header[i].substr(delimiter + 1);
            }

            // Get Global.Time
            if (!logData.timestamps.empty())
            {
                vectorN_t timeBuffer = Eigen::Matrix<int64_t, 1, Eigen::Dynamic>::Map(
                    logData.timestamps.data(), logData.timestamps.size()).cast<float64_t>() / logData.timeUnit;
                PyObject * valuePyTime(getNumpyReference(timeBuffer));
                variables[logData.header[lastConstantIdx + 1]] = bp::object(bp::handle<>(
                    PyArray_FROM_OF(valuePyTime, NPY_ARRAY_ENSURECOPY)));
                Py_XDECREF(valuePyTime);
            }
            else
            {
                npy_intp dims[1] = {npy_intp(0)};
                variables[logData.header[lastConstantIdx + 1]] = bp::object(bp::handle<>(
                    PyArray_SimpleNew(1, dims, NPY_FLOAT64)));
            }

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

                    /* One must make copies with PyArray_FROM_OF instead of using
                       raw pointer for floatMatrix and setting NPY_ARRAY_OWNDATA
                       because otherwise Python is not able to free the memory
                       associated with each columns independently. Moreover, one
                       must decrease manually the counter reference for some reasons... */
                    PyObject * valuePyInt(getNumpyReference(intVector));
                    variables[header_i] = bp::object(bp::handle<>(
                        PyArray_FROM_OF(valuePyInt, NPY_ARRAY_ENSURECOPY)));
                    Py_XDECREF(valuePyInt);
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

                    PyObject * valuePyFloat(getNumpyReference(floatVector));
                    variables[header_i] = bp::object(bp::handle<>(
                        PyArray_FROM_OF(valuePyFloat, NPY_ARRAY_ENSURECOPY)));
                    Py_XDECREF(valuePyFloat);
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

    // ***************************** PyEngineVisitor ***********************************

    struct PyEngineVisitor
        : public bp::def_visitor<PyEngineVisitor>
    {
    public:
        ///////////////////////////////////////////////////////////////////////////////
        /// \brief Expose C++ API through the visitor.
        ///////////////////////////////////////////////////////////////////////////////
        template<class PyClass>
        void visit(PyClass& cl) const
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
            TimeStateRefFctPyWrapper<pinocchio::Force> forceFct(forcePy);
            self.registerForceProfile(frameName, std::move(forceFct));
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
}  // End of namespace python.
}  // End of namespace jiminy.

#endif  // SIMULATOR_PYTHON_H
