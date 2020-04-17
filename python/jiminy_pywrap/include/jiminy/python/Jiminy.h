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
#include "jiminy/core/control/ControllerFunctor.h"
#include "jiminy/core/telemetry/TelemetryData.h"
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
    bp::handle<> FctPyWrapperArgToPython(T const & arg) = delete; // Do NOT provide default implementation

    template<>
    bp::handle<> FctPyWrapperArgToPython<float64_t>(float64_t const & arg)
    {
        return bp::handle<>(PyFloat_FromDouble(arg));
    }

    template<>
    bp::handle<> FctPyWrapperArgToPython<vectorN_t>(vectorN_t const & arg)
    {
        return bp::handle<>(getNumpyReference(const_cast<vectorN_t &>(arg)));
    }

    template<>
    bp::handle<> FctPyWrapperArgToPython<Eigen::Ref<vectorN_t const> >(Eigen::Ref<vectorN_t const> const & arg)
    {
        // Pass the arguments by reference (be careful const qualifiers are lost)
        return bp::handle<>(getNumpyReference(arg));
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
        outPtr_(new OutputArg),
        outData_(setDataInternalBuffer(outPtr_)),
        outPyPtr_(nullptr)
        {
            outPyPtr_ = getNumpyReference(outData_);
        }

        // Copy constructor, same as the normal constructor
        FctPyWrapper(FctPyWrapper const & other) :
        funcPyPtr_(other.funcPyPtr_),
        outPtr_(new OutputArg),
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
                                                  Eigen::Ref<vectorN_t const> /* q */,
                                                  Eigen::Ref<vectorN_t const> /* v */>;

    template<typename T>
    using TimeStateFctPyWrapper = FctPyWrapper<T /* OutputType */,
                                               float64_t /* t */,
                                               vectorN_t /* q */,
                                               vectorN_t /* v */>;

    template<typename T>
    using TimeBistateRefFctPyWrapper = FctPyWrapper<T /* OutputType */,
                                                    float64_t /* t */,
                                                    Eigen::Ref<vectorN_t const> /* q1 */,
                                                    Eigen::Ref<vectorN_t const> /* v1 */,
                                                    Eigen::Ref<vectorN_t const> /* q2 */,
                                                    Eigen::Ref<vectorN_t const> /* v2 */ >;

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
                auto & sensorDataTypeByName = self.at(sensorType).get<IndexByName>();
                auto sensorDataIt = sensorDataTypeByName.find(sensorName);
                Eigen::Ref<vectorN_t const> const & sensorDataValue = sensorDataIt->value;
                bp::handle<> valuePy(getNumpyReference(sensorDataValue));
                return bp::object(valuePy);
            }
            catch (...)
            {
                PyErr_SetString(PyExc_KeyError, "The key does not exist.");
                return bp::object();
            }
        }

        static matrixN_t getSub(sensorsDataMap_t       & self,
                                std::string      const & sensorType)
        {
            matrixN_t data;
            auto const & sensorsDataType = self.at(sensorType);
            auto sensorDataIt = sensorsDataType.begin();
            data.resize(sensorDataIt->value.size(), sensorsDataType.size());
            data.col(sensorDataIt->idx) = sensorDataIt->value;
            sensorDataIt++;
            for (; sensorDataIt != sensorsDataType.end(); sensorDataIt++)
            {
                data.col(sensorDataIt->idx) = sensorDataIt->value;
            }
            return data;
        }

        static bool_t contains(sensorsDataMap_t       & self,
                               bp::tuple        const & sensorInfo)
        {
            std::string const sensorType = bp::extract<std::string>(sensorInfo[0]);
            std::string const sensorName = bp::extract<std::string>(sensorInfo[1]);
            auto const & sensorDataType = self.find(sensorType);
            if (sensorDataType != self.end())
            {
                auto & sensorDataTypeByName = sensorDataType->second.get<IndexByName>();
                auto sensorDataIt = sensorDataTypeByName.find(sensorName);
                if (sensorDataIt != sensorDataTypeByName.end())
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
            for (auto const & sensorDataType : self)
            {
                sensorsValue.append(getSub(self, sensorDataType.first));
            }
            return sensorsValue;
        }

        static bp::list items(sensorsDataMap_t & self)
        {
            bp::list sensorsDataPy;
            for (auto const & sensorDataType : self)
            {
                sensorsDataPy.append(bp::make_tuple(sensorDataType.first,
                                                    getSub(self, sensorDataType.first)));
            }
            return sensorsDataPy;
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

    // **************************** ControllerFctWrapper *******************************

    struct ControllerFctWrapper
    {
    public:
        ControllerFctWrapper(bp::object const & objPy) : funcPyPtr_(objPy) {}
        void operator() (float64_t        const & t,
                         vectorN_t        const & q,
                         vectorN_t        const & v,
                         sensorsDataMap_t const & sensorsData,
                         vectorN_t              & uCommand)
        {
            // Pass the arguments by reference (be careful const qualifiers are lost).
            bp::handle<> qPy(getNumpyReference(const_cast<vectorN_t &>(q)));
            bp::handle<> vPy(getNumpyReference(const_cast<vectorN_t &>(v)));
            bp::handle<> uCommandPy(getNumpyReference(const_cast<vectorN_t &>(uCommand)));
            funcPyPtr_(t, qPy, vPy, boost::ref(sensorsData), uCommandPy);
        }
    private:
        bp::object funcPyPtr_;
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
                    .add_property("joint_position_idx", bp::make_function(&AbstractMotorBase::getJointPositionIdx,
                                                        bp::return_value_policy<bp::copy_const_reference>()))
                    .add_property("joint_velocity_idx", bp::make_function(&AbstractMotorBase::getJointVelocityIdx,
                                                        bp::return_value_policy<bp::copy_const_reference>()))
                    .add_property("torque_limit", bp::make_function(&AbstractMotorBase::getTorqueLimit,
                                                  bp::return_value_policy<bp::copy_const_reference>()))
                    .add_property("rotor_inertia", bp::make_function(&AbstractMotorBase::getRotorInertia,
                                                   bp::return_value_policy<bp::copy_const_reference>()))
                    ;
            }

            template<class Q = TMotor>
            static enable_if_t<!std::is_same<Q, AbstractMotorBase>::value, void>
            visit(PyClass& cl)
            {
                visitAbstract(cl);

                cl
                    .def("initialize", &TMotor::initialize)
                    ;
            }

            template<class Q = TMotor>
            static enable_if_t<std::is_same<Q, AbstractMotorBase>::value, void>
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
                    .add_property("get_jacobian", bp::make_function(&AbstractConstraint::getJacobian,
                                                  bp::return_value_policy<bp::copy_const_reference>()))
                    .add_property("get_drift", bp::make_function(&AbstractConstraint::getDrift,
                                               bp::return_value_policy<bp::copy_const_reference>()))
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
                    ;
            }

            template<class Q = TSensor>
            static enable_if_t<!std::is_same<Q, AbstractSensorBase>::value, void>
            visit(PyClass& cl)
            {
                visitAbstract(cl);

                cl
                    .def("initialize", &TSensor::initialize)
                    .def_readonly("type", &TSensor::type_)
                    .add_static_property("fieldnames", bp::make_getter(&TSensor::fieldNames_,
                                                       bp::return_value_policy<bp::return_by_value>()))
                    ;
            }

            template<class Q = TSensor>
            static enable_if_t<std::is_same<Q, AbstractSensorBase>::value, void>
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

            bp::class_<ForceSensor, bp::bases<AbstractSensorBase>,
                       std::shared_ptr<ForceSensor>,
                       boost::noncopyable>("ForceSensor", bp::init<std::string>())
                .def(PySensorVisitor());

            bp::class_<EncoderSensor, bp::bases<AbstractSensorBase>,
                       std::shared_ptr<EncoderSensor>,
                       boost::noncopyable>("EncoderSensor", bp::init<std::string>())
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
                .def("add_contact_points", &PyModelVisitor::addContactPoints,
                                           (bp::arg("self"),
                                            bp::arg("frame_names") = std::vector<std::string>()))
                .def("remove_contact_points", &PyModelVisitor::removeContactPoints,
                                              (bp::arg("self"), "frame_names"))

                .def("get_flexible_state_from_rigid", &PyModelVisitor::getFlexibleStateFromRigid,
                                                      (bp::arg("self"), "rigid_state"))
                .def("get_rigid_state_from_flexible", &PyModelVisitor::getRigidStateFromFlexible,
                                                      (bp::arg("self"), "flexible_state"))

                .add_property("pinocchio_model", bp::make_getter(&Robot::pncModel_,
                                                 bp::return_internal_reference<>()))
                .add_property("pinocchio_data", bp::make_getter(&Robot::pncData_,
                                                bp::return_internal_reference<>()))
                .add_property("pinocchio_model_th", bp::make_getter(&Robot::pncModelRigidOrig_,
                                                    bp::return_internal_reference<>()))
                .add_property("pinocchio_data_th", bp::make_getter(&Robot::pncDataRigidOrig_,
                                                   bp::return_internal_reference<>()))

                .add_property("is_initialized", bp::make_function(&Robot::getIsInitialized,
                                                bp::return_value_policy<bp::copy_const_reference>()))
                .add_property("urdf_path", bp::make_function(&Robot::getUrdfPath,
                                           bp::return_value_policy<bp::copy_const_reference>()))
                .add_property("has_freeflyer", bp::make_function(&Robot::getHasFreeflyer,
                                               bp::return_value_policy<bp::copy_const_reference>()))
                .add_property("is_flexible", &PyModelVisitor::isFlexibleModelEnable)
                .add_property("nq", bp::make_function(&Robot::nq,
                                    bp::return_value_policy<bp::copy_const_reference>()))
                .add_property("nv", bp::make_function(&Robot::nv,
                                    bp::return_value_policy<bp::copy_const_reference>()))
                .add_property("nx", bp::make_function(&Robot::nx,
                                    bp::return_value_policy<bp::copy_const_reference>()))

                .add_property("contact_frames_names", bp::make_function(&Robot::getContactFramesNames,
                                                      bp::return_value_policy<bp::copy_const_reference>()))
                .add_property("contact_frames_idx", bp::make_function(&Robot::getContactFramesIdx,
                                                    bp::return_value_policy<bp::copy_const_reference>()))
                .add_property("rigid_joints_names", bp::make_function(&Robot::getRigidJointsNames,
                                                    bp::return_value_policy<bp::copy_const_reference>()))
                .add_property("rigid_joints_position_idx", bp::make_function(&Robot::getRigidJointsPositionIdx,
                                                           bp::return_value_policy<bp::copy_const_reference>()))
                .add_property("rigid_joints_velocity_idx", bp::make_function(&Robot::getRigidJointsVelocityIdx,
                                                           bp::return_value_policy<bp::copy_const_reference>()))
                .add_property("flexible_joints_names", bp::make_function(&Robot::getFlexibleJointsNames,
                                                       bp::return_value_policy<bp::copy_const_reference>()))

                .add_property("position_limit_upper", bp::make_function(&Robot::getPositionLimitMin,
                                                      bp::return_value_policy<bp::copy_const_reference>()))
                .add_property("position_limit_lower", bp::make_function(&Robot::getPositionLimitMax,
                                                      bp::return_value_policy<bp::copy_const_reference>()))
                .add_property("velocity_limit", bp::make_function(&Robot::getVelocityLimit,
                                                bp::return_value_policy<bp::copy_const_reference>()))

                .add_property("logfile_position_headers", bp::make_function(&Robot::getPositionFieldnames,
                                                          bp::return_value_policy<bp::copy_const_reference>()))
                .add_property("logfile_velocity_headers", bp::make_function(&Robot::getVelocityFieldnames,
                                                          bp::return_value_policy<bp::copy_const_reference>()))
                .add_property("logfile_acceleration_headers", bp::make_function(&Robot::getAccelerationFieldnames,
                                                              bp::return_value_policy<bp::copy_const_reference>()))
                ;
        }

        static hresult_t addContactPoints(Robot          & self,
                                          bp::list const & frameNamesPy)
        {
            auto frameNames = convertFromPython<std::vector<std::string> >(frameNamesPy);
            return self.addContactPoints(frameNames);
        }

        static hresult_t removeContactPoints(Robot          & self,
                                             bp::list const & frameNamesPy)
        {
            auto frameNames = convertFromPython<std::vector<std::string> >(frameNamesPy);
            return self.removeContactPoints(frameNames);
        }

        static vectorN_t getFlexibleStateFromRigid(Robot           & self,
                                                   vectorN_t const & xRigid)
        {
            vectorN_t xFlexible;
            self.getFlexibleStateFromRigid(xRigid, xFlexible);
            return xFlexible;
        }

        static vectorN_t getRigidStateFromFlexible(Robot           & self,
                                                   vectorN_t const & xFlexible)
        {
            vectorN_t xRigid;
            self.getRigidStateFromFlexible(xFlexible, xRigid);
            return xRigid;
        }

        ///////////////////////////////////////////////////////////////////////////////
        /// \brief      Getters and Setters
        ///////////////////////////////////////////////////////////////////////////////

        static bool_t isFlexibleModelEnable(Robot & self)
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
                                    bp::arg("has_freeflyer") = false))

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
                .add_property("motors_torques", bp::make_function(&Robot::getMotorsTorques,
                                                bp::return_value_policy<bp::copy_const_reference>()))

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

                .add_property("motors_names", bp::make_function(&Robot::getMotorsNames,
                                              bp::return_value_policy<bp::copy_const_reference>()))
                .add_property("motors_position_idx", &Robot::getMotorsPositionIdx)
                .add_property("motors_velocity_idx", &Robot::getMotorsVelocityIdx)
                .add_property("sensors_names", bp::make_function(
                    static_cast<
                        std::unordered_map<std::string, std::vector<std::string> > const & (Robot::*)(void) const
                    >(&Robot::getSensorsNames),
                    bp::return_value_policy<bp::copy_const_reference>()))

                .add_property("torque_limit", &Robot::getTorqueLimit)
                .add_property("motor_inertia", &Robot::getMotorInertia)

                .add_property("logfile_motor_torque_headers", bp::make_function(&Robot::getMotorTorqueFieldnames,
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
                .def("register_variable", &PyAbstractControllerVisitor::registerVariable,
                                          (bp::arg("self"), "fieldname", "value"),
                                          "@copydoc AbstractController::registerVariable")
                .def("register_variable", &PyAbstractControllerVisitor::registerVariableVector,
                                          (bp::arg("self"), "fieldnames", "values"))
                .def("register_constant", &PyAbstractControllerVisitor::registerConstant,
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
                float64_t const * data = (float64_t *) PyArray_DATA(reinterpret_cast<PyArrayObject *>(dataPy));
                return self.registerVariable(fieldName, *data);
            }
            else
            {
                std::cout << "Error - PyAbstractControllerVisitor::registerVariable - 'value' input must have type 'numpy.ndarray'." << std::endl;
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
                Eigen::Map<vectorN_t> data((float64_t *) PyArray_DATA(dataPyArray), PyArray_SIZE(dataPyArray));
                return self.registerVariable(fieldnames, data);
            }
            else
            {
                std::cout << "Error - PyAbstractControllerVisitor::registerVariableVector - 'values' input must have type 'numpy.ndarray'." << std::endl;
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
                    std::cout << "Error - PyAbstractControllerVisitor::registerConstant - The only dtype supported for 'numpy.ndarray' is float." << std::endl;
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
                    std::cout << "Error - PyAbstractControllerVisitor::registerConstant - The max number of dims supported for 'numpy.ndarray' is 2." << std::endl;
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
            #if PY_VERSION_HEX >= 0x03000000
            else if (PyBytes_Check(dataPy))
            {
                return self.registerConstant(fieldName, PyBytes_AsString(dataPy));
            }
            else if (PyUnicode_Check(dataPy))
            {
                return self.registerConstant(fieldName, PyUnicode_AsUTF8(dataPy));
            }
            #else
            else if (PyString_Check(dataPy))
            {
                return self.registerConstant(fieldName, PyString_AsString(dataPy));
            }
            #endif
            else
            {
                std::cout << "Error - PyAbstractControllerVisitor::registerConstant - 'value' type is unsupported." << std::endl;
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
        using CtrlFunctor = ControllerFunctor<ControllerFctWrapper, ControllerFctWrapper>;

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
                                (bp::arg("command_function"), "internal_dynamics_function")));
                ;
        }

        static std::shared_ptr<CtrlFunctor> factory(bp::object & commandPy,
                                                    bp::object & internalDynamicsPy)
        {
            ControllerFctWrapper commandFct(commandPy);
            ControllerFctWrapper internalDynamicsFct(internalDynamicsPy);
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
                .add_property("x", bp::make_getter(&stepperState_t::x,
                                   bp::return_value_policy<bp::copy_non_const_reference>()))
                .add_property("dxdt", bp::make_getter(&stepperState_t::dxdt,
                                      bp::return_value_policy<bp::copy_non_const_reference>()))
                ;
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
                                   bp::return_value_policy<bp::copy_non_const_reference>()))
                .add_property("v", bp::make_getter(&systemState_t::v,
                                   bp::return_value_policy<bp::copy_non_const_reference>()))
                .add_property("qDot", bp::make_getter(&systemState_t::qDot,
                                      bp::return_value_policy<bp::copy_non_const_reference>()))
                .add_property("a", bp::make_getter(&systemState_t::a,
                                   bp::return_value_policy<bp::copy_non_const_reference>()))
                .add_property("u", bp::make_getter(&systemState_t::u,
                                   bp::return_value_policy<bp::copy_non_const_reference>()))
                .add_property("u_motor", bp::make_getter(&systemState_t::uMotor,
                                         bp::return_value_policy<bp::copy_non_const_reference>()))
                .add_property("u_command", bp::make_getter(&systemState_t::uCommand,
                                           bp::return_value_policy<bp::copy_non_const_reference>()))
                .add_property("u_internal", bp::make_getter(&systemState_t::uInternal,
                                            bp::return_value_policy<bp::copy_non_const_reference>()))
                .add_property("f_external", bp::make_getter(&systemState_t::fExternal,
                                            bp::return_value_policy<bp::copy_non_const_reference>()))
                ;
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

    // ***************************** PySystemDataVisitor ***********************************

    struct PySystemDataVisitor
        : public bp::def_visitor<PySystemDataVisitor>
    {
    public:
        ///////////////////////////////////////////////////////////////////////////////
        /// \brief Expose C++ API through the visitor.
        ///////////////////////////////////////////////////////////////////////////////
        template<class PyClass>
        void visit(PyClass& cl) const
        {
            cl
                .add_property("name", bp::make_getter(&systemDataHolder_t::name,
                                      bp::return_value_policy<bp::copy_non_const_reference>()))
                .add_property("robot", bp::make_getter(&systemDataHolder_t::robot,
                                       bp::return_internal_reference<>()))
                .add_property("controller", bp::make_getter(&systemDataHolder_t::controller,
                                            bp::return_internal_reference<>()))
                .add_property("callbackFct", bp::make_getter(&systemDataHolder_t::callbackFct,
                                             bp::return_internal_reference<>()))
                ;
        }

        ///////////////////////////////////////////////////////////////////////////////
        /// \brief Expose.
        ///////////////////////////////////////////////////////////////////////////////
        static void expose()
        {
            bp::class_<systemDataHolder_t,
                       std::shared_ptr<systemDataHolder_t>,
                       boost::noncopyable>("systemData", bp::no_init)
                .def(PySystemDataVisitor());
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
                              (bp::arg("self"), "x_init",
                               bp::arg("reset_random_generator") = false,
                               bp::arg("remove_forces") = false))
                .def("step", &PyEngineMultiRobotVisitor::step,
                             (bp::arg("self"), bp::arg("dt_desired") = -1))
                .def("stop", &EngineMultiRobot::stop, (bp::arg("self")))
                .def("simulate", &PyEngineMultiRobotVisitor::simulate,
                                 (bp::arg("self"), "end_time", "x_init"))

                .def("get_log", &PyEngineMultiRobotVisitor::getLog)
                .def("write_log", &PyEngineMultiRobotVisitor::writeLog,
                                  (bp::arg("self"), "filename",
                                   bp::arg("isModeBinary") = true))
                .def("read_log", &PyEngineMultiRobotVisitor::parseLogBinary, (bp::arg("filename")))
                .staticmethod("read_log")

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
                .def("get_system_state", bp::make_function(&EngineMultiRobot::getSystemState,
                                         bp::return_internal_reference<>(),
                                         (bp::arg("self"), "system_name")))
                .add_property("stepper_state", bp::make_function(&EngineMultiRobot::getStepperState,
                                               bp::return_internal_reference<>()))
                ;
        }

        static hresult_t addSystemWithoutController(EngineMultiRobot             & self,
                                                    std::string            const & systemName,
                                                    std::shared_ptr<Robot> const & robot)
        {
            auto commandFct = [](float64_t        const & t,
                                 vectorN_t        const & q,
                                 vectorN_t        const & v,
                                 sensorsDataMap_t const & sensorsData,
                                 vectorN_t              & uCommand) {};
            auto internalDynamicsFct = [](float64_t        const & t,
                                          vectorN_t        const & q,
                                          vectorN_t        const & v,
                                          sensorsDataMap_t const & sensorsData,
                                          vectorN_t              & uCommand) {};
            callbackFunctor_t callbackFct = [](float64_t const & t,
                                               vectorN_t const & q,
                                               vectorN_t const & v) -> bool_t
                                            {
                                                return true;
                                            };
            auto controller = std::make_shared<
                ControllerFunctor<decltype(commandFct),
                                  decltype(internalDynamicsFct)>
            >(commandFct, internalDynamicsFct);
            controller->initialize(robot.get());
            return self.addSystem(systemName, robot, controller, callbackFct);
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
            return self.addSystem(systemName, robot, controller, callbackFct);
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

        static systemDataHolder_t & getSystem(EngineMultiRobot  & self,
                                              std::string const & systemName)
        {
            systemDataHolder_t * system;
            self.getSystem(systemName, system);
            return *system;
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
                systemName1, systemName2, frameName1, frameName2, forceFct);
        }

        static hresult_t start(EngineMultiRobot       & self,
                               bp::object       const & xInit,
                               bool             const & resetRandomGenerator,
                               bool             const & removeForces)
        {
            return self.start(convertFromPython<std::map<std::string, vectorN_t> >(xInit),
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
                                  bp::object       const & xInit)
        {
            return self.simulate(endTime,
                                 convertFromPython<std::map<std::string, vectorN_t> >(xInit));
        }

        static void writeLog(EngineMultiRobot       & self,
                             std::string      const & filename,
                             bool_t           const & isModeBinary)
        {
            if (isModeBinary)
            {
                self.writeLogBinary(filename);
            }
            else
            {
                self.writeLogTxt(filename);
            }
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

        static bp::tuple formatLog(std::vector<std::string>             const & header,
                                   std::vector<float64_t>                     & timestamps,
                                   std::vector<std::vector<int32_t> >         & intData,
                                   std::vector<std::vector<float32_t> >       & floatData,
                                   bool_t                               const & clear_memory = true)
        {
            bp::dict constants;
            bp::dict data;

            // Get constants
            int32_t const lastConstantIdx = std::distance(
                header.begin(), std::find(header.begin(), header.end(), START_COLUMNS));
            for (int32_t i = 1; i < lastConstantIdx; i++)
            {
                int32_t const delimiter = header[i].find("=");
                constants[header[i].substr(0, delimiter)] = header[i].substr(delimiter + 1);
            }

            // Get Global.Time
            if (!timestamps.empty())
            {
                Eigen::Ref<vectorN_t> timeBuffer = vectorN_t::Map(
                    timestamps.data(), timestamps.size());
                PyObject * valuePyTime(getNumpyReference(timeBuffer));
                data[header[lastConstantIdx + 1]] = bp::object(bp::handle<>(
                    PyArray_FROM_OF(valuePyTime, NPY_ARRAY_ENSURECOPY)));
                Py_XDECREF(valuePyTime);
            }

            // Get intergers
            Eigen::Matrix<int32_t, Eigen::Dynamic, Eigen::Dynamic> intMatrix;
            if (!intData.empty())
            {
                intMatrix.resize(timestamps.size(), intData[0].size());
                for (uint32_t i=0; i<intData.size(); i++)
                {
                    intMatrix.row(i) = Eigen::Matrix<int32_t, 1, Eigen::Dynamic>::Map(
                        intData[i].data(), intData[0].size());
                }
                if (clear_memory)
                {
                    intData.clear();
                }

                for (uint32_t i=0; i<intMatrix.cols(); i++)
                {
                    Eigen::Ref<Eigen::Matrix<int32_t, -1, 1> > intCol(intMatrix.col(i));
                    PyObject * valuePyInt(getNumpyReference(intCol));
                    std::string const & header_i = header[i + (lastConstantIdx + 1) + 1];
                    /* One must make copies with PyArray_FROM_OF instead of using
                       raw pointer for floatMatrix and setting NPY_ARRAY_OWNDATA
                       because otherwise Python is not able to free the memory
                       associated with each columns independently. Moreover, one
                       must decrease manually the counter reference for some reasons... */
                    data[header_i] = bp::object(bp::handle<>(
                        PyArray_FROM_OF(valuePyInt, NPY_ARRAY_ENSURECOPY)));
                    Py_XDECREF(valuePyInt);
                }
            }

            // Get floats
            Eigen::Matrix<float32_t, Eigen::Dynamic, Eigen::Dynamic> floatMatrix;
            if (!floatData.empty())
            {
                floatMatrix.resize(timestamps.size(), floatData[0].size());
                for (uint32_t i=0; i<floatData.size(); i++)
                {
                    floatMatrix.row(i) = Eigen::Matrix<float32_t, 1, Eigen::Dynamic>::Map(
                        floatData[i].data(), floatData[0].size());
                }
                if (clear_memory)
                {
                    floatData.clear();
                }

                for (uint32_t i=0; i<floatMatrix.cols(); i++)
                {
                    Eigen::Ref<Eigen::Matrix<float32_t, -1, 1> > floatCol(floatMatrix.col(i));
                    PyObject * valuePyFloat(getNumpyReference(floatCol));
                    std::string const & header_i =
                        header[i + (lastConstantIdx + 1) + 1 + intData[0].size()];
                    data[header_i] = bp::object(bp::handle<>(
                        PyArray_FROM_OF(valuePyFloat, NPY_ARRAY_ENSURECOPY)));
                    Py_XDECREF(valuePyFloat);
                }
            }

            return bp::make_tuple(data, constants);
        }

        static bp::tuple getLog(EngineMultiRobot & self)
        {
            std::vector<std::string> header;
            std::vector<float64_t> timestamps;
            std::vector<std::vector<int32_t> > intData;
            std::vector<std::vector<float32_t> > floatData;
            self.getLogDataRaw(header, timestamps, intData, floatData);
            return formatLog(header, timestamps, intData, floatData);
        }

        static bp::tuple parseLogBinary(std::string const & filename)
        {
            std::vector<std::string> header;
            std::vector<float64_t> timestamps;
            std::vector<std::vector<int32_t> > intData;
            std::vector<std::vector<float32_t> > floatData;
            hresult_t returnCode = EngineMultiRobot::parseLogBinaryRaw(
                filename, header, timestamps, intData, floatData);
            if (returnCode == hresult_t::SUCCESS)
            {
                return formatLog(header, timestamps, intData, floatData);
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

                .def("start",
                    static_cast<
                        hresult_t (Engine::*)(vectorN_t const &, bool_t const &, bool_t const &, bool_t const &)
                    >(&Engine::start),
                    (bp::arg("self"), "x_init",
                     bp::arg("is_state_theoretical") = false,
                     bp::arg("reset_random_generator") = false,
                     bp::arg("remove_forces") = false))
                .def("simulate",
                    static_cast<
                        hresult_t (Engine::*)(float64_t const &, vectorN_t const &, bool_t const &)
                    >(&Engine::simulate),
                    (bp::arg("self"), "end_time", "x_init", bp::arg("is_state_theoretical") = false))

                .def("register_force_impulse", &PyEngineVisitor::registerForceImpulse,
                                               (bp::arg("self"), "frame_name", "t", "dt", "F"))
                .def("register_force_profile", &PyEngineVisitor::registerForceProfile,
                                               (bp::arg("self"), "frame_name", "force_function"))

                .add_property("is_initialized", bp::make_function(&Engine::getIsInitialized,
                                                bp::return_value_policy<bp::copy_const_reference>()))
                .add_property("robot",
                    static_cast<
                        std::shared_ptr<Robot> (Engine::*)(void)
                    >(&Engine::getRobot))
                .add_property("controller",
                    static_cast<
                        std::shared_ptr<AbstractController> (Engine::*)(void)
                    >(&Engine::getController))
                .add_property("stepper_state", bp::make_function(&Engine::getStepperState,
                                               bp::return_internal_reference<>()))
                .add_property("system_state", bp::make_function(
                    static_cast<
                        systemState_t const & (Engine::*)(void) const
                    >(&Engine::getSystemState),
                    bp::return_internal_reference<>()))
                ;
        }

        static hresult_t initializeWithoutController(Engine                       & self,
                                                     std::shared_ptr<Robot> const & robot)
        {
            auto commandFct = [](float64_t        const & t,
                                 vectorN_t        const & q,
                                 vectorN_t        const & v,
                                 sensorsDataMap_t const & sensorsData,
                                 vectorN_t              & uCommand) {};
            auto internalDynamicsFct = [](float64_t        const & t,
                                          vectorN_t        const & q,
                                          vectorN_t        const & v,
                                          sensorsDataMap_t const & sensorsData,
                                          vectorN_t              & uCommand) {};
            callbackFunctor_t callbackFct = [](float64_t const & t,
                                               vectorN_t const & q,
                                               vectorN_t const & v) -> bool_t
                                            {
                                                return true;
                                            };
            auto controller = std::make_shared<
                ControllerFunctor<decltype(commandFct),
                                  decltype(internalDynamicsFct)>
            >(commandFct, internalDynamicsFct);
            controller->initialize(robot.get());
            return self.initialize(robot, controller, callbackFct);
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
            return self.initialize(robot, controller, callbackFct);
        }

        static hresult_t initializeWithCallback(Engine                                    & self,
                                                std::shared_ptr<Robot>              const & robot,
                                                std::shared_ptr<AbstractController> const & controller,
                                                bp::object                          const & callbackPy)
        {
            TimeStateFctPyWrapper<bool_t> callbackFct(callbackPy);
            return self.initialize(robot, controller, std::move(callbackFct));
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
