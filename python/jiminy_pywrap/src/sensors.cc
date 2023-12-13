#include "jiminy/core/traits.h"
#include "jiminy/core/hardware/abstract_sensor.h"
#include "jiminy/core/hardware/basic_sensors.h"

#include "pinocchio/bindings/python/fwd.hpp"

#include "jiminy/python/utilities.h"
#include "jiminy/python/sensors.h"


namespace jiminy::python
{
    namespace bp = boost::python;

    // ******************************* SensorsDataMap ********************************

    struct PySensorsDataMapVisitor : public bp::def_visitor<PySensorsDataMapVisitor>
    {
    public:
        /// \brief Expose C++ API through the visitor.
        template<class PyClass>
        void visit(PyClass & cl) const
        {
            // clang-format off
            cl
                /* with_custodian_and_ward_postcall is used to tie the lifetime of the Python
                   object with the one of the C++ reference, so that the Python object does not
                   get deleted while the C++ object is not. */
                .def("__init__", &PySensorsDataMapVisitor::factoryWrapper,
                                 bp::with_custodian_and_ward_postcall<1, 2>(),
                                 (bp::arg("self"), "sensors_data_dict"))
                .def("__len__", &PySensorsDataMapVisitor::len,
                                (bp::arg("self")))
                .def("__getitem__", &PySensorsDataMapVisitor::getItem,
                                    bp::return_value_policy<result_converter<false>>(),
                                    (bp::arg("self"), "sensor_info"))
                .def("__getitem__", &PySensorsDataMapVisitor::getItemSplit,
                                    bp::return_value_policy<result_converter<false>>(),
                                    (bp::arg("self"), "sensor_type", "sensor_name"))
                .def("__getitem__", &PySensorsDataMapVisitor::getSub,
                                    bp::return_value_policy<result_converter<false>>(),
                                    (bp::arg("self"), "sensor_type"))
                /* Using '__iter__' is discouraged because it has very poor efficiency due to
                   the overhead of translating 'StopIteration' exception when reaching the end. */
                .def("__iter__", bp::range<bp::return_value_policy<result_converter<false>>>(
                                 static_cast<
                                     SensorsDataMap::iterator (SensorsDataMap::*)(void)
                                 >(&SensorsDataMap::begin),
                                 static_cast<
                                     SensorsDataMap::iterator (SensorsDataMap::*)(void)
                                 >(&SensorsDataMap::end)))
                .def("__contains__", &PySensorsDataMapVisitor::contains,
                                     (bp::arg("self"), "key"))
                .def("__repr__", &PySensorsDataMapVisitor::repr)
                .def("keys", &PySensorsDataMapVisitor::keys,
                             (bp::arg("self")))
                .def("keys", &PySensorsDataMapVisitor::keysSensorType,
                             (bp::arg("self"), "sensor_type"))
                .def("values", &PySensorsDataMapVisitor::values,
                               (bp::arg("self")))
                .def("items", &PySensorsDataMapVisitor::items,
                              (bp::arg("self")))
                ;
            // clang-format on
        }

        static bp::ssize_t len(SensorsDataMap & self) { return self.size(); }

        static const Eigen::Ref<const Eigen::VectorXd> & getItem(SensorsDataMap & self,
                                                                 const bp::tuple & sensorInfo)
        {
            const std::string sensorType = bp::extract<std::string>(sensorInfo[0]);
            const std::string sensorName = bp::extract<std::string>(sensorInfo[1]);
            return PySensorsDataMapVisitor::getItemSplit(self, sensorType, sensorName);
        }

        static const Eigen::Ref<const Eigen::VectorXd> & getItemSplit(
            SensorsDataMap & self, const std::string & sensorType, const std::string & sensorName)
        {
            try
            {
                auto & sensorsDataTypeByName = self.at(sensorType).get<IndexByName>();
                auto sensorDataIt = sensorsDataTypeByName.find(sensorName);
                if (sensorDataIt == sensorsDataTypeByName.end())
                {
                    throw std::runtime_error("");
                }
                return sensorDataIt->value;
            }
            catch (...)
            {
                std::ostringstream errorMsg;
                errorMsg << "The key pair ('" << sensorType << "', '" << sensorName
                         << "') does not exist.";
                PyErr_SetString(PyExc_KeyError, errorMsg.str().c_str());
                throw bp::error_already_set();
            }
        }

        static const Eigen::MatrixXd & getSub(SensorsDataMap & self,
                                              const std::string & sensorType)
        {
            try
            {
                return self.at(sensorType).getAll();
            }
            catch (...)
            {
                std::ostringstream errorMsg;
                errorMsg << "The key '" << sensorType << "' does not exist.";
                PyErr_SetString(PyExc_KeyError, errorMsg.str().c_str());
                throw bp::error_already_set();
            }
        }

        static bool_t contains(SensorsDataMap & self, const bp::tuple & sensorInfo)
        {
            const std::string sensorType = bp::extract<std::string>(sensorInfo[0]);
            const std::string sensorName = bp::extract<std::string>(sensorInfo[1]);
            const auto & sensorsDataType = self.find(sensorType);
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

        static bp::list keys(SensorsDataMap & self)
        {
            bp::list sensorsInfo;
            for (auto & sensorData : self)
            {
                sensorsInfo.append(sensorData.first);
            }
            return sensorsInfo;
        }

        static bp::list keysSensorType(SensorsDataMap & self, const std::string & sensorType)
        {
            bp::list sensorsInfo;
            for (auto & sensorData : self.at(sensorType))
            {
                sensorsInfo.append(sensorData.name);
            }
            return sensorsInfo;
        }

        static bp::list values(SensorsDataMap & self)
        {
            bp::list sensorsValue;
            for (const auto & sensorsDataType : self)
            {
                sensorsValue.append(convertToPython(sensorsDataType.second.getAll(), false));
            }
            return sensorsValue;
        }

        static bp::list items(SensorsDataMap & self)
        {
            bp::list sensorsDataPy;
            for (const auto & sensorsDataType : self)
            {
                sensorsDataPy.append(convertToPython(sensorsDataType, false));
            }
            return sensorsDataPy;
        }

        static std::string repr(SensorsDataMap & self)
        {
            std::stringstream s;
            Eigen::IOFormat HeavyFmt(5, 1, ", ", "", "", "", "[", "]\n");

            for (const auto & sensorsDataType : self)
            {
                const std::string & sensorTypeName = sensorsDataType.first;
                s << sensorTypeName << ":\n";
                for (const auto & sensorData : sensorsDataType.second)
                {
                    const std::string & sensorName = sensorData.name;
                    const std::size_t & sensorIdx = sensorData.idx;
                    const Eigen::Ref<const Eigen::VectorXd> & sensorDataValue = sensorData.value;
                    s << "    (" << sensorIdx << ") " << sensorName << ": "
                      << sensorDataValue.transpose().format(HeavyFmt);
                }
            }
            return s.str();
        }

        static std::shared_ptr<SensorsDataMap> factory(bp::dict & sensorDataPy)
        {
            auto sensorData = convertFromPython<SensorsDataMap>(sensorDataPy);
            return std::make_shared<SensorsDataMap>(std::move(sensorData));
        }

        static void factoryWrapper(bp::object & self, bp::dict & sensorDataPy)
        {
            auto constructor = bp::make_constructor(&PySensorsDataMapVisitor::factory);
            constructor(self, sensorDataPy);
        }

        static void expose()
        {
            // clang-format off
            bp::class_<SensorsDataMap,
                       std::shared_ptr<SensorsDataMap>,
                       boost::noncopyable>("sensorsData", bp::no_init)
                .def(PySensorsDataMapVisitor());
            // clang-format on
        }
    };

    BOOST_PYTHON_VISITOR_EXPOSE(SensorsDataMap)

    // ************************* PyAbstractSensorVisitor ******************************

    struct PyAbstractSensorVisitor : public bp::def_visitor<PyAbstractSensorVisitor>
    {
    public:
        /// \brief Expose C++ API through the visitor.
        template<class PyClass>
        void visit(PyClass & cl) const
        {
            // clang-format off
            cl
                .ADD_PROPERTY_GET_WITH_POLICY("is_initialized",
                                              &AbstractSensorBase::getIsInitialized,
                                              bp::return_value_policy<bp::return_by_value>())

                .ADD_PROPERTY_GET_WITH_POLICY("type",
                                              &AbstractSensorBase::getType,
                                              bp::return_value_policy<bp::return_by_value>())
                .ADD_PROPERTY_GET_WITH_POLICY("fieldnames",
                                              &AbstractSensorBase::getFieldnames,
                                              bp::return_value_policy<result_converter<true>>())

                .ADD_PROPERTY_GET_WITH_POLICY("name",
                                              &AbstractSensorBase::getName,
                                              bp::return_value_policy<bp::return_by_value>())
                .ADD_PROPERTY_GET_WITH_POLICY("idx",
                                              &AbstractSensorBase::getIdx,
                                              bp::return_value_policy<bp::return_by_value>())
                .ADD_PROPERTY_GET_SET_WITH_POLICY("data",
                                                  static_cast<
                                                      Eigen::Ref<const Eigen::VectorXd> (AbstractSensorBase::*)(void) const
                                                  >(&AbstractSensorBase::get),
                                                  bp::return_value_policy<result_converter<false>>(),
                                                  static_cast<
                                                      hresult_t (AbstractSensorBase::*)(const Eigen::MatrixBase<Eigen::VectorXd> &)
                                                  >(&AbstractSensorBase::set))

                .def("set_options", &PyAbstractSensorVisitor::setOptions)
                .def("get_options", &AbstractSensorBase::getOptions)

                .def("__repr__", &PyAbstractSensorVisitor::repr)
                ;
            // clang-format on
        }

    public:
        static std::string repr(AbstractSensorBase & self)
        {
            std::stringstream s;
            s << "type: " << self.getType() << "\n";
            s << "name: " << self.getName() << "\n";
            s << "idx: " << self.getIdx() << "\n";
            s << "data:\n    ";
            const std::vector<std::string> & fieldnames = self.getFieldnames();
            const Eigen::Ref<const Eigen::VectorXd> & sensorDataValue =
                const_cast<const AbstractSensorBase &>(self).get();
            for (std::size_t i = 0; i < fieldnames.size(); ++i)
            {
                const std::string & field = fieldnames[i];
                float64_t value = sensorDataValue[i];
                if (i > 0)
                {
                    s << ", ";
                }
                s << field << ": " << value;
            }
            return s.str();
        }

        static hresult_t setOptions(AbstractSensorBase & self, const bp::dict & configPy)
        {
            GenericConfig config = self.getOptions();
            convertFromPython(configPy, config);
            return self.setOptions(config);
        }

        static void expose()
        {
            // clang-format off
            bp::class_<AbstractSensorBase,
                       std::shared_ptr<AbstractSensorBase>,
                       boost::noncopyable>("AbstractSensor", bp::no_init)
                .def(PyAbstractSensorVisitor());
            // clang-format on
        }
    };

    BOOST_PYTHON_VISITOR_EXPOSE(AbstractSensor)

    // ************************** PyBasicSensorsVisitor ******************************

    struct PyBasicSensorsVisitor : public bp::def_visitor<PyBasicSensorsVisitor>
    {
    public:
        template<typename PyClass>
        static void visitBasicSensors(PyClass & cl)
        {
            using DerivedSensor = typename PyClass::wrapped_type;

            // clang-format off
            cl
                .def("initialize", &DerivedSensor::initialize)
                .def_readonly("type", &DerivedSensor::type_)
                .def_readonly("has_prefix", &DerivedSensor::areFieldnamesGrouped_)
                .add_static_property("fieldnames", bp::make_getter(&DerivedSensor::fieldnames_,
                                                   bp::return_value_policy<result_converter<true>>()))
                ;
            // clang-format on
        }

        template<typename PyClass>
        static std::enable_if_t<std::is_same_v<typename PyClass::wrapped_type, ImuSensor> ||
                                    std::is_same_v<typename PyClass::wrapped_type, ContactSensor>,
                                void>
        visit(PyClass & cl)
        {
            using DerivedSensor = typename PyClass::wrapped_type;

            visitBasicSensors(cl);

            // clang-format off
            cl
                .ADD_PROPERTY_GET_WITH_POLICY("frame_name",
                                              &DerivedSensor::getFrameName,
                                              bp::return_value_policy<bp::return_by_value>())
                .ADD_PROPERTY_GET_WITH_POLICY("frame_idx",
                                              &DerivedSensor::getFrameIdx,
                                              bp::return_value_policy<bp::return_by_value>())
                ;
            // clang-format on
        }

        template<typename PyClass>
        static std::enable_if_t<std::is_same_v<typename PyClass::wrapped_type, ForceSensor>, void>
        visit(PyClass & cl)
        {
            visitBasicSensors(cl);

            // clang-format off
            cl
                .ADD_PROPERTY_GET_WITH_POLICY("frame_name",
                                              &ForceSensor::getFrameName,
                                              bp::return_value_policy<bp::return_by_value>())
                .ADD_PROPERTY_GET_WITH_POLICY("frame_idx",
                                              &ForceSensor::getFrameIdx,
                                              bp::return_value_policy<bp::return_by_value>())
                .ADD_PROPERTY_GET_WITH_POLICY("joint_idx",
                                              &ForceSensor::getJointIdx,
                                              bp::return_value_policy<bp::return_by_value>())
                ;
            // clang-format on
        }

        template<typename PyClass>
        static std::enable_if_t<std::is_same_v<typename PyClass::wrapped_type, EncoderSensor>, void>
        visit(PyClass & cl)
        {
            visitBasicSensors(cl);

            // clang-format off
            cl
                .ADD_PROPERTY_GET_WITH_POLICY("joint_name",
                                              &EncoderSensor::getJointName,
                                              bp::return_value_policy<bp::return_by_value>())
                .ADD_PROPERTY_GET_WITH_POLICY("joint_idx",
                                              &EncoderSensor::getJointIdx,
                                              bp::return_value_policy<bp::return_by_value>())
                .ADD_PROPERTY_GET_WITH_POLICY("joint_type",
                                              &EncoderSensor::getJointType,
                                              bp::return_value_policy<bp::return_by_value>())
                ;
            // clang-format on
        }

        template<typename PyClass>
        static std::enable_if_t<std::is_same_v<typename PyClass::wrapped_type, EffortSensor>, void>
        visit(PyClass & cl)
        {
            visitBasicSensors(cl);

            // clang-format off
            cl
                .ADD_PROPERTY_GET_WITH_POLICY("motor_name",
                                              &EffortSensor::getMotorName,
                                              bp::return_value_policy<bp::return_by_value>())
                .ADD_PROPERTY_GET_WITH_POLICY("motor_idx",
                                              &EffortSensor::getMotorIdx,
                                              bp::return_value_policy<bp::return_by_value>())
                ;
            // clang-format on
        }

        static void expose()
        {
            // clang-format off
            bp::class_<ImuSensor, bp::bases<AbstractSensorBase>,
                       std::shared_ptr<ImuSensor>,
                       boost::noncopyable>("ImuSensor",
                       bp::init<const std::string &>(
                       (bp::arg("self"), "frame_name")))
                .def(PyBasicSensorsVisitor());

            bp::class_<ContactSensor, bp::bases<AbstractSensorBase>,
                       std::shared_ptr<ContactSensor>,
                       boost::noncopyable>("ContactSensor",
                       bp::init<const std::string &>(
                       (bp::arg("self"), "frame_name")))
                .def(PyBasicSensorsVisitor());

            bp::class_<ForceSensor, bp::bases<AbstractSensorBase>,
                       std::shared_ptr<ForceSensor>,
                       boost::noncopyable>("ForceSensor",
                       bp::init<const std::string &>(
                       (bp::arg("self"), "frame_name")))
                .def(PyBasicSensorsVisitor());

            bp::class_<EncoderSensor, bp::bases<AbstractSensorBase>,
                       std::shared_ptr<EncoderSensor>,
                       boost::noncopyable>("EncoderSensor",
                       bp::init<const std::string &>(
                       (bp::arg("self"), "joint_name")))
                .def(PyBasicSensorsVisitor());

            bp::class_<EffortSensor, bp::bases<AbstractSensorBase>,
                       std::shared_ptr<EffortSensor>,
                       boost::noncopyable>("EffortSensor",
                       bp::init<const std::string &>(
                       (bp::arg("self"), "joint_name")))
                .def(PyBasicSensorsVisitor());
            // clang-format on
        }
    };

    BOOST_PYTHON_VISITOR_EXPOSE(BasicSensors)
}
