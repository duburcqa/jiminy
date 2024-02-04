#include "jiminy/core/telemetry/fwd.h"
#include "jiminy/core/hardware/abstract_sensor.h"
#include "jiminy/core/hardware/basic_sensors.h"

#include "pinocchio/bindings/python/fwd.hpp"

#include "jiminy/python/utilities.h"
#include "jiminy/python/sensors.h"


namespace jiminy::python
{
    namespace bp = boost::python;

    // ********************************* SensorMeasurementTree ********************************* //

    struct PySensorMeasurementTreeVisitor : public bp::def_visitor<PySensorMeasurementTreeVisitor>
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
                .def("__init__", &PySensorMeasurementTreeVisitor::factoryWrapper,
                                 bp::with_custodian_and_ward_postcall<1, 2>(),
                                 (bp::arg("self"), "sensor_measurements"))
                .def("__len__", &PySensorMeasurementTreeVisitor::len,
                                (bp::arg("self")))
                .def("__getitem__", &PySensorMeasurementTreeVisitor::getItem,
                                    bp::return_value_policy<result_converter<false>>(),
                                    (bp::arg("self"), "sensor_info"))
                .def("__getitem__", &PySensorMeasurementTreeVisitor::getItemSplit,
                                    bp::return_value_policy<result_converter<false>>(),
                                    (bp::arg("self"), "sensor_type", "sensor_name"))
                .def("__getitem__", &PySensorMeasurementTreeVisitor::getSub,
                                    bp::return_value_policy<result_converter<false>>(),
                                    (bp::arg("self"), "sensor_type"))
                /* Using '__iter__' is discouraged because it has very poor efficiency due to
                   the overhead of translating 'StopIteration' exception when reaching the end. */
                .def("__iter__", bp::range<bp::return_value_policy<result_converter<false>>>(
                                 static_cast<
                                     SensorMeasurementTree::iterator (SensorMeasurementTree::*)(void)
                                 >(&SensorMeasurementTree::begin),
                                 static_cast<
                                     SensorMeasurementTree::iterator (SensorMeasurementTree::*)(void)
                                 >(&SensorMeasurementTree::end)))
                .def("__contains__", &PySensorMeasurementTreeVisitor::contains,
                                     (bp::arg("self"), "key"))
                .def("__repr__", &PySensorMeasurementTreeVisitor::repr)
                .def("keys", &PySensorMeasurementTreeVisitor::keys,
                             (bp::arg("self")))
                .def("keys", &PySensorMeasurementTreeVisitor::keysSensorType,
                             (bp::arg("self"), "sensor_type"))
                .def("values", &PySensorMeasurementTreeVisitor::values,
                               (bp::arg("self")))
                .def("items", &PySensorMeasurementTreeVisitor::items,
                              (bp::arg("self")))
                ;
            // clang-format on
        }

        static bp::ssize_t len(SensorMeasurementTree & self) { return self.size(); }

        static const Eigen::Ref<const Eigen::VectorXd> & getItem(SensorMeasurementTree & self,
                                                                 const bp::tuple & sensorInfo)
        {
            const std::string sensorType = bp::extract<std::string>(sensorInfo[0]);
            const std::string sensorName = bp::extract<std::string>(sensorInfo[1]);
            return PySensorMeasurementTreeVisitor::getItemSplit(self, sensorType, sensorName);
        }

        static const Eigen::Ref<const Eigen::VectorXd> & getItemSplit(
            SensorMeasurementTree & self,
            const std::string & sensorType,
            const std::string & sensorName)
        {
            try
            {
                auto & SensorMeasurementTreeByName = self.at(sensorType).get<IndexByName>();
                auto sensorDataIt = SensorMeasurementTreeByName.find(sensorName);
                if (sensorDataIt == SensorMeasurementTreeByName.end())
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

        static const Eigen::MatrixXd & getSub(SensorMeasurementTree & self,
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

        static bool contains(SensorMeasurementTree & self, const bp::tuple & sensorInfo)
        {
            const std::string sensorType = bp::extract<std::string>(sensorInfo[0]);
            const std::string sensorName = bp::extract<std::string>(sensorInfo[1]);
            const auto & SensorMeasurementTree = self.find(sensorType);
            if (SensorMeasurementTree != self.end())
            {
                auto & SensorMeasurementTreeByName =
                    SensorMeasurementTree->second.get<IndexByName>();
                auto sensorDataIt = SensorMeasurementTreeByName.find(sensorName);
                if (sensorDataIt != SensorMeasurementTreeByName.end())
                {
                    return true;
                }
            }
            return false;
        }

        static bp::list keys(SensorMeasurementTree & self)
        {
            bp::list sensorsInfo;
            for (auto & sensorData : self)
            {
                sensorsInfo.append(sensorData.first);
            }
            return sensorsInfo;
        }

        static bp::list keysSensorType(SensorMeasurementTree & self,
                                       const std::string & sensorType)
        {
            bp::list sensorsInfo;
            for (auto & sensorData : self.at(sensorType))
            {
                sensorsInfo.append(sensorData.name);
            }
            return sensorsInfo;
        }

        static bp::list values(SensorMeasurementTree & self)
        {
            bp::list sensorsValue;
            for (const auto & SensorMeasurementTree : self)
            {
                sensorsValue.append(convertToPython(SensorMeasurementTree.second.getAll(), false));
            }
            return sensorsValue;
        }

        static bp::list items(SensorMeasurementTree & self)
        {
            bp::list sensorsDataPy;
            for (const auto & SensorMeasurementTree : self)
            {
                sensorsDataPy.append(convertToPython(SensorMeasurementTree, false));
            }
            return sensorsDataPy;
        }

        static std::string repr(SensorMeasurementTree & self)
        {
            std::stringstream s;
            Eigen::IOFormat HeavyFmt(5, 1, ", ", "", "", "", "[", "]\n");

            for (const auto & SensorMeasurementTree : self)
            {
                const std::string & sensorTypeName = SensorMeasurementTree.first;
                s << sensorTypeName << ":\n";
                for (const auto & sensorData : SensorMeasurementTree.second)
                {
                    const std::string & sensorName = sensorData.name;
                    std::size_t sensorIndex = sensorData.index;
                    const Eigen::Ref<const Eigen::VectorXd> & sensorDataValue = sensorData.value;
                    s << "    (" << sensorIndex << ") " << sensorName << ": "
                      << sensorDataValue.transpose().format(HeavyFmt);
                }
            }
            return s.str();
        }

        static std::shared_ptr<SensorMeasurementTree> factory(bp::dict & sensorDataPy)
        {
            auto sensorData = convertFromPython<SensorMeasurementTree>(sensorDataPy);
            return std::make_shared<SensorMeasurementTree>(std::move(sensorData));
        }

        static void factoryWrapper(bp::object & self, bp::dict & sensorDataPy)
        {
            auto constructor = bp::make_constructor(&PySensorMeasurementTreeVisitor::factory);
            constructor(self, sensorDataPy);
        }

        static void expose()
        {
            // clang-format off
            bp::class_<SensorMeasurementTree,
                       std::shared_ptr<SensorMeasurementTree>,
                       boost::noncopyable>("SensorMeasurementTree", bp::no_init)
                .def(PySensorMeasurementTreeVisitor());
            // clang-format on
        }
    };

    BOOST_PYTHON_VISITOR_EXPOSE(SensorMeasurementTree)

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
                .ADD_PROPERTY_GET_WITH_POLICY("index",
                                              &AbstractSensorBase::getIndex,
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
            s << "index: " << self.getIndex() << "\n";
            s << "data:\n    ";
            const std::vector<std::string> & fieldnames = self.getFieldnames();
            const Eigen::Ref<const Eigen::VectorXd> & sensorDataValue =
                const_cast<const AbstractSensorBase &>(self).get();
            for (std::size_t i = 0; i < fieldnames.size(); ++i)
            {
                const std::string & field = fieldnames[i];
                const double value = sensorDataValue[i];
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
                .ADD_PROPERTY_GET_WITH_POLICY("frame_index",
                                              &DerivedSensor::getFrameIndex,
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
                .ADD_PROPERTY_GET_WITH_POLICY("frame_index",
                                              &ForceSensor::getFrameIndex,
                                              bp::return_value_policy<bp::return_by_value>())
                .ADD_PROPERTY_GET_WITH_POLICY("joint_index",
                                              &ForceSensor::getJointIndex,
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
                .ADD_PROPERTY_GET_WITH_POLICY("joint_index",
                                              &EncoderSensor::getJointIndex,
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
                .ADD_PROPERTY_GET_WITH_POLICY("motor_index",
                                              &EffortSensor::getMotorIndex,
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
