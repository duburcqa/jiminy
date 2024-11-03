#include "jiminy/core/telemetry/fwd.h"
#include "jiminy/core/hardware/abstract_sensor.h"
#include "jiminy/core/hardware/basic_sensors.h"

#define NO_IMPORT_ARRAY
#include "jiminy/python/fwd.h"
#include "jiminy/python/utilities.h"
#include "jiminy/python/sensors.h"


namespace jiminy::python
{
    namespace bp = boost::python;

    // ********************************* SensorMeasurementTree ********************************* //

    namespace internal::sensor_measurement_tree
    {
        static bp::ssize_t len(SensorMeasurementTree & self)
        {
            return self.size();
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
                    JIMINY_THROW(std::runtime_error, "");
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

        static const Eigen::Ref<const Eigen::VectorXd> & getItem(SensorMeasurementTree & self,
                                                                 const bp::tuple & sensorInfo)
        {
            const std::string sensorType = bp::extract<std::string>(sensorInfo[0]);
            const std::string sensorName = bp::extract<std::string>(sensorInfo[1]);
            return internal::sensor_measurement_tree::getItemSplit(self, sensorType, sensorName);
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
            return std::make_shared<SensorMeasurementTree>(
                convertFromPython<SensorMeasurementTree>(sensorDataPy));
        }

        static void factoryWrapper(bp::object & self, bp::dict & sensorDataPy)
        {
            auto constructor = bp::make_constructor(&internal::sensor_measurement_tree::factory);
            constructor(self, sensorDataPy);
        }
    }

    void exposeSensorMeasurementTree()
    {
        bp::class_<SensorMeasurementTree,
                   std::shared_ptr<SensorMeasurementTree>,
                   boost::noncopyable>("SensorMeasurementTree", bp::no_init)
            /* with_custodian_and_ward_postcall is used to tie the lifetime of the Python
               object with the one of the C++ reference, so that the Python object does not
               get deleted while the C++ object is not. */
            .def("__init__",
                 &internal::sensor_measurement_tree::factoryWrapper,
                 bp::with_custodian_and_ward_postcall<1, 2>(),
                 (bp::arg("self"), "sensor_measurements"))
            .def("__len__", &internal::sensor_measurement_tree::len, (bp::arg("self")))
            .def("__getitem__",
                 &internal::sensor_measurement_tree::getItem,
                 bp::return_value_policy<result_converter<false>>(),
                 (bp::arg("self"), "sensor_info"))
            .def("__getitem__",
                 &internal::sensor_measurement_tree::getItemSplit,
                 bp::return_value_policy<result_converter<false>>(),
                 (bp::arg("self"), "sensor_type", "sensor_name"))
            .def("__getitem__",
                 &internal::sensor_measurement_tree::getSub,
                 bp::return_value_policy<result_converter<false>>(),
                 (bp::arg("self"), "sensor_type"))
            /* Using '__iter__' is discouraged because it has very poor efficiency due to
               the overhead of translating 'StopIteration' exception when reaching the end. */
            .def("__iter__",
                 bp::range<bp::return_value_policy<result_converter<false>>>(
                     static_cast<SensorMeasurementTree::iterator (SensorMeasurementTree::*)(void)>(
                         &SensorMeasurementTree::begin),
                     static_cast<SensorMeasurementTree::iterator (SensorMeasurementTree::*)(void)>(
                         &SensorMeasurementTree::end)))
            .def("__contains__",
                 &internal::sensor_measurement_tree::contains,
                 (bp::arg("self"), "key"))
            .def("__repr__", &internal::sensor_measurement_tree::repr)
            .def("keys", &internal::sensor_measurement_tree::keys, (bp::arg("self")))
            .def("keys",
                 &internal::sensor_measurement_tree::keysSensorType,
                 (bp::arg("self"), "sensor_type"))
            .def("values", &internal::sensor_measurement_tree::values, (bp::arg("self")))
            .def("items", &internal::sensor_measurement_tree::items, (bp::arg("self")));
    }

    // ************************************* AbstractSensor ************************************ //

    namespace internal::abstract_sensor
    {
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

        static void setOptions(AbstractSensorBase & self, const bp::dict & configPy)
        {
            GenericConfig config = self.getOptions();
            convertFromPython(configPy, config);
            return self.setOptions(config);
        }
    }

    void exposeAbstractSensor()
    {
        bp::class_<AbstractSensorBase, std::shared_ptr<AbstractSensorBase>, boost::noncopyable>(
            "AbstractSensor", bp::no_init)
            .ADD_PROPERTY_GET("is_attached", &AbstractSensorBase::getIsAttached)
            .ADD_PROPERTY_GET("is_initialized", &AbstractSensorBase::getIsInitialized)

            .ADD_PROPERTY_GET_WITH_POLICY("type",
                                          &AbstractSensorBase::getType,
                                          bp::return_value_policy<bp::return_by_value>())
            .ADD_PROPERTY_GET_WITH_POLICY("fieldnames",
                                          &AbstractSensorBase::getFieldnames,
                                          bp::return_value_policy<result_converter<true>>())

            .ADD_PROPERTY_GET_WITH_POLICY("name",
                                          &AbstractSensorBase::getName,
                                          bp::return_value_policy<bp::return_by_value>())
            .ADD_PROPERTY_GET("index", &AbstractSensorBase::getIndex)
            .ADD_PROPERTY_GET_SET_WITH_POLICY(
                "data",
                static_cast<Eigen::Ref<const Eigen::VectorXd> (AbstractSensorBase::*)(void) const>(
                    &AbstractSensorBase::get),
                bp::return_value_policy<result_converter<false>>(),
                static_cast<void (AbstractSensorBase::*)(
                    const Eigen::MatrixBase<Eigen::VectorXd> &)>(&AbstractSensorBase::set))

            .def("set_options", &internal::abstract_sensor::setOptions)
            .def("get_options",
                 &AbstractSensorBase::getOptions,
                 bp::return_value_policy<bp::return_by_value>())

            .def("__repr__", &internal::abstract_sensor::repr);
    }

    // ************************************** BasicSensors ************************************* //

    namespace internal::encoder
    {
        void Initialize(
            EncoderSensor & self, const bp::object & motorNamePy, const bp::object & jointNamePy)
        {
            if (!(motorNamePy.is_none() ^ jointNamePy.is_none()))
            {
                throw std::invalid_argument(
                    "Either 'motor_name' or 'joint_name' must be specified but not both.");
            }

            if (jointNamePy.is_none())
            {
                const std::string motorName = bp::extract<std::string>(motorNamePy);
                self.initialize(motorName, false);
            }
            else
            {
                const std::string jointName = bp::extract<std::string>(jointNamePy);
                self.initialize(jointName, true);
            }
        }
    }

    struct PyBasicSensorsVisitor : public bp::def_visitor<PyBasicSensorsVisitor>
    {
    public:
        template<typename PyClass>
        static void visitBasicSensors(PyClass & cl)
        {
            using DerivedSensor = typename PyClass::wrapped_type;

            // clang-format off
            cl
                .def_readonly("type", &DerivedSensor::type_)
                .add_static_property("fieldnames", bp::make_getter(&DerivedSensor::fieldnames_,
                                                   bp::return_value_policy<result_converter<true>>()))
                ;
            // clang-format on
        }

        template<typename DerivedSensor>
        inline static constexpr bool isFrameBasedSensor =
            std::disjunction_v<std::is_same<DerivedSensor, ImuSensor>,
                               std::is_same<DerivedSensor, ContactSensor>,
                               std::is_same<DerivedSensor, ForceSensor>>;

        template<typename PyClass, typename DerivedSensor = typename PyClass::wrapped_type>
        static std::enable_if_t<!isFrameBasedSensor<DerivedSensor>, void> visit(PyClass & cl)
        {
            visitBasicSensors(cl);
        }

        template<typename PyClass, typename DerivedSensor = typename PyClass::wrapped_type>
        static std::enable_if_t<isFrameBasedSensor<DerivedSensor>, void> visit(PyClass & cl)
        {
            visitBasicSensors(cl);

            cl.def("initialize", &DerivedSensor::initialize, (bp::arg("self"), "frame_name"))
                .ADD_PROPERTY_GET_WITH_POLICY("frame_name",
                                              &DerivedSensor::getFrameName,
                                              bp::return_value_policy<bp::return_by_value>())
                .ADD_PROPERTY_GET("frame_index", &DerivedSensor::getFrameIndex);
        }

        static void expose()
        {
            bp::class_<ImuSensor,
                       bp::bases<AbstractSensorBase>,
                       std::shared_ptr<ImuSensor>,
                       boost::noncopyable>(
                "ImuSensor", bp::init<const std::string &>((bp::arg("self"), "name")))
                .def(PyBasicSensorsVisitor());

            bp::class_<ContactSensor,
                       bp::bases<AbstractSensorBase>,
                       std::shared_ptr<ContactSensor>,
                       boost::noncopyable>(
                "ContactSensor", bp::init<const std::string &>((bp::arg("self"), "name")))
                .def(PyBasicSensorsVisitor());

            bp::class_<ForceSensor,
                       bp::bases<AbstractSensorBase>,
                       std::shared_ptr<ForceSensor>,
                       boost::noncopyable>(
                "ForceSensor", bp::init<const std::string &>((bp::arg("self"), "name")))
                .def(PyBasicSensorsVisitor())
                .ADD_PROPERTY_GET("joint_index", &ForceSensor::getJointIndex);

            bp::class_<EncoderSensor,
                       bp::bases<AbstractSensorBase>,
                       std::shared_ptr<EncoderSensor>,
                       boost::noncopyable>(
                "EncoderSensor", bp::init<const std::string &>((bp::arg("self"), "name")))
                .def(PyBasicSensorsVisitor())
                .def("initialize",
                     &internal::encoder::Initialize,
                     (bp::arg("self"),
                      bp::arg("motor_name") = bp::object(),
                      bp::arg("joint_name") = bp::object()))
                .ADD_PROPERTY_GET_WITH_POLICY("joint_name",
                                              &EncoderSensor::getJointName,
                                              bp::return_value_policy<bp::return_by_value>())
                .ADD_PROPERTY_GET("joint_index", &EncoderSensor::getJointIndex)
                .ADD_PROPERTY_GET_WITH_POLICY("motor_name",
                                              &EncoderSensor::getMotorName,
                                              bp::return_value_policy<bp::return_by_value>())
                .ADD_PROPERTY_GET("motor_index", &EncoderSensor::getMotorIndex);

            bp::class_<EffortSensor,
                       bp::bases<AbstractSensorBase>,
                       std::shared_ptr<EffortSensor>,
                       boost::noncopyable>(
                "EffortSensor", bp::init<const std::string &>((bp::arg("self"), "name")))
                .def(PyBasicSensorsVisitor())
                .def("initialize", &EffortSensor::initialize, (bp::arg("self"), "motor_name"))
                .ADD_PROPERTY_GET_WITH_POLICY("motor_name",
                                              &EffortSensor::getMotorName,
                                              bp::return_value_policy<bp::return_by_value>())
                .ADD_PROPERTY_GET("motor_index", &EffortSensor::getMotorIndex);
        }
    };

    void exposeBasicSensors()
    {
        PyBasicSensorsVisitor::expose();
    }
}
