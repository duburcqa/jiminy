#include "jiminy/core/robot/AbstractSensor.h"
#include "jiminy/core/robot/BasicSensors.h"
#include "jiminy/core/Types.h"

#include <eigenpy/eigenpy.hpp>
#include <boost/python.hpp>

#include "jiminy/python/Utilities.h"
#include "jiminy/python/Sensors.h"


namespace jiminy
{
namespace python
{
    namespace bp = boost::python;

    // ******************************* sensorsDataMap_t ********************************

    struct PySensorsDataMapVisitor
        : public bp::def_visitor<PySensorsDataMapVisitor>
    {
    public:
        ///////////////////////////////////////////////////////////////////////////////
        /// \brief Expose C++ API through the visitor.
        ///////////////////////////////////////////////////////////////////////////////
        template<class PyClass>
        void visit(PyClass & cl) const
        {
            cl
                .def("__init__", &PySensorsDataMapVisitor::factoryWrapper,
                                 bp::with_custodian_and_ward_postcall<1, 2>(),
                                 (bp::arg("self"), "sensors_data_dict"))  // with_custodian_and_ward_postcall is used to tie the lifetime of the Python object with the one of the C++ reference, so that the Python object does not get deleted while the C++ object is not
                .def("__len__", &PySensorsDataMapVisitor::len,
                                (bp::arg("self")))
                .def("__getitem__", &PySensorsDataMapVisitor::getItem,
                                    (bp::arg("self"), "(sensor_type, sensor_name)"))
                .def("__getitem__", &PySensorsDataMapVisitor::getItemSplit,
                                    (bp::arg("self"), "sensor_type", "sensor_name"))
                .def("__getitem__", &PySensorsDataMapVisitor::getSub,
                                    (bp::arg("self"), "sensor_type"))
                .def("__iter__", bp::iterator<sensorsDataMap_t>())
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
        }

        static bp::ssize_t len(sensorsDataMap_t & self)
        {
            return self.size();
        }

        static bp::object getItem(sensorsDataMap_t        & self,
                                  bp::tuple         const & sensorInfo)
        {
            std::string const sensorType = bp::extract<std::string>(sensorInfo[0]);
            std::string const sensorName = bp::extract<std::string>(sensorInfo[1]);
            return PySensorsDataMapVisitor::getItemSplit(self, sensorType, sensorName);
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
                return convertToPython(sensorDataValue, false);
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
                return convertToPython(sensorsDataType.getAll(), false);
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
                sensorsValue.append(convertToPython(sensorsDataType.second.getAll(), false));
            }
            return sensorsValue;
        }

        static bp::list items(sensorsDataMap_t & self)
        {
            bp::list sensorsDataPy;
            for (auto const & sensorsDataType : self)
            {
                sensorsDataPy.append(bp::make_tuple(
                    sensorsDataType.first,
                    convertToPython(sensorsDataType.second.getAll(), false)));
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
                    std::size_t const & sensorIdx = sensorData.idx;
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
            auto constructor = bp::make_constructor(&PySensorsDataMapVisitor::factory);
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
                .def(PySensorsDataMapVisitor());
        }
    };

    BOOST_PYTHON_VISITOR_EXPOSE(SensorsDataMap)

    // ************************* PyAbstractSensorVisitor ******************************

    struct PyAbstractSensorVisitor
        : public bp::def_visitor<PyAbstractSensorVisitor>
    {
    public:
        ///////////////////////////////////////////////////////////////////////////////
        /// \brief Expose C++ API through the visitor.
        ///////////////////////////////////////////////////////////////////////////////

        template<class PyClass>
        void visit(PyClass & cl) const
        {
            cl
                .add_property("is_initialized", bp::make_function(&AbstractSensorBase::getIsInitialized,
                                                bp::return_value_policy<bp::copy_const_reference>()))

                .add_property("type", bp::make_function(&AbstractSensorBase::getType,
                                      bp::return_value_policy<bp::copy_const_reference>()))
                .add_property("fieldnames", bp::make_function(&AbstractSensorBase::getFieldnames,
                                            bp::return_value_policy<bp::return_by_value>()))

                .add_property("name", bp::make_function(&AbstractSensorBase::getName,
                                      bp::return_value_policy<bp::copy_const_reference>()))
                .add_property("idx", bp::make_function(&AbstractSensorBase::getIdx,
                                     bp::return_value_policy<bp::copy_const_reference>()))
                .add_property("data", bp::make_function(
                    static_cast<
                        Eigen::Ref<vectorN_t const> (AbstractSensorBase::*)(void) const
                    >(&AbstractSensorBase::get),
                    bp::return_value_policy<result_converter<false> >()),
                    static_cast<
                        hresult_t (AbstractSensorBase::*)(Eigen::MatrixBase<vectorN_t> const &)
                    >(&AbstractSensorBase::set))

                .def("set_options", &PyAbstractSensorVisitor::setOptions)
                .def("get_options", &AbstractSensorBase::getOptions)

                .def("__repr__", &PyAbstractSensorVisitor::repr)
                ;
        }

    public:
        static std::string repr(AbstractSensorBase & self)
        {
            std::stringstream s;
            s << "type: " << self.getType() << "\n";
            s << "name: " << self.getName() << "\n";
            s << "idx: " << self.getIdx() << "\n";
            s << "data:\n    ";
            std::vector<std::string> const & fieldnames = self.getFieldnames();
            Eigen::Ref<vectorN_t const> const & sensorDataValue =
                const_cast<AbstractSensorBase const &>(self).get();
            for (std::size_t i = 0; i<fieldnames.size(); ++i)
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

        static void setOptions(AbstractSensorBase       & self,
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
            bp::class_<AbstractSensorBase,
                       std::shared_ptr<AbstractSensorBase>,
                       boost::noncopyable>("AbstractSensor", bp::no_init)
                .def(PyAbstractSensorVisitor());
        }
    };

    BOOST_PYTHON_VISITOR_EXPOSE(AbstractSensor)

    // ************************** PyBasicSensorsVisitor ******************************

    struct PyBasicSensorsVisitor
        : public bp::def_visitor<PyBasicSensorsVisitor>
    {
    public:
        ///////////////////////////////////////////////////////////////////////////////
        /// \brief Expose C++ API through the visitor.
        ///////////////////////////////////////////////////////////////////////////////

        template<class PyClass>
        class PyBasicSensorsVisitorImpl
        {
        public:
            using TSensor = typename PyClass::wrapped_type;

            static void visitBasicSensors(PyClass & cl)
            {
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
            visit(PyClass & cl)
            {
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
            visit(PyClass & cl)
            {
                visitBasicSensors(cl);

                cl
                    .add_property("frame_name", bp::make_function(&ForceSensor::getFrameName,
                                                bp::return_value_policy<bp::copy_const_reference>()))
                    .add_property("frame_idx", bp::make_function(&ForceSensor::getFrameIdx,
                                               bp::return_value_policy<bp::copy_const_reference>()))
                    .add_property("joint_idx", bp::make_function(&ForceSensor::getJointIdx,
                                               bp::return_value_policy<bp::return_by_value>()))
                    ;
            }

            template<class Q = TSensor>
            static std::enable_if_t<std::is_same<Q, EncoderSensor>::value, void>
            visit(PyClass & cl)
            {
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
            visit(PyClass & cl)
            {
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
        void visit(PyClass & cl) const
        {
            PyBasicSensorsVisitorImpl<PyClass>::visit(cl);
        }

        ///////////////////////////////////////////////////////////////////////////////
        /// \brief Expose.
        ///////////////////////////////////////////////////////////////////////////////
        static void expose()
        {
            bp::class_<ImuSensor, bp::bases<AbstractSensorBase>,
                       std::shared_ptr<ImuSensor>,
                       boost::noncopyable>("ImuSensor",
                       bp::init<std::string const &>(
                       bp::args("self", "frame_name")))
                .def(PyBasicSensorsVisitor());

            bp::class_<ContactSensor, bp::bases<AbstractSensorBase>,
                       std::shared_ptr<ContactSensor>,
                       boost::noncopyable>("ContactSensor",
                       bp::init<std::string const &>(
                       bp::args("self", "frame_name")))
                .def(PyBasicSensorsVisitor());

            bp::class_<ForceSensor, bp::bases<AbstractSensorBase>,
                       std::shared_ptr<ForceSensor>,
                       boost::noncopyable>("ForceSensor",
                       bp::init<std::string const &>(
                       bp::args("self", "frame_name")))
                .def(PyBasicSensorsVisitor());

            bp::class_<EncoderSensor, bp::bases<AbstractSensorBase>,
                       std::shared_ptr<EncoderSensor>,
                       boost::noncopyable>("EncoderSensor",
                       bp::init<std::string const &>(
                       bp::args("self", "joint_name")))
                .def(PyBasicSensorsVisitor());

            bp::class_<EffortSensor, bp::bases<AbstractSensorBase>,
                       std::shared_ptr<EffortSensor>,
                       boost::noncopyable>("EffortSensor",
                       bp::init<std::string const &>(
                       bp::args("self", "joint_name")))
                .def(PyBasicSensorsVisitor());
        }
    };

    BOOST_PYTHON_VISITOR_EXPOSE(BasicSensors)
}  // End of namespace python.
}  // End of namespace jiminy.
