///////////////////////////////////////////////////////////////////////////////////////////////
///
/// \brief          Generic interface for any sensor.
///
///                 Any sensor must inherit from this base class and implement its virtual
///                 methods.
///
///                 Each sensor added to a Jiminy Robot is downcasted as an instance of
///                 AbstractSensor and polymorphism is used to call the actual implementations.
///
///////////////////////////////////////////////////////////////////////////////////////////////

#ifndef JIMINY_ABSTRACT_SENSOR_H
#define JIMINY_ABSTRACT_SENSOR_H

#include "jiminy/core/telemetry/TelemetrySender.h"
#include "jiminy/core/Types.h"

#include <boost/circular_buffer.hpp>


namespace jiminy
{
    class TelemetryData;
    class Robot;

    class AbstractSensorBase;

    ///////////////////////////////////////////////////////////////////////////////////////////////
    ///
    /// \brief      Structure holding the data for every sensors of a given type.
    ///
    /// \details    Every sensors of a given type must have the same 'behavior', e.g. the same
    ///             delay interpolation order and output type. However, their physical properties
    ///             may defer, such as the delay, the noise level or the bias. This enable us to
    ///             optimize the efficiency of data storage by gathering the state of every sensor
    ///             of the given type in Eigen Vectors by simply adding an extra dimension
    ///             corresponding to the sensor ID.
    ///
    ///////////////////////////////////////////////////////////////////////////////////////////////
    struct SensorSharedDataHolder_t
    {
        SensorSharedDataHolder_t(void) :
        time_(),
        data_(),
        sensors_(),
        num_(0),
        delayMax_(0.0)
        {
            // Empty.
        };

        ~SensorSharedDataHolder_t(void) = default;

        boost::circular_buffer_space_optimized<float64_t> time_;    ///< Circular buffer of the stored timesteps
        boost::circular_buffer_space_optimized<matrixN_t> data_;    ///< Circular buffer of past sensor data
        std::vector<AbstractSensorBase *> sensors_;                 ///< Vector of pointers to the sensors
        int32_t num_;                                               ///< Number of sensors of that type
        float64_t delayMax_;                                        ///< Maximum delay over all the sensors
    };

    class AbstractSensorBase: public std::enable_shared_from_this<AbstractSensorBase>
    {
        /* Using friend to avoid double delegation, which would make public
           the attach whereas only robot is able to call it.
           TODO: remove friend declaration and use pluggin mechanism instead.
           It consist in populating a factory method in Robot at runtime with
           lambda function able to create each type of sensors. These lambda
           functions are registered by each sensor using static method. */
        friend Robot;

    protected:
        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Dictionary gathering the configuration options shared between sensors
        ///////////////////////////////////////////////////////////////////////////////////////////////
        virtual configHolder_t getDefaultSensorOptions(void)
        {
            configHolder_t config;
            config["noiseStd"] = vectorN_t();
            config["bias"] = vectorN_t();
            config["delay"] = 0.0;
            config["delayInterpolationOrder"] = 0U;

            return config;
        };

    public:
        struct abstractSensorOptions_t
        {
            vectorN_t const noiseStd;   ///< Standard deviation of the noise of the sensor
            vectorN_t const bias;       ///< Bias of the sensor
            float64_t const delay;      ///< Delay of the sensor
            uint32_t  const delayInterpolationOrder; ///< Order of the interpolation used to compute delayed sensor data. [0: Zero-order holder, 1: Linear interpolation]

            abstractSensorOptions_t(configHolder_t const & options) :
            noiseStd(boost::get<vectorN_t>(options.at("noiseStd"))),
            bias(boost::get<vectorN_t>(options.at("bias"))),
            delay(boost::get<float64_t>(options.at("delay"))),
            delayInterpolationOrder(boost::get<uint32_t>(options.at("delayInterpolationOrder")))
            {
                // Empty.
            }
        };

    public:
        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Forbid the copy of the class
        ///////////////////////////////////////////////////////////////////////////////////////////////
        AbstractSensorBase(AbstractSensorBase const & abstractSensor) = delete;
        AbstractSensorBase & operator = (AbstractSensorBase const & other) = delete;

        ///////////////////////////////////////////////////////////////////////////////////////////////
        ///
        /// \brief      Constructor
        ///
        /// \param[in]  name    Name of the sensor
        ///
        ///////////////////////////////////////////////////////////////////////////////////////////////
        AbstractSensorBase(std::string const & name);
        virtual ~AbstractSensorBase(void) = default;

        template <typename T>
        std::shared_ptr<T> shared_from(T* derived) {
            assert(this == derived);
            return std::static_pointer_cast<T>(shared_from_this());
        }

        ///////////////////////////////////////////////////////////////////////////////////////////////
        ///
        /// \brief Reset the internal state of the sensor.
        ///
        /// \details  This method resets the internal state of the sensor and unset the configuration
        ///           of the telemetry.
        ///
        /// \remark   This method is not intended to be called manually. The Robot to which the
        ///           sensor is added is taking care of it when its own `reset` method is called.
        ///
        ///////////////////////////////////////////////////////////////////////////////////////////////
        virtual void reset(void) = 0;

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief    Refresh the proxies.
        ///
        /// \remark   This method is not intended to be called manually. The Robot to which the
        ///           motor is added is taking care of it when its own `refresh` method is called.
        ///////////////////////////////////////////////////////////////////////////////////////////////
        virtual hresult_t refreshProxies(void) = 0;

        ///////////////////////////////////////////////////////////////////////////////////////////////
        ///
        /// \brief      Configure the telemetry of the sensor.
        ///
        /// \details    This method connects the controller-specific telemetry sender to a given
        ///             telemetry data (which is unique for a given exoskeleton robot), so that it is
        ///             later possible to register the variables that one want to monitor. Finally,
        ///             the telemetry recoder logs every registered variables at each timestep in a
        ///             memory buffer.
        ///
        /// \remark     This method is not intended to be called manually. The Robot to which the
        ///             sensor is added is taking care of it before flushing the telemetry data
        ///             at the end of each simulation steps.
        ///
        /// \param[in]  telemetryData       Shared pointer to the robot-wide telemetry data object
        ///
        /// \return     Return code to determine whether the execution of the method was successful.
        ///
        ///////////////////////////////////////////////////////////////////////////////////////////////
        virtual hresult_t configureTelemetry(std::shared_ptr<TelemetryData> telemetryData,
                                             std::string const & objectPrefixName = "");

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Update the internal buffers of the telemetry associated with variables
        ///             monitored by the sensor.
        ///////////////////////////////////////////////////////////////////////////////////////////////
        void updateTelemetry(void);

        ///////////////////////////////////////////////////////////////////////////////////////////////
        ///
        /// \brief      Update the internal buffers of the telemetry associated with variables
        ///             monitored every sensors of the same type than the current one.
        ///
        /// \remark     This method is not intended to be called manually. The Robot to which the
        ///             sensor is added is taking care of it before flushing the telemetry data at
        //              the end of each simulation steps.
        ///
        ///////////////////////////////////////////////////////////////////////////////////////////////
        virtual void updateTelemetryAll(void) = 0;

        ///////////////////////////////////////////////////////////////////////////////////////////////
        ///
        /// \brief      Set the configuration options of the sensor.
        ///
        /// \param[in]  sensorOptions   Dictionary with the parameters of the sensor
        ///
        ///////////////////////////////////////////////////////////////////////////////////////////////
        virtual hresult_t setOptions(configHolder_t const & sensorOptions);

        ///////////////////////////////////////////////////////////////////////////////////////////////
        ///
        /// \brief      Set the same configuration options of any sensor of the same type than the
        ///             current one.
        ///
        /// \param[in]  sensorOptions   Dictionary with the parameters used for any sensor
        ///
        ///////////////////////////////////////////////////////////////////////////////////////////////
        virtual hresult_t setOptionsAll(configHolder_t const & sensorOptions) = 0;

        ///////////////////////////////////////////////////////////////////////////////////////////////
        ///
        /// \brief      Get the configuration options of the sensor.
        ///
        /// \return     Dictionary with the parameters of the sensor
        ///
        ///////////////////////////////////////////////////////////////////////////////////////////////
        configHolder_t getOptions(void) const;

        ///////////////////////////////////////////////////////////////////////////////////////////////
        ///
        /// \brief      Request the sensor to record data based of the input data.
        ///
        /// \details    It assumes that the internal state of the robot is consistent with the
        ///             input arguments.
        ///
        /// \param[in]  t       Current time
        /// \param[in]  q       Current configuration vector
        /// \param[in]  v       Current velocity vector
        /// \param[in]  a       Current acceleration vector
        /// \param[in]  uMotor  Current motor torque vector
        ///
        /// \return     Return code to determine whether the execution of the method was successful.
        ///
        ///////////////////////////////////////////////////////////////////////////////////////////////
        virtual hresult_t set(float64_t                   const & t,
                              Eigen::Ref<vectorN_t const> const & q,
                              Eigen::Ref<vectorN_t const> const & v,
                              Eigen::Ref<vectorN_t const> const & a,
                              vectorN_t                   const & uMotor) = 0;

        ///////////////////////////////////////////////////////////////////////////////////////////////
        ///
        /// \brief      Request every sensors of the same type than the current one to record data
        ///             based of the input data.
        ///
        /// \details    It assumes that the internal state of the robot is consistent with the
        ///             input arguments.
        ///
        /// \remark     This method is not intended to be called manually. The Robot to which the
        ///             sensor is added is taking care of it while updating the state of the sensors.
        ///
        /// \param[in]  t       Current time
        /// \param[in]  q       Current configuration vector
        /// \param[in]  v       Current velocity vector
        /// \param[in]  a       Current acceleration vector
        /// \param[in]  u       Current motor torque vector
        ///
        /// \return     Return code to determine whether the execution of the method was successful.
        ///
        ///////////////////////////////////////////////////////////////////////////////////////////////
        virtual hresult_t setAll(float64_t                   const & t,
                                 Eigen::Ref<vectorN_t const> const & q,
                                 Eigen::Ref<vectorN_t const> const & v,
                                 Eigen::Ref<vectorN_t const> const & a,
                                 vectorN_t                   const & uMotor) = 0;

        ///////////////////////////////////////////////////////////////////////////////////////////////
        ///
        /// \brief      Get the measurement of the sensor at the current time.
        ///
        /// \details    Note that the current time corresponds to the last time sensor data was
        ///             recorded. If the delay of the sensor is nonzero, then an interpolation method
        ///             is used to compute the delayed measurement based on a buffer of previously
        ///             recorded non-delayed data.
        ///
        /// \return     Eigen reference to a Eigen Vector where to store of sensor measurement.
        ///             It can be an actual vectorN_t, or the extraction of a column vector from
        ///             a higher dimensional tensor.
        ///
        ///////////////////////////////////////////////////////////////////////////////////////////////
        virtual vectorN_t const * get(void) = 0;

        ///////////////////////////////////////////////////////////////////////////////////////////////
        ///
        /// \brief      Get the measurements of all the sensors of the same type than the current one
        ///             at the current time.
        ///
        /// \details    Note that the current time corresponds to the last time sensor data was
        ///             recorded. If the delay of a sensor is nonzero, then an interpolation method
        ///             is used to compute the delayed measurement based on a buffer of previously
        ///             recorded non-delayed data.
        ///
        /// \return     Eigen matrix where to store of measurement of all the sensors.
        ///
        ///////////////////////////////////////////////////////////////////////////////////////////////
        virtual matrixN_t getAll(void) = 0;

        ///////////////////////////////////////////////////////////////////////////////////////////////
        ///
        /// \brief      Get isInitialized_.
        ///
        /// \details    It is a flag used to determine if the sensor has been initialized.
        ///
        /// \remark     Note that a sensor can be considered initialized even if its telemetry is
        ///             not properly configured. If not, it is the only thing to do before being ready
        ///             to use.
        ///
        ///////////////////////////////////////////////////////////////////////////////////////////////
        bool_t const & getIsInitialized(void) const;

        ///////////////////////////////////////////////////////////////////////////////////////////////
        ///
        /// \brief      Get isAttached_.
        ///
        /// \details    It is a flag used to determine if the sensor has been attached to a robot.
        ///
        ///////////////////////////////////////////////////////////////////////////////////////////////
        bool_t const & getIsAttached(void) const;

        ///////////////////////////////////////////////////////////////////////////////////////////////
        ///
        /// \brief      Get isTelemetryConfigured_.
        ///
        /// \details    It is a flag used to determine if the telemetry of the controller has been
        ///             initialized.
        ///
        ///////////////////////////////////////////////////////////////////////////////////////////////
        bool_t const & getIsTelemetryConfigured(void) const;

        ///////////////////////////////////////////////////////////////////////////////////////////////
        ///
        /// \brief      Get name_.
        ///
        /// \details    It is the name of the sensor.
        ///
        ///////////////////////////////////////////////////////////////////////////////////////////////
        std::string const & getName(void) const;

        ///////////////////////////////////////////////////////////////////////////////////////////////
        ///
        /// \brief      Get sensorId_.
        ///
        /// \details    It is the index of the sensor of the global shared buffer.
        ///
        ///////////////////////////////////////////////////////////////////////////////////////////////
        virtual int32_t const & getIdx(void) const = 0;

        ///////////////////////////////////////////////////////////////////////////////////////////////
        ///
        /// \brief      Get type_.
        ///
        /// \details    It is the type of the sensor.
        ///
        ///////////////////////////////////////////////////////////////////////////////////////////////
        virtual std::string const & getType(void) const = 0;

        ///////////////////////////////////////////////////////////////////////////////////////////////
        ///
        /// \brief      It is the size of the sensor's data vector.
        ///
        ///////////////////////////////////////////////////////////////////////////////////////////////
        virtual uint32_t getSize(void) const = 0;

        ///////////////////////////////////////////////////////////////////////////////////////////////
        ///
        /// \brief      Get the name of each element of the data measured by the sensor
        ///
        ///////////////////////////////////////////////////////////////////////////////////////////////
        virtual std::vector<std::string> const & getFieldnames(void) const = 0;

    protected:
        ///////////////////////////////////////////////////////////////////////////////////////////////
        ///
        /// \brief    Attach the sensor to a robot
        ///
        /// \details  This method must be called before initializing the sensor.
        ///
        ///////////////////////////////////////////////////////////////////////////////////////////////
        virtual hresult_t attach(Robot const * robot,
                                 SensorSharedDataHolder_t * sharedHolder) = 0;

        ///////////////////////////////////////////////////////////////////////////////////////////////
        ///
        /// \brief    Detach the sensor from the robot
        ///
        ///////////////////////////////////////////////////////////////////////////////////////////////
        virtual hresult_t detach(void) = 0;

        ///////////////////////////////////////////////////////////////////////////////////////////////
        ///
        /// \brief      Get a Eigen Reference to a Eigen Vector corresponding to the last data
        ///             recorded (or being recorded) by the sensor.
        ///
        /// \details    More precisely, it corresponds to the memory associated with the most recent
        ///             data in the buffer of previously recorded (non-delayed) data. It does not have
        ///             to be filled up at this stage.
        ///
        /// \return     Eigen Reference to a Eigen Vector corresponding to the last data recorded
        ///
        ///////////////////////////////////////////////////////////////////////////////////////////////
        virtual Eigen::Ref<vectorN_t> data(void) = 0;
        static Eigen::Ref<vectorN_t> data(AbstractSensorBase * base) { return base->data(); }

        ///////////////////////////////////////////////////////////////////////////////////////////////
        ///
        /// \brief      Get the name of the sensor in the telemetry.
        ///
        /// \details    Note that the telemetry management is independent for each sensor instead of
        ///             gatering them by type. Nevertheles, the element recorded by a given sensor
        ///             are prefixed with the type and name of the sensor, so that it appears as if
        ///             they were actually gathered.
        ///
        ///////////////////////////////////////////////////////////////////////////////////////////////
        virtual std::string getTelemetryName(void) const = 0;

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Update the measurement buffer.
        ///////////////////////////////////////////////////////////////////////////////////////////////
        virtual hresult_t updateDataBuffer(void) = 0;
        static hresult_t updateDataBuffer(AbstractSensorBase * base) { return base->updateDataBuffer(); }

    public:
        std::unique_ptr<abstractSensorOptions_t const> baseSensorOptions_;    ///< Structure with the parameters of the sensor

    protected:
        configHolder_t sensorOptionsHolder_;    ///< Dictionary with the parameters of the sensor
        bool_t isInitialized_;                  ///< Flag to determine whether the sensor has been initialized or not
        bool_t isAttached_;                     ///< Flag to determine whether the sensor is attached to a robot
        bool_t isTelemetryConfigured_;          ///< Flag to determine whether the telemetry of the sensor has been initialized or not
        Robot const * robot_;                   ///< Robot for which the command and internal dynamics
        std::string name_;                      ///< Name of the sensor
        vectorN_t data_;                        ///< Measurement buffer to avoid recomputing the same "current" measurement multiple times

    private:
        TelemetrySender telemetrySender_;       ///< Telemetry sender of the sensor used to register and update telemetry variables
    };

    template<class T>
    class AbstractSensorTpl : public AbstractSensorBase
    {
    public:
        // Disable the copy of the class
        AbstractSensorTpl(AbstractSensorTpl const & abstractSensor) = delete;
        AbstractSensorTpl & operator = (AbstractSensorTpl const & other) = delete;

    public:
        AbstractSensorTpl(std::string const & name);
        virtual ~AbstractSensorTpl(void);

        auto shared_from_this() { return shared_from(this); }
        auto shared_from_this() const { return shared_from(this); }

        virtual void reset(void) override;
        void updateTelemetryAll(void) override final;

        virtual hresult_t setOptions(configHolder_t const & sensorOptions) override;
        virtual hresult_t setOptionsAll(configHolder_t const & sensorOptions) override final;
        virtual int32_t const & getIdx(void) const override final;
        virtual std::string const & getType(void) const override final;
        virtual std::vector<std::string> const & getFieldnames(void) const final;
        virtual uint32_t getSize(void) const override final;

        virtual vectorN_t const * get(void) override final;
        virtual matrixN_t getAll(void) override final;
        virtual hresult_t setAll(float64_t                   const & t,
                                 Eigen::Ref<vectorN_t const> const & q,
                                 Eigen::Ref<vectorN_t const> const & v,
                                 Eigen::Ref<vectorN_t const> const & a,
                                 vectorN_t                   const & uMotor) override final;

    protected:
        using AbstractSensorBase::data;
        virtual Eigen::Ref<vectorN_t> data(void) override final;

    private:
        virtual hresult_t attach(Robot const * robot,
                                 SensorSharedDataHolder_t * sharedHolder) override final;
        virtual hresult_t detach(void) override final;
        virtual std::string getTelemetryName(void) const override final;
        using AbstractSensorBase::updateDataBuffer;
        virtual hresult_t updateDataBuffer(void) override final;
        void clearDataBuffer(void);

    public:
        /* Be careful, the static variables must be const since the 'static'
           keyword binds all the sensors together, even if they are associated
           to complete separated robots. */
        static std::string const type_;
        static std::vector<std::string> const fieldNames_;
        static bool_t const areFieldnamesGrouped_;

    protected:
        int32_t sensorId_;

    private:
        using AbstractSensorBase::data_;
        SensorSharedDataHolder_t * sharedHolder_;
    };
}

#include "jiminy/core/robot/AbstractSensor.tpp"

#endif //end of JIMINY_ABSTRACT_SENSOR_H
