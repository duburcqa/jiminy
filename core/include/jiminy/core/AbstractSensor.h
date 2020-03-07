///////////////////////////////////////////////////////////////////////////////////////////////
///
/// \brief          Generic interface for any sensor.
///
///                 Any sensor must inherit from this base class and implement its virtual
///                 methods.
///
///                 Each sensor added to a Jiminy Model is downcasted as an instance of
///                 AbstractSensor and polymorphism is used to call the actual implementations.
///
///////////////////////////////////////////////////////////////////////////////////////////////

#ifndef JIMINY_ABSTRACT_SENSOR_H
#define JIMINY_ABSTRACT_SENSOR_H

#include <boost/circular_buffer.hpp>

#include "jiminy/core/Utilities.h"
#include "jiminy/core/TelemetrySender.h"
#include "jiminy/core/Types.h"

namespace jiminy
{
    static uint8_t const MIN_DELAY_BUFFER_RESERVE(20); ///< Minimum memory allocation is memory is full and the older data stored is dated less than the desired delay
    static uint8_t const MAX_DELAY_BUFFER_EXCEED(20);  ///< Maximum number of data stored allowed to be dated more than the desired delay

    class TelemetryData;
    class Model;

    class AbstractSensorBase;
    template<typename> class AbstractSensorTpl;

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
        uint8_t num_;                                               ///< Number of sensors of that type
        float64_t delayMax_;                                        ///< Maximum delay over all the sensors
    };

    class AbstractSensorBase
    {
        friend class Model;

        template<typename> friend class AbstractSensorTpl;

    public:
        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Dictionary gathering the configuration options shared between sensors
        ///////////////////////////////////////////////////////////////////////////////////////////////
        virtual configHolder_t getDefaultOptions(void)
        {
            configHolder_t config;
            config["noiseStd"] = vectorN_t();
            config["bias"] = vectorN_t();
            config["delay"] = 0.0;
            config["delayInterpolationOrder"] = 0U;

            return config;
        };

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
        /// \param[in]  model   Model of the system
        /// \param[in]  name    Name of the sensor
        ///
        ///////////////////////////////////////////////////////////////////////////////////////////////
        AbstractSensorBase(Model       const & model,
                           std::string const & name);

        virtual ~AbstractSensorBase(void) = default;

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
        /// \brief      Set the configuration options of the sensor.
        ///
        /// \param[in]  sensorOptions   Dictionary with the parameters of the sensor
        ///
        ///////////////////////////////////////////////////////////////////////////////////////////////
        virtual void setOptions(configHolder_t const & sensorOptions);

        ///////////////////////////////////////////////////////////////////////////////////////////////
        ///
        /// \brief      Set the same configuration options of any sensor of the same type than the
        ///             current one.
        ///
        /// \param[in]  sensorOptions   Dictionary with the parameters used for any sensor
        ///
        ///////////////////////////////////////////////////////////////////////////////////////////////
        virtual void setOptionsAll(configHolder_t const & sensorOptions) = 0;

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
        virtual std::string const & getName(void) const;

        ///////////////////////////////////////////////////////////////////////////////////////////////
        ///
        /// \brief      Get sensorId_.
        ///
        /// \details    It is the index of the sensor of the global shared buffer.
        ///
        ///////////////////////////////////////////////////////////////////////////////////////////////
        virtual uint8_t const & getId(void) const = 0;

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
        virtual std::vector<std::string> const & getFieldNames(void) const = 0;

    protected:
        ///////////////////////////////////////////////////////////////////////////////////////////////
        ///
        /// \brief Reset the internal state of the sensor.
        ///
        /// \details  This method resets the internal state of the sensor and unset the configuration
        ///           of the telemetry.
        ///
        /// \remark   This method is not intended to be called manually. The Model to which the
        ///           sensor is added is taking care of it when its own `reset` method is called.
        ///
        ///////////////////////////////////////////////////////////////////////////////////////////////
        virtual void reset(void) = 0;

        ///////////////////////////////////////////////////////////////////////////////////////////////
        ///
        /// \brief      Configure the telemetry of the sensor.
        ///
        /// \details    This method connects the controller-specific telemetry sender to a given
        ///             telemetry data (which is unique for a given exoskeleton model), so that it is
        ///             later possible to register the variables that one want to monitor. Finally,
        ///             the telemetry recoder logs every registered variables at each timestep in a
        ///             memory buffer.
        ///
        /// \remark     This method is not intended to be called manually. The Model to which the
        ///             sensor is added is taking care of it before flushing the telemetry data
        ///             at the end of each simulation steps.
        ///
        /// \param[in]  telemetryData       Shared pointer to the model-wide telemetry data object
        ///
        /// \return     Return code to determine whether the execution of the method was successful.
        ///
        ///////////////////////////////////////////////////////////////////////////////////////////////
        virtual result_t configureTelemetry(std::shared_ptr<TelemetryData> const & telemetryData);

        ///////////////////////////////////////////////////////////////////////////////////////////////
        ///
        /// \brief      Update the internal buffers of the telemetry associated with variables
        ///             monitored every sensors of the same type than the current one.
        ///
        /// \remark     This method is not intended to be called manually. The Model to which the
        ///             sensor is added is taking care of it before flushing the telemetry data at
        //              the end of each simulation steps.
        ///
        ///////////////////////////////////////////////////////////////////////////////////////////////
        virtual void updateTelemetryAll(void) = 0;

        ///////////////////////////////////////////////////////////////////////////////////////////////
        ///
        /// \brief      Request every sensors of the same type than the current one to record data
        ///             based of the input data.
        ///
        /// \details    It assumes that the internal state of the model is consistent with the
        ///             input arguments.
        ///
        /// \remark     This method is not intended to be called manually. The Model to which the
        ///             sensor is added is taking care of it while updating the state of the sensors.
        ///
        /// \param[in]  t       Current time
        /// \param[in]  q       Current configuration vector
        /// \param[in]  v       Current velocity vector
        /// \param[in]  a       Current acceleration vector
        /// \param[in]  u       Current torque vector
        ///
        /// \return     Return code to determine whether the execution of the method was successful.
        ///
        ///////////////////////////////////////////////////////////////////////////////////////////////
        virtual result_t setAll(float64_t const & t,
                                vectorN_t const & q,
                                vectorN_t const & v,
                                vectorN_t const & a,
                                vectorN_t const & u) = 0;

    private:
        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Update the internal buffers of the telemetry associated with variables
        ///             monitored by the sensor.
        ///////////////////////////////////////////////////////////////////////////////////////////////
        void updateTelemetry(void);

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

        ///////////////////////////////////////////////////////////////////////////////////////////////
        ///
        /// \brief      Request the sensor to record data based of the input data.
        ///
        /// \details    It assumes that the internal state of the model is consistent with the
        ///             input arguments.
        ///
        /// \param[in]  t       Current time
        /// \param[in]  q       Current configuration vector
        /// \param[in]  v       Current velocity vector
        /// \param[in]  a       Current acceleration vector
        /// \param[in]  u       Current torque vector
        ///
        /// \return     Return code to determine whether the execution of the method was successful.
        ///
        ///////////////////////////////////////////////////////////////////////////////////////////////
        virtual result_t set(float64_t const & t,
                             vectorN_t const & q,
                             vectorN_t const & v,
                             vectorN_t const & a,
                             vectorN_t const & u) = 0;

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
        virtual result_t updateDataBuffer(void) = 0;

    public:
        std::unique_ptr<abstractSensorOptions_t const> sensorOptions_; ///< Structure with the parameters of the sensor

    protected:
        configHolder_t sensorOptionsHolder_;    ///< Dictionary with the parameters of the sensor
        TelemetrySender telemetrySender_;       ///< Telemetry sender of the sensor used to register and update telemetry variables
        bool_t isInitialized_;                  ///< Flag to determine whether the controller has been initialized or not
        bool_t isTelemetryConfigured_;          ///< Flag to determine whether the telemetry of the controller has been initialized or not
        Model const * model_;                   ///< Model of the system for which the command and internal dynamics

    private:
        std::string name_;                      ///< Name of the sensor
        vectorN_t data_;                        ///< Measurement buffer to avoid recomputing the same "current" measurement multiple times
    };

    template<class T>
    class AbstractSensorTpl : public AbstractSensorBase
    {
    public:
        // Disable the copy of the class
        AbstractSensorTpl(AbstractSensorTpl const & abstractSensor) = delete;
        AbstractSensorTpl & operator = (AbstractSensorTpl const & other) = delete;

    public:
        AbstractSensorTpl(Model       const & model,
                          std::shared_ptr<SensorSharedDataHolder_t> const & dataHolder,
                          std::string const & name);
        virtual ~AbstractSensorTpl(void);

        virtual void reset(void) override;

        virtual void setOptions(configHolder_t const & sensorOptions) override;
        virtual void setOptionsAll(configHolder_t const & sensorOptions) override final;
        virtual uint8_t const & getId(void) const override final;
        virtual std::string const & getType(void) const override final;
        virtual std::vector<std::string> const & getFieldNames(void) const final;
        virtual uint32_t getSize(void) const override final;

        virtual vectorN_t const * get(void) override final;
        virtual matrixN_t getAll(void) override final;
        virtual result_t setAll(float64_t const & t,
                                vectorN_t const & q,
                                vectorN_t const & v,
                                vectorN_t const & a,
                                vectorN_t const & u) override final;
        void updateTelemetryAll(void) override final;

    protected:
        virtual std::string getTelemetryName(void) const override final;

        virtual Eigen::Ref<vectorN_t> data(void) override final;

    private:
        virtual result_t updateDataBuffer(void) override final;

    public:
        /* Be careful, the variable must be const if static since the 'static'
           keyword binds all the sensors together, even if they are associated
           to complete separated models! */
        static std::string const type_;
        static bool_t const areFieldNamesGrouped_;
        static std::vector<std::string> const fieldNames_;

    private:
        std::shared_ptr<SensorSharedDataHolder_t> sharedHolder_;
        uint8_t sensorId_;
    };
}

#include "jiminy/core/AbstractSensor.tpp"

#endif //end of JIMINY_ABSTRACT_SENSOR_H
