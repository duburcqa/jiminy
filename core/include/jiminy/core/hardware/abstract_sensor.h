#ifndef JIMINY_ABSTRACT_SENSOR_H
#define JIMINY_ABSTRACT_SENSOR_H

#include "jiminy/core/fwd.h"
#include "jiminy/core/utilities/random.h"  // `PCG32`

#include <boost/circular_buffer.hpp>
#include <boost/functional/hash.hpp>
#include <boost/multi_index/hashed_index.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/sequenced_index.hpp>
#include <boost/multi_index/tag.hpp>
#include <boost/multi_index_container.hpp>


namespace jiminy
{
    class TelemetryData;
    class TelemetrySender;
    class Robot;
    class AbstractSensorBase;

    /// \brief Structure gathering data shared amongst all sensors of a given type.
    ///
    /// \details All sensors of a given type must have the same 'behavior', e.g. the same delay
    ///          interpolation order and output type. However, their physical properties may
    ///          differ, such as the delay, the noise level or the bias. This enables to optimize
    ///          their memory layout by stacking their respective state in Eigen Matrices. This
    ///          way, performing the same operation on all sensors of the given type would be much
    ///          faster thanks to memory locality and vectorized computing via SIMD instructions.
    struct SensorSharedDataHolder_t
    {
        /// \brief Circular buffer of the stored timesteps.
        boost::circular_buffer<double> time_;
        /// \brief Circular buffer of past sensor real data.
        boost::circular_buffer<Eigen::MatrixXd> data_;
        /// \brief Current sensor measurements.
        Eigen::MatrixXd dataMeasured_;
        /// \brief Vector of pointers to the sensors.
        std::vector<AbstractSensorBase *> sensors_;
        /// \brief Number of sensors currently sharing this buffer.
        std::size_t num_;
        /// \brief Maximum delay over all the sensors.
        double delayMax_;
    };

    /// \brief Generic interface for any sensor.
    ///
    /// \details Any sensor must inherit from this base class and implement its virtual methods.
    class JIMINY_DLLAPI AbstractSensorBase :
    public std::enable_shared_from_this<AbstractSensorBase>
    {
        /* Using friend to avoid double delegation, which would make public the attach, whereas
           only robot is able to call it.

           TODO: Remove friend declaration and use plugin mechanism instead. It consist in
           populating a factory method in Robot at runtime with lambda function able to create each
           type of sensors. These lambda functions are registered by each sensor using static
           method. */
        friend Robot;

        template<typename T>
        friend class AbstractSensorTpl;

    public:
        /// \brief Dictionary gathering the configuration options shared between sensors
        virtual GenericConfig getDefaultSensorOptions()
        {
            GenericConfig config;
            config["noiseStd"] = Eigen::VectorXd{};
            config["bias"] = Eigen::VectorXd{};
            config["delay"] = 0.0;
            config["jitter"] = 0.0;
            config["delayInterpolationOrder"] = 1U;

            return config;
        };

        struct abstractSensorOptions_t
        {
            /// \brief Standard deviation of the noise of the sensor.
            const Eigen::VectorXd noiseStd;
            /// \brief Bias of the sensor.
            const Eigen::VectorXd bias;
            /// \brief Delay of the sensor.
            const double delay;
            /// \brief Jitter of the sensor.
            const double jitter;
            /// \brief Order of the interpolation used to compute delayed sensor data.
            ///
            /// \details [0: Zero-order holder, 1: Linear interpolation].
            const uint32_t delayInterpolationOrder;

            abstractSensorOptions_t(const GenericConfig & options) :
            noiseStd(boost::get<Eigen::VectorXd>(options.at("noiseStd"))),
            bias(boost::get<Eigen::VectorXd>(options.at("bias"))),
            delay(boost::get<double>(options.at("delay"))),
            jitter(boost::get<double>(options.at("jitter"))),
            delayInterpolationOrder(boost::get<uint32_t>(options.at("delayInterpolationOrder")))
            {
            }
        };

    public:
        DISABLE_COPY(AbstractSensorBase)

    public:
        /// \param[in] name Name of the sensor
        explicit AbstractSensorBase(const std::string & name) noexcept;
        virtual ~AbstractSensorBase();

        /// \brief Reset the internal state of the sensors.
        ///
        /// \details This method resets the internal state of the sensor and unset the
        ///          configuration of the telemetry.
        ///
        /// \remark This method is not intended to be called manually. The Robot to which the
        ///         sensor is added is taking care of it when its own `reset` method is called.
        virtual hresult_t resetAll(uint32_t seed) = 0;

        /// \brief Refresh the proxies.
        ///
        /// \remark This method is not intended to be called manually. The Robot to which the motor
        ///         is added is taking care of it when its own `refresh` method is called.
        virtual hresult_t refreshProxies() = 0;

        /// \brief Configure the telemetry of the sensor.
        ///
        /// \details This method connects the controller-specific telemetry sender to a given
        ///          telemetry data (which is unique for a given exoskeleton robot), so that it is
        ///          later possible to register the variables that one want to monitor. Finally,
        ///          the telemetry recorder logs every registered variables at each timestep in a
        ///          memory buffer.
        ///
        /// \remark This method is not intended to be called manually. The Robot to which the
        ///         sensor is added is taking care of it before flushing the telemetry data at the
        ///         end of each simulation steps.
        ///
        /// \param[in] telemetryData Shared pointer to the robot-wide telemetry data object
        ///
        /// \return Return code to determine whether the execution of the method was successful.
        virtual hresult_t configureTelemetry(std::shared_ptr<TelemetryData> telemetryData,
                                             const std::string & objectPrefixName = {});

        /// \brief Update the internal buffers of the telemetry associated with variables monitored
        ///        by the sensor.
        void updateTelemetry();

        /// \brief Update the internal buffers of the telemetry associated with variables monitored
        ///        every sensors of the same type than the current one.
        ///
        /// \remarks This method is not intended to be called manually. The Robot to which the
        ///          sensor is added is taking care of it before flushing the telemetry data at the
        ///          end of each simulation steps.
        virtual void updateTelemetryAll() = 0;

        /// \brief Set the configuration options of the sensor.
        ///
        /// \param[in] sensorOptions Dictionary with the parameters of the sensor.
        virtual hresult_t setOptions(const GenericConfig & sensorOptions);

        /// \brief Set the same configuration options of any sensor of the same type than the
        ///        current one.
        ///
        /// \param[in] sensorOptions Dictionary with the parameters used for any sensor.
        virtual hresult_t setOptionsAll(const GenericConfig & sensorOptions) = 0;

        /// \brief Configuration options of the sensor.
        GenericConfig getOptions() const noexcept;

        template<typename DerivedType>
        hresult_t set(const Eigen::MatrixBase<DerivedType> & value);

        /// \brief Measurement of the sensor at the current time.
        ///
        /// \details Note that the current time corresponds to the last time sensor data was
        ///          recorded. If the delay of the sensor is nonzero, then an interpolation method
        ///          is used to compute the delayed measurement based on a buffer of previously
        ///          recorded non-delayed data.
        ///
        /// \return Eigen reference to a Eigen Vector where to store of sensor measurement. It can
        ///         be an actual Eigen::VectorXd, or the extraction of a column vector from a
        ///         higher dimensional tensor.
        virtual Eigen::Ref<const Eigen::VectorXd> get() const = 0;

        /// \brief Whether the sensor has been initialized.
        ///
        /// \remark Note that a sensor can be considered initialized even if its telemetry is not
        ///         properly configured. If not, it must be done before being ready to use.
        bool getIsInitialized() const;

        /// \brief Whether the sensor has been attached to a robot.
        bool getIsAttached() const;

        /// \brief Whether the telemetry of the controller has been initialized.
        bool getIsTelemetryConfigured() const;

        /// \brief Name of the sensor.
        const std::string & getName() const;

        /// \brief Index of the sensor of the global shared buffer.
        virtual std::size_t getIdx() const = 0;

        /// \brief Type of the sensor.
        virtual const std::string & getType() const = 0;

        /// \brief It is the size of the sensor's data vector.
        virtual std::size_t getSize() const = 0;

        /// \brief Name of each element of the data measured by the sensor.
        virtual const std::vector<std::string> & getFieldnames() const = 0;

    protected:
        /// \brief Request every sensors of the same type than the current one to record data based
        ///        of the input data.
        ///
        /// \details It assumes that the internal state of the robot is consistent with the input
        ///          arguments.
        ///
        /// \remarks This method is not intended to be called manually. The Robot to which the
        ///          sensor is added is taking care of it while updating the state of the sensors.
        ///
        /// \param[in] t Current time.
        /// \param[in] q Current configuration of the robot.
        /// \param[in] v Current velocity of the robot.
        /// \param[in] a Current acceleration of the robot.
        /// \param[in] uMotor Current motor efforts.
        /// \param[in] fExternal Current external forces applied on the robot.
        ///
        /// \return Return code to determine whether the execution of the method was successful.
        virtual hresult_t setAll(double t,
                                 const Eigen::VectorXd & q,
                                 const Eigen::VectorXd & v,
                                 const Eigen::VectorXd & a,
                                 const Eigen::VectorXd & uMotor,
                                 const ForceVector & fExternal) = 0;

        /// \brief Request the sensor to record data based of the input data.
        ///
        /// \details It assumes that the internal state of the robot is consistent with the input
        ///          arguments.
        ///
        /// \param[in] t Current time.
        /// \param[in] q Current configuration of the robot.
        /// \param[in] v Current velocity of the robot.
        /// \param[in] a Current acceleration of the robot.
        /// \param[in] uMotor Current motor efforts.
        /// \param[in] fExternal Current external forces applied on the robot.
        ///
        /// \return Return code to determine whether the execution of the method was successful.
        virtual hresult_t set(double t,
                              const Eigen::VectorXd & q,
                              const Eigen::VectorXd & v,
                              const Eigen::VectorXd & a,
                              const Eigen::VectorXd & uMotor,
                              const ForceVector & fExternal) = 0;

        /// \brief Attach the sensor to a robot.
        ///
        /// \details This method must be called before initializing the sensor.
        virtual hresult_t attach(std::weak_ptr<const Robot> robot,
                                 SensorSharedDataHolder_t * sharedHolder) = 0;

        /// \brief Detach the sensor from the robot.
        virtual hresult_t detach() = 0;

        /// \brief Eigen Reference to a Eigen Vector corresponding to the last data recorded (or
        ///        being recorded) by the sensor.
        ///
        /// \details More precisely, it corresponds to the memory associated with the most recent
        ///          data in the buffer of previously recorded (non-delayed) data. It does not have
        ///          to be filled up at this stage.
        ///
        /// \return Eigen Reference to a Eigen Vector corresponding to the last data recorded.
        virtual Eigen::Ref<Eigen::VectorXd> data() = 0;

        virtual Eigen::Ref<Eigen::VectorXd> get() = 0;

        /// \brief Name of the sensor in the telemetry.
        ///
        /// \details Note that the telemetry management is independent for each sensor instead of
        ///          gathering them by type. Nevertheless, the element recorded by a given sensor
        ///          are prefixed with the type and name of the sensor, so that it appears as if
        ///          they were actually gathered.
        virtual std::string getTelemetryName() const = 0;

        /// \brief Set the measurement buffer with the real data interpolated at the current time.
        virtual hresult_t interpolateData() = 0;

        /// \brief Add white noise and bias to the measurement buffer.
        virtual void measureData();

        /// \brief Set the measurement buffer with true data, but skewed with white noise and bias.
        virtual hresult_t measureDataAll() = 0;

    public:
        /// \brief Structure with the parameters of the sensor
        std::unique_ptr<const abstractSensorOptions_t> baseSensorOptions_{nullptr};

    protected:
        /// \brief Dictionary with the parameters of the sensor.
        GenericConfig sensorOptionsHolder_{};
        /// \brief Flag to determine whether the sensor has been initialized.
        bool isInitialized_{false};
        /// \brief Flag to determine whether the sensor is attached to a robot.
        bool isAttached_{false};
        /// \brief Flag to determine whether the telemetry of the sensor has been initialized.
        bool isTelemetryConfigured_{false};
        /// \brief Robot for which the command and internal dynamics Name of the sensor.
        std::weak_ptr<const Robot> robot_{};
        /// \brief Random number generator used internally for sampling measurement noise.
        PCG32 generator_;
        /// \brief Name of the sensor.
        std::string name_;

    private:
        /// \brief Telemetry sender of the sensor used to register and update telemetry variables.
        std::unique_ptr<TelemetrySender> telemetrySender_;
    };

    template<typename T>
    class AbstractSensorTpl : public AbstractSensorBase
    {
    public:
        DISABLE_COPY(AbstractSensorTpl)

    public:
        using AbstractSensorBase::AbstractSensorBase;
        virtual ~AbstractSensorTpl();

        auto shared_from_this() { return shared_from(this); }
        auto shared_from_this() const { return shared_from(this); }

        hresult_t resetAll(uint32_t seed) override final;
        void updateTelemetryAll() override final;

        virtual hresult_t setOptionsAll(const GenericConfig & sensorOptions) override final;
        virtual std::size_t getIdx() const override final;
        virtual const std::string & getType() const override final;
        virtual const std::vector<std::string> & getFieldnames() const final;
        virtual std::size_t getSize() const override final;

        virtual Eigen::Ref<const Eigen::VectorXd> get() const override final;

    protected:
        virtual hresult_t setAll(double t,
                                 const Eigen::VectorXd & q,
                                 const Eigen::VectorXd & v,
                                 const Eigen::VectorXd & a,
                                 const Eigen::VectorXd & uMotor,
                                 const ForceVector & fExternal) override final;
        virtual Eigen::Ref<Eigen::VectorXd> get() override final;
        virtual Eigen::Ref<Eigen::VectorXd> data() override final;

    private:
        virtual hresult_t attach(std::weak_ptr<const Robot> robot,
                                 SensorSharedDataHolder_t * sharedHolder) override final;
        virtual hresult_t detach() override final;
        virtual std::string getTelemetryName() const override final;
        virtual hresult_t interpolateData() override final;
        virtual hresult_t measureDataAll() override final;

    public:
        /* Be careful, the static variables must be const since the 'static' keyword binds all the
           sensors together, even if they are associated to complete separated robots. */
        static const std::string type_;
        static const std::vector<std::string> fieldnames_;
        static const bool areFieldnamesGrouped_;

    protected:
        std::size_t sensorIdx_{0};

    private:
        SensorSharedDataHolder_t * sharedHolder_{nullptr};
    };
}

#include "jiminy/core/hardware/abstract_sensor.hxx"

#endif  // JIMINY_ABSTRACT_SENSOR_H
