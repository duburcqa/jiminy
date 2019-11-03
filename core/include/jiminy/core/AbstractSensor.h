#ifndef SIMU_ABSTRACT_SENSOR_H
#define SIMU_ABSTRACT_SENSOR_H

#include "jiminy/core/Utilities.h"
#include "jiminy/core/TelemetrySender.h"
#include "jiminy/core/Model.h"
#include "jiminy/core/Types.h"

namespace jiminy
{
    static uint8_t const MIN_DELAY_BUFFER_RESERVE(20);
    static uint8_t const MAX_DELAY_BUFFER_EXCEED(20);

    class TelemetryData;
    template<typename> class AbstractSensorTpl;

    class AbstractSensorBase
    {
        template<typename> friend class AbstractSensorTpl;

    public:
        virtual configHolder_t getDefaultOptions(void)
        {
            configHolder_t config;
            config["rawData"] = false;
            config["noiseStd"] = vectorN_t();
            config["bias"] = vectorN_t();
            config["delay"] = 0.0;
            config["delayInterpolationOrder"] = 0U; // [0: Zero-order holder, 1: Linear interpolation]

            return config;
        };

        struct abstractSensorOptions_t
        {
            bool      const rawData;
            vectorN_t const noiseStd;
            vectorN_t const bias;
            float64_t const delay;
            uint32_t  const delayInterpolationOrder;

            abstractSensorOptions_t(configHolder_t const & options) :
            rawData(boost::get<bool>(options.at("rawData"))),
            noiseStd(boost::get<vectorN_t>(options.at("noiseStd"))),
            bias(boost::get<vectorN_t>(options.at("bias"))),
            delay(boost::get<float64_t>(options.at("delay"))),
            delayInterpolationOrder(boost::get<uint32_t>(options.at("delayInterpolationOrder")))
            {
                // Empty.
            }
        };

    public:
        // Disable the copy of the class
        AbstractSensorBase(AbstractSensorBase const & abstractSensor) = delete;
        AbstractSensorBase & operator = (AbstractSensorBase const & other) = delete;

    public:
        AbstractSensorBase(Model       const & model,
                           std::string const & name);
        virtual ~AbstractSensorBase(void);

        virtual void reset(void) = 0;
        virtual result_t configureTelemetry(std::shared_ptr<TelemetryData> const & telemetryData);

        configHolder_t getOptions(void);
        virtual void setOptions(configHolder_t const & sensorOptions);
        virtual void setOptionsAll(configHolder_t const & sensorOptions) = 0;
        bool const & getIsInitialized(void) const;
        bool const & getIsTelemetryConfigured(void) const;
        virtual std::string const & getName(void) const;
        virtual std::string const & getType(void) const = 0;
        virtual std::vector<std::string> const & getFieldNames(void) const = 0;

        virtual result_t get(Eigen::Ref<vectorN_t> data) = 0;
        virtual result_t getAll(matrixN_t & data) = 0;
        virtual result_t setAll(float64_t const & t,
                                vectorN_t const & q,
                                vectorN_t const & v,
                                vectorN_t const & a,
                                vectorN_t const & u) = 0;
        virtual void updateTelemetryAll(void) = 0;

    protected:
        virtual std::string getTelemetryName(void) const = 0;

        virtual matrixN_t::ColXpr data(void) = 0;
        virtual result_t set(float64_t const & t,
                             vectorN_t const & q,
                             vectorN_t const & v,
                             vectorN_t const & a,
                             vectorN_t const & u) = 0;
        void updateTelemetry(void);

    public:
        std::unique_ptr<abstractSensorOptions_t const> sensorOptions_;

    protected:
        configHolder_t sensorOptionsHolder_;
        TelemetrySender telemetrySender_;
        bool isInitialized_;
        bool isTelemetryConfigured_;
        Model const * model_; // Raw pointer to avoid managing its deletion

    private:
        std::string name_;
        vectorN_t data_;
        bool isDataUpToDate_;
    };

    template<class T>
    class AbstractSensorTpl : public AbstractSensorBase
    {
    public:
        // Disable the copy of the class
        AbstractSensorTpl(AbstractSensorTpl const & abstractSensor) = delete;
        AbstractSensorTpl & operator = (AbstractSensorTpl const & other) = delete;

    public:
        AbstractSensorTpl(Model                               const & model,
                          std::shared_ptr<SensorDataHolder_t> const & dataHolder,
                          std::string                         const & name);
        virtual ~AbstractSensorTpl(void);

        virtual void reset(void) override;

        virtual void setOptions(configHolder_t const & sensorOptions) override;
        virtual void setOptionsAll(configHolder_t const & sensorOptions) override;
        virtual std::string const & getType(void) const override;
        std::vector<std::string> const & getFieldNames(void) const;
        uint32_t getSize(void) const;

        virtual result_t get(Eigen::Ref<vectorN_t> data) override; // Eigen::Ref<vectorN_t> = anything that looks like a vectorN_t
        virtual result_t getAll(matrixN_t & data) override;
        virtual result_t setAll(float64_t const & t,
                                vectorN_t const & q,
                                vectorN_t const & v,
                                vectorN_t const & a,
                                vectorN_t const & u) override;
        void updateTelemetryAll(void) override;

    protected:
        virtual std::string getTelemetryName(void) const override;

        virtual matrixN_t::ColXpr data(void) override;

    public:
        static std::string const type_;
        static bool const areFieldNamesGrouped_;
        static std::vector<std::string> const fieldNamesPostProcess_;
        static std::vector<std::string> const fieldNamesPreProcess_;
        static float64_t delayMax_;

    private:
        std::shared_ptr<SensorDataHolder_t> dataHolder_;
        uint32_t sensorId_;
    };
}

#include "jiminy/core/AbstractSensor.tcc"

#endif //end of SIMU_ABSTRACT_SENSOR_H
