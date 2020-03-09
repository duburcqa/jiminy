#ifndef JIMINY_BASIC_SENSORS_H
#define JIMINY_BASIC_SENSORS_H

#include "jiminy/core/AbstractSensor.h"


namespace jiminy
{
    class Model;

    class ImuSensor : public AbstractSensorTpl<ImuSensor>
    {
    public:
        ImuSensor(Model       const & model,
                  std::shared_ptr<SensorSharedDataHolder_t> const & dataHolder,
                  std::string const & name);
        ~ImuSensor(void) = default;

        result_t initialize(std::string const & frameName);

        virtual result_t refreshProxies(void) override;

        std::string const & getFrameName(void) const;

    private:
        virtual result_t set(float64_t const & t,
                             vectorN_t const & q,
                             vectorN_t const & v,
                             vectorN_t const & a,
                             vectorN_t const & u) override;

    private:
        std::string frameName_;
        int32_t frameIdx_;
    };

    class ForceSensor : public AbstractSensorTpl<ForceSensor>
    {
    public:
        ForceSensor(Model       const & model,
                    std::shared_ptr<SensorSharedDataHolder_t> const & dataHolder,
                    std::string const & name);
        ~ForceSensor(void) = default;

        result_t initialize(std::string const & frameName);

        virtual result_t refreshProxies(void) override;

        std::string const & getFrameName(void) const;

    private:
        virtual result_t set(float64_t const & t,
                             vectorN_t const & q,
                             vectorN_t const & v,
                             vectorN_t const & a,
                             vectorN_t const & u);

    private:
        std::string frameName_;
        int32_t frameIdx_;
    };

    class EncoderSensor : public AbstractSensorTpl<EncoderSensor>
    {
    public:
        EncoderSensor(Model       const & model,
                      std::shared_ptr<SensorSharedDataHolder_t> const & dataHolder,
                      std::string const & name);
        ~EncoderSensor(void) = default;

        result_t initialize(std::string const & jointName);

        virtual result_t refreshProxies(void) override;

        std::string const & getJointName(void) const;

    private:
        virtual result_t set(float64_t const & t,
                             vectorN_t const & q,
                             vectorN_t const & v,
                             vectorN_t const & a,
                             vectorN_t const & u);

    private:
        std::string jointName_;
        int32_t jointPositionIdx_;
        int32_t jointVelocityIdx_;
    };
}

#endif //end of JIMINY_BASIC_SENSORS_H