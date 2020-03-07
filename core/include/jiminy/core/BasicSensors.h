#ifndef JIMINY_STANDARD_SENSORS_H
#define JIMINY_STANDARD_SENSORS_H

#include <iostream>

#include "jiminy/core/AbstractSensor.h"


namespace jiminy
{
    class Model;

    class ImuSensor : public AbstractSensorTpl<ImuSensor>
    {
    public:
        ImuSensor(Model                                 const & model,
                  std::shared_ptr<SensorSharedHolder_t> const & dataHolder,
                  std::string                           const & name);
        ~ImuSensor(void) = default;

        result_t initialize(std::string const & frameName);
        virtual void reset(void) override;

        std::string getFrameName(void) const;

    private:
        result_t set(float64_t const & t,
                     vectorN_t const & q,
                     vectorN_t const & v,
                     vectorN_t const & a,
                     vectorN_t const & u) override;

    private:
        std::string frameName_;
        int32_t     frameIdx_;
    };

    class ForceSensor : public AbstractSensorTpl<ForceSensor>
    {
    public:
        ForceSensor(Model                                 const & model,
                    std::shared_ptr<SensorSharedHolder_t> const & dataHolder,
                    std::string                           const & name);
        ~ForceSensor(void) = default;

        result_t initialize(std::string const & frameName);
        virtual void reset(void) override;

        std::string getFrameName(void) const;

    private:
        result_t set(float64_t const & t,
                     vectorN_t const & q,
                     vectorN_t const & v,
                     vectorN_t const & a,
                     vectorN_t const & u);

    private:
        std::string frameName_;
        int32_t     frameIdx_;
    };

    class EncoderSensor : public AbstractSensorTpl<EncoderSensor>
    {
    public:
        EncoderSensor(Model                                 const & model,
                      std::shared_ptr<SensorSharedHolder_t> const & dataHolder,
                      std::string                           const & name);
        ~EncoderSensor(void) = default;

        result_t initialize(std::string const & motorName);
        virtual void reset(void) override;

        std::string getMotorName(void) const;

    private:
        result_t set(float64_t const & t,
                     vectorN_t const & q,
                     vectorN_t const & v,
                     vectorN_t const & a,
                     vectorN_t const & u);

    private:
        std::string motorName_;
        int32_t     motorPositionIdx_;
        int32_t     motorVelocityIdx_;
    };
}

#endif //end of JIMINY_STANDARD_SENSORS_H