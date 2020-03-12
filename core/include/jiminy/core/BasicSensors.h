#ifndef JIMINY_BASIC_SENSORS_H
#define JIMINY_BASIC_SENSORS_H

#include "jiminy/core/AbstractSensor.h"


namespace jiminy
{
    class Model;

    class ImuSensor : public AbstractSensorTpl<ImuSensor>
    {
    public:
        ImuSensor(std::string const & name);
        ~ImuSensor(void) = default;

        result_t initialize(std::string const & frameName);

        virtual result_t refreshProxies(void) override;

        std::string const & getFrameName(void) const;

    private:
        virtual result_t set(float64_t const & t,
                             vectorN_t const & q,
                             vectorN_t const & v,
                             vectorN_t const & a,
                             vectorN_t const & uMotor) override;

    private:
        std::string frameName_;
        int32_t frameIdx_;
    };

    class ForceSensor : public AbstractSensorTpl<ForceSensor>
    {
    public:
        ForceSensor(std::string const & name);
        ~ForceSensor(void) = default;

        result_t initialize(std::string const & frameName);

        virtual result_t refreshProxies(void) override;

        std::string const & getFrameName(void) const;

    private:
        virtual result_t set(float64_t const & t,
                             vectorN_t const & q,
                             vectorN_t const & v,
                             vectorN_t const & a,
                             vectorN_t const & uMotor);

    private:
        std::string frameName_;
        int32_t frameIdx_;
    };

    class EncoderSensor : public AbstractSensorTpl<EncoderSensor>
    {
    public:
        EncoderSensor(std::string const & name);
        ~EncoderSensor(void) = default;

        result_t initialize(std::string const & jointName);

        virtual result_t refreshProxies(void) override;

        std::string const & getJointName(void) const;

    private:
        virtual result_t set(float64_t const & t,
                             vectorN_t const & q,
                             vectorN_t const & v,
                             vectorN_t const & a,
                             vectorN_t const & uMotor);

    private:
        std::string jointName_;
        int32_t jointPositionIdx_;
        int32_t jointVelocityIdx_;
    };

    class TorqueSensor : public AbstractSensorTpl<TorqueSensor>
    {
    public:
        TorqueSensor(std::string const & name);
        ~TorqueSensor(void) = default;

        result_t initialize(std::string const & motorName);

        virtual result_t refreshProxies(void) override;

        std::string const & getMotorName(void) const;

    private:
        virtual result_t set(float64_t const & t,
                             vectorN_t const & q,
                             vectorN_t const & v,
                             vectorN_t const & a,
                             vectorN_t const & uMotor);

    private:
        std::string motorName_;
        int32_t motorIdx_;
    };
}

#endif //end of JIMINY_BASIC_SENSORS_H