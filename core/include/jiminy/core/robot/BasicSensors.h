#ifndef JIMINY_BASIC_SENSORS_H
#define JIMINY_BASIC_SENSORS_H

#include "jiminy/core/robot/AbstractMotor.h"

#include "jiminy/core/robot/AbstractSensor.h"


namespace jiminy
{
    class Robot;

    class ImuSensor : public AbstractSensorTpl<ImuSensor>
    {
    public:
        ImuSensor(std::string const & name);
        ~ImuSensor(void) = default;

        auto shared_from_this() { return shared_from(this); }

        hresult_t initialize(std::string const & frameName);

        virtual hresult_t refreshProxies(void) override;

        std::string const & getFrameName(void) const;

    private:
        virtual hresult_t set(float64_t const & t,
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

        auto shared_from_this() { return shared_from(this); }

        hresult_t initialize(std::string const & frameName);

        virtual hresult_t refreshProxies(void) override;

        std::string const & getFrameName(void) const;

    private:
        virtual hresult_t set(float64_t const & t,
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

        auto shared_from_this() { return shared_from(this); }

        hresult_t initialize(std::string const & jointName);

        virtual hresult_t refreshProxies(void) override;

        std::string const & getJointName(void) const;

    private:
        virtual hresult_t set(float64_t const & t,
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

        auto shared_from_this() { return shared_from(this); }

        hresult_t initialize(std::string const & motorName);

        virtual hresult_t refreshProxies(void) override;

        std::string const & getMotorName(void) const;

    private:
        virtual hresult_t set(float64_t const & t,
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