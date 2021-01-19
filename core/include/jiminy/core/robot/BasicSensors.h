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

        virtual hresult_t setOptions(configHolder_t const & sensorOptions) final override;
        virtual hresult_t refreshProxies(void) final override;

        std::string const & getFrameName(void) const;
        int32_t const & getFrameIdx(void) const;

    private:
        virtual hresult_t set(float64_t const & t,
                              vectorN_t const & q,
                              vectorN_t const & v,
                              vectorN_t const & a,
                              vectorN_t const & uMotor) final override;
        virtual void skewMeasurement(void) final override;

    private:
        std::string frameName_;
        int32_t frameIdx_;
        quaternion_t sensorRotationBias_;  ///< Sensor rotation bias.
    };

    class ContactSensor : public AbstractSensorTpl<ContactSensor>
    {
    public:
        ContactSensor(std::string const & name);
        ~ContactSensor(void) = default;

        auto shared_from_this() { return shared_from(this); }

        hresult_t initialize(std::string const & frameName);

        virtual hresult_t refreshProxies(void) final override;

        std::string const & getFrameName(void) const;
        int32_t const & getFrameIdx(void) const;

    private:
        virtual hresult_t set(float64_t const & t,
                              vectorN_t const & q,
                              vectorN_t const & v,
                              vectorN_t const & a,
                              vectorN_t const & uMotor) final override;

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

        virtual hresult_t refreshProxies(void) final override;

        std::string const & getFrameName(void) const;
        int32_t const & getFrameIdx(void) const;
        int32_t getJointIdx(void) const;

    private:
        virtual hresult_t set(float64_t const & t,
                              vectorN_t const & q,
                              vectorN_t const & v,
                              vectorN_t const & a,
                              vectorN_t const & uMotor) final override;

    private:
        std::string frameName_;
        int32_t frameIdx_;
        int32_t parentJointIdx_;
    };

    class EncoderSensor : public AbstractSensorTpl<EncoderSensor>
    {
    public:
        EncoderSensor(std::string const & name);
        ~EncoderSensor(void) = default;

        auto shared_from_this() { return shared_from(this); }

        hresult_t initialize(std::string const & jointName);

        virtual hresult_t refreshProxies(void) final override;

        std::string const & getJointName(void) const;
        int32_t const & getJointIdx(void) const;
        joint_t const & getJointType(void) const;

    private:
        virtual hresult_t set(float64_t const & t,
                              vectorN_t const & q,
                              vectorN_t const & v,
                              vectorN_t const & a,
                              vectorN_t const & uMotor) final override;

    private:
        std::string jointName_;
        int32_t jointIdx_;
        joint_t jointType_;
    };

    class EffortSensor : public AbstractSensorTpl<EffortSensor>
    {
    public:
        EffortSensor(std::string const & name);
        ~EffortSensor(void) = default;

        auto shared_from_this() { return shared_from(this); }

        hresult_t initialize(std::string const & motorName);

        virtual hresult_t refreshProxies(void) final override;

        std::string const & getMotorName(void) const;
        int32_t const & getMotorIdx(void) const;

    private:
        virtual hresult_t set(float64_t const & t,
                              vectorN_t const & q,
                              vectorN_t const & v,
                              vectorN_t const & a,
                              vectorN_t const & uMotor) final override;

    private:
        std::string motorName_;
        int32_t motorIdx_;
    };
}

#endif //end of JIMINY_BASIC_SENSORS_H