#ifndef JIMINY_BASIC_SENSORS_H
#define JIMINY_BASIC_SENSORS_H

#include "jiminy/core/hardware/abstract_motor.h"
#include "jiminy/core/hardware/abstract_sensor.h"


namespace jiminy
{
    class Robot;

    class ImuSensor : public AbstractSensorTpl<ImuSensor>
    {
    public:
        ImuSensor(const std::string & name);
        virtual ~ImuSensor(void) = default;

        auto shared_from_this() { return shared_from(this); }

        hresult_t initialize(const std::string & frameName);

        virtual hresult_t setOptions(const configHolder_t & sensorOptions) final override;
        virtual hresult_t refreshProxies(void) final override;

        const std::string & getFrameName(void) const;
        const frameIndex_t & getFrameIdx(void) const;

    private:
        virtual hresult_t set(const float64_t & t,
                              const vectorN_t & q,
                              const vectorN_t & v,
                              const vectorN_t & a,
                              const vectorN_t & uMotor,
                              const forceVector_t & fExternal) final override;
        virtual void measureData(void) final override;

    private:
        std::string frameName_;
        frameIndex_t frameIdx_;
        /// \brief Sensor inverse rotation bias.
        matrix3_t sensorRotationBiasInv_;
    };

    class ContactSensor : public AbstractSensorTpl<ContactSensor>
    {
    public:
        ContactSensor(const std::string & name);
        virtual ~ContactSensor(void) = default;

        auto shared_from_this() { return shared_from(this); }

        hresult_t initialize(const std::string & frameName);

        virtual hresult_t refreshProxies(void) final override;

        const std::string & getFrameName(void) const;
        const frameIndex_t & getFrameIdx(void) const;

    private:
        virtual hresult_t set(const float64_t & t,
                              const vectorN_t & q,
                              const vectorN_t & v,
                              const vectorN_t & a,
                              const vectorN_t & uMotor,
                              const forceVector_t & fExternal) final override;

    private:
        std::string frameName_;
        frameIndex_t frameIdx_;
    };

    class ForceSensor : public AbstractSensorTpl<ForceSensor>
    {
    public:
        ForceSensor(const std::string & name);
        virtual ~ForceSensor(void) = default;

        auto shared_from_this() { return shared_from(this); }

        hresult_t initialize(const std::string & frameName);

        virtual hresult_t refreshProxies(void) final override;

        const std::string & getFrameName(void) const;
        const frameIndex_t & getFrameIdx(void) const;
        jointIndex_t getJointIdx(void) const;

    private:
        virtual hresult_t set(const float64_t & t,
                              const vectorN_t & q,
                              const vectorN_t & v,
                              const vectorN_t & a,
                              const vectorN_t & uMotor,
                              const forceVector_t & fExternal) final override;

    private:
        std::string frameName_;
        frameIndex_t frameIdx_;
        jointIndex_t parentJointIdx_;
        pinocchio::Force f_;
    };

    class EncoderSensor : public AbstractSensorTpl<EncoderSensor>
    {
    public:
        EncoderSensor(const std::string & name);
        virtual ~EncoderSensor(void) = default;

        auto shared_from_this() { return shared_from(this); }

        hresult_t initialize(const std::string & jointName);

        virtual hresult_t refreshProxies(void) final override;

        const std::string & getJointName(void) const;
        const jointIndex_t & getJointIdx(void) const;
        const joint_t & getJointType(void) const;

    private:
        virtual hresult_t set(const float64_t & t,
                              const vectorN_t & q,
                              const vectorN_t & v,
                              const vectorN_t & a,
                              const vectorN_t & uMotor,
                              const forceVector_t & fExternal) final override;

    private:
        std::string jointName_;
        jointIndex_t jointIdx_;
        joint_t jointType_;
    };

    class EffortSensor : public AbstractSensorTpl<EffortSensor>
    {
    public:
        EffortSensor(const std::string & name);
        virtual ~EffortSensor(void) = default;

        auto shared_from_this() { return shared_from(this); }

        hresult_t initialize(const std::string & motorName);

        virtual hresult_t refreshProxies(void) final override;

        const std::string & getMotorName(void) const;
        const std::size_t & getMotorIdx(void) const;

    private:
        virtual hresult_t set(const float64_t & t,
                              const vectorN_t & q,
                              const vectorN_t & v,
                              const vectorN_t & a,
                              const vectorN_t & uMotor,
                              const forceVector_t & fExternal) final override;

    private:
        std::string motorName_;
        std::size_t motorIdx_;
    };
}

#endif  // end of JIMINY_BASIC_SENSORS_H