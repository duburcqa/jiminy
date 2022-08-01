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
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    public:
        ImuSensor(std::string const & name);
        ~ImuSensor(void) = default;

        auto shared_from_this() { return shared_from(this); }

        hresult_t initialize(std::string const & frameName);

        virtual hresult_t setOptions(configHolder_t const & sensorOptions) final override;
        virtual hresult_t refreshProxies(void) final override;

        std::string const & getFrameName(void) const;
        frameIndex_t const & getFrameIdx(void) const;

    private:
        virtual hresult_t set(float64_t     const & t,
                              vectorN_t     const & q,
                              vectorN_t     const & v,
                              vectorN_t     const & a,
                              vectorN_t     const & uMotor,
                              forceVector_t const & fExternal) final override;
        virtual void measureData(void) final override;

    private:
        std::string frameName_;
        frameIndex_t frameIdx_;
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
        frameIndex_t const & getFrameIdx(void) const;

    private:
        virtual hresult_t set(float64_t     const & t,
                              vectorN_t     const & q,
                              vectorN_t     const & v,
                              vectorN_t     const & a,
                              vectorN_t     const & uMotor,
                              forceVector_t const & fExternal) final override;

    private:
        std::string frameName_;
        frameIndex_t frameIdx_;
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
        frameIndex_t const & getFrameIdx(void) const;
        jointIndex_t getJointIdx(void) const;

    private:
        virtual hresult_t set(float64_t     const & t,
                              vectorN_t     const & q,
                              vectorN_t     const & v,
                              vectorN_t     const & a,
                              vectorN_t     const & uMotor,
                              forceVector_t const & fExternal) final override;

    private:
        std::string frameName_;
        frameIndex_t frameIdx_;
        jointIndex_t parentJointIdx_;
        pinocchio::Force f_;
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
        jointIndex_t const & getJointIdx(void) const;
        joint_t const & getJointType(void) const;

    private:
        virtual hresult_t set(float64_t     const & t,
                              vectorN_t     const & q,
                              vectorN_t     const & v,
                              vectorN_t     const & a,
                              vectorN_t     const & uMotor,
                              forceVector_t const & fExternal) final override;

    private:
        std::string jointName_;
        jointIndex_t jointIdx_;
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
        std::size_t const & getMotorIdx(void) const;

    private:
        virtual hresult_t set(float64_t     const & t,
                              vectorN_t     const & q,
                              vectorN_t     const & v,
                              vectorN_t     const & a,
                              vectorN_t     const & uMotor,
                              forceVector_t const & fExternal) final override;

    private:
        std::string motorName_;
        std::size_t motorIdx_;
    };
}

#endif //end of JIMINY_BASIC_SENSORS_H