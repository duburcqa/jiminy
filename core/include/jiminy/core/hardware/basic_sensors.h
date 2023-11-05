#ifndef JIMINY_BASIC_SENSORS_H
#define JIMINY_BASIC_SENSORS_H

#include "jiminy/core/macros.h"
#include "jiminy/core/hardware/abstract_sensor.h"


namespace jiminy
{
    class Robot;

    class JIMINY_DLLAPI ImuSensor : public AbstractSensorTpl<ImuSensor>
    {
    public:
        ImuSensor(const std::string & name);
        virtual ~ImuSensor() = default;

        auto shared_from_this() { return shared_from(this); }

        hresult_t initialize(const std::string & frameName);

        virtual hresult_t setOptions(const configHolder_t & sensorOptions) final override;
        virtual hresult_t refreshProxies() final override;

        const std::string & getFrameName() const;
        const frameIndex_t & getFrameIdx() const;

    private:
        virtual hresult_t set(const float64_t & t,
                              const Eigen::VectorXd & q,
                              const Eigen::VectorXd & v,
                              const Eigen::VectorXd & a,
                              const Eigen::VectorXd & uMotor,
                              const forceVector_t & fExternal) final override;
        virtual void measureData() final override;

    private:
        std::string frameName_;
        frameIndex_t frameIdx_;
        /// \brief Sensor inverse rotation bias.
        Eigen::Matrix3d sensorRotationBiasInv_;
    };

    class JIMINY_DLLAPI ContactSensor : public AbstractSensorTpl<ContactSensor>
    {
    public:
        ContactSensor(const std::string & name);
        virtual ~ContactSensor() = default;

        auto shared_from_this() { return shared_from(this); }

        hresult_t initialize(const std::string & frameName);

        virtual hresult_t refreshProxies() final override;

        const std::string & getFrameName() const;
        const frameIndex_t & getFrameIdx() const;

    private:
        virtual hresult_t set(const float64_t & t,
                              const Eigen::VectorXd & q,
                              const Eigen::VectorXd & v,
                              const Eigen::VectorXd & a,
                              const Eigen::VectorXd & uMotor,
                              const forceVector_t & fExternal) final override;

    private:
        std::string frameName_;
        frameIndex_t frameIdx_;
    };

    class JIMINY_DLLAPI ForceSensor : public AbstractSensorTpl<ForceSensor>
    {
    public:
        ForceSensor(const std::string & name);
        virtual ~ForceSensor() = default;

        auto shared_from_this() { return shared_from(this); }

        hresult_t initialize(const std::string & frameName);

        virtual hresult_t refreshProxies() final override;

        const std::string & getFrameName() const;
        const frameIndex_t & getFrameIdx() const;
        jointIndex_t getJointIdx() const;

    private:
        virtual hresult_t set(const float64_t & t,
                              const Eigen::VectorXd & q,
                              const Eigen::VectorXd & v,
                              const Eigen::VectorXd & a,
                              const Eigen::VectorXd & uMotor,
                              const forceVector_t & fExternal) final override;

    private:
        std::string frameName_;
        frameIndex_t frameIdx_;
        jointIndex_t parentJointIdx_;
        pinocchio::Force f_;
    };

    class JIMINY_DLLAPI EncoderSensor : public AbstractSensorTpl<EncoderSensor>
    {
    public:
        EncoderSensor(const std::string & name);
        virtual ~EncoderSensor() = default;

        auto shared_from_this() { return shared_from(this); }

        hresult_t initialize(const std::string & jointName);

        virtual hresult_t refreshProxies() final override;

        const std::string & getJointName() const;
        const jointIndex_t & getJointIdx() const;
        const joint_t & getJointType() const;

    private:
        virtual hresult_t set(const float64_t & t,
                              const Eigen::VectorXd & q,
                              const Eigen::VectorXd & v,
                              const Eigen::VectorXd & a,
                              const Eigen::VectorXd & uMotor,
                              const forceVector_t & fExternal) final override;

    private:
        std::string jointName_;
        jointIndex_t jointIdx_;
        joint_t jointType_;
    };

    class JIMINY_DLLAPI EffortSensor : public AbstractSensorTpl<EffortSensor>
    {
    public:
        EffortSensor(const std::string & name);
        virtual ~EffortSensor() = default;

        auto shared_from_this() { return shared_from(this); }

        hresult_t initialize(const std::string & motorName);

        virtual hresult_t refreshProxies() final override;

        const std::string & getMotorName() const;
        const std::size_t & getMotorIdx() const;

    private:
        virtual hresult_t set(const float64_t & t,
                              const Eigen::VectorXd & q,
                              const Eigen::VectorXd & v,
                              const Eigen::VectorXd & a,
                              const Eigen::VectorXd & uMotor,
                              const forceVector_t & fExternal) final override;

    private:
        std::string motorName_;
        std::size_t motorIdx_;
    };
}

#endif  // end of JIMINY_BASIC_SENSORS_H