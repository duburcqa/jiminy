#ifndef JIMINY_BASIC_SENSORS_H
#define JIMINY_BASIC_SENSORS_H

#include "jiminy/core/fwd.h"
#include "jiminy/core/utilities/helpers.h"
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

        virtual hresult_t setOptions(const GenericConfig & sensorOptions) final override;
        virtual hresult_t refreshProxies() final override;

        const std::string & getFrameName() const;
        pinocchio::FrameIndex getFrameIdx() const;

    private:
        virtual hresult_t set(float64_t t,
                              const Eigen::VectorXd & q,
                              const Eigen::VectorXd & v,
                              const Eigen::VectorXd & a,
                              const Eigen::VectorXd & uMotor,
                              const ForceVector & fExternal) final override;
        virtual void measureData() final override;

    private:
        std::string frameName_;
        pinocchio::FrameIndex frameIdx_;
        /// \brief Sensor inverse rotation bias.
        Eigen::Matrix3d sensorRotationBiasInv_;
    };

    template<>
    const std::string AbstractSensorTpl<ImuSensor>::type_;
    template<>
    const bool_t AbstractSensorTpl<ImuSensor>::areFieldnamesGrouped_;
    template<>
    const std::vector<std::string> AbstractSensorTpl<ImuSensor>::fieldnames_;

    class JIMINY_DLLAPI ContactSensor : public AbstractSensorTpl<ContactSensor>
    {
    public:
        ContactSensor(const std::string & name);
        virtual ~ContactSensor() = default;

        auto shared_from_this() { return shared_from(this); }

        hresult_t initialize(const std::string & frameName);

        virtual hresult_t refreshProxies() final override;

        const std::string & getFrameName() const;
        pinocchio::FrameIndex getFrameIdx() const;

    private:
        virtual hresult_t set(float64_t t,
                              const Eigen::VectorXd & q,
                              const Eigen::VectorXd & v,
                              const Eigen::VectorXd & a,
                              const Eigen::VectorXd & uMotor,
                              const ForceVector & fExternal) final override;

    private:
        std::string frameName_;
        pinocchio::FrameIndex frameIdx_;
    };

    template<>
    const std::string AbstractSensorTpl<ContactSensor>::type_;
    template<>
    const bool_t AbstractSensorTpl<ContactSensor>::areFieldnamesGrouped_;
    template<>
    const std::vector<std::string> AbstractSensorTpl<ContactSensor>::fieldnames_;

    class JIMINY_DLLAPI ForceSensor : public AbstractSensorTpl<ForceSensor>
    {
    public:
        ForceSensor(const std::string & name);
        virtual ~ForceSensor() = default;

        auto shared_from_this() { return shared_from(this); }

        hresult_t initialize(const std::string & frameName);

        virtual hresult_t refreshProxies() final override;

        const std::string & getFrameName() const;
        pinocchio::FrameIndex getFrameIdx() const;
        pinocchio::JointIndex getJointIdx() const;

    private:
        virtual hresult_t set(float64_t t,
                              const Eigen::VectorXd & q,
                              const Eigen::VectorXd & v,
                              const Eigen::VectorXd & a,
                              const Eigen::VectorXd & uMotor,
                              const ForceVector & fExternal) final override;

    private:
        std::string frameName_;
        pinocchio::FrameIndex frameIdx_;
        pinocchio::JointIndex parentJointIdx_;
        pinocchio::Force f_;
    };

    template<>
    const std::string AbstractSensorTpl<ForceSensor>::type_;
    template<>
    const bool_t AbstractSensorTpl<ForceSensor>::areFieldnamesGrouped_;
    template<>
    const std::vector<std::string> AbstractSensorTpl<ForceSensor>::fieldnames_;

    class JIMINY_DLLAPI EncoderSensor : public AbstractSensorTpl<EncoderSensor>
    {
    public:
        EncoderSensor(const std::string & name);
        virtual ~EncoderSensor() = default;

        auto shared_from_this() { return shared_from(this); }

        hresult_t initialize(const std::string & jointName);

        virtual hresult_t refreshProxies() final override;

        const std::string & getJointName() const;
        pinocchio::JointIndex getJointIdx() const;
        JointModelType getJointType() const;

    private:
        virtual hresult_t set(float64_t t,
                              const Eigen::VectorXd & q,
                              const Eigen::VectorXd & v,
                              const Eigen::VectorXd & a,
                              const Eigen::VectorXd & uMotor,
                              const ForceVector & fExternal) final override;

    private:
        std::string jointName_;
        pinocchio::JointIndex jointIdx_;
        JointModelType jointType_;
    };

    template<>
    const std::string AbstractSensorTpl<EncoderSensor>::type_;
    template<>
    const bool_t AbstractSensorTpl<EncoderSensor>::areFieldnamesGrouped_;
    template<>
    const std::vector<std::string> AbstractSensorTpl<EncoderSensor>::fieldnames_;

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
        virtual hresult_t set(float64_t t,
                              const Eigen::VectorXd & q,
                              const Eigen::VectorXd & v,
                              const Eigen::VectorXd & a,
                              const Eigen::VectorXd & uMotor,
                              const ForceVector & fExternal) final override;

    private:
        std::string motorName_;
        std::size_t motorIdx_;
    };

    template<>
    const std::string AbstractSensorTpl<EffortSensor>::type_;
    template<>
    const bool_t AbstractSensorTpl<EffortSensor>::areFieldnamesGrouped_;
    template<>
    const std::vector<std::string> AbstractSensorTpl<EffortSensor>::fieldnames_;
}

#endif  // end of JIMINY_BASIC_SENSORS_H