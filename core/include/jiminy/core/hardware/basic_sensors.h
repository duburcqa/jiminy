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
        using AbstractSensorTpl<ImuSensor>::AbstractSensorTpl;
        virtual ~ImuSensor() = default;

        auto shared_from_this() { return shared_from(this); }

        hresult_t initialize(const std::string & frameName);

        virtual hresult_t setOptions(const GenericConfig & sensorOptions) final override;
        virtual hresult_t refreshProxies() final override;

        const std::string & getFrameName() const;
        pinocchio::FrameIndex getFrameIdx() const;

    private:
        virtual hresult_t set(double t,
                              const Eigen::VectorXd & q,
                              const Eigen::VectorXd & v,
                              const Eigen::VectorXd & a,
                              const Eigen::VectorXd & uMotor,
                              const ForceVector & fExternal) final override;
        virtual void measureData() final override;

    private:
        std::string frameName_{};
        pinocchio::FrameIndex frameIdx_{0};
        /// \brief Sensor inverse rotation bias.
        Eigen::Matrix3d sensorRotationBiasInv_{Eigen::Matrix3d::Identity()};
    };

    template<>
    const std::string AbstractSensorTpl<ImuSensor>::type_;
    template<>
    const std::vector<std::string> AbstractSensorTpl<ImuSensor>::fieldnames_;
    template<>
    const bool AbstractSensorTpl<ImuSensor>::areFieldnamesGrouped_;

    class JIMINY_DLLAPI ContactSensor : public AbstractSensorTpl<ContactSensor>
    {
    public:
        using AbstractSensorTpl<ContactSensor>::AbstractSensorTpl;
        virtual ~ContactSensor() = default;

        auto shared_from_this() { return shared_from(this); }

        hresult_t initialize(const std::string & frameName);

        virtual hresult_t refreshProxies() final override;

        const std::string & getFrameName() const;
        pinocchio::FrameIndex getFrameIdx() const;

    private:
        virtual hresult_t set(double t,
                              const Eigen::VectorXd & q,
                              const Eigen::VectorXd & v,
                              const Eigen::VectorXd & a,
                              const Eigen::VectorXd & uMotor,
                              const ForceVector & fExternal) final override;

    private:
        std::string frameName_{};
        pinocchio::FrameIndex frameIdx_{0};
    };

    template<>
    const std::string AbstractSensorTpl<ContactSensor>::type_;
    template<>
    const std::vector<std::string> AbstractSensorTpl<ContactSensor>::fieldnames_;
    template<>
    const bool AbstractSensorTpl<ContactSensor>::areFieldnamesGrouped_;

    class JIMINY_DLLAPI ForceSensor : public AbstractSensorTpl<ForceSensor>
    {
    public:
        using AbstractSensorTpl<ForceSensor>::AbstractSensorTpl;
        virtual ~ForceSensor() = default;

        auto shared_from_this() { return shared_from(this); }

        hresult_t initialize(const std::string & frameName);

        virtual hresult_t refreshProxies() final override;

        const std::string & getFrameName() const;
        pinocchio::FrameIndex getFrameIdx() const;
        pinocchio::JointIndex getJointModelIdx() const;

    private:
        virtual hresult_t set(double t,
                              const Eigen::VectorXd & q,
                              const Eigen::VectorXd & v,
                              const Eigen::VectorXd & a,
                              const Eigen::VectorXd & uMotor,
                              const ForceVector & fExternal) final override;

    private:
        std::string frameName_{};
        pinocchio::FrameIndex frameIdx_{0};
        pinocchio::JointIndex parentJointModelIdx_{0};
        pinocchio::Force f_{};
    };

    template<>
    const std::string AbstractSensorTpl<ForceSensor>::type_;
    template<>
    const std::vector<std::string> AbstractSensorTpl<ForceSensor>::fieldnames_;
    template<>
    const bool AbstractSensorTpl<ForceSensor>::areFieldnamesGrouped_;

    class JIMINY_DLLAPI EncoderSensor : public AbstractSensorTpl<EncoderSensor>
    {
    public:
        using AbstractSensorTpl<EncoderSensor>::AbstractSensorTpl;
        virtual ~EncoderSensor() = default;

        auto shared_from_this() { return shared_from(this); }

        hresult_t initialize(const std::string & jointName);

        virtual hresult_t refreshProxies() final override;

        const std::string & getJointName() const;
        pinocchio::JointIndex getJointModelIdx() const;
        JointModelType getJointType() const;

    private:
        virtual hresult_t set(double t,
                              const Eigen::VectorXd & q,
                              const Eigen::VectorXd & v,
                              const Eigen::VectorXd & a,
                              const Eigen::VectorXd & uMotor,
                              const ForceVector & fExternal) final override;

    private:
        std::string jointName_{};
        pinocchio::JointIndex jointModelIdx_{0};
        JointModelType jointType_{JointModelType::UNSUPPORTED};
    };

    template<>
    const std::string AbstractSensorTpl<EncoderSensor>::type_;
    template<>
    const std::vector<std::string> AbstractSensorTpl<EncoderSensor>::fieldnames_;
    template<>
    const bool AbstractSensorTpl<EncoderSensor>::areFieldnamesGrouped_;

    class JIMINY_DLLAPI EffortSensor : public AbstractSensorTpl<EffortSensor>
    {
    public:
        using AbstractSensorTpl<EffortSensor>::AbstractSensorTpl;
        virtual ~EffortSensor() = default;

        auto shared_from_this() { return shared_from(this); }

        hresult_t initialize(const std::string & motorName);

        virtual hresult_t refreshProxies() final override;

        const std::string & getMotorName() const;
        std::size_t getMotorIdx() const;

    private:
        virtual hresult_t set(double t,
                              const Eigen::VectorXd & q,
                              const Eigen::VectorXd & v,
                              const Eigen::VectorXd & a,
                              const Eigen::VectorXd & uMotor,
                              const ForceVector & fExternal) final override;

    private:
        std::string motorName_{};
        std::size_t motorIdx_{0};
    };

    template<>
    const std::string AbstractSensorTpl<EffortSensor>::type_;
    template<>
    const std::vector<std::string> AbstractSensorTpl<EffortSensor>::fieldnames_;
    template<>
    const bool AbstractSensorTpl<EffortSensor>::areFieldnamesGrouped_;
}

#endif  // end of JIMINY_BASIC_SENSORS_H