#ifndef JIMINY_BASIC_SENSORS_H
#define JIMINY_BASIC_SENSORS_H

#include "jiminy/core/fwd.h"
#include "jiminy/core/utilities/helpers.h"
#include "jiminy/core/hardware/abstract_sensor.h"


namespace jiminy
{
    class Robot;

    class JIMINY_DLLAPI ImuSensor final : public AbstractSensorTpl<ImuSensor>
    {
    public:
        using AbstractSensorTpl<ImuSensor>::AbstractSensorTpl;

        auto shared_from_this() { return shared_from(this); }

        hresult_t initialize(const std::string & frameName);

        hresult_t setOptions(const GenericConfig & sensorOptions) override;
        hresult_t refreshProxies() override;

        const std::string & getFrameName() const;
        pinocchio::FrameIndex getFrameIdx() const;

    private:
        hresult_t set(double t,
                      const Eigen::VectorXd & q,
                      const Eigen::VectorXd & v,
                      const Eigen::VectorXd & a,
                      const Eigen::VectorXd & uMotor,
                      const ForceVector & fExternal) override;
        void measureData() override;

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

    class JIMINY_DLLAPI ContactSensor final : public AbstractSensorTpl<ContactSensor>
    {
    public:
        using AbstractSensorTpl<ContactSensor>::AbstractSensorTpl;

        auto shared_from_this() { return shared_from(this); }

        hresult_t initialize(const std::string & frameName);

        hresult_t refreshProxies() override;

        const std::string & getFrameName() const;
        pinocchio::FrameIndex getFrameIdx() const;

    private:
        hresult_t set(double t,
                      const Eigen::VectorXd & q,
                      const Eigen::VectorXd & v,
                      const Eigen::VectorXd & a,
                      const Eigen::VectorXd & uMotor,
                      const ForceVector & fExternal) override;

    private:
        std::string frameName_{};
        pinocchio::FrameIndex frameIdx_{0};
        std::size_t contactForceIdx_{0};
    };

    template<>
    const std::string AbstractSensorTpl<ContactSensor>::type_;
    template<>
    const std::vector<std::string> AbstractSensorTpl<ContactSensor>::fieldnames_;
    template<>
    const bool AbstractSensorTpl<ContactSensor>::areFieldnamesGrouped_;

    class JIMINY_DLLAPI ForceSensor final : public AbstractSensorTpl<ForceSensor>
    {
    public:
        using AbstractSensorTpl<ForceSensor>::AbstractSensorTpl;

        auto shared_from_this() { return shared_from(this); }

        hresult_t initialize(const std::string & frameName);

        hresult_t refreshProxies() override;

        const std::string & getFrameName() const;
        pinocchio::FrameIndex getFrameIdx() const;
        pinocchio::JointIndex getJointModelIdx() const;

    private:
        hresult_t set(double t,
                      const Eigen::VectorXd & q,
                      const Eigen::VectorXd & v,
                      const Eigen::VectorXd & a,
                      const Eigen::VectorXd & uMotor,
                      const ForceVector & fExternal) override;

    private:
        std::string frameName_{};
        pinocchio::FrameIndex frameIdx_{0};
        pinocchio::JointIndex parentJointModelIdx_{0};
        static_map_t<std::size_t, pinocchio::SE3> contactForcesIdxAndPlacement_{};
        pinocchio::Force f_{};
    };

    template<>
    const std::string AbstractSensorTpl<ForceSensor>::type_;
    template<>
    const std::vector<std::string> AbstractSensorTpl<ForceSensor>::fieldnames_;
    template<>
    const bool AbstractSensorTpl<ForceSensor>::areFieldnamesGrouped_;

    class JIMINY_DLLAPI EncoderSensor final : public AbstractSensorTpl<EncoderSensor>
    {
    public:
        using AbstractSensorTpl<EncoderSensor>::AbstractSensorTpl;

        auto shared_from_this() { return shared_from(this); }

        hresult_t initialize(const std::string & jointName);

        hresult_t refreshProxies() override;

        const std::string & getJointName() const;
        pinocchio::JointIndex getJointModelIdx() const;
        JointModelType getJointType() const;

    private:
        hresult_t set(double t,
                      const Eigen::VectorXd & q,
                      const Eigen::VectorXd & v,
                      const Eigen::VectorXd & a,
                      const Eigen::VectorXd & uMotor,
                      const ForceVector & fExternal) override;

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

    class JIMINY_DLLAPI EffortSensor final : public AbstractSensorTpl<EffortSensor>
    {
    public:
        using AbstractSensorTpl<EffortSensor>::AbstractSensorTpl;

        auto shared_from_this() { return shared_from(this); }

        hresult_t initialize(const std::string & motorName);

        hresult_t refreshProxies() override;

        const std::string & getMotorName() const;
        std::size_t getMotorIdx() const;

    private:
        hresult_t set(double t,
                      const Eigen::VectorXd & q,
                      const Eigen::VectorXd & v,
                      const Eigen::VectorXd & a,
                      const Eigen::VectorXd & uMotor,
                      const ForceVector & fExternal) override;

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