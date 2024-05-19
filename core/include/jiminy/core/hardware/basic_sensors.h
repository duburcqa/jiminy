#ifndef JIMINY_BASIC_SENSORS_H
#define JIMINY_BASIC_SENSORS_H

#include "jiminy/core/fwd.h"
#include "jiminy/core/utilities/helpers.h"
#include "jiminy/core/hardware/abstract_sensor.h"


namespace jiminy
{
    class Robot;

    class ImuSensor;
#if defined EXPORT_SYMBOLS || (!defined _WIN32 && !defined __CYGWIN__)
    template<>
    const std::string JIMINY_DLLAPI AbstractSensorTpl<ImuSensor>::type_;
    template<>
    const std::vector<std::string> JIMINY_DLLAPI AbstractSensorTpl<ImuSensor>::fieldnames_;
#endif
    template class JIMINY_TEMPLATE_INSTANTIATION_DLLAPI AbstractSensorTpl<ImuSensor>;

    class JIMINY_DLLAPI ImuSensor final : public AbstractSensorTpl<ImuSensor>
    {
    public:
        using AbstractSensorTpl<ImuSensor>::AbstractSensorTpl;

        void initialize(const std::string & frameName);

        void setOptions(const GenericConfig & sensorOptions) override;
        void refreshProxies() override;

        const std::string & getFrameName() const;
        pinocchio::FrameIndex getFrameIndex() const;

    private:
        void set(double t,
                 const Eigen::VectorXd & q,
                 const Eigen::VectorXd & v,
                 const Eigen::VectorXd & a,
                 const Eigen::VectorXd & uMotor,
                 const ForceVector & fExternal) override;
        void measureData() override;

    private:
        std::string frameName_{};
        pinocchio::FrameIndex frameIndex_{0};
        /// \brief Sensor inverse rotation bias.
        Eigen::Matrix3d sensorRotationBiasInv_{Eigen::Matrix3d::Identity()};
    };

    class ContactSensor;
#if defined EXPORT_SYMBOLS || (!defined _WIN32 && !defined __CYGWIN__)
    template<>
    const std::string JIMINY_DLLAPI AbstractSensorTpl<ContactSensor>::type_;
    template<>
    const std::vector<std::string> JIMINY_DLLAPI AbstractSensorTpl<ContactSensor>::fieldnames_;
#endif
    template class JIMINY_TEMPLATE_INSTANTIATION_DLLAPI AbstractSensorTpl<ContactSensor>;

    class JIMINY_DLLAPI ContactSensor final : public AbstractSensorTpl<ContactSensor>
    {
    public:
        using AbstractSensorTpl<ContactSensor>::AbstractSensorTpl;

        void initialize(const std::string & frameName);

        void setOptions(const GenericConfig & sensorOptions) override;
        void refreshProxies() override;

        const std::string & getFrameName() const;
        pinocchio::FrameIndex getFrameIndex() const;

    private:
        void set(double t,
                 const Eigen::VectorXd & q,
                 const Eigen::VectorXd & v,
                 const Eigen::VectorXd & a,
                 const Eigen::VectorXd & uMotor,
                 const ForceVector & fExternal) override;

    private:
        std::string frameName_{};
        pinocchio::FrameIndex frameIndex_{0};
        std::size_t contactIndex_{0};
    };

    class ForceSensor;
#if defined EXPORT_SYMBOLS || (!defined _WIN32 && !defined __CYGWIN__)
    template<>
    const std::string JIMINY_DLLAPI AbstractSensorTpl<ForceSensor>::type_;
    template<>
    const std::vector<std::string> JIMINY_DLLAPI AbstractSensorTpl<ForceSensor>::fieldnames_;
#endif
    template class JIMINY_TEMPLATE_INSTANTIATION_DLLAPI AbstractSensorTpl<ForceSensor>;

    class JIMINY_DLLAPI ForceSensor final : public AbstractSensorTpl<ForceSensor>
    {
    public:
        using AbstractSensorTpl<ForceSensor>::AbstractSensorTpl;

        void initialize(const std::string & frameName);

        void setOptions(const GenericConfig & sensorOptions) override;
        void refreshProxies() override;

        const std::string & getFrameName() const;
        pinocchio::FrameIndex getFrameIndex() const;
        pinocchio::JointIndex getJointIndex() const;

    private:
        void set(double t,
                 const Eigen::VectorXd & q,
                 const Eigen::VectorXd & v,
                 const Eigen::VectorXd & a,
                 const Eigen::VectorXd & uMotor,
                 const ForceVector & fExternal) override;

    private:
        std::string frameName_{};
        pinocchio::FrameIndex frameIndex_{0};
        pinocchio::JointIndex parentJointIndex_{0};
        static_map_t<std::size_t, pinocchio::SE3> contactIndexPlacementPairs_{};
        pinocchio::Force f_{};
    };

    class EncoderSensor;
#if defined EXPORT_SYMBOLS || (!defined _WIN32 && !defined __CYGWIN__)
    template<>
    const std::string JIMINY_DLLAPI AbstractSensorTpl<EncoderSensor>::type_;
    template<>
    const std::vector<std::string> JIMINY_DLLAPI AbstractSensorTpl<EncoderSensor>::fieldnames_;
#endif
    template class JIMINY_TEMPLATE_INSTANTIATION_DLLAPI AbstractSensorTpl<EncoderSensor>;

    class JIMINY_DLLAPI EncoderSensor final : public AbstractSensorTpl<EncoderSensor>
    {
    public:
        using AbstractSensorTpl<EncoderSensor>::AbstractSensorTpl;

        void initialize(const std::string & motorOrJointName, bool isJointSide = false);

        void setOptions(const GenericConfig & sensorOptions) override;
        void refreshProxies() override;

        const std::string & getMotorName() const;
        std::size_t getMotorIndex() const;
        const std::string & getJointName() const;
        pinocchio::JointIndex getJointIndex() const;

    private:
        void set(double t,
                 const Eigen::VectorXd & q,
                 const Eigen::VectorXd & v,
                 const Eigen::VectorXd & a,
                 const Eigen::VectorXd & uMotor,
                 const ForceVector & fExternal) override;

    private:
        std::string jointName_{};
        pinocchio::JointIndex jointIndex_{0};
        std::string motorName_{};
        std::size_t motorIndex_{0};
        bool isJointSide_{false};

        JointModelType jointType_{JointModelType::UNSUPPORTED};
        std::size_t jointPositionIndex_{0};
        std::size_t joinVelocityIndex_{0};
        double mechanicalReduction_{1.0};
    };

    class EffortSensor;
#if defined EXPORT_SYMBOLS || (!defined _WIN32 && !defined __CYGWIN__)
    template<>
    const std::string JIMINY_DLLAPI AbstractSensorTpl<EffortSensor>::type_;
    template<>
    const std::vector<std::string> JIMINY_DLLAPI AbstractSensorTpl<EffortSensor>::fieldnames_;
#endif
    template class JIMINY_TEMPLATE_INSTANTIATION_DLLAPI AbstractSensorTpl<EffortSensor>;

    class JIMINY_DLLAPI EffortSensor final : public AbstractSensorTpl<EffortSensor>
    {
    public:
        using AbstractSensorTpl<EffortSensor>::AbstractSensorTpl;

        void initialize(const std::string & motorName);

        void setOptions(const GenericConfig & sensorOptions) override;
        void refreshProxies() override;

        const std::string & getMotorName() const;
        std::size_t getMotorIndex() const;

    private:
        void set(double t,
                 const Eigen::VectorXd & q,
                 const Eigen::VectorXd & v,
                 const Eigen::VectorXd & a,
                 const Eigen::VectorXd & uMotor,
                 const ForceVector & fExternal) override;

    private:
        std::string motorName_{};
        std::size_t motorIndex_{0};
    };
}

#endif  // end of JIMINY_BASIC_SENSORS_H