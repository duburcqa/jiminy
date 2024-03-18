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

    template<>
    const std::string JIMINY_EXPL_TPL_INST_DLLEXPORT AbstractSensorTpl<ImuSensor>::type_;
    template<>
    const std::vector<std::string> JIMINY_EXPL_TPL_INST_DLLEXPORT
        AbstractSensorTpl<ImuSensor>::fieldnames_;
    template<>
    const bool JIMINY_EXPL_TPL_INST_DLLEXPORT AbstractSensorTpl<ImuSensor>::areFieldnamesGrouped_;
    extern template class JIMINY_EXPL_TPL_INST_DLLIMPORT AbstractSensorTpl<ImuSensor>;

    class JIMINY_DLLAPI ContactSensor final : public AbstractSensorTpl<ContactSensor>
    {
    public:
        using AbstractSensorTpl<ContactSensor>::AbstractSensorTpl;

        auto shared_from_this() { return shared_from(this); }

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

    template<>
    const std::string JIMINY_EXPL_TPL_INST_DLLEXPORT AbstractSensorTpl<ContactSensor>::type_;
    template<>
    const std::vector<std::string> JIMINY_EXPL_TPL_INST_DLLEXPORT
        AbstractSensorTpl<ContactSensor>::fieldnames_;
    template<>
    const bool JIMINY_EXPL_TPL_INST_DLLEXPORT
        AbstractSensorTpl<ContactSensor>::areFieldnamesGrouped_;
    extern template class JIMINY_EXPL_TPL_INST_DLLIMPORT AbstractSensorTpl<ContactSensor>;

    class JIMINY_DLLAPI ForceSensor final : public AbstractSensorTpl<ForceSensor>
    {
    public:
        using AbstractSensorTpl<ForceSensor>::AbstractSensorTpl;

        auto shared_from_this() { return shared_from(this); }

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

    template<>
    const std::string JIMINY_EXPL_TPL_INST_DLLEXPORT AbstractSensorTpl<ForceSensor>::type_;
    template<>
    const std::vector<std::string> JIMINY_EXPL_TPL_INST_DLLEXPORT
        AbstractSensorTpl<ForceSensor>::fieldnames_;
    template<>
    const bool JIMINY_EXPL_TPL_INST_DLLEXPORT
        AbstractSensorTpl<ForceSensor>::areFieldnamesGrouped_;
    extern template class JIMINY_EXPL_TPL_INST_DLLIMPORT AbstractSensorTpl<ForceSensor>;

    class JIMINY_DLLAPI EncoderSensor final : public AbstractSensorTpl<EncoderSensor>
    {
    public:
        using AbstractSensorTpl<EncoderSensor>::AbstractSensorTpl;

        auto shared_from_this() { return shared_from(this); }

        void initialize(const std::string & jointName);

        void setOptions(const GenericConfig & sensorOptions) override;
        void refreshProxies() override;

        const std::string & getJointName() const;
        pinocchio::JointIndex getJointIndex() const;
        JointModelType getJointType() const;

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
        JointModelType jointType_{JointModelType::UNSUPPORTED};
    };

    template<>
    const std::string JIMINY_EXPL_TPL_INST_DLLEXPORT AbstractSensorTpl<EncoderSensor>::type_;
    template<>
    const std::vector<std::string> JIMINY_EXPL_TPL_INST_DLLEXPORT
        AbstractSensorTpl<EncoderSensor>::fieldnames_;
    template<>
    const bool JIMINY_EXPL_TPL_INST_DLLEXPORT
        AbstractSensorTpl<EncoderSensor>::areFieldnamesGrouped_;
    extern template class JIMINY_EXPL_TPL_INST_DLLIMPORT AbstractSensorTpl<EncoderSensor>;

    class JIMINY_DLLAPI EffortSensor final : public AbstractSensorTpl<EffortSensor>
    {
    public:
        using AbstractSensorTpl<EffortSensor>::AbstractSensorTpl;

        auto shared_from_this() { return shared_from(this); }

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

    template<>
    const std::string JIMINY_EXPL_TPL_INST_DLLEXPORT AbstractSensorTpl<EffortSensor>::type_;
    template<>
    const std::vector<std::string> JIMINY_EXPL_TPL_INST_DLLEXPORT
        AbstractSensorTpl<EffortSensor>::fieldnames_;
    template<>
    const bool JIMINY_EXPL_TPL_INST_DLLEXPORT
        AbstractSensorTpl<EffortSensor>::areFieldnamesGrouped_;
    extern template class JIMINY_EXPL_TPL_INST_DLLIMPORT AbstractSensorTpl<EffortSensor>;
}

#endif  // end of JIMINY_BASIC_SENSORS_H