#ifndef JIMINY_ROBOT_H
#define JIMINY_ROBOT_H

#include "jiminy/core/fwd.h"
#include "jiminy/core/hardware/fwd.h"
#include "jiminy/core/utilities/helpers.h"
#include "jiminy/core/robot/model.h"


namespace jiminy
{
    struct MotorSharedStorage;
    class AbstractMotorBase;
    struct SensorSharedStorage;
    class AbstractSensorBase;
    class AbstractController;
    class TelemetryData;
    class MutexLocal;
    class LockGuardLocal;

    class JIMINY_DLLAPI Robot : public Model
    {
    public:
        virtual GenericConfig getDefaultRobotOptions()
        {
            GenericConfig config;
            config["model"] = getDefaultModelOptions();
            config["motors"] = GenericConfig{};
            config["sensors"] = GenericConfig{};
            config["controller"] = GenericConfig{};
            config["telemetry"] = GenericConfig{};

            return config;
        };

    public:
        using MotorVector = std::vector<std::shared_ptr<AbstractMotorBase>>;
        using SensorVector = std::vector<std::shared_ptr<AbstractSensorBase>>;
        using SensorTree = std::unordered_map<std::string, SensorVector>;

    public:
        JIMINY_DISABLE_COPY(Robot)

    public:
        /// \param[in] name Name of the Robot.
        explicit Robot(const std::string & name = "") noexcept;
        virtual ~Robot();

        auto shared_from_this() { return shared_from(this); }
        auto shared_from_this() const { return shared_from(this); }

        void initialize(
            const pinocchio::Model & pinocchioModel,
            const std::optional<pinocchio::GeometryModel> & collisionModel = std::nullopt,
            const std::optional<pinocchio::GeometryModel> & visualModel = std::nullopt);
        void initialize(const std::string & urdfPath,
                        bool hasFreeflyer = true,
                        const std::vector<std::string> & meshPackageDirs = {},
                        bool loadVisualMeshes = false);

        const std::string & getName() const;

        void attachMotor(std::shared_ptr<AbstractMotorBase> motor);
        std::shared_ptr<AbstractMotorBase> getMotor(const std::string & motorName);
        std::weak_ptr<const AbstractMotorBase> getMotor(const std::string & motorName) const;
        const MotorVector & getMotors() const;
        void detachMotor(const std::string & motorName);
        void detachMotors(std::vector<std::string> motorNames = {});
        void attachSensor(std::shared_ptr<AbstractSensorBase> sensor);
        std::shared_ptr<AbstractSensorBase> getSensor(const std::string & sensorType,
                                                      const std::string & sensorName);
        std::weak_ptr<const AbstractSensorBase> getSensor(const std::string & sensorType,
                                                          const std::string & sensorName) const;
        const SensorTree & getSensors() const;
        void detachSensor(const std::string & sensorType, const std::string & sensorName);
        void detachSensors(const std::string & sensorType = {});

        void setController(const std::shared_ptr<AbstractController> & controller);
        std::shared_ptr<AbstractController> getController();
        std::weak_ptr<const AbstractController> getController() const;

        void computeMotorEfforts(double t,
                                 const Eigen::VectorXd & q,
                                 const Eigen::VectorXd & v,
                                 const Eigen::VectorXd & a,
                                 const Eigen::VectorXd & command);
        const Eigen::VectorXd & getMotorEfforts() const;
        double getMotorEffort(const std::string & motorName) const;

        /// \warning It assumes that kinematic quantities have been updated previously and are
        ///          consistent with the following input arguments. If not, one must call
        ///          `pinocchio::forwardKinematics` and `pinocchio::updateFramePlacements`
        ///          beforehand.
        void computeSensorMeasurements(double t,
                                       const Eigen::VectorXd & q,
                                       const Eigen::VectorXd & v,
                                       const Eigen::VectorXd & a,
                                       const Eigen::VectorXd & uMotor,
                                       const ForceVector & fExternal);
        SensorMeasurementTree getSensorMeasurements() const;
        Eigen::Ref<const Eigen::VectorXd> getSensorMeasurement(
            const std::string & sensorType, const std::string & sensorName) const;

        void setOptions(const GenericConfig & robotOptions);
        const GenericConfig & getOptions() const noexcept;
        void setModelOptions(const GenericConfig & modelOptions);
        const GenericConfig & getModelOptions() const noexcept;

        /// \remark This method does not have to be called manually before running a simulation.
        ///         The Engine is taking care of it.
        virtual void reset(const uniform_random_bit_generator_ref<uint32_t> & g) override;
        virtual void configureTelemetry(std::shared_ptr<TelemetryData> telemetryData);
        void updateTelemetry();
        bool getIsTelemetryConfigured() const;

        const std::vector<std::string> & getMotorNames() const;
        std::vector<pinocchio::JointIndex> getMotorJointIndices() const;
        std::vector<std::vector<Eigen::Index>> getMotorsPositionIndices() const;
        std::vector<Eigen::Index> getMotorVelocityIndices() const;
        const std::unordered_map<std::string, std::vector<std::string>> & getSensorNames() const;
        const std::vector<std::string> & getSensorNames(const std::string & sensorType) const;

        const Eigen::VectorXd & getCommandLimit() const;

        const std::vector<std::string> & getLogCommandFieldnames() const;
        const std::vector<std::string> & getLogMotorEffortFieldnames() const;

        // Getters without 'get' prefix for consistency with pinocchio C++ API
        Eigen::Index nmotors() const;

        std::unique_ptr<LockGuardLocal> getLock();
        bool getIsLocked() const;

    protected:
        void refreshMotorProxies();
        void refreshSensorProxies();
        void refreshProxies() override;

        void initializeExtendedModel() override;

    private:
        void detachMotor(const std::string & motorName, bool triggerReset);

    protected:
        bool isTelemetryConfigured_{false};
        std::shared_ptr<TelemetryData> telemetryData_{nullptr};
        /// \brief Motors attached to the robot.
        MotorVector motors_{};
        /// \brief Sensors attached to the robot.
        SensorTree sensors_{};
        /// \brief Whether the robot options are guaranteed to be up-to-date.
        mutable bool areRobotOptionsRefreshed_{false};
        /// \brief Dictionary with the parameters of the robot.
        mutable GenericConfig robotOptionsGeneric_{};
        /// \brief Name of the motors.
        std::vector<std::string> motorNames_{};
        /// \brief Name of the sensors.
        std::unordered_map<std::string, std::vector<std::string>> sensorNames_{};
        /// \brief Fieldnames of the command.
        std::vector<std::string> logCommandFieldnames_{};
        /// \brief Fieldnames of the motors effort.
        std::vector<std::string> logMotorEffortFieldnames_{};
        /// \brief Number of motors.
        Eigen::Index nmotors_{0};
        /// \brief Controller of the robot.
        std::shared_ptr<AbstractController> controller_{nullptr};
        /// \brief Name of the robot.
        std::string name_;

    private:
        std::unique_ptr<MutexLocal> mutexLocal_{std::make_unique<MutexLocal>()};
        std::shared_ptr<MotorSharedStorage> motorSharedStorage_;
        std::unordered_map<std::string, std::shared_ptr<SensorSharedStorage>>
            sensorSharedStorageMap_{};
    };
}

#endif  // end of JIMINY_ROBOT_H
