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
    class TelemetryData;
    class MutexLocal;
    class LockGuardLocal;

    class JIMINY_DLLAPI Robot : public Model
    {
    public:
        using MotorVector = std::vector<std::shared_ptr<AbstractMotorBase>>;
        using SensorVector = std::vector<std::shared_ptr<AbstractSensorBase>>;
        using SensorTree = std::unordered_map<std::string, SensorVector>;

    public:
        DISABLE_COPY(Robot)

    public:
        explicit Robot() noexcept;
        virtual ~Robot();

        auto shared_from_this() { return shared_from(this); }
        auto shared_from_this() const { return shared_from(this); }

        void initialize(const pinocchio::Model & pinocchioModel,
                        const pinocchio::GeometryModel & collisionModel,
                        const pinocchio::GeometryModel & visualModel);
        void initialize(const std::string & urdfPath,
                        bool hasFreeflyer = true,
                        const std::vector<std::string> & meshPackageDirs = {},
                        bool loadVisualMeshes = false);

        void attachMotor(std::shared_ptr<AbstractMotorBase> motor);
        std::shared_ptr<AbstractMotorBase> getMotor(const std::string & motorName);
        std::weak_ptr<const AbstractMotorBase> getMotor(const std::string & motorName) const;
        const MotorVector & getMotors() const;
        void detachMotor(const std::string & motorName);
        void detachMotors(std::vector<std::string> motorsNames = {});
        void attachSensor(std::shared_ptr<AbstractSensorBase> sensor);
        std::shared_ptr<AbstractSensorBase> getSensor(const std::string & sensorType,
                                                      const std::string & sensorName);
        std::weak_ptr<const AbstractSensorBase> getSensor(const std::string & sensorType,
                                                          const std::string & sensorName) const;
        const SensorTree & getSensors() const;
        void detachSensor(const std::string & sensorType, const std::string & sensorName);
        void detachSensors(const std::string & sensorType = {});

        void computeMotorEfforts(double t,
                                 const Eigen::VectorXd & q,
                                 const Eigen::VectorXd & v,
                                 const Eigen::VectorXd & a,
                                 const Eigen::VectorXd & command);
        const Eigen::VectorXd & getMotorEfforts() const;
        double getMotorEffort(const std::string & motorName) const;
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
        GenericConfig getOptions() const noexcept;
        void setMotorOptions(const std::string & motorName, const GenericConfig & motorOptions);
        void setMotorsOptions(const GenericConfig & motorsOptions);
        GenericConfig getMotorOptions(const std::string & motorName) const;
        GenericConfig getMotorsOptions() const;
        void setSensorOptions(const std::string & sensorType,
                              const std::string & sensorName,
                              const GenericConfig & sensorOptions);
        void setSensorsOptions(const std::string & sensorType,
                               const GenericConfig & sensorsOptions);
        void setSensorsOptions(const GenericConfig & sensorsOptions);
        GenericConfig getSensorOptions(const std::string & sensorType,
                                       const std::string & sensorName) const;
        GenericConfig getSensorsOptions(const std::string & sensorType) const;
        GenericConfig getSensorsOptions() const;
        void setModelOptions(const GenericConfig & modelOptions);
        GenericConfig getModelOptions() const;
        void setTelemetryOptions(const GenericConfig & telemetryOptions);
        GenericConfig getTelemetryOptions() const;

        void dumpOptions(const std::string & filepath) const;
        void loadOptions(const std::string & filepath);

        /// \remarks Those methods are not intended to be called manually. The Engine is taking
        ///          care of it.
        virtual void reset(const uniform_random_bit_generator_ref<uint32_t> & g) override;
        virtual void configureTelemetry(std::shared_ptr<TelemetryData> telemetryData,
                                        const std::string & prefix = {});
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
        virtual void refreshProxies() override;

    protected:
        bool isTelemetryConfigured_{false};
        std::shared_ptr<TelemetryData> telemetryData_{nullptr};
        MotorVector motors_{};
        SensorTree sensors_{};
        std::unordered_map<std::string, bool> sensorTelemetryOptions_{};
        /// \brief Name of the motors.
        std::vector<std::string> motorNames_{};
        /// \brief Name of the sensors.
        std::unordered_map<std::string, std::vector<std::string>> sensorNames_{};
        /// \brief Fieldnames of the command.
        std::vector<std::string> logCommandFieldnames_{};
        /// \brief Fieldnames of the motors effort.
        std::vector<std::string> logMotorEffortFieldnames_{};
        /// \brief The number of motors.
        Eigen::Index nmotors_{0};

    private:
        std::unique_ptr<MutexLocal> mutexLocal_{std::make_unique<MutexLocal>()};
        std::shared_ptr<MotorSharedStorage> motorSharedStorage_;
        std::unordered_map<std::string, std::shared_ptr<SensorSharedStorage>>
            sensorSharedStorageMap_{};
    };
}

#endif  // end of JIMINY_ROBOT_H
