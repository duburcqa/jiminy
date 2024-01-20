#ifndef JIMINY_ROBOT_H
#define JIMINY_ROBOT_H

#include "jiminy/core/fwd.h"
#include "jiminy/core/hardware/fwd.h"
#include "jiminy/core/utilities/helpers.h"
#include "jiminy/core/robot/model.h"


namespace jiminy
{
    struct MotorSharedDataHolder_t;
    class AbstractMotorBase;
    struct SensorSharedDataHolder_t;
    class AbstractSensorBase;
    class TelemetryData;
    class MutexLocal;
    class LockGuardLocal;

    class JIMINY_DLLAPI Robot : public Model
    {
    public:
        using motorsHolder_t = std::vector<std::shared_ptr<AbstractMotorBase>>;
        using sensorsHolder_t = std::vector<std::shared_ptr<AbstractSensorBase>>;
        using sensorsGroupHolder_t = std::unordered_map<std::string, sensorsHolder_t>;
        using sensorsSharedHolder_t =
            std::unordered_map<std::string, std::shared_ptr<SensorSharedDataHolder_t>>;

    public:
        DISABLE_COPY(Robot)

    public:
        explicit Robot() noexcept;
        virtual ~Robot();

        auto shared_from_this() { return shared_from(this); }
        auto shared_from_this() const { return shared_from(this); }

        hresult_t initialize(const pinocchio::Model & pncModel,
                             const pinocchio::GeometryModel & collisionModel,
                             const pinocchio::GeometryModel & visualModel);
        hresult_t initialize(const std::string & urdfPath,
                             bool hasFreeflyer = true,
                             const std::vector<std::string> & meshPackageDirs = {},
                             bool loadVisualMeshes = false);

        hresult_t attachMotor(std::shared_ptr<AbstractMotorBase> motor);
        hresult_t getMotor(const std::string & motorName,
                           std::shared_ptr<AbstractMotorBase> & motor);
        hresult_t getMotor(const std::string & motorName,
                           std::weak_ptr<const AbstractMotorBase> & motor) const;
        const motorsHolder_t & getMotors() const;
        hresult_t detachMotor(const std::string & motorName);
        hresult_t detachMotors(const std::vector<std::string> & motorsNames = {});
        hresult_t attachSensor(std::shared_ptr<AbstractSensorBase> sensor);
        hresult_t getSensor(const std::string & sensorType,
                            const std::string & sensorName,
                            std::shared_ptr<AbstractSensorBase> & sensor);
        hresult_t getSensor(const std::string & sensorType,
                            const std::string & sensorName,
                            std::weak_ptr<const AbstractSensorBase> & sensor) const;
        const sensorsGroupHolder_t & getSensors() const;
        hresult_t detachSensor(const std::string & sensorType, const std::string & sensorName);
        hresult_t detachSensors(const std::string & sensorType = {});

        void computeMotorsEfforts(double t,
                                  const Eigen::VectorXd & q,
                                  const Eigen::VectorXd & v,
                                  const Eigen::VectorXd & a,
                                  const Eigen::VectorXd & command);
        const Eigen::VectorXd & getMotorsEfforts() const;
        double getMotorEffort(const std::string & motorName) const;
        void setSensorsData(double t,
                            const Eigen::VectorXd & q,
                            const Eigen::VectorXd & v,
                            const Eigen::VectorXd & a,
                            const Eigen::VectorXd & uMotor,
                            const ForceVector & fExternal);

        SensorsDataMap getSensorsData() const;
        Eigen::Ref<const Eigen::VectorXd> getSensorData(const std::string & sensorType,
                                                        const std::string & sensorName) const;

        hresult_t setOptions(const GenericConfig & robotOptions);
        GenericConfig getOptions() const noexcept;
        hresult_t setMotorOptions(const std::string & motorName,
                                  const GenericConfig & motorOptions);
        hresult_t setMotorsOptions(const GenericConfig & motorsOptions);
        hresult_t getMotorOptions(const std::string & motorName,
                                  GenericConfig & motorOptions) const;
        GenericConfig getMotorsOptions() const;
        hresult_t setSensorOptions(const std::string & sensorType,
                                   const std::string & sensorName,
                                   const GenericConfig & sensorOptions);
        hresult_t setSensorsOptions(const std::string & sensorType,
                                    const GenericConfig & sensorsOptions);
        hresult_t setSensorsOptions(const GenericConfig & sensorsOptions);
        hresult_t getSensorOptions(const std::string & sensorType,
                                   const std::string & sensorName,
                                   GenericConfig & sensorOptions) const;
        hresult_t getSensorsOptions(const std::string & sensorType,
                                    GenericConfig & sensorsOptions) const;
        GenericConfig getSensorsOptions() const;
        hresult_t setModelOptions(const GenericConfig & modelOptions);
        GenericConfig getModelOptions() const;
        hresult_t setTelemetryOptions(const GenericConfig & telemetryOptions);
        GenericConfig getTelemetryOptions() const;

        hresult_t dumpOptions(const std::string & filepath) const;
        hresult_t loadOptions(const std::string & filepath);

        /// \remarks Those methods are not intended to be called manually. The Engine is taking
        ///          care of it.
        virtual void reset(const uniform_random_bit_generator_ref<uint32_t> & g) override;
        virtual hresult_t configureTelemetry(std::shared_ptr<TelemetryData> telemetryData,
                                             const std::string & objectPrefixName = {});
        void updateTelemetry();
        bool getIsTelemetryConfigured() const;

        const std::vector<std::string> & getMotorsNames() const;
        std::vector<pinocchio::JointIndex> getMotorsModelIdx() const;
        std::vector<std::vector<Eigen::Index>> getMotorsPositionIdx() const;
        std::vector<Eigen::Index> getMotorsVelocityIdx() const;
        const std::unordered_map<std::string, std::vector<std::string>> & getSensorsNames() const;
        const std::vector<std::string> & getSensorsNames(const std::string & sensorType) const;

        const Eigen::VectorXd & getCommandLimit() const;

        const std::vector<std::string> & getCommandFieldnames() const;
        const std::vector<std::string> & getMotorEffortFieldnames() const;

        // Getters without 'get' prefix for consistency with pinocchio C++ API
        uint64_t nmotors() const;

        hresult_t getLock(std::unique_ptr<LockGuardLocal> & lock);
        bool getIsLocked() const;

    protected:
        hresult_t refreshMotorsProxies();
        hresult_t refreshSensorsProxies();
        virtual hresult_t refreshProxies() override;

    protected:
        bool isTelemetryConfigured_{false};
        std::shared_ptr<TelemetryData> telemetryData_{nullptr};
        motorsHolder_t motorsHolder_{};
        sensorsGroupHolder_t sensorsGroupHolder_{};
        std::unordered_map<std::string, bool> sensorTelemetryOptions_{};
        /// \brief Name of the motors.
        std::vector<std::string> motorsNames_{};
        /// \brief Name of the sensors.
        std::unordered_map<std::string, std::vector<std::string>> sensorsNames_{};
        /// \brief Fieldnames of the command.
        std::vector<std::string> logFieldnamesCommand_{};
        /// \brief Fieldnames of the motors effort.
        std::vector<std::string> logFieldnamesMotorEffort_{};
        /// \brief The number of motors.
        uint64_t nmotors_{0U};

    private:
        std::unique_ptr<MutexLocal> mutexLocal_{std::make_unique<MutexLocal>()};
        std::shared_ptr<MotorSharedDataHolder_t> motorsSharedHolder_;
        sensorsSharedHolder_t sensorsSharedHolder_{};
    };
}

#endif  // end of JIMINY_ROBOT_H
