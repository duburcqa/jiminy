#ifndef JIMINY_ROBOT_H
#define JIMINY_ROBOT_H

#include "jiminy/core/macros.h"
#include "jiminy/core/types.h"
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
        Robot();
        virtual ~Robot();

        auto shared_from_this() { return shared_from(this); }
        auto shared_from_this() const { return shared_from(this); }

        hresult_t initialize(const pinocchio::Model & pncModel,
                             const pinocchio::GeometryModel & collisionModel,
                             const pinocchio::GeometryModel & visualModel);
        hresult_t initialize(const std::string & urdfPath,
                             const bool_t & hasFreeflyer = true,
                             const std::vector<std::string> & meshPackageDirs = {},
                             const bool_t & loadVisualMeshes = false);

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

        void computeMotorsEfforts(const float64_t & t,
                                  const Eigen::VectorXd & q,
                                  const Eigen::VectorXd & v,
                                  const Eigen::VectorXd & a,
                                  const Eigen::VectorXd & command);
        const Eigen::VectorXd & getMotorsEfforts() const;
        const float64_t & getMotorEffort(const std::string & motorName) const;
        void setSensorsData(const float64_t & t,
                            const Eigen::VectorXd & q,
                            const Eigen::VectorXd & v,
                            const Eigen::VectorXd & a,
                            const Eigen::VectorXd & uMotor,
                            const forceVector_t & fExternal);

        sensorsDataMap_t getSensorsData() const;
        Eigen::Ref<const Eigen::VectorXd> getSensorData(const std::string & sensorType,
                                                        const std::string & sensorName) const;

        hresult_t setOptions(const configHolder_t & robotOptions);
        configHolder_t getOptions() const;
        hresult_t setMotorOptions(const std::string & motorName,
                                  const configHolder_t & motorOptions);
        hresult_t setMotorsOptions(const configHolder_t & motorsOptions);
        hresult_t getMotorOptions(const std::string & motorName,
                                  configHolder_t & motorOptions) const;
        configHolder_t getMotorsOptions() const;
        hresult_t setSensorOptions(const std::string & sensorType,
                                   const std::string & sensorName,
                                   const configHolder_t & sensorOptions);
        hresult_t setSensorsOptions(const std::string & sensorType,
                                    const configHolder_t & sensorsOptions);
        hresult_t setSensorsOptions(const configHolder_t & sensorsOptions);
        hresult_t getSensorOptions(const std::string & sensorType,
                                   const std::string & sensorName,
                                   configHolder_t & sensorOptions) const;
        hresult_t getSensorsOptions(const std::string & sensorType,
                                    configHolder_t & sensorsOptions) const;
        configHolder_t getSensorsOptions() const;
        hresult_t setModelOptions(const configHolder_t & modelOptions);
        configHolder_t getModelOptions() const;
        hresult_t setTelemetryOptions(const configHolder_t & telemetryOptions);
        configHolder_t getTelemetryOptions() const;

        hresult_t dumpOptions(const std::string & filepath) const;
        hresult_t loadOptions(const std::string & filepath);

        /// \remarks Those methods are not intended to be called manually. The Engine is taking
        ///          care of it.
        virtual void reset() override;
        virtual hresult_t configureTelemetry(std::shared_ptr<TelemetryData> telemetryData,
                                             const std::string & objectPrefixName = "");
        void updateTelemetry();
        const bool_t & getIsTelemetryConfigured() const;

        const std::vector<std::string> & getMotorsNames() const;
        std::vector<jointIndex_t> getMotorsModelIdx() const;
        std::vector<std::vector<int32_t>> getMotorsPositionIdx() const;
        std::vector<int32_t> getMotorsVelocityIdx() const;
        const std::unordered_map<std::string, std::vector<std::string>> & getSensorsNames() const;
        const std::vector<std::string> & getSensorsNames(const std::string & sensorType) const;

        const Eigen::VectorXd & getCommandLimit() const;

        const std::vector<std::string> & getCommandFieldnames() const;
        const std::vector<std::string> & getMotorEffortFieldnames() const;

        // Getters without 'get' prefix for consistency with pinocchio C++ API
        const uint64_t & nmotors() const;

        hresult_t getLock(std::unique_ptr<LockGuardLocal> & lock);
        const bool_t & getIsLocked() const;

    protected:
        hresult_t refreshMotorsProxies();
        hresult_t refreshSensorsProxies();
        virtual hresult_t refreshProxies() override;

    protected:
        bool_t isTelemetryConfigured_;
        std::shared_ptr<TelemetryData> telemetryData_;
        motorsHolder_t motorsHolder_;
        sensorsGroupHolder_t sensorsGroupHolder_;
        std::unordered_map<std::string, bool_t> sensorTelemetryOptions_;
        /// \brief Name of the motors.
        std::vector<std::string> motorsNames_;
        /// \brief Name of the sensors.
        std::unordered_map<std::string, std::vector<std::string>> sensorsNames_;
        /// \brief Fieldnames of the command.
        std::vector<std::string> logFieldnamesCommand_;
        /// \brief Fieldnames of the motors effort.
        std::vector<std::string> logFieldnamesMotorEffort_;
        /// \brief The number of motors.
        uint64_t nmotors_;

    private:
        std::unique_ptr<MutexLocal> mutexLocal_;
        std::shared_ptr<MotorSharedDataHolder_t> motorsSharedHolder_;
        sensorsSharedHolder_t sensorsSharedHolder_;
    };
}

#endif  // end of JIMINY_ROBOT_H
