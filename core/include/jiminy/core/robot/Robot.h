#ifndef JIMINY_ROBOT_H
#define JIMINY_ROBOT_H

#include "jiminy/core/robot/Model.h"
#include "jiminy/core/Types.h"


namespace jiminy
{
    struct MotorSharedDataHolder_t;
    class AbstractMotorBase;
    struct SensorSharedDataHolder_t;
    class AbstractSensorBase;
    class TelemetryData;
    class MutexLocal;
    class LockGuardLocal;

    class Robot : public Model
    {
    public:
        using motorsHolder_t = std::vector<std::shared_ptr<AbstractMotorBase> >;
        using sensorsHolder_t = std::vector<std::shared_ptr<AbstractSensorBase> >;
        using sensorsGroupHolder_t = std::unordered_map<std::string, sensorsHolder_t>;
        using sensorsSharedHolder_t = std::unordered_map<std::string, std::shared_ptr<SensorSharedDataHolder_t> >;

    public:
        // Disable the copy of the class
        Robot(Robot const & robot) = delete;
        Robot & operator = (Robot const & other) = delete;

    public:
        Robot(void);
        virtual ~Robot(void);

        auto shared_from_this() { return shared_from(this); }
        auto shared_from_this() const { return shared_from(this); }

        hresult_t initialize(std::string const & urdfPath,
                             bool_t const & hasFreeflyer = true,
                             std::vector<std::string> const & meshPackageDirs = {});

        hresult_t attachMotor(std::shared_ptr<AbstractMotorBase> motor);
        hresult_t getMotor(std::string const & motorName,
                           std::shared_ptr<AbstractMotorBase> & motor);
        hresult_t getMotor(std::string const & motorName,
                           std::weak_ptr<AbstractMotorBase const> & motor) const;
        motorsHolder_t const & getMotors(void) const;
        hresult_t detachMotor(std::string const & motorName);
        hresult_t detachMotors(std::vector<std::string> const & motorsNames = {});
        hresult_t attachSensor(std::shared_ptr<AbstractSensorBase> sensor);
        hresult_t getSensor(std::string const & sensorType,
                            std::string const & sensorName,
                            std::shared_ptr<AbstractSensorBase> & sensor);
        hresult_t getSensor(std::string const & sensorType,
                            std::string const & sensorName,
                            std::weak_ptr<AbstractSensorBase const> & sensor) const;
        sensorsGroupHolder_t const & getSensors(void) const;
        hresult_t detachSensor(std::string const & sensorType,
                              std::string const & sensorName);
        hresult_t detachSensors(std::string const & sensorType = {});

        void computeMotorsEfforts(float64_t const & t,
                                  vectorN_t const & q,
                                  vectorN_t const & v,
                                  vectorN_t const & a,
                                  vectorN_t const & command);
        vectorN_t const & getMotorsEfforts(void) const;
        float64_t const & getMotorEffort(std::string const & motorName) const;
        void setSensorsData(float64_t const & t,
                            vectorN_t const & q,
                            vectorN_t const & v,
                            vectorN_t const & a,
                            vectorN_t const & uMotor);

        sensorsDataMap_t getSensorsData(void) const;
        Eigen::Ref<vectorN_t const> getSensorData(std::string const & sensorType,
                                                  std::string const & sensorName) const;

        hresult_t setOptions(configHolder_t const & robotOptions);
        configHolder_t getOptions(void) const;
        hresult_t setMotorOptions(std::string    const & motorName,
                                  configHolder_t const & motorOptions);
        hresult_t setMotorsOptions(configHolder_t const & motorsOptions);
        hresult_t getMotorOptions(std::string    const & motorName,
                                  configHolder_t       & motorOptions) const;
        configHolder_t getMotorsOptions(void) const;
        hresult_t setSensorOptions(std::string    const & sensorType,
                                   std::string    const & sensorName,
                                   configHolder_t const & sensorOptions);
        hresult_t setSensorsOptions(std::string    const & sensorType,
                                    configHolder_t const & sensorsOptions);
        hresult_t setSensorsOptions(configHolder_t const & sensorsOptions);
        hresult_t getSensorOptions(std::string    const & sensorType,
                                   std::string    const & sensorName,
                                   configHolder_t       & sensorOptions) const;
        hresult_t getSensorsOptions(std::string    const & sensorType,
                                    configHolder_t       & sensorsOptions) const;
        configHolder_t getSensorsOptions(void) const;
        hresult_t setModelOptions(configHolder_t const & modelOptions);
        configHolder_t getModelOptions(void) const;
        hresult_t setTelemetryOptions(configHolder_t const & telemetryOptions);
        configHolder_t getTelemetryOptions(void) const;

        hresult_t dumpOptions(std::string const & filepath) const;
        hresult_t loadOptions(std::string const & filepath);

        // Those methods are not intended to be called manually. The Engine is taking care of it.
        virtual void reset(void) override;
        virtual hresult_t configureTelemetry(std::shared_ptr<TelemetryData> telemetryData,
                                             std::string const & objectPrefixName = "");
        void updateTelemetry(void);
        bool_t const & getIsTelemetryConfigured(void) const;

        std::vector<std::string> const & getMotorsNames(void) const;
        std::vector<int32_t> getMotorsModelIdx(void) const;
        std::vector<std::vector<int32_t> > getMotorsPositionIdx(void) const;
        std::vector<int32_t> getMotorsVelocityIdx(void) const;
        std::unordered_map<std::string, std::vector<std::string> > const & getSensorsNames(void) const;
        std::vector<std::string> const & getSensorsNames(std::string const & sensorType) const;

        vectorN_t const & getCommandLimit(void) const;
        vectorN_t const & getArmatures(void) const;

        std::vector<std::string> const & getCommandFieldnames(void) const;
        std::vector<std::string> const & getMotorEffortFieldnames(void) const;

        // Getters without 'get' prefix for consistency with pinocchio C++ API
        int32_t const & nmotors(void) const;

        hresult_t getLock(std::unique_ptr<LockGuardLocal> & lock);
        bool_t const & getIsLocked(void) const;

    protected:
        hresult_t refreshMotorsProxies(void);
        hresult_t refreshSensorsProxies(void);
        virtual hresult_t refreshProxies(void) override;

    protected:
        bool_t isTelemetryConfigured_;
        std::shared_ptr<TelemetryData> telemetryData_;
        motorsHolder_t motorsHolder_;
        sensorsGroupHolder_t sensorsGroupHolder_;
        std::unordered_map<std::string, bool_t> sensorTelemetryOptions_;
        std::vector<std::string> motorsNames_;                                      ///< Name of the motors
        std::unordered_map<std::string, std::vector<std::string> > sensorsNames_;   ///< Name of the sensors
        std::vector<std::string> commandFieldnames_;                                ///< Fieldnames of the command
        std::vector<std::string> motorEffortFieldnames_;                            ///< Fieldnames of the motors effort
        int32_t nmotors_;                                                           ///< The number of motors

    private:
        std::unique_ptr<MutexLocal> mutexLocal_;
        std::shared_ptr<MotorSharedDataHolder_t> motorsSharedHolder_;
        sensorsSharedHolder_t sensorsSharedHolder_;
    };
}

#endif //end of JIMINY_ROBOT_H
