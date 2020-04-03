#ifndef JIMINY_ROBOT_H
#define JIMINY_ROBOT_H

#include "jiminy/core/robot/Model.h"
#include "jiminy/core/Utilities.h"
#include "jiminy/core/Types.h"


namespace jiminy
{
    struct MotorSharedDataHolder_t;
    class AbstractMotorBase;
    class AbstractConstraint;
    struct SensorSharedDataHolder_t;
    class AbstractSensorBase;
    class TelemetryData;

    class Robot : public Model
    {
    public:
        using motorsHolder_t = std::vector<std::shared_ptr<AbstractMotorBase> >;
        using sensorsHolder_t = std::vector<std::shared_ptr<AbstractSensorBase> >;
        using sensorsGroupHolder_t = std::unordered_map<std::string, sensorsHolder_t>;

    public:
        // Disable the copy of the class
        Robot(Robot const & robot) = delete;
        Robot & operator = (Robot const & other) = delete;

    public:
        Robot(void);
        virtual ~Robot(void);

        hresult_t initialize(std::string const & urdfPath,
                             bool_t      const & hasFreeflyer = true);

        hresult_t attachMotor(std::shared_ptr<AbstractMotorBase> motor);
        hresult_t getMotor(std::string const & motorName,
                           std::shared_ptr<AbstractMotorBase> & motor);
        hresult_t getMotor(std::string       const   & motorName,
                           AbstractMotorBase const * & motor) const;
        motorsHolder_t const & getMotors(void) const;
        hresult_t detachMotor(std::string const & motorName);
        hresult_t detachMotors(std::vector<std::string> const & motorsNames = {});
        hresult_t attachSensor(std::shared_ptr<AbstractSensorBase> sensor);
        hresult_t getSensor(std::string const & sensorType,
                            std::string const & sensorName,
                            std::shared_ptr<AbstractSensorBase> & sensor);
        hresult_t getSensor(std::string        const   & sensorType,
                            std::string        const   & sensorName,
                            AbstractSensorBase const * & sensor) const;
        sensorsGroupHolder_t const & getSensors(void) const;
        hresult_t detachSensor(std::string const & sensorType,
                              std::string const & sensorName);
        hresult_t detachSensors(std::string const & sensorType = {});

        void computeMotorsTorques(float64_t                   const & t,
                                  Eigen::Ref<vectorN_t const> const & q,
                                  Eigen::Ref<vectorN_t const> const & v,
                                  vectorN_t                   const & a, // Do Not use Eigen::Ref for the acceleration to avoid memory allocation by the engine for a temporary
                                  vectorN_t                   const & u);
        vectorN_t const & getMotorsTorques(void) const;
        float64_t const & getMotorTorque(std::string const & motorName) const;
        void setSensorsData(float64_t                   const & t,
                            Eigen::Ref<vectorN_t const> const & q,
                            Eigen::Ref<vectorN_t const> const & v,
                            Eigen::Ref<vectorN_t const> const & a,
                            vectorN_t                   const & u);

        /// \brief Add a kinematic constraint to the robot.
        ///
        /// \param[in] constraintName Unique name identifying the kinematic constraint.
        /// \param[in] constraint Constraint to add.
        hresult_t addConstraint(std::string const & constraintName,
                                std::shared_ptr<AbstractConstraint> constraint);

        /// \brief Remove a kinematic constraint form the system.
        ///
        /// \param[in] constraintName Unique name identifying the kinematic constraint.
        hresult_t removeConstraint(std::string const & constraintName);

        /// \brief Compute jacobian and drift associated to all the constraints.
        ///
        /// \note  It is assumed frames forward kinematics has already been called.
        ///
        /// \param[in] q    Joint position.
        /// \param[in] v    Joint velocity.
        /// \param[out] jacobianOut Output jacobian matrix.
        /// \param[out] driftOut    Output drift vector.
        void computeConstraints(Eigen::Ref<vectorN_t const> const & q,
                                Eigen::Ref<vectorN_t const> const & v,
                                matrixN_t & jacobianOut,
                                vectorN_t & driftOut);

        /// \brief Returns true if at least one constraint is active on the robot.
        inline bool_t hasConstraint()
        {
            return !constraintsHolder_.empty();
        }

        sensorsDataMap_t getSensorsData(void) const;
        matrixN_t getSensorsData(std::string const & sensorType) const;
        vectorN_t getSensorData(std::string const & sensorType,
                                std::string const & motorName) const;

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
        std::vector<int32_t> getMotorsPositionIdx(void) const;
        std::vector<int32_t> getMotorsVelocityIdx(void) const;
        std::unordered_map<std::string, std::vector<std::string> > const & getSensorsNames(void) const;
        std::vector<std::string> const & getSensorsNames(std::string const & sensorType) const;

        vectorN_t getTorqueLimit(void) const;
        vectorN_t getMotorInertia(void) const;

        std::vector<std::string> const & getMotorTorqueFieldnames(void) const;

        hresult_t getLock(std::unique_ptr<MutexLocal::LockGuardLocal> & lock);
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
        std::vector<std::string> motorsNames_;              ///< Name of the motors of the robot
        std::unordered_map<std::string, std::vector<std::string> > sensorsNames_;   ///<Name of the sensors of the robot
        std::vector<std::string> motorTorqueFieldnames_;    ///< Fieldnames of the torques of the motors

        static_map_t<std::string, std::shared_ptr<AbstractConstraint>> constraintsHolder_;

    private:
        MutexLocal mutexLocal_;
        std::shared_ptr<MotorSharedDataHolder_t> motorsSharedHolder_;
        std::unordered_map<std::string, std::shared_ptr<SensorSharedDataHolder_t> > sensorsSharedHolder_;
    };
}

#endif //end of JIMINY_ROBOT_H
