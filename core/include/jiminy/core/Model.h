#ifndef JIMINY_MODEL_H
#define JIMINY_MODEL_H

#include <string>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "pinocchio/multibody/model.hpp"
#include "pinocchio/algorithm/frames.hpp"

#include "jiminy/core/Utilities.h"
#include "jiminy/core/Types.h"


namespace jiminy
{
    std::string const JOINT_PREFIX_BASE("current");
    std::string const FREE_FLYER_PREFIX_BASE_NAME(JOINT_PREFIX_BASE + "Freeflyer");
    std::string const FLEXIBLE_JOINT_SUFFIX = "FlexibleJoint";

    struct MotorSharedDataHolder_t;
    class AbstractMotorBase;
    struct SensorSharedDataHolder_t;
    class AbstractSensorBase;
    class TelemetryData;
    class Engine;

    class Model
    {
    public:
        // Disable the copy of the class
        Model(Model const & model) = delete;
        Model & operator = (Model const & other) = delete;

    public:
        virtual configHolder_t getDefaultJointOptions()
        {
            configHolder_t config;
            config["enablePositionLimit"] = true;
            config["positionLimitFromUrdf"] = true;
            config["positionLimitMin"] = vectorN_t();
            config["positionLimitMax"] = vectorN_t();
            config["enableVelocityLimit"] = true;
            config["velocityLimitFromUrdf"] = true;
            config["velocityLimit"] = vectorN_t();

            return config;
        };

        struct jointOptions_t
        {
            bool_t    const enablePositionLimit;
            bool_t    const positionLimitFromUrdf;
            vectorN_t const positionLimitMin;         ///< Min position limit of all the actual joints, namely without freeflyer and flexible joints if any
            vectorN_t const positionLimitMax;
            bool_t    const enableVelocityLimit;
            bool_t    const velocityLimitFromUrdf;
            vectorN_t const velocityLimit;

            jointOptions_t(configHolder_t const & options) :
            enablePositionLimit(boost::get<bool_t>(options.at("enablePositionLimit"))),
            positionLimitFromUrdf(boost::get<bool_t>(options.at("positionLimitFromUrdf"))),
            positionLimitMin(boost::get<vectorN_t>(options.at("positionLimitMin"))),
            positionLimitMax(boost::get<vectorN_t>(options.at("positionLimitMax"))),
            enableVelocityLimit(boost::get<bool_t>(options.at("enableVelocityLimit"))),
            velocityLimitFromUrdf(boost::get<bool_t>(options.at("velocityLimitFromUrdf"))),
            velocityLimit(boost::get<vectorN_t>(options.at("velocityLimit")))
            {
                // Empty.
            }
        };

        virtual configHolder_t getDefaultDynamicsOptions()
        {
            // Add extra options or update default values
            configHolder_t config;
            config["inertiaBodiesBiasStd"] = 0.0;
            config["massBodiesBiasStd"] = 0.0;
            config["centerOfMassPositionBodiesBiasStd"] = 0.0;
            config["relativePositionBodiesBiasStd"] = 0.0;
            config["enableFlexibleModel"] = true;
            config["flexibilityConfig"] = flexibilityConfig_t();

            return config;
        };

        struct dynamicsOptions_t
        {
            float64_t           const inertiaBodiesBiasStd;
            float64_t           const massBodiesBiasStd;
            float64_t           const centerOfMassPositionBodiesBiasStd;
            float64_t           const relativePositionBodiesBiasStd;
            bool_t              const enableFlexibleModel;
            flexibilityConfig_t const flexibilityConfig;

            dynamicsOptions_t(configHolder_t const & options) :
            inertiaBodiesBiasStd(boost::get<float64_t>(options.at("inertiaBodiesBiasStd"))),
            massBodiesBiasStd(boost::get<float64_t>(options.at("massBodiesBiasStd"))),
            centerOfMassPositionBodiesBiasStd(boost::get<float64_t>(options.at("centerOfMassPositionBodiesBiasStd"))),
            relativePositionBodiesBiasStd(boost::get<float64_t>(options.at("relativePositionBodiesBiasStd"))),
            enableFlexibleModel(boost::get<bool_t>(options.at("enableFlexibleModel"))),
            flexibilityConfig(boost::get<flexibilityConfig_t>(options.at("flexibilityConfig")))
            {
                // Empty.
            }
        };

        virtual configHolder_t getDefaultOptions()
        {
            configHolder_t config;
            config["dynamics"] = getDefaultDynamicsOptions();
            config["joints"] = getDefaultJointOptions();

            return config;
        };

        struct modelOptions_t
        {
            dynamicsOptions_t const dynamics;
            jointOptions_t const joints;

            modelOptions_t(configHolder_t const & options) :
            dynamics(boost::get<configHolder_t>(options.at("dynamics"))),
            joints(boost::get<configHolder_t>(options.at("joints")))
            {
                // Empty.
            }
        };

    public:
        using motorsHolder_t = std::unordered_map<std::string, std::shared_ptr<AbstractMotorBase> >;
        using sensorsHolder_t = std::unordered_map<std::string, std::shared_ptr<AbstractSensorBase> >;
        using sensorsGroupHolder_t = std::unordered_map<std::string, sensorsHolder_t>;

    public:
        Model(void);
        virtual ~Model(void);

        result_t initialize(std::string const & urdfPath,
                            bool_t      const & hasFreeflyer = true);

        result_t addContactPoints(std::vector<std::string> const & frameNames);
        result_t removeContactPoints(std::vector<std::string> const & frameNames = {});
        result_t attachMotor(std::shared_ptr<AbstractMotorBase> const & motor);
        result_t getMotor(std::string const & motorName,
                          std::shared_ptr<AbstractMotorBase const> & motor) const;
        motorsHolder_t const & getMotors(void) const;
        result_t detachMotor(std::string const & motorName);
        result_t detachMotors(std::vector<std::string> const & motorsNames = {});
        result_t attachSensor(std::shared_ptr<AbstractSensorBase> const & sensor);
        result_t getSensor(std::string const & sensorType,
                           std::string const & sensorName,
                           std::shared_ptr<AbstractSensorBase const> & sensor) const;
        sensorsGroupHolder_t const & getSensors(void) const;
        result_t detachSensor(std::string const & sensorType,
                              std::string const & sensorName);
        result_t detachSensors(std::string const & sensorType = {});

        void computeMotorsTorques(float64_t const & t,
                                  vectorN_t const & q,
                                  vectorN_t const & v,
                                  vectorN_t const & a,
                                  vectorN_t const & u);
        vectorN_t const & getMotorsTorques(void) const;
        float64_t const & getMotorTorque(std::string const & motorName) const;
        void setSensorsData(float64_t const & t,
                            vectorN_t const & q,
                            vectorN_t const & v,
                            vectorN_t const & a,
                            vectorN_t const & u);
        sensorsDataMap_t getSensorsData(void) const;
        matrixN_t getSensorsData(std::string const & sensorType) const;
        vectorN_t getSensorData(std::string const & sensorType,
                                std::string const & motorName) const;

        result_t setOptions(configHolder_t mdlOptions); // Make a copy !
        configHolder_t getOptions(void) const;
        result_t setMotorOptions(std::string    const & motorName,
                                 configHolder_t const & motorOptions);
        result_t setMotorsOptions(configHolder_t const & motorsOptions);
        result_t getMotorOptions(std::string    const & motorName,
                                 configHolder_t       & motorOptions) const;
        result_t getMotorsOptions(configHolder_t & motorsOptions) const;
        result_t setSensorOptions(std::string    const & sensorType,
                                  std::string    const & sensorName,
                                  configHolder_t const & sensorOptions);
        result_t setSensorsOptions(std::string    const & sensorType,
                                   configHolder_t const & sensorsOptions);
        result_t setSensorsOptions(configHolder_t const & sensorsOptions);
        result_t getSensorOptions(std::string    const & sensorType,
                                  std::string    const & sensorName,
                                  configHolder_t       & sensorOptions) const;
        result_t getSensorsOptions(std::string    const & sensorType,
                                   configHolder_t       & sensorsOptions) const;
        result_t getSensorsOptions(configHolder_t & sensorsOptions) const;
        result_t setTelemetryOptions(configHolder_t const & telemetryOptions);
        result_t getTelemetryOptions(configHolder_t & telemetryOptions) const;

        // Those methods are not intended to be called manually. The Engine is taking care of it.
        virtual void reset(void);
        virtual result_t configureTelemetry(std::shared_ptr<TelemetryData> const & telemetryData);
        void updateTelemetry(void);
        bool_t const & getIsTelemetryConfigured(void) const;

        bool_t const & getIsInitialized(void) const;
        std::string const & getUrdfPath(void) const;
        bool_t const & getHasFreeflyer(void) const;
        // Getters without 'get' prefix for consistency with pinocchio C++ API
        uint32_t const & nq(void) const;
        uint32_t const & nv(void) const;
        uint32_t const & nx(void) const;

        std::vector<std::string> const & getContactFramesNames(void) const;
        std::vector<int32_t> const & getContactFramesIdx(void) const;
        std::vector<std::string> const & getMotorsNames(void) const;
        std::vector<int32_t> getMotorsModelIdx(void) const;
        std::vector<int32_t> getMotorsPositionIdx(void) const;
        std::vector<int32_t> getMotorsVelocityIdx(void) const;
        std::unordered_map<std::string, std::vector<std::string> > getSensorsNames(void) const;
        std::vector<std::string> getSensorsNames(std::string const & sensorType) const;
        std::vector<std::string> const & getRigidJointsNames(void) const;
        std::vector<int32_t> const & getRigidJointsModelIdx(void) const;
        std::vector<int32_t> const & getRigidJointsPositionIdx(void) const;
        std::vector<int32_t> const & getRigidJointsVelocityIdx(void) const;
        std::vector<std::string> const & getFlexibleJointsNames(void) const;
        std::vector<int32_t> const & getFlexibleJointsModelIdx(void) const;

        vectorN_t const & getPositionLimitMin(void) const;
        vectorN_t const & getPositionLimitMax(void) const;
        vectorN_t const & getVelocityLimit(void) const;
        vectorN_t getTorqueLimit(void) const;
        vectorN_t getMotorInertia(void) const;

        std::vector<std::string> const & getPositionFieldNames(void) const;
        std::vector<std::string> const & getVelocityFieldNames(void) const;
        std::vector<std::string> const & getAccelerationFieldNames(void) const;
        std::vector<std::string> const & getMotorTorqueFieldNames(void) const;

        result_t getLock(std::unique_ptr<MutexLocal::LockGuardLocal> & lock);
        bool_t const & getIsLocked(void) const;

    protected:
        result_t loadUrdfModel(std::string const & urdfPath,
                               bool_t      const & hasFreeflyer);
        result_t generateModelFlexible(void);
        result_t generateModelBiased(void);
        result_t refreshContactProxies(void);
        result_t refreshMotorProxies(void);
        virtual result_t refreshProxies(void);

    public:
        pinocchio::Model pncModel_;
        mutable pinocchio::Data pncData_;
        pinocchio::Model pncModelRigidOrig_;
        pinocchio::Data pncDataRigidOrig_;
        std::unique_ptr<modelOptions_t const> mdlOptions_;
        forceVector_t contactForces_;                       ///< Buffer storing the contact forces

    protected:
        bool_t isInitialized_;
        bool_t isTelemetryConfigured_;
        std::string urdfPath_;
        bool_t hasFreeflyer_;
        configHolder_t mdlOptionsHolder_;

        std::shared_ptr<TelemetryData> telemetryData_;
        motorsHolder_t motorsHolder_;
        sensorsGroupHolder_t sensorsGroupHolder_;
        std::unordered_map<std::string, bool_t> sensorTelemetryOptions_;

        std::vector<std::string> contactFramesNames_;       ///< Name of the frames of the contact points of the model
        std::vector<int32_t> contactFramesIdx_;             ///< Indices of the contact frames in the frame list of the model
        std::vector<std::string> motorsNames_;              ///< Joint name of the motors of the model
        std::vector<std::string> rigidJointsNames_;         ///< Name of the actual joints of the model, not taking into account the freeflyer
        std::vector<int32_t> rigidJointsModelIdx_;          ///< Index of the actual joints in the pinocchio model
        std::vector<int32_t> rigidJointsPositionIdx_;       ///< All the indices of the actual joints in the configuration vector of the model
        std::vector<int32_t> rigidJointsVelocityIdx_;       ///< All the indices of the actual joints in the velocity vector of the model
        std::vector<std::string> flexibleJointsNames_;      ///< Name of the flexibility joints of the model regardless of whether the flexibilities are enable
        std::vector<int32_t> flexibleJointsModelIdx_;       ///< Index of the flexibility joints in the pinocchio model regardless of whether the flexibilities are enable

        vectorN_t positionLimitMin_;
        vectorN_t positionLimitMax_;
        vectorN_t velocityLimit_;

        std::vector<std::string> positionFieldNames_;       ///< Fieldnames of the elements in the configuration vector of the rigid model
        std::vector<std::string> velocityFieldNames_;       ///< Fieldnames of the elements in the velocity vector of the rigid model
        std::vector<std::string> accelerationFieldNames_;   ///< Fieldnames of the elements in the acceleration vector of the rigid model
        std::vector<std::string> motorTorqueFieldNames_;    ///< Fieldnames of the torques of the motors

    private:
        MutexLocal mutexLocal_;
        pinocchio::Model pncModelFlexibleOrig_;
        std::shared_ptr<MotorSharedDataHolder_t> motorsSharedHolder_;
        std::unordered_map<std::string, std::shared_ptr<SensorSharedDataHolder_t> > sensorsSharedHolder_;
        uint32_t nq_;
        uint32_t nv_;
        uint32_t nx_;
    };
}

#endif //end of JIMINY_MODEL_H
