#ifndef SIMU_MODEL_H
#define SIMU_MODEL_H

#include <string>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "pinocchio/multibody/model.hpp"
#include "pinocchio/algorithm/frames.hpp"

#include <boost/circular_buffer.hpp>

#include "jiminy/core/Types.h"


namespace jiminy
{
    std::string const JOINT_PREFIX_BASE("current");
    std::string const FREE_FLYER_PREFIX_BASE_NAME(JOINT_PREFIX_BASE + "FreeFlyer");
    std::string const FLEXIBLE_JOINT_SUFFIX = "FlexibleJoint";

    class Engine;
    class AbstractSensorBase;
    class TelemetryData;
    struct SensorDataHolder_t;

    class Model
    {
        friend Engine;

    public:
        // Disable the copy of the class
        Model(Model const & model) = delete;
        Model & operator = (Model const & other) = delete;

    public:
        virtual configHolder_t getDefaultJointOptions()
        {
            configHolder_t config;
            config["positionLimitFromUrdf"] = true;
            config["positionLimitMin"] = vectorN_t();
            config["positionLimitMax"] = vectorN_t();
            config["velocityLimitFromUrdf"] = true;
            config["velocityLimit"] = vectorN_t();

            return config;
        };

        struct jointOptions_t
        {
            bool      const positionLimitFromUrdf;
            vectorN_t const positionLimitMin;
            vectorN_t const positionLimitMax;
            bool      const velocityLimitFromUrdf;
            vectorN_t const velocityLimit;

            jointOptions_t(configHolder_t const & options) :
            positionLimitFromUrdf(boost::get<bool>(options.at("positionLimitFromUrdf"))),
            positionLimitMin(boost::get<vectorN_t>(options.at("positionLimitMin"))),
            positionLimitMax(boost::get<vectorN_t>(options.at("positionLimitMax"))),
            velocityLimitFromUrdf(boost::get<bool>(options.at("velocityLimitFromUrdf"))),
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
            config["flexibleJointsNames"] = std::vector<std::string>();
            config["flexibleJointsStiffness"] = std::vector<vectorN_t>();
            config["flexibleJointsDamping"] = std::vector<vectorN_t>();

            return config;
        };

        struct dynamicsOptions_t
        {
            float64_t                const inertiaBodiesBiasStd;
            float64_t                const massBodiesBiasStd;
            float64_t                const centerOfMassPositionBodiesBiasStd;
            float64_t                const relativePositionBodiesBiasStd;
            bool                     const enableFlexibleModel;
            std::vector<std::string> const flexibleJointsNames;
            std::vector<vectorN_t>   const flexibleJointsStiffness;
            std::vector<vectorN_t>   const flexibleJointsDamping;

            dynamicsOptions_t(configHolder_t const & options) :
            inertiaBodiesBiasStd(boost::get<float64_t>(options.at("inertiaBodiesBiasStd"))),
            massBodiesBiasStd(boost::get<float64_t>(options.at("massBodiesBiasStd"))),
            centerOfMassPositionBodiesBiasStd(boost::get<float64_t>(options.at("centerOfMassPositionBodiesBiasStd"))),
            relativePositionBodiesBiasStd(boost::get<float64_t>(options.at("relativePositionBodiesBiasStd"))),
            enableFlexibleModel(boost::get<bool>(options.at("enableFlexibleModel"))),
            flexibleJointsNames(boost::get<std::vector<std::string> >(options.at("flexibleJointsNames"))),
            flexibleJointsStiffness(boost::get<std::vector<vectorN_t> >(options.at("flexibleJointsStiffness"))),
            flexibleJointsDamping(boost::get<std::vector<vectorN_t> >(options.at("flexibleJointsDamping")))
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
        Model(void);
        virtual ~Model(void);

        result_t initialize(std::string              const & urdfPath,
                            std::vector<std::string> const & contactFramesNames,
                            std::vector<std::string> const & motorsNames,
                            bool                     const & hasFreeflyer = true);
        virtual void reset(void);

        virtual result_t configureTelemetry(std::shared_ptr<TelemetryData> const & telemetryData);
        void updateTelemetry(void);

        template<typename TSensor>
        result_t addSensor(std::string              const & sensorName,
                           std::shared_ptr<TSensor>       & sensor);
        result_t removeSensor(std::string const & sensorType,
                              std::string const & sensorName);
        result_t removeSensors(std::string const & sensorType);

        configHolder_t getOptions(void) const;
        result_t setOptions(configHolder_t mdlOptions); // Make a copy !
        result_t getSensorOptions(std::string    const & sensorType,
                                  std::string    const & sensorName,
                                  configHolder_t       & sensorOptions) const;
        result_t getSensorsOptions(std::string    const & sensorType,
                                   configHolder_t       & sensorsOptions) const;
        result_t getSensorsOptions(configHolder_t & sensorsOptions) const;
        result_t setSensorOptions(std::string    const & sensorType,
                                  std::string    const & sensorName,
                                  configHolder_t const & sensorOptions);
        result_t setSensorsOptions(std::string    const & sensorType,
                                   configHolder_t const & sensorsOptions);
        result_t setSensorsOptions(configHolder_t const & sensorsOptions);
        result_t getTelemetryOptions(configHolder_t & telemetryOptions) const;
        result_t setTelemetryOptions(configHolder_t const & telemetryOptions);
        bool const & getIsInitialized(void) const;
        bool const & getIsTelemetryConfigured(void) const;
        std::string const & getUrdfPath(void) const;
        bool const & getHasFreeFlyer(void) const;
        std::unordered_map<std::string, std::vector<std::string> > getSensorsNames(void) const;
        result_t getSensorsData(std::vector<matrixN_t> & data) const;
        result_t getSensorsData(std::string const & sensorType,
                                matrixN_t         & data) const;
        result_t getSensorData(std::string const & sensorType,
                               std::string const & sensorName,
                               vectorN_t         & data) const;
        void setSensorsData(float64_t const & t,
                            vectorN_t const & q,
                            vectorN_t const & v,
                            vectorN_t const & a,
                            vectorN_t const & u);
        std::vector<int32_t> const & getContactFramesIdx(void) const;
        std::vector<std::string> const & getMotorsNames(void) const;
        std::vector<int32_t> const & getMotorsPositionIdx(void) const;
        std::vector<int32_t> const & getMotorsVelocityIdx(void) const;
        std::vector<std::string> const & getRigidJointsNames(void) const;
        std::vector<int32_t> const & getRigidJointsPositionIdx(void) const;
        std::vector<int32_t> const & getRigidJointsVelocityIdx(void) const;
        std::vector<std::string> const & getFlexibleJointsNames(void) const;
        std::vector<int32_t> const & getFlexibleJointsPositionIdx(void) const;
        std::vector<int32_t> const & getFlexibleJointsVelocityIdx(void) const;
        std::vector<std::string> const & getPositionFieldNames(void) const;
        std::vector<std::string> const & getVelocityFieldNames(void) const;
        std::vector<std::string> const & getAccelerationFieldNames(void) const;
        std::vector<std::string> const & getMotorTorqueFieldNames(void) const;
        vectorN_t const & getPositionLimitMin(void) const;
        vectorN_t const & getPositionLimitMax(void) const;
        vectorN_t const & getVelocityLimit(void) const;

        // Getter without keywords for consistency with pinocchio C++ API
        uint32_t const & nq(void) const;
        uint32_t const & nv(void) const;
        uint32_t const & nx(void) const;

        template<typename TSensor>
        result_t getSensor(std::string              const & sensorType,
                           std::string              const & sensorName,
                           std::shared_ptr<TSensor>       & sensor);

    protected:
        result_t loadUrdfModel(std::string const & urdfPath,
                               bool        const & hasFreeflyer);
        result_t generateFlexibleModel(void);
        result_t generateBiasedModel(void);
        result_t generateFieldNames(void);

    public:
        pinocchio::Model pncModel_;
        pinocchio::Data pncData_;
        std::unique_ptr<modelOptions_t const> mdlOptions_;
        pinocchio::container::aligned_vector<pinocchio::Force> contactForces_; // Buffer to store the contact forces

    protected:
        bool isInitialized_;
        bool isTelemetryConfigured_;
        std::string urdfPath_;
        bool hasFreeflyer_;
        configHolder_t mdlOptionsHolder_;

        std::shared_ptr<TelemetryData> telemetryData_;
        sensorsGroupHolder_t sensorsGroupHolder_;
        std::unordered_map<std::string, bool> sensorTelemetryOptions_;

        std::vector<std::string> contactFramesNames_;       // Name of the frames of the contact points of the model
        std::vector<int32_t> contactFramesIdx_;             // Indices of the contact frames in the frame list of the model
        std::vector<std::string> motorsNames_;              // Joint name of the motors of the model
        std::vector<int32_t> motorsPositionIdx_;            // First indices of the motors in the configuration vector of the model
        std::vector<int32_t> motorsVelocityIdx_;            // First indices of the motors in the velocity vector of the model
        std::vector<std::string> rigidJointsNames_;         // Name of the actual joints of the model
        std::vector<int32_t> rigidJointsPositionIdx_;       // Indices of the actual joints in the configuration vector of the model
        std::vector<int32_t> rigidJointsVelocityIdx_;       // Indices of the actual joints in the velocity vector of the model
        std::vector<std::string> flexibleJointsNames_;      // Name of the flexibility joints of the model
        std::vector<int32_t> flexibleJointsPositionIdx_;    // First indices of the flexibility joints in the configuration vector of the model
        std::vector<int32_t> flexibleJointsVelocityIdx_;    // First indices of the flexibility joints in the velocity vector of the model

        vectorN_t positionLimitMin_;
        vectorN_t positionLimitMax_;
        vectorN_t velocityLimit_;

        std::vector<std::string> positionFieldNames_;       // Fieldnames of the elements in the configuration vector of the rigid model
        std::vector<std::string> velocityFieldNames_;       // Fieldnames of the elements in the velocity vector of the rigid model
        std::vector<std::string> accelerationFieldNames_;   // Fieldnames of the elements in the acceleration vector of the rigid model
        std::vector<std::string> motorTorqueFieldNames_;    // Fieldnames of the torques of the motors

    private:
        pinocchio::Model pncModelRigidOrig_;
        pinocchio::Model pncModelFlexibleOrig_;
        std::unordered_map<std::string, std::shared_ptr<SensorDataHolder_t> > sensorsDataHolder_;
        uint32_t nq_;
        uint32_t nv_;
        uint32_t nx_;
    };

    struct SensorDataHolder_t
    {
        SensorDataHolder_t(void) :
        time_(),
        data_(),
        counters_(),
        sensors_(),
        num_()
        {
            // Empty.
        };

        ~SensorDataHolder_t(void)
        {
            // Empty.
        };

        boost::circular_buffer_space_optimized<float64_t> time_;
        boost::circular_buffer_space_optimized<matrixN_t> data_;
        std::vector<uint32_t> counters_;
        std::vector<AbstractSensorBase *> sensors_;
        uint32_t num_;
    };
}

#include "jiminy/core/Model.tcc"

#endif //end of SIMU_MODEL_H
