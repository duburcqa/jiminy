#ifndef JIMINY_MODEL_H
#define JIMINY_MODEL_H

#include "pinocchio/multibody/model.hpp"
#include "pinocchio/algorithm/frames.hpp"

#include "jiminy/core/Types.h"


namespace jiminy
{
    class Model
    {
    public:

    protected:
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

        virtual configHolder_t getDefaultModelOptions()
        {
            configHolder_t config;
            config["dynamics"] = getDefaultDynamicsOptions();
            config["joints"] = getDefaultJointOptions();

            return config;
        };

    public:
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
        // Disable the copy of the class
        Model(Model const & robot) = delete;
        Model & operator = (Model const & other) = delete;

    public:
        Model(void);
        virtual ~Model(void) = default;

        hresult_t initialize(std::string const & urdfPath,
                             bool_t      const & hasFreeflyer = true);

        hresult_t addContactPoints(std::vector<std::string> const & frameNames);
        hresult_t removeContactPoints(std::vector<std::string> const & frameNames = {});

        hresult_t setOptions(configHolder_t modelOptions); // Make a copy !
        configHolder_t getOptions(void) const;

        /// This method are not intended to be called manually. The Engine is taking care of it.
        virtual void reset(void);

        bool_t const & getIsInitialized(void) const;
        std::string const & getUrdfPath(void) const;
        bool_t const & getHasFreeflyer(void) const;
        // Getters without 'get' prefix for consistency with pinocchio C++ API
        uint32_t const & nq(void) const;
        uint32_t const & nv(void) const;
        uint32_t const & nx(void) const;

        std::vector<std::string> const & getContactFramesNames(void) const;
        std::vector<int32_t> const & getContactFramesIdx(void) const;
        std::vector<std::string> const & getRigidJointsNames(void) const;
        std::vector<int32_t> const & getRigidJointsModelIdx(void) const;
        std::vector<int32_t> const & getRigidJointsPositionIdx(void) const;
        std::vector<int32_t> const & getRigidJointsVelocityIdx(void) const;
        std::vector<std::string> const & getFlexibleJointsNames(void) const;
        std::vector<int32_t> const & getFlexibleJointsModelIdx(void) const;

        vectorN_t const & getPositionLimitMin(void) const;
        vectorN_t const & getPositionLimitMax(void) const;
        vectorN_t const & getVelocityLimit(void) const;

        std::vector<std::string> const & getPositionFieldnames(void) const;
        std::vector<std::string> const & getVelocityFieldnames(void) const;
        std::vector<std::string> const & getAccelerationFieldnames(void) const;

        hresult_t getFlexibleStateFromRigid(vectorN_t const & xRigid,
                                            vectorN_t       & xFlex) const;
        hresult_t getRigidStateFromFlexible(vectorN_t const & xFlex,
                                            vectorN_t       & xRigid) const;

    protected:
        hresult_t loadUrdfModel(std::string const & urdfPath,
                                bool_t      const & hasFreeflyer);
        hresult_t generateModelFlexible(void);
        hresult_t generateModelBiased(void);
        hresult_t refreshContactsProxies(void);
        virtual hresult_t refreshProxies(void);

    public:
        pinocchio::Model pncModel_;
        mutable pinocchio::Data pncData_;
        pinocchio::Model pncModelRigidOrig_;
        pinocchio::Data pncDataRigidOrig_;
        std::unique_ptr<modelOptions_t const> mdlOptions_;
        forceVector_t contactForces_;                       ///< Buffer storing the contact forces

    protected:
        bool_t isInitialized_;
        std::string urdfPath_;
        bool_t hasFreeflyer_;
        configHolder_t mdlOptionsHolder_;

        std::vector<std::string> contactFramesNames_;       ///< Name of the frames of the contact points of the robot
        std::vector<int32_t> contactFramesIdx_;             ///< Indices of the contact frames in the frame list of the robot
        std::vector<std::string> rigidJointsNames_;         ///< Name of the actual joints of the robot, not taking into account the freeflyer
        std::vector<int32_t> rigidJointsModelIdx_;          ///< Index of the actual joints in the pinocchio robot
        std::vector<int32_t> rigidJointsPositionIdx_;       ///< All the indices of the actual joints in the configuration vector of the robot
        std::vector<int32_t> rigidJointsVelocityIdx_;       ///< All the indices of the actual joints in the velocity vector of the robot
        std::vector<std::string> flexibleJointsNames_;      ///< Name of the flexibility joints of the robot regardless of whether the flexibilities are enable
        std::vector<int32_t> flexibleJointsModelIdx_;       ///< Index of the flexibility joints in the pinocchio robot regardless of whether the flexibilities are enable

        vectorN_t positionLimitMin_;
        vectorN_t positionLimitMax_;
        vectorN_t velocityLimit_;

        std::vector<std::string> positionFieldnames_;       ///< Fieldnames of the elements in the configuration vector of the rigid robot
        std::vector<std::string> velocityFieldnames_;       ///< Fieldnames of the elements in the velocity vector of the rigid robot
        std::vector<std::string> accelerationFieldnames_;   ///< Fieldnames of the elements in the acceleration vector of the rigid robot

    private:
        pinocchio::Model pncModelFlexibleOrig_;
        uint32_t nq_;
        uint32_t nv_;
        uint32_t nx_;
    };
}

#endif //end of JIMINY_MODEL_H
