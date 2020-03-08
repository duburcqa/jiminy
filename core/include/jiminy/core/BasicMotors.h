#ifndef JIMINY_BASIC_MOTORS_H
#define JIMINY_BASIC_MOTORS_H

#include "jiminy/core/AbstractMotor.h"


namespace jiminy
{
    class Model;

    class SimpleMotor : public AbstractMotor
    {
    public:
        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Dictionary gathering the configuration options shared between motors
        ///////////////////////////////////////////////////////////////////////////////////////////////
        virtual configHolder_t getDefaultOptions(void) override
        {
            configHolder_t config;
            config["enableFriction"] = false;
            config["frictionViscousPositive"] = 0.0;
            config["frictionViscousNegative"] = 0.0;
            config["frictionDryPositive"] = 0.0;
            config["frictionDryNegative"] = 0.0;
            config["frictionDrySlope"] = 0.0;

            return config;
        };

        struct motorOptions_t : public abstractMotorOptions_t
        {
            bool_t    const enableFriction;             ///< Flag to enable the joint friction. It is always negative.
            float64_t const frictionViscousPositive;    ///< Viscous coefficient of the joint friction for positive velocity. It is always negative.
            float64_t const frictionViscousNegative;    ///< Viscous coefficient of the joint friction for negative velocity. It is always negative.
            float64_t const frictionDryPositive;        ///< Dry coefficient of the joint friction for positive velocity, which corresponds to the positive dry friction at saturation. It is always negative.
            float64_t const frictionDryNegative;        ///< Dry coefficient of the joint friction for negative velocity, which corresponds to the negative dry friction at saturation. It is always negative.
            float64_t const frictionDrySlope;           ///< Slope of the Tanh of the joint velocity that saturate the dry friction at frictionDry. It is always positive (no dry friction when equal to zero).

            motorOptions_t(configHolder_t const & options) :
            abstractMotorOptions_t(options),
            enableFriction(boost::get<bool_t>(options.at("enableFriction"))),
            frictionViscousPositive(boost::get<float64_t>(options.at("frictionViscousPositive"))),
            frictionViscousNegative(boost::get<float64_t>(options.at("frictionViscousNegative"))),
            frictionDryPositive(boost::get<float64_t>(options.at("frictionDryPositive"))),
            frictionDryNegative(boost::get<float64_t>(options.at("frictionDryNegative"))),
            frictionDrySlope(boost::get<float64_t>(options.at("frictionDrySlope")))
            {
                // Empty.
            }
        };

    public:
        SimpleMotor(Model       const & model,
                    std::shared_ptr<MotorSharedDataHolder_t> const & dataHolder,
                    std::string const & name);
        ~SimpleMotor(void) = default;

        result_t initialize(std::string const & jointName);
        virtual void reset(void) override;

        virtual result_t setOptions(configHolder_t motorOptions);

        std::string const & getJointName(void) const;

    private:
        virtual result_t computeEffort(float64_t const & t,
                                       vectorN_t const & q,
                                       vectorN_t const & v,
                                       vectorN_t const & a,
                                       vectorN_t const & u) override;

    private:
        std::unique_ptr<motorOptions_t const> motorOptions_;

        std::string jointName_;
        int32_t motorPositionIdx_;
        int32_t motorVelocityIdx_;
    };
}

#endif //end of JIMINY_BASIC_MOTORS_H