#ifndef JIMINY_BASIC_MOTORS_H
#define JIMINY_BASIC_MOTORS_H

#include "jiminy/core/robot/AbstractMotor.h"


namespace jiminy
{
    class SimpleMotor : public AbstractMotorBase
    {
    public:
        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Dictionary gathering the configuration options shared between motors
        ///////////////////////////////////////////////////////////////////////////////////////////////
        virtual configHolder_t getDefaultMotorOptions(void) override
        {
            // Add extra options or update default values
            configHolder_t config = AbstractMotorBase::getDefaultMotorOptions();

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
        SimpleMotor(std::string const & name);
        virtual ~SimpleMotor(void) = default;

        auto shared_from_this() { return shared_from(this); }
        auto shared_from_this() const { return shared_from(this); }

        hresult_t initialize(std::string const & jointName);

        virtual hresult_t setOptions(configHolder_t const & motorOptions) final override;

    private:
        virtual hresult_t computeEffort(float64_t const & t,
                                        Eigen::VectorBlock<vectorN_t const> const & q,
                                        float64_t const & v,
                                        float64_t const & a,
                                        float64_t command) final override;

    private:
        std::unique_ptr<motorOptions_t const> motorOptions_;
    };
}

#endif //end of JIMINY_BASIC_MOTORS_H