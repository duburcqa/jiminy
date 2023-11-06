#ifndef JIMINY_BASIC_MOTORS_H
#define JIMINY_BASIC_MOTORS_H

#include "jiminy/core/macros.h"
#include "jiminy/core/hardware/abstract_motor.h"


namespace jiminy
{
    class JIMINY_DLLAPI SimpleMotor : public AbstractMotorBase
    {
    public:
        /// \brief Dictionary gathering the configuration options shared between motors.
        virtual configHolder_t getDefaultMotorOptions() override
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
            /// \brief Flag to enable the joint friction.
            ///
            /// \pre Must be negative.
            const bool_t enableFriction;
            /// \brief Viscous coefficient of the joint friction for positive velocity.
            ///
            /// \pre Must be negative.
            const float64_t frictionViscousPositive;
            /// \brief Viscous coefficient of the joint friction for negative velocity.
            ///
            /// \pre Must be negative.
            const float64_t frictionViscousNegative;
            /// \brief Dry coefficient of the joint friction for positive velocity, which
            ///        corresponds to the positive dry friction at saturation.
            ///
            /// \pre Must be negative.
            const float64_t frictionDryPositive;
            /// \brief Dry coefficient of the joint friction for negative velocity, which
            ///        corresponds to the negative dry friction at saturation.
            ///
            /// \pre Must be negative.
            const float64_t frictionDryNegative;
            /// \brief Slope of the Tanh of the joint velocity that saturates the dry friction.
            ///
            /// \pre Must be negative.
            const float64_t frictionDrySlope;

            motorOptions_t(const configHolder_t & options) :
            abstractMotorOptions_t(options),
            enableFriction(boost::get<bool_t>(options.at("enableFriction"))),
            frictionViscousPositive(boost::get<float64_t>(options.at("frictionViscousPositive"))),
            frictionViscousNegative(boost::get<float64_t>(options.at("frictionViscousNegative"))),
            frictionDryPositive(boost::get<float64_t>(options.at("frictionDryPositive"))),
            frictionDryNegative(boost::get<float64_t>(options.at("frictionDryNegative"))),
            frictionDrySlope(boost::get<float64_t>(options.at("frictionDrySlope")))
            {
            }
        };

    public:
        SimpleMotor(const std::string & name);
        virtual ~SimpleMotor() = default;

        auto shared_from_this() { return shared_from(this); }
        auto shared_from_this() const { return shared_from(this); }

        hresult_t initialize(const std::string & jointName);

        virtual hresult_t setOptions(const configHolder_t & motorOptions) final override;

    private:
        virtual hresult_t computeEffort(const float64_t & t,
                                        const Eigen::VectorBlock<const Eigen::VectorXd> & q,
                                        const float64_t & v,
                                        const float64_t & a,
                                        float64_t command) final override;

    private:
        std::unique_ptr<const motorOptions_t> motorOptions_;
    };
}

#endif  // end of JIMINY_BASIC_MOTORS_H