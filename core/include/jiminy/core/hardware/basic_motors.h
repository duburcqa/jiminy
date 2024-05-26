#ifndef JIMINY_BASIC_MOTORS_H
#define JIMINY_BASIC_MOTORS_H

#include "jiminy/core/fwd.h"
#include "jiminy/core/utilities/helpers.h"
#include "jiminy/core/hardware/abstract_motor.h"


namespace jiminy
{
    class JIMINY_DLLAPI SimpleMotor : public AbstractMotorBase
    {
    public:
        /// \brief Dictionary gathering the configuration options shared between motors.
        virtual GenericConfig getDefaultMotorOptions() override
        {
            // Add extra options or update default values
            GenericConfig config = AbstractMotorBase::getDefaultMotorOptions();

            config["enableVelocityLimit"] = false;
            config["velocityEffortInvSlope"] = 0.0;
            config["enableEffortLimit"] = true;
            config["enableFriction"] = false;
            config["frictionViscousPositive"] = 0.0;
            config["frictionViscousNegative"] = 0.0;
            config["frictionDryPositive"] = 0.0;
            config["frictionDryNegative"] = 0.0;
            config["frictionDrySlope"] = 0.0;

            return config;
        };

        /// \brief Simple motor that does not required specifying advanced motor constant (phase
        ///        inductance, back-EMF...), nor hardware properties such as battery voltage.
        ///
        /// \details The motor torque is applied instantaneously, but may be saturated to make into
        ///          account the maximum torque at zero velocity and the maximum velocity at zero
        ///          load. A simple linear model is used for the maximum velocity. For reference:
        ///          https://things-in-motion.blogspot.com/2019/05/understanding-bldc-pmsm-electric-motors.html
        ///
        /// \sa For details about the modelling of PMSM motors (ie brushless motors), see:
        ///     https://github.com/matthieuvigne/nemo_bldc/blob/1c9073114a70d762b8d13774e7da984afd48bd32/src/nemo_bldc/doc/BrushlessMotorPhysics.pdf
        struct SimpleMotorOptions : public AbstractMotorOptions
        {
            /// \brief Wether velocity limit is enabled.
            const bool enableVelocityLimit;
            /// \brief Inverse constant decrease rate of the maximum torque wrt velocity when
            ///        approaching the maximum velocity. The maximum torque is equal to zero at
            ///        maximum velocity.
            const double velocityEffortInvSlope;
            /// \brief Wether effort limit is enabled.
            const bool enableEffortLimit;
            /// \brief Wether joint friction is enabled.
            const bool enableFriction;
            /// \brief Viscous coefficient of the joint friction for positive velocity.
            ///
            /// \pre Must be negative.
            const double frictionViscousPositive;
            /// \brief Viscous coefficient of the joint friction for negative velocity.
            ///
            /// \pre Must be negative.
            const double frictionViscousNegative;
            /// \brief Dry coefficient of the joint friction for positive velocity, which
            ///        corresponds to the positive dry friction at saturation.
            ///
            /// \pre Must be negative.
            const double frictionDryPositive;
            /// \brief Dry coefficient of the joint friction for negative velocity, which
            ///        corresponds to the negative dry friction at saturation.
            ///
            /// \pre Must be negative.
            const double frictionDryNegative;
            /// \brief Slope of the Tanh of the joint velocity that saturates the dry friction.
            ///
            /// \pre Must be negative.
            const double frictionDrySlope;

            SimpleMotorOptions(const GenericConfig & options) :
            AbstractMotorOptions(options),
            enableVelocityLimit(boost::get<bool>(options.at("enableVelocityLimit"))),
            velocityEffortInvSlope{boost::get<double>(options.at("velocityEffortInvSlope"))},
            enableEffortLimit(boost::get<bool>(options.at("enableEffortLimit"))),
            enableFriction{boost::get<bool>(options.at("enableFriction"))},
            frictionViscousPositive{boost::get<double>(options.at("frictionViscousPositive"))},
            frictionViscousNegative{boost::get<double>(options.at("frictionViscousNegative"))},
            frictionDryPositive{boost::get<double>(options.at("frictionDryPositive"))},
            frictionDryNegative{boost::get<double>(options.at("frictionDryNegative"))},
            frictionDrySlope{boost::get<double>(options.at("frictionDrySlope"))}
            {
            }
        };

    public:
        explicit SimpleMotor(const std::string & name);
        virtual ~SimpleMotor() = default;

        void initialize(const std::string & jointName);

        void setOptions(const GenericConfig & motorOptions) override;

    private:
        void computeEffort(double t,
                           const Eigen::VectorBlock<const Eigen::VectorXd> & q,
                           double v,
                           double a,
                           double command) override;

    private:
        std::unique_ptr<const SimpleMotorOptions> motorOptions_{nullptr};
    };
}

#endif  // end of JIMINY_BASIC_MOTORS_H