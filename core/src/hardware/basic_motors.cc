#include "jiminy/core/utilities/helpers.h"
#include "jiminy/core/robot/robot.h"

#include "jiminy/core/hardware/basic_motors.h"


namespace jiminy
{
    SimpleMotor::SimpleMotor(const std::string & name) :
    AbstractMotorBase(name)
    {
        /* AbstractMotorBase constructor calls the base implementations of the virtual methods
           since the derived class is not available at this point. Thus it must be called
           explicitly in the constructor. */
        motorOptionsGeneric_ = getDefaultMotorOptions();
        setOptions(getOptions());
    }

    void SimpleMotor::initialize(const std::string & jointName)
    {
        // Update joint name
        jointName_ = jointName;
        isInitialized_ = true;

        /* Try refreshing proxies if possible, restore internals before throwing exception if not.
           Note that it will raise an exception if a simulation is already running. */
        try
        {
            refreshProxies();
        }
        catch (...)
        {
            jointName_.clear();
            isInitialized_ = false;
            throw;
        }
    }

    void SimpleMotor::setOptions(const GenericConfig & motorOptions)
    {
        // Check if the friction parameters make sense
        // Make sure the user-defined position limit has the right dimension
        if (boost::get<double>(motorOptions.at("velocityEffortInvSlope")) < 0.0)
        {
            JIMINY_THROW(std::invalid_argument, "'velocityEffortInvSlope' must be positive.");
        }
        if (boost::get<double>(motorOptions.at("frictionViscousPositive")) > 0.0)
        {
            JIMINY_THROW(std::invalid_argument, "'frictionViscousPositive' must be negative.");
        }
        if (boost::get<double>(motorOptions.at("frictionViscousNegative")) > 0.0)
        {
            JIMINY_THROW(std::invalid_argument, "'frictionViscousNegative' must be negative.");
        }
        if (boost::get<double>(motorOptions.at("frictionDryPositive")) > 0.0)
        {
            JIMINY_THROW(std::invalid_argument, "'frictionDryPositive' must be negative.");
        }
        if (boost::get<double>(motorOptions.at("frictionDryNegative")) > 0.0)
        {
            JIMINY_THROW(std::invalid_argument, "'frictionDryNegative' must be negative.");
        }
        if (boost::get<double>(motorOptions.at("frictionDrySlope")) < 0.0)
        {
            JIMINY_THROW(std::invalid_argument, "'frictionDrySlope' must be positive.");
        }

        // Check that velocity limit is not enabled with effort limit
        if (boost::get<bool>(motorOptions.at("enableVelocityLimit")) &&
            !boost::get<bool>(motorOptions.at("enableEffortLimit")))
        {
            JIMINY_THROW(std::invalid_argument,
                         "'enableVelocityLimit' cannot be enabled without 'enableEffortLimit'.");
        }

        // Update class-specific "strongly typed" accessor for fast and convenient access
        motorOptions_ = std::make_unique<const SimpleMotorOptions>(motorOptions);

        // Update base "strongly typed" option accessor and inherited polymorphic accessor
        AbstractMotorBase::setOptions(motorOptions);
    }

    void SimpleMotor::computeEffort(double /* t */,
                                    const Eigen::VectorBlock<const Eigen::VectorXd> & /* q */,
                                    double v,
                                    double /* a */,
                                    double command)
    {
        if (!isInitialized_)
        {
            JIMINY_THROW(bad_control_flow,
                         "Motor not initialized. Impossible to compute actual motor effort.");
        }

        // Extract motor data
        const auto & [uMotor, uTransmission] = data();

        // Compute the velocity at motor-level
        const double vMotor = motorOptions_->mechanicalReduction * v;

        /* Compute the motor effort, taking into account velocity and effort limits.
           It is the output of the motor on joint side, ie after the transmission. */
        double effortMin = -INF;
        double effortMax = INF;
        if (motorOptions_->enableEffortLimit)
        {
            effortMin = -effortLimit_;
            effortMax = effortLimit_;
            if (motorOptions_->enableVelocityLimit)
            {
                const double velocityDelta = effortLimit_ * motorOptions_->velocityEffortInvSlope;
                if (velocityDelta > 0.0)
                {
                    const double velocityThr = std::max(velocityLimit_ - velocityDelta, 0.0);
                    effortMin *= std::clamp(
                        (velocityLimit_ + vMotor) / (velocityLimit_ - velocityThr), 0.0, 1.0);
                    effortMax *= std::clamp(
                        (velocityLimit_ - vMotor) / (velocityLimit_ - velocityThr), 0.0, 1.0);
                }
            }
        }
        uMotor = std::clamp(command, effortMin, effortMax);

        // Translate motor effort on joint side
        uTransmission = motorOptions_->mechanicalReduction * uMotor;

        // Add transmission friction if enable
        if (motorOptions_->enableFriction)
        {
            if (v > 0.0)
            {
                uTransmission +=
                    motorOptions_->frictionViscousPositive * v +
                    motorOptions_->frictionDryPositive * tanh(motorOptions_->frictionDrySlope * v);
            }
            else
            {
                uTransmission +=
                    motorOptions_->frictionViscousNegative * v +
                    motorOptions_->frictionDryNegative * tanh(motorOptions_->frictionDrySlope * v);
            }
        }
    }
}
