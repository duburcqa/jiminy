#include "jiminy/core/utilities/helpers.h"
#include "jiminy/core/robot/robot.h"

#include "jiminy/core/hardware/basic_motors.h"


namespace jiminy
{
    SimpleMotor::SimpleMotor(const std::string & name) noexcept :
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
        // Make sure that no simulation is already running
        // TODO: This check should be enforced by AbstractMotor somehow
        auto robot = robot_.lock();
        if (robot && robot->getIsLocked())
        {
            JIMINY_THROW(bad_control_flow,
                         "Robot already locked, probably because a simulation is running. "
                         "Please stop it before refreshing motor proxies.");
        }

        // Update joint name
        jointName_ = jointName;
        isInitialized_ = true;

        // Try refreshing proxies if possible, restore internals before throwing exception if not
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

        /* Compute the motor effort, taking into account the limit, if any.
           It is the output of the motor on joint side, ie after the transmission. */
        if (motorOptions_->enableCommandLimit)
        {
            command = std::clamp(command, -commandLimit_, commandLimit_);
        }
        data() = motorOptions_->mechanicalReduction * command;

        /* Add friction to the joints associated with the motor if enable.
           It is computed on joint side instead of the motor. */
        if (motorOptions_->enableFriction)
        {
            if (v > 0)
            {
                data() +=
                    motorOptions_->frictionViscousPositive * v +
                    motorOptions_->frictionDryPositive * tanh(motorOptions_->frictionDrySlope * v);
            }
            else
            {
                data() +=
                    motorOptions_->frictionViscousNegative * v +
                    motorOptions_->frictionDryNegative * tanh(motorOptions_->frictionDrySlope * v);
            }
        }
    }
}
