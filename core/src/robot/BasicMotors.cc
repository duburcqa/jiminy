#include <algorithm>

#include "jiminy/core/utilities/Helpers.h"

#include "jiminy/core/robot/BasicMotors.h"


namespace jiminy
{
    SimpleMotor::SimpleMotor(std::string const & name) :
    AbstractMotorBase(name),
    motorOptions_(nullptr)
    {
        /* AbstractMotorBase constructor calls the base implementations of
           the virtual methods since the derived class is not available at
           this point. Thus it must be called explicitly in the constructor. */
        setOptions(getDefaultMotorOptions());
    }

    hresult_t SimpleMotor::initialize(std::string const & jointName)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        jointName_ = jointName;
        isInitialized_ = true;
        returnCode = refreshProxies();

        if (returnCode != hresult_t::SUCCESS)
        {
            jointName_.clear();
            isInitialized_ = false;
        }

        return returnCode;
    }

    hresult_t SimpleMotor::setOptions(configHolder_t const & motorOptions)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        returnCode = AbstractMotorBase::setOptions(motorOptions);

        // Check if the friction parameters make sense
        if (returnCode == hresult_t::SUCCESS)
        {
            // Make sure the user-defined position limit has the right dimension
            if (boost::get<float64_t>(motorOptions.at("frictionViscousPositive")) > 0.0)
            {
                PRINT_ERROR("'frictionViscousPositive' must be negative.");
                returnCode = hresult_t::ERROR_BAD_INPUT;
            }
            if (boost::get<float64_t>(motorOptions.at("frictionViscousNegative")) > 0.0)
            {
                PRINT_ERROR("'frictionViscousNegative' must be negative.");
                returnCode = hresult_t::ERROR_BAD_INPUT;
            }
            if (boost::get<float64_t>(motorOptions.at("frictionDryPositive")) > 0.0)
            {
                PRINT_ERROR("'frictionDryPositive' must be negative.");
                returnCode = hresult_t::ERROR_BAD_INPUT;
            }
            if (boost::get<float64_t>(motorOptions.at("frictionDryNegative")) > 0.0)
            {
                PRINT_ERROR("'frictionDryNegative' must be negative.");
                returnCode = hresult_t::ERROR_BAD_INPUT;
            }
            if (boost::get<float64_t>(motorOptions.at("frictionDrySlope")) < 0.0)
            {
                PRINT_ERROR("'frictionDrySlope' must be positive.");
                returnCode = hresult_t::ERROR_BAD_INPUT;
            }
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            motorOptions_ = std::make_unique<motorOptions_t const>(motorOptions);
        }

        return returnCode;
    }

    hresult_t SimpleMotor::computeEffort(float64_t const & /* t */,
                                         Eigen::VectorBlock<vectorN_t const> const & /* q */,
                                         float64_t const & v,
                                         float64_t const & /* a */,
                                         float64_t command)
    {
        if (!isInitialized_)
        {
            PRINT_ERROR("Motor not initialized. Impossible to compute actual motor effort.");
            return hresult_t::ERROR_INIT_FAILED;
        }

        /* Compute the motor effort, taking into account the limit, if any.
           It is the output of the motor on joint side, ie after the transmission. */
        if (motorOptions_->enableCommandLimit)
        {
            command = clamp(command, -getCommandLimit(), getCommandLimit());
        }
        data() = motorOptions_->mechanicalReduction * command;

        /* Add friction to the joints associated with the motor if enable.
           It is computed on joint side instead of the motor. */
        if (motorOptions_->enableFriction)
        {
            float64_t const & vJoint = v;
            if (vJoint > 0)
            {
                data() += motorOptions_->frictionViscousPositive * vJoint
                        + motorOptions_->frictionDryPositive * tanh(motorOptions_->frictionDrySlope * vJoint);
            }
            else
            {
                data() += motorOptions_->frictionViscousNegative * vJoint
                        + motorOptions_->frictionDryNegative * tanh(motorOptions_->frictionDrySlope * vJoint);
            }
        }

        return hresult_t::SUCCESS;
    }
}
