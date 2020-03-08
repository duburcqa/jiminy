#include "jiminy/core/AbstractMotor.h"
#include "jiminy/core/Model.h"


namespace jiminy
{
    AbstractMotor::AbstractMotor(Model       const & model,
                                 std::shared_ptr<MotorSharedDataHolder_t> const & sharedHolder,
                                 std::string const & name) :
    motorOptions_(nullptr),
    baseMotorOptionsHolder_(),
    isInitialized_(false),
    model_(&model),
    sharedHolder_(sharedHolder),
    name_(name),
    motorId_(sharedHolder_->num_)
    {
        // Initialize the options
        setOptions(getDefaultOptions());

        // Add the motor to the data holder
        ++sharedHolder_->num_;
        sharedHolder_->motors_.push_back(this);
    }

    AbstractMotor::~AbstractMotor(void)
    {
        // Remove associated col in the global data buffer
        if(motorId_ < sharedHolder_->num_ - 1)
        {
            int8_t motorShift = sharedHolder_->num_ - motorId_ - 1;
            sharedHolder_->data_.segment(motorId_, motorShift) =
                sharedHolder_->data_.segment(motorId_ + 1, motorShift).eval(); // eval to avoid aliasing
        }
        sharedHolder_->data_.conservativeResize(sharedHolder_->num_ - 1);

        // Shift the motor ids
        for (uint8_t i = motorId_ + 1; i < sharedHolder_->num_; i++)
        {
            --sharedHolder_->motors_[i]->motorId_;
        }

        // Remove the deprecated elements of the global containers
        sharedHolder_->motors_.erase(sharedHolder_->motors_.begin() + motorId_);

        // Update the total number of motors left
        --sharedHolder_->num_;
    }

    void AbstractMotor::reset(void)
    {
        sharedHolder_->data_ = vectorN_t::Zero(sharedHolder_->num_);
    }

    float64_t & AbstractMotor::data(void)
    {
        return sharedHolder_->data_[motorId_];
    }

    float64_t const & AbstractMotor::get(void) const
    {
        return sharedHolder_->data_[motorId_];
    }

    vectorN_t const & AbstractMotor::getAll(void) const
    {
        return sharedHolder_->data_;
    }

    configHolder_t AbstractMotor::getOptions(void) const
    {
        return baseMotorOptionsHolder_;
    }

    result_t AbstractMotor::setOptions(configHolder_t const & motorOptions)
    {
        baseMotorOptionsHolder_ = motorOptions;
        motorOptions_ = std::make_unique<abstractMotorOptions_t const>(baseMotorOptionsHolder_);
        return result_t::SUCCESS;
    }

    result_t AbstractMotor::setOptionsAll(configHolder_t const & motorOptions)
    {
        result_t returnCode = result_t::SUCCESS;

        for (AbstractMotor * motor : sharedHolder_->motors_)
        {
            if (returnCode == result_t::SUCCESS)
            {
                returnCode = motor->setOptions(motorOptions);
            }
        }

        return returnCode;
    }

    bool_t const & AbstractMotor::getIsInitialized(void) const
    {
        return isInitialized_;
    }

    std::string const & AbstractMotor::getName(void) const
    {
        return name_;
    }

    uint8_t const & AbstractMotor::getId(void) const
    {
        return motorId_;
    }

    result_t AbstractMotor::computeAllEfforts(float64_t const & t,
                                              vectorN_t const & q,
                                              vectorN_t const & v,
                                              vectorN_t const & a,
                                              vectorN_t const & u)
    {
        result_t returnCode = result_t::SUCCESS;

        // Compute the motors' output
        for (AbstractMotor * motor : sharedHolder_->motors_)
        {
            if (returnCode == result_t::SUCCESS)
            {
                // Compute the actual torque
                returnCode = motor->computeEffort(t, q, v, a, u);
            }
        }

        return returnCode;
    }
}
