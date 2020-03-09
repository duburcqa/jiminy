#include "jiminy/core/AbstractMotor.h"
#include "jiminy/core/Model.h"
#include "jiminy/core/Utilities.h"


namespace jiminy
{
    AbstractMotor::AbstractMotor(Model             & model,
                                 std::shared_ptr<MotorSharedDataHolder_t> const & sharedHolder,
                                 std::string const & name) :
    baseMotorOptions_(nullptr),
    motorOptionsHolder_(),
    isInitialized_(false),
    model_(&model),
    sharedHolder_(sharedHolder),
    name_(name),
    motorId_(sharedHolder_->num_),
    jointName_(),
    jointVelocityIdx_(),
    torqueLimit_()
    {
        // Initialize the options
        setOptions(getDefaultOptions());

        // Add the motor to the data holder
        ++sharedHolder_->num_;
        sharedHolder_->motors_.push_back(this);

        // Generate a new data buffer taking into account the new motor
        clearDataBuffer();
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

        // Generate a new data buffer taking into account the new motor
        clearDataBuffer();
    }

    result_t AbstractMotor::initialize(std::string const & jointName)
    {
        result_t returnCode = result_t::SUCCESS;

        if (returnCode == result_t::SUCCESS)
        {
            jointName_ = jointName;
            returnCode = refreshProxies();
        }

        if (returnCode == result_t::SUCCESS)
        {
            returnCode = model_->refreshMotorProxies();
        }

        if (returnCode == result_t::SUCCESS)
        {
            isInitialized_ = true;
        }

        return returnCode;
    }

    void AbstractMotor::clearDataBuffer(void)
    {
        sharedHolder_->data_ = vectorN_t::Zero(sharedHolder_->num_);
    }

    void AbstractMotor::reset(void)
    {
        // Clear the data buffer
        clearDataBuffer();

        // Refresh proxies that are model-dependent
        refreshProxies();
    }

    result_t AbstractMotor::refreshProxies(void)
    {
        result_t returnCode = result_t::SUCCESS;

        if (model_->getIsInitialized())
        {
            std::cout << "Error - AbstractMotor::refreshProxies - Model not initialized. Impossible to refresh model-dependent proxies." << std::endl;
            returnCode =  result_t::ERROR_INIT_FAILED;
        }

        if (returnCode == result_t::SUCCESS)
        {
            returnCode = ::jiminy::getJointVelocityIdx(model_->pncModel_, jointName_, jointVelocityIdx_);
        }

        if (returnCode == result_t::SUCCESS)
        {
            // Get the motor torque limits from the URDF or the user options.
            if (baseMotorOptions_->torqueLimitFromUrdf)
            {
                torqueLimit_ = model_->pncModel_.effortLimit[jointVelocityIdx_];
            }
            else
            {
                torqueLimit_ = baseMotorOptions_->torqueLimit;
            }
        }

        return returnCode;
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
        return motorOptionsHolder_;
    }

    result_t AbstractMotor::setOptions(configHolder_t const & motorOptions)
    {

        // Check if the internal buffers must be updated
        bool_t internalBuffersMustBeUpdated = false;
        bool_t const & torqueLimitFromUrdf = boost::get<bool_t>(motorOptions.at("torqueLimitFromUrdf"));
        if (!torqueLimitFromUrdf)
        {
            float64_t const & torqueLimit = boost::get<float64_t>(motorOptions.at("torqueLimit"));
            internalBuffersMustBeUpdated |= std::abs(torqueLimit - baseMotorOptions_->torqueLimit) > EPS;
        }
        internalBuffersMustBeUpdated |= (baseMotorOptions_->torqueLimitFromUrdf != torqueLimitFromUrdf);

        /* Check that the user-defined motor intertia is positive,
           then check if the internal model's buffers to must be updated. */
        bool_t modelBuffersMustBeUpdated = false;
        if (boost::get<bool_t>(motorOptions.at("enableMotorInertia")))
        {
            float64_t const & motorInertia = boost::get<float64_t>(motorOptions.at("motorInertia"));
            if(motorInertia < 0.0)
            {
                std::cout << "Error - Model::setOptions - Every values in 'motorInertia' must be positive." << std::endl;
                return result_t::ERROR_BAD_INPUT;
            }
            modelBuffersMustBeUpdated |= std::abs(motorInertia - baseMotorOptions_->motorInertia) > EPS;
        }

        // Update the motor's options
        motorOptionsHolder_ = motorOptions;
        baseMotorOptions_ = std::make_unique<abstractMotorOptions_t const>(motorOptionsHolder_);

        // Refresh the proxies if the model is initialized
        if (model_->getIsInitialized())
        {
            if (modelBuffersMustBeUpdated)
            {
                model_->refreshMotorProxies();
            }
            if (internalBuffersMustBeUpdated)
            {
                refreshProxies();
            }
        }

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

    std::string const & AbstractMotor::getJointName(void) const
    {
        return jointName_;
    }

    int32_t const & AbstractMotor::getJointVelocityIdx(void) const
    {
        return jointVelocityIdx_;
    }

    float64_t const & AbstractMotor::getTorqueLimit(void) const
    {
        return torqueLimit_;
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
