#include "jiminy/core/AbstractMotor.h"
#include "jiminy/core/Model.h"
#include "jiminy/core/Utilities.h"


namespace jiminy
{
    AbstractMotorBase::AbstractMotorBase(std::string const & name) :
    baseMotorOptions_(nullptr),
    motorOptionsHolder_(),
    isInitialized_(false),
    isAttached_(false),
    model_(nullptr),
    name_(name),
    motorId_(-1),
    jointName_(),
    jointModelIdx_(-1),
    jointPositionIdx_(-1),
    jointVelocityIdx_(-1),
    torqueLimit_(0.0),
    rotorInertia_(0.0),
    sharedHolder_(nullptr)
    {
        // Initialize the options
        setOptions(getDefaultOptions());
    }

    AbstractMotorBase::~AbstractMotorBase(void)
    {
        // Detach the sensor before deleting it if necessary
        if (isAttached_)
        {
            detach();
        }
    }

    result_t AbstractMotorBase::attach(Model const * model,
                                       std::shared_ptr<MotorSharedDataHolder_t> & sharedHolder)
    {
        if (isAttached_)
        {
            std::cout << "Error - AbstractMotorBase::attach - Motor already attached to a model. Please 'detach' method before attaching it." << std::endl;
            return result_t::ERROR_GENERIC;
        }

        // Copy references to the model and shared data
        model_ = model;
        sharedHolder_ = sharedHolder.get();

        // Get an Id
        motorId_ = sharedHolder_->num_;

        // Add the motor to the shared data
        sharedHolder_->data_.conservativeResize(Eigen::NoChange, sharedHolder_->num_ + 1);
        sharedHolder_->motors_.push_back(this);
        ++sharedHolder_->num_;

        // Update the flag
        isAttached_ = true;

        return result_t::SUCCESS;
    }

    result_t AbstractMotorBase::detach(void)
    {
        // Delete the part of the shared memory associated with the motor

        if (!isAttached_)
        {
            std::cout << "Error - AbstractMotorBase::detach - Motor not attached to any model." << std::endl;
            return result_t::ERROR_GENERIC;
        }

        // Remove associated col in the global data buffer
        if(motorId_ < sharedHolder_->num_ - 1)
        {
            int32_t motorShift = sharedHolder_->num_ - motorId_ - 1;
            sharedHolder_->data_.segment(motorId_, motorShift) =
                sharedHolder_->data_.segment(motorId_ + 1, motorShift).eval(); // eval to avoid aliasing
        }
        sharedHolder_->data_.conservativeResize(sharedHolder_->num_ - 1);

        // Shift the motor ids
        for (int32_t i = motorId_ + 1; i < sharedHolder_->num_; i++)
        {
            --sharedHolder_->motors_[i]->motorId_;
        }

        // Remove the deprecated elements of the global containers
        sharedHolder_->motors_.erase(sharedHolder_->motors_.begin() + motorId_);

        // Update the total number of motors left
        --sharedHolder_->num_;

        // Clear the references to the model and shared data
        model_ = nullptr;
        sharedHolder_ = nullptr;

        // Update the flag
        isAttached_ = false;

        return result_t::SUCCESS;
    }

    void AbstractMotorBase::reset(void)
    {
        // Clear the data buffer
        clearDataBuffer();

        // Refresh proxies that are model-dependent
        refreshProxies();
    }

    result_t AbstractMotorBase::setOptions(configHolder_t const & motorOptions)
    {
        // Check if the internal buffers must be updated
        bool_t internalBuffersMustBeUpdated = false;
        if (isInitialized_)
        {
            bool_t const & torqueLimitFromUrdf = boost::get<bool_t>(motorOptions.at("torqueLimitFromUrdf"));
            if (!torqueLimitFromUrdf)
            {
                float64_t const & torqueLimit = boost::get<float64_t>(motorOptions.at("torqueLimit"));
                internalBuffersMustBeUpdated |= std::abs(torqueLimit - baseMotorOptions_->torqueLimit) > EPS;
            }
            internalBuffersMustBeUpdated |= (baseMotorOptions_->torqueLimitFromUrdf != torqueLimitFromUrdf);
        }

        // Update the motor's options
        motorOptionsHolder_ = motorOptions;
        baseMotorOptions_ = std::make_unique<abstractMotorOptions_t const>(motorOptionsHolder_);

        // Refresh the proxies if the model is initialized
        if (isAttached_ && internalBuffersMustBeUpdated && model_->getIsInitialized())
        {
            refreshProxies();
        }

        return result_t::SUCCESS;
    }

    configHolder_t AbstractMotorBase::getOptions(void) const
    {
        return motorOptionsHolder_;
    }

    result_t AbstractMotorBase::refreshProxies(void)
    {
        result_t returnCode = result_t::SUCCESS;

        if (!model_->getIsInitialized())
        {
            std::cout << "Error - AbstractMotorBase::refreshProxies - Model not initialized. Impossible to refresh model-dependent proxies." << std::endl;
            returnCode =  result_t::ERROR_INIT_FAILED;
        }

        if (returnCode == result_t::SUCCESS)
        {
            if (!isInitialized_)
            {
                std::cout << "Error - AbstractMotorBase::refreshProxies - Motor not initialized. Impossible to refresh model-dependent proxies." << std::endl;
                returnCode = result_t::ERROR_INIT_FAILED;
            }
        }

        if (returnCode == result_t::SUCCESS)
        {
            returnCode = ::jiminy::getJointModelIdx(model_->pncModel_, jointName_, jointModelIdx_);
        }

        if (returnCode == result_t::SUCCESS)
        {
            ::jiminy::getJointPositionIdx(model_->pncModel_, jointName_, jointPositionIdx_);
            ::jiminy::getJointVelocityIdx(model_->pncModel_, jointName_, jointVelocityIdx_);

            // Get the motor torque limits from the URDF or the user options.
            if (baseMotorOptions_->torqueLimitFromUrdf)
            {
                torqueLimit_ = model_->pncModel_.effortLimit[jointVelocityIdx_];
            }
            else
            {
                torqueLimit_ = baseMotorOptions_->torqueLimit;
            }

            // Get the rotor inertia
            if (baseMotorOptions_->enableRotorInertia)
            {
                rotorInertia_ = baseMotorOptions_->rotorInertia;
            }
            else
            {
                rotorInertia_ = 0.0;
            }
        }

        return returnCode;
    }

    float64_t & AbstractMotorBase::data(void)
    {
        return sharedHolder_->data_[motorId_];
    }

    float64_t const & AbstractMotorBase::get(void) const
    {
        return sharedHolder_->data_[motorId_];
    }

    vectorN_t const & AbstractMotorBase::getAll(void) const
    {
        return sharedHolder_->data_;
    }

    result_t AbstractMotorBase::setOptionsAll(configHolder_t const & motorOptions)
    {
        result_t returnCode = result_t::SUCCESS;

        for (AbstractMotorBase * motor : sharedHolder_->motors_)
        {
            if (returnCode == result_t::SUCCESS)
            {
                returnCode = motor->setOptions(motorOptions);
            }
        }

        return returnCode;
    }

    bool_t const & AbstractMotorBase::getIsInitialized(void) const
    {
        return isInitialized_;
    }

    std::string const & AbstractMotorBase::getName(void) const
    {
        return name_;
    }

    int32_t const & AbstractMotorBase::getIdx(void) const
    {
        return motorId_;
    }

    std::string const & AbstractMotorBase::getJointName(void) const
    {
        return jointName_;
    }

    int32_t const & AbstractMotorBase::getJointModelIdx(void) const
    {
        return jointModelIdx_;
    }

    int32_t const & AbstractMotorBase::getJointPositionIdx(void) const
    {
        return jointPositionIdx_;
    }

    int32_t const & AbstractMotorBase::getJointVelocityIdx(void) const
    {
        return jointVelocityIdx_;
    }

    float64_t const & AbstractMotorBase::getTorqueLimit(void) const
    {
        return torqueLimit_;
    }

    float64_t const & AbstractMotorBase::getRotorInertia(void) const
    {
        return rotorInertia_;
    }

    void AbstractMotorBase::clearDataBuffer(void)
    {
        sharedHolder_->data_ = vectorN_t::Zero(sharedHolder_->num_);
    }

    result_t AbstractMotorBase::computeAllEffort(float64_t const & t,
                                                 vectorN_t const & q,
                                                 vectorN_t const & v,
                                                 vectorN_t const & a,
                                                 vectorN_t const & uCommand)
    {
        result_t returnCode = result_t::SUCCESS;

        // Compute the motors' output
        for (AbstractMotorBase * motor : sharedHolder_->motors_)
        {
            if (returnCode == result_t::SUCCESS)
            {
                // Compute the actual torque
                returnCode = motor->computeEffort(t,
                                                  q[motor->getJointPositionIdx()],
                                                  v[motor->getJointVelocityIdx()],
                                                  a[motor->getJointVelocityIdx()],
                                                  uCommand[motor->getIdx()]);
            }
        }

        return returnCode;
    }
}
