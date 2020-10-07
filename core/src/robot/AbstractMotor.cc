#include "jiminy/core/robot/Robot.h"
#include "jiminy/core/Utilities.h"

#include "jiminy/core/robot/AbstractMotor.h"


namespace jiminy
{
    AbstractMotorBase::AbstractMotorBase(std::string const & name) :
    baseMotorOptions_(nullptr),
    motorOptionsHolder_(),
    isInitialized_(false),
    isAttached_(false),
    robot_(nullptr),
    name_(name),
    motorIdx_(-1),
    jointName_(),
    jointModelIdx_(-1),
    jointType_(joint_t::NONE),
    jointPositionIdx_(-1),
    jointVelocityIdx_(-1),
    effortLimit_(0.0),
    rotorInertia_(0.0),
    sharedHolder_(nullptr)
    {
        // Initialize the options
        setOptions(getDefaultMotorOptions());
    }

    AbstractMotorBase::~AbstractMotorBase(void)
    {
        // Detach the sensor before deleting it if necessary
        if (isAttached_)
        {
            detach();
        }
    }

    hresult_t AbstractMotorBase::attach(Robot const * robot,
                                        MotorSharedDataHolder_t * sharedHolder)
    {
        if (isAttached_)
        {
            std::cout << "Error - AbstractMotorBase::attach - Motor already attached to a robot. Please 'detach' method before attaching it." << std::endl;
            return hresult_t::ERROR_GENERIC;
        }

        robot_ = robot;
        sharedHolder_ = sharedHolder;

        // Get an index
        motorIdx_ = sharedHolder_->num_;

        // Add a value for the motor to the shared data buffer
        sharedHolder_->data_.conservativeResize(sharedHolder_->num_ + 1);
        sharedHolder_->data_.tail<1>().setZero();

        // Add the motor to the shared memory
        sharedHolder_->motors_.push_back(this);
        ++sharedHolder_->num_;

        // Update the flag
        isAttached_ = true;

        return hresult_t::SUCCESS;
    }

    hresult_t AbstractMotorBase::detach(void)
    {
        // Delete the part of the shared memory associated with the motor

        if (!isAttached_)
        {
            std::cout << "Error - AbstractMotorBase::detach - Motor not attached to any robot." << std::endl;
            return hresult_t::ERROR_GENERIC;
        }

        // Remove associated col in the global data buffer
        if (motorIdx_ < sharedHolder_->num_ - 1)
        {
            int32_t motorShift = sharedHolder_->num_ - motorIdx_ - 1;
            sharedHolder_->data_.segment(motorIdx_, motorShift) =
                sharedHolder_->data_.segment(motorIdx_ + 1, motorShift).eval(); // eval to avoid aliasing
        }
        sharedHolder_->data_.conservativeResize(sharedHolder_->num_ - 1);

        // Shift the motor ids
        for (int32_t i = motorIdx_ + 1; i < sharedHolder_->num_; ++i)
        {
            --sharedHolder_->motors_[i]->motorIdx_;
        }

        // Remove the motor to the shared memory
        sharedHolder_->motors_.erase(sharedHolder_->motors_.begin() + motorIdx_);
        --sharedHolder_->num_;

        // Clear the references to the robot and shared data
        robot_ = nullptr;
        sharedHolder_ = nullptr;

        // Update the flag
        isAttached_ = false;

        return hresult_t::SUCCESS;
    }

    void AbstractMotorBase::resetAll(void)
    {
        // Clear the shared data buffer
        sharedHolder_->data_.setZero();

        // Update motor scope information
        for (AbstractMotorBase * motor : sharedHolder_->motors_)
        {
            // Refresh proxies that are robot-dependent
            motor->refreshProxies();
        }
    }

    hresult_t AbstractMotorBase::setOptions(configHolder_t const & motorOptions)
    {
        // Check if the internal buffers must be updated
        bool_t internalBuffersMustBeUpdated = false;
        if (isInitialized_)
        {
            bool_t const & effortLimitFromUrdf = boost::get<bool_t>(motorOptions.at("effortLimitFromUrdf"));
            if (!effortLimitFromUrdf)
            {
                float64_t const & effortLimit = boost::get<float64_t>(motorOptions.at("effortLimit"));
                internalBuffersMustBeUpdated |= std::abs(effortLimit - baseMotorOptions_->effortLimit) > EPS;
            }
            internalBuffersMustBeUpdated |= (baseMotorOptions_->effortLimitFromUrdf != effortLimitFromUrdf);
        }

        // Update the motor's options
        motorOptionsHolder_ = motorOptions;
        baseMotorOptions_ = std::make_unique<abstractMotorOptions_t const>(motorOptionsHolder_);

        // Refresh the proxies if the robot is initialized
        if (internalBuffersMustBeUpdated && robot_->getIsInitialized() && isAttached_)
        {
            refreshProxies();
        }

        return hresult_t::SUCCESS;
    }

    configHolder_t AbstractMotorBase::getOptions(void) const
    {
        return motorOptionsHolder_;
    }

    hresult_t AbstractMotorBase::refreshProxies(void)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (!robot_->getIsInitialized())
        {
            std::cout << "Error - AbstractMotorBase::refreshProxies - Robot not initialized. Impossible to refresh model-dependent proxies." << std::endl;
            returnCode =  hresult_t::ERROR_INIT_FAILED;
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            if (!isInitialized_)
            {
                std::cout << "Error - AbstractMotorBase::refreshProxies - Motor not initialized. Impossible to refresh model-dependent proxies." << std::endl;
                returnCode = hresult_t::ERROR_INIT_FAILED;
            }
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            returnCode = ::jiminy::getJointModelIdx(robot_->pncModel_, jointName_, jointModelIdx_);
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            returnCode = getJointTypeFromIdx(robot_->pncModel_, jointModelIdx_, jointType_);
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            // Motors are only supported for linear and rotary joints
            if (jointType_ != joint_t::LINEAR && jointType_ != joint_t::ROTARY && jointType_ != joint_t::ROTARY_UNBOUNDED)
            {
                std::cout << "Error - AbstractMotorBase::refreshProxies - A motor can only be associated with a 1-dof linear or rotary joint." << std::endl;
                returnCode =  hresult_t::ERROR_BAD_INPUT;
            }
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            ::jiminy::getJointPositionIdx(robot_->pncModel_, jointName_, jointPositionIdx_);
            ::jiminy::getJointVelocityIdx(robot_->pncModel_, jointName_, jointVelocityIdx_);

            // Get the motor effort limits from the URDF or the user options.
            if (baseMotorOptions_->effortLimitFromUrdf)
            {
                effortLimit_ = robot_->pncModel_.effortLimit[jointVelocityIdx_] * baseMotorOptions_->mechanicalReduction;
            }
            else
            {
                effortLimit_ = baseMotorOptions_->effortLimit;
            }

            // Get the rotor inertia
            if (baseMotorOptions_->enableRotorInertia)
            {
                rotorInertia_ = baseMotorOptions_->rotorInertia * baseMotorOptions_->mechanicalReduction;
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
        return sharedHolder_->data_[motorIdx_];
    }

    float64_t const & AbstractMotorBase::get(void) const
    {
        return sharedHolder_->data_[motorIdx_];
    }

    vectorN_t const & AbstractMotorBase::getAll(void) const
    {
        return sharedHolder_->data_;
    }

    hresult_t AbstractMotorBase::setOptionsAll(configHolder_t const & motorOptions)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        for (AbstractMotorBase * motor : sharedHolder_->motors_)
        {
            if (returnCode == hresult_t::SUCCESS)
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
        return motorIdx_;
    }

    std::string const & AbstractMotorBase::getJointName(void) const
    {
        return jointName_;
    }

    int32_t const & AbstractMotorBase::getJointModelIdx(void) const
    {
        return jointModelIdx_;
    }

    joint_t const & AbstractMotorBase::getJointType(void) const
    {
        return jointType_;
    }

    int32_t const & AbstractMotorBase::getJointPositionIdx(void) const
    {
        return jointPositionIdx_;
    }

    int32_t const & AbstractMotorBase::getJointVelocityIdx(void) const
    {
        return jointVelocityIdx_;
    }

    float64_t const & AbstractMotorBase::getEffortLimit(void) const
    {
        return effortLimit_;
    }

    float64_t const & AbstractMotorBase::getRotorInertia(void) const
    {
        return rotorInertia_;
    }

    hresult_t AbstractMotorBase::computeEffortAll(float64_t const & t,
                                                  vectorN_t const & q,
                                                  vectorN_t const & v,
                                                  vectorN_t const & a,
                                                  vectorN_t const & uCommand)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        // Compute the actual effort of every motor
        for (AbstractMotorBase * motor : sharedHolder_->motors_)
        {
            if (returnCode == hresult_t::SUCCESS)
            {
                uint8_t nq_motor;
                if (motor->getJointType() == joint_t::ROTARY_UNBOUNDED)
                {
                    nq_motor = 2;
                }
                else
                {
                    nq_motor = 1;
                }
                returnCode = motor->computeEffort(t,
                                                  q.segment(motor->getJointPositionIdx(), nq_motor),
                                                  v[motor->getJointVelocityIdx()],
                                                  a[motor->getJointVelocityIdx()],
                                                  uCommand[motor->getIdx()]);
            }
        }

        return returnCode;
    }
}
