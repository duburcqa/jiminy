#include "jiminy/core/robot/Robot.h"
#include "jiminy/core/Macros.h"

#include "jiminy/core/robot/AbstractMotor.h"


namespace jiminy
{
    AbstractMotorBase::AbstractMotorBase(std::string const & name) :
    baseMotorOptions_(nullptr),
    motorOptionsHolder_(),
    isInitialized_(false),
    isAttached_(false),
    robot_(),
    name_(name),
    motorIdx_(-1),
    jointName_(),
    jointModelIdx_(-1),
    jointType_(joint_t::NONE),
    jointPositionIdx_(-1),
    jointVelocityIdx_(-1),
    commandLimit_(0.0),
    armature_(0.0),
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

    hresult_t AbstractMotorBase::attach(std::weak_ptr<Robot const> robot,
                                        MotorSharedDataHolder_t * sharedHolder)
    {
        // Make sure the motor is not already attached
        if (isAttached_)
        {
            PRINT_ERROR("Motor already attached to a robot. Please 'detach' method before attaching it.");
            return hresult_t::ERROR_GENERIC;
        }

        // Make sure the robot still exists
        if (robot.expired())
        {
            PRINT_ERROR("Robot pointer expired or unset.");
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
            PRINT_ERROR("Motor not attached to any robot.");
            return hresult_t::ERROR_GENERIC;
        }

        if (auto robot = robot_.lock())
        {
            // Remove associated col in the global data buffer
            if (motorIdx_ < sharedHolder_->num_ - 1)
            {
                int32_t motorShift = sharedHolder_->num_ - motorIdx_ - 1;
                sharedHolder_->data_.segment(motorIdx_, motorShift) =
                    sharedHolder_->data_.segment(motorIdx_ + 1, motorShift).eval();  // eval to avoid aliasing
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
        }

        // Clear the references to the robot and shared data
        robot_.reset();
        sharedHolder_ = nullptr;

        // Unset the Id
        motorIdx_ = -1;

        // Update the flag
        isAttached_ = false;

        return hresult_t::SUCCESS;
    }

    hresult_t AbstractMotorBase::resetAll(void)
    {
        // Make sure the robot still exists
        if (robot_.expired())
        {
            PRINT_ERROR("Robot has been deleted. Impossible to reset the motors.");
            return hresult_t::ERROR_GENERIC;
        }

        // Clear the shared data buffer
        sharedHolder_->data_.setZero();

        // Update motor scope information
        for (AbstractMotorBase * motor : sharedHolder_->motors_)
        {
            // Refresh proxies that are robot-dependent
            motor->refreshProxies();  //
        }

        return hresult_t::SUCCESS;
    }

    hresult_t AbstractMotorBase::setOptions(configHolder_t const & motorOptions)
    {
        // Check if the internal buffers must be updated
        bool_t internalBuffersMustBeUpdated = false;
        if (isInitialized_)
        {
            bool_t const & commandLimitFromUrdf = boost::get<bool_t>(motorOptions.at("commandLimitFromUrdf"));
            if (!commandLimitFromUrdf)
            {
                float64_t const & commandLimit = boost::get<float64_t>(motorOptions.at("commandLimit"));
                internalBuffersMustBeUpdated |= std::abs(commandLimit - baseMotorOptions_->commandLimit) > EPS;
            }
            internalBuffersMustBeUpdated |= (baseMotorOptions_->commandLimitFromUrdf != commandLimitFromUrdf);
        }

        // Update the motor's options
        motorOptionsHolder_ = motorOptions;
        baseMotorOptions_ = std::make_unique<abstractMotorOptions_t const>(motorOptionsHolder_);

        // Refresh the proxies if the robot is initialized if available
        if (auto robot = robot_.lock())
        {
            if (internalBuffersMustBeUpdated && robot->getIsInitialized() && isAttached_)
            {
                refreshProxies();
            }
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

        if (!isAttached_)
        {
            PRINT_ERROR("Motor not attached to any robot. Impossible to refresh proxies.");
            returnCode = hresult_t::ERROR_INIT_FAILED;
        }

        auto robot = robot_.lock();
        if (returnCode == hresult_t::SUCCESS)
        {
            if (!robot)
            {
                PRINT_ERROR("Robot has been deleted. Impossible to refresh proxies.");
                returnCode = hresult_t::ERROR_GENERIC;
            }
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            if (!isInitialized_)
            {
                PRINT_ERROR("Motor not initialized. Impossible to refresh proxies.");
                returnCode = hresult_t::ERROR_INIT_FAILED;
            }
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            if (!robot->getIsInitialized())
            {
                PRINT_ERROR("Robot not initialized. Impossible to refresh proxies.");
                returnCode = hresult_t::ERROR_INIT_FAILED;
            }
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            returnCode = ::jiminy::getJointModelIdx(robot->pncModel_, jointName_, jointModelIdx_);
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            returnCode = getJointTypeFromIdx(robot->pncModel_, jointModelIdx_, jointType_);
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            // Motors are only supported for linear and rotary joints
            if (jointType_ != joint_t::LINEAR && jointType_ != joint_t::ROTARY && jointType_ != joint_t::ROTARY_UNBOUNDED)
            {
                PRINT_ERROR("A motor can only be associated with a 1-dof linear or rotary joint.");
                returnCode = hresult_t::ERROR_BAD_INPUT;
            }
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            ::jiminy::getJointPositionIdx(robot->pncModel_, jointName_, jointPositionIdx_);
            ::jiminy::getJointVelocityIdx(robot->pncModel_, jointName_, jointVelocityIdx_);

            // Get the motor effort limits from the URDF or the user options.
            if (baseMotorOptions_->commandLimitFromUrdf)
            {
                commandLimit_ = robot->pncModel_.effortLimit[jointVelocityIdx_] / baseMotorOptions_->mechanicalReduction;
            }
            else
            {
                commandLimit_ = baseMotorOptions_->commandLimit;
            }

            // Get the rotor inertia
            if (baseMotorOptions_->enableArmature)
            {
                armature_ = baseMotorOptions_->armature;
            }
            else
            {
                armature_ = 0.0;
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

    float64_t const & AbstractMotorBase::getCommandLimit(void) const
    {
        return commandLimit_;
    }

    float64_t const & AbstractMotorBase::getArmature(void) const
    {
        return armature_;
    }

    hresult_t AbstractMotorBase::computeEffortAll(float64_t const & t,
                                                  vectorN_t const & q,
                                                  vectorN_t const & v,
                                                  vectorN_t const & a,
                                                  vectorN_t const & command)
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
                                                  command[motor->getIdx()]);
            }
        }

        return returnCode;
    }
}
