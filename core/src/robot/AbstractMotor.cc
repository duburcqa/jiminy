#include "jiminy/core/robot/Robot.h"
#include "jiminy/core/Macros.h"

#include "jiminy/core/utilities/Pinocchio.h"
#include "jiminy/core/robot/AbstractMotor.h"


namespace jiminy
{
    AbstractMotorBase::AbstractMotorBase(std::string const & name) :
    baseMotorOptions_(nullptr),
    motorOptionsHolder_(),
    isInitialized_(false),
    isAttached_(false),
    robot_(),
    notifyRobot_(),
    name_(name),
    motorIdx_(-1),
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
                                        std::function<hresult_t(AbstractMotorBase & /*motor*/)> notifyRobot,
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

        // Copy references to the robot and shared data
        robot_ = robot;
        notifyRobot_ = notifyRobot;
        sharedHolder_ = sharedHolder;

        // Get an index
        motorIdx_ = sharedHolder_->num_;

        // Add values for the motor to the shared data buffer
        for (vectorN_t & data : std::array<vectorN_t &, 4>{{
                sharedHolder_->position_,
                sharedHolder_->velocity_,
                sharedHolder_->acceleration_,
                sharedHolder_->effort_}})
        {
            data.conservativeResize(sharedHolder_->num_ + 1);
            data.tail<1>().setZero();
        }
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

        for (vectorN_t & data : std::array<vectorN_t &, 4>{{
            sharedHolder_->position_,
            sharedHolder_->velocity_,
            sharedHolder_->acceleration_,
            sharedHolder_->effort_}})
        {
            // Remove associated col in the global data buffer
            if (motorIdx_ < sharedHolder_->num_ - 1)
            {
                int32_t motorShift = sharedHolder_->num_ - motorIdx_ - 1;
                data.segment(motorIdx_, motorShift) =
                    data.segment(motorIdx_ + 1, motorShift).eval();  // eval to avoid aliasing
            }
            data.conservativeResize(sharedHolder_->num_ - 1);
        }

        // Shift the motor ids
        for (int32_t i = motorIdx_ + 1; i < sharedHolder_->num_; ++i)
        {
            --sharedHolder_->motors_[i]->motorIdx_;
        }

        // Remove the motor to the shared memory
        sharedHolder_->motors_.erase(sharedHolder_->motors_.begin() + motorIdx_);
        --sharedHolder_->num_;

        // Clear the references to the robot and shared data
        robot_.reset();
        notifyRobot_ = nullptr;
        sharedHolder_ = nullptr;

        // Unset the Id
        motorIdx_ = -1;

        // Update the flag
        isAttached_ = false;

        return hresult_t::SUCCESS;
    }

    hresult_t AbstractMotorBase::resetAll(void)
    {
        // Make sure the motor is attached to a robot
        if (!isAttached_)
        {
            PRINT_ERROR("Motor not attached to any robot.");
            return hresult_t::ERROR_GENERIC;
        }

        // Make sure the robot still exists
        if (robot_.expired())
        {
            PRINT_ERROR("Robot has been deleted. Impossible to reset the motors.");
            return hresult_t::ERROR_GENERIC;
        }

        // Clear the shared data buffer
        for (vectorN_t & data : std::array<vectorN_t &, 4>{{
            sharedHolder_->position_,
            sharedHolder_->velocity_,
            sharedHolder_->acceleration_,
            sharedHolder_->effort_}})
        {
            data.setZero();
        }

        // Update motor scope information
        for (AbstractMotorBase * motor : sharedHolder_->motors_)
        {
            // Refresh proxies that are robot-dependent
            motor->refreshProxies();
        }

        return hresult_t::SUCCESS;
    }

    hresult_t AbstractMotorBase::setOptions(configHolder_t const & motorOptions)
    {
        // Check if the internal buffers must be updated
        bool_t internalBuffersMustBeUpdated = false;
        if (isInitialized_)
        {
            // Check if armature has changed
            bool_t const & enableArmature = boost::get<bool_t>(motorOptions.at("enableArmature"));
            internalBuffersMustBeUpdated |= (baseMotorOptions_->enableArmature != enableArmature);
            if (enableArmature)
            {
                float64_t const & armature = boost::get<float64_t>(motorOptions.at("armature"));
                internalBuffersMustBeUpdated |= std::abs(armature - baseMotorOptions_->armature) > EPS;
            }

            // Check if command limit has changed
            bool_t const & commandLimitFromUrdf = boost::get<bool_t>(motorOptions.at("commandLimitFromUrdf"));
            internalBuffersMustBeUpdated |= (baseMotorOptions_->commandLimitFromUrdf != commandLimitFromUrdf);
            if (!commandLimitFromUrdf)
            {
                float64_t const & commandLimit = boost::get<float64_t>(motorOptions.at("commandLimit"));
                internalBuffersMustBeUpdated |= std::abs(commandLimit - baseMotorOptions_->commandLimit) > EPS;
            }
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
            if (!robot->getIsInitialized())
            {
                PRINT_ERROR("Robot not initialized. Impossible to refresh proxies.");
                returnCode = hresult_t::ERROR_INIT_FAILED;
            }
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            ::jiminy::getJointVelocityIdx(robot->pncModel_, jointName_, jointVelocityIdx_);

            // Get the motor effort limits from the URDF or the user options.
            if (baseMotorOptions_->commandLimitFromUrdf)
            {
                commandLimit_ = robot->pncModel_.effortLimit[jointVelocityIdx_];
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

            // Propagate the user-defined motor inertia at Pinocchio model level
            if (notifyRobot_)
            {
                returnCode = notifyRobot_(*this);
            }
        }

        return returnCode;
    }

    float64_t & AbstractMotorBase::q(void)
    {
        return sharedHolder_->position_[motorIdx_];
    }

    float64_t & AbstractMotorBase::v(void)
    {
        return sharedHolder_->velocity_[motorIdx_];
    }
    float64_t & AbstractMotorBase::a(void)
    {
        return sharedHolder_->acceleration_[motorIdx_];
    }
    float64_t & AbstractMotorBase::u(void)
    {
        return sharedHolder_->effort_[motorIdx_];
    }

    float64_t const & AbstractMotorBase::getPosition(void)
    {
        return sharedHolder_->position_[motorIdx_];
    }

    float64_t const & AbstractMotorBase::getVelocity(void);
    {
        return sharedHolder_->velocity_[motorIdx_];
    }

    float64_t const & AbstractMotorBase::getAcceleration(void);
    {
        return sharedHolder_->acceleration_[motorIdx_];
    }

    float64_t const & AbstractMotorBase::getEffort(void);
    {
        return sharedHolder_->effort_[motorIdx_];
    }

    hresult_t AbstractMotorBase::setOptionsAll(configHolder_t const & motorOptions)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        // Make sure the motor is attached to a robot
        if (!isAttached_)
        {
            PRINT_ERROR("Motor not attached to any robot.");
            returnCode = hresult_t::ERROR_GENERIC;
        }

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

    float64_t const & AbstractMotorBase::getCommandLimit(void) const
    {
        return commandLimit_;
    }

    float64_t const & AbstractMotorBase::getArmature(void) const
    {
        return armature_;
    }

    hresult_t AbstractMotorBase::computeEffortAll(vectorN_t const & command)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        // Make sure the motor is attached to a robot
        if (!isAttached_)
        {
            PRINT_ERROR("Motor not attached to any robot.");
            returnCode = hresult_t::ERROR_GENERIC;
        }

        // Compute the actual effort of every motor
        for (AbstractMotorBase * motor : sharedHolder_->motors_)
        {
            if (returnCode == hresult_t::SUCCESS)
            {
                returnCode = motor->computeEffort(command[motor->getIdx()]);
            }
        }

        return returnCode;
    }
}
