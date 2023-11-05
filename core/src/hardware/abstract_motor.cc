#include "jiminy/core/robot/robot.h"
#include "jiminy/core/utilities/pinocchio.h"

#include "jiminy/core/hardware/abstract_motor.h"


namespace jiminy
{
    AbstractMotorBase::AbstractMotorBase(const std::string & name) :
    baseMotorOptions_(nullptr),
    motorOptionsHolder_(),
    isInitialized_(false),
    isAttached_(false),
    robot_(),
    notifyRobot_(),
    name_(name),
    motorIdx_(0),
    jointName_(),
    jointModelIdx_(0),
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

    AbstractMotorBase::~AbstractMotorBase()
    {
        // Detach the sensor before deleting it if necessary
        if (isAttached_)
        {
            detach();
        }
    }

    hresult_t AbstractMotorBase::attach(
        std::weak_ptr<const Robot> robot,
        std::function<hresult_t(AbstractMotorBase & /*motor*/)> notifyRobot,
        MotorSharedDataHolder_t * sharedHolder)
    {
        // Make sure the motor is not already attached
        if (isAttached_)
        {
            PRINT_ERROR(
                "Motor already attached to a robot. Please 'detach' method before attaching it.");
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

    hresult_t AbstractMotorBase::detach()
    {
        // Delete the part of the shared memory associated with the motor

        if (!isAttached_)
        {
            PRINT_ERROR("Motor not attached to any robot.");
            return hresult_t::ERROR_GENERIC;
        }

        // Remove associated col in the global data buffer
        if (motorIdx_ < sharedHolder_->num_ - 1)
        {
            const Eigen::Index motorShift =
                static_cast<Eigen::Index>(sharedHolder_->num_ - motorIdx_ - 1);
            sharedHolder_->data_.segment(motorIdx_, motorShift) =
                sharedHolder_->data_.tail(motorShift);
        }
        sharedHolder_->data_.conservativeResize(sharedHolder_->num_ - 1);

        // Shift the motor ids
        for (std::size_t i = motorIdx_ + 1; i < sharedHolder_->num_; ++i)
        {
            --sharedHolder_->motors_[i]->motorIdx_;
        }

        // Remove the motor to the shared memory
        sharedHolder_->motors_.erase(std::next(sharedHolder_->motors_.begin(), motorIdx_));
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

    hresult_t AbstractMotorBase::resetAll()
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
        sharedHolder_->data_.setZero();

        // Update motor scope information
        for (AbstractMotorBase * motor : sharedHolder_->motors_)
        {
            // Refresh proxies that are robot-dependent
            motor->refreshProxies();
        }

        return hresult_t::SUCCESS;
    }

    hresult_t AbstractMotorBase::setOptions(const configHolder_t & motorOptions)
    {
        // Check if the internal buffers must be updated
        bool_t internalBuffersMustBeUpdated = false;
        if (isInitialized_)
        {
            // Check if armature has changed
            const bool_t & enableArmature = boost::get<bool_t>(motorOptions.at("enableArmature"));
            internalBuffersMustBeUpdated |= (baseMotorOptions_->enableArmature != enableArmature);
            if (enableArmature)
            {
                const float64_t & armature = boost::get<float64_t>(motorOptions.at("armature"));
                internalBuffersMustBeUpdated |=  //
                    std::abs(armature - baseMotorOptions_->armature) > EPS;
            }

            // Check if command limit has changed
            const bool_t & commandLimitFromUrdf =
                boost::get<bool_t>(motorOptions.at("commandLimitFromUrdf"));
            internalBuffersMustBeUpdated |=
                (baseMotorOptions_->commandLimitFromUrdf != commandLimitFromUrdf);
            if (!commandLimitFromUrdf)
            {
                const float64_t & commandLimit =
                    boost::get<float64_t>(motorOptions.at("commandLimit"));
                internalBuffersMustBeUpdated |=
                    std::abs(commandLimit - baseMotorOptions_->commandLimit) > EPS;
            }
        }

        // Update the motor's options
        motorOptionsHolder_ = motorOptions;
        baseMotorOptions_ = std::make_unique<const abstractMotorOptions_t>(motorOptionsHolder_);

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

    configHolder_t AbstractMotorBase::getOptions() const
    {
        return motorOptionsHolder_;
    }

    hresult_t AbstractMotorBase::refreshProxies()
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
            if (jointType_ != joint_t::LINEAR && jointType_ != joint_t::ROTARY &&
                jointType_ != joint_t::ROTARY_UNBOUNDED)
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
                int32_t jointVelocityOrigIdx;
                ::jiminy::getJointVelocityIdx(
                    robot->pncModelOrig_, jointName_, jointVelocityOrigIdx);
                commandLimit_ = robot->pncModelOrig_.effortLimit[jointVelocityOrigIdx] /
                                baseMotorOptions_->mechanicalReduction;
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

    const float64_t & AbstractMotorBase::get() const
    {
        static float64_t dataEmpty;
        if (isAttached_)
        {
            return sharedHolder_->data_[motorIdx_];
        }
        return dataEmpty;
    }

    float64_t & AbstractMotorBase::data()
    {
        return sharedHolder_->data_[motorIdx_];
    }

    const Eigen::VectorXd & AbstractMotorBase::getAll() const
    {
        return sharedHolder_->data_;
    }

    hresult_t AbstractMotorBase::setOptionsAll(const configHolder_t & motorOptions)
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

    const bool_t & AbstractMotorBase::getIsInitialized() const
    {
        return isInitialized_;
    }

    const std::string & AbstractMotorBase::getName() const
    {
        return name_;
    }

    const std::size_t & AbstractMotorBase::getIdx() const
    {
        return motorIdx_;
    }

    const std::string & AbstractMotorBase::getJointName() const
    {
        return jointName_;
    }

    const jointIndex_t & AbstractMotorBase::getJointModelIdx() const
    {
        return jointModelIdx_;
    }

    const joint_t & AbstractMotorBase::getJointType() const
    {
        return jointType_;
    }

    const int32_t & AbstractMotorBase::getJointPositionIdx() const
    {
        return jointPositionIdx_;
    }

    const int32_t & AbstractMotorBase::getJointVelocityIdx() const
    {
        return jointVelocityIdx_;
    }

    const float64_t & AbstractMotorBase::getCommandLimit() const
    {
        return commandLimit_;
    }

    const float64_t & AbstractMotorBase::getArmature() const
    {
        return armature_;
    }

    hresult_t AbstractMotorBase::computeEffortAll(const float64_t & t,
                                                  const Eigen::VectorXd & q,
                                                  const Eigen::VectorXd & v,
                                                  const Eigen::VectorXd & a,
                                                  const Eigen::VectorXd & command)
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
                uint8_t nq_motor;
                if (motor->getJointType() == joint_t::ROTARY_UNBOUNDED)
                {
                    nq_motor = 2;
                }
                else
                {
                    nq_motor = 1;
                }
                returnCode =
                    motor->computeEffort(t,
                                         q.segment(motor->getJointPositionIdx(), nq_motor),
                                         v[motor->getJointVelocityIdx()],
                                         a[motor->getJointVelocityIdx()],
                                         command[motor->getIdx()]);
            }
        }

        return returnCode;
    }
}
