#include "jiminy/core/robot/robot.h"
#include "jiminy/core/utilities/pinocchio.h"

#include "jiminy/core/hardware/abstract_motor.h"


namespace jiminy
{
    AbstractMotorBase::AbstractMotorBase(const std::string & name) noexcept :
    name_{name}
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

    void AbstractMotorBase::attach(std::weak_ptr<const Robot> robot,
                                   std::function<void(AbstractMotorBase & /*motor*/)> notifyRobot,
                                   MotorSharedStorage * sharedStorage)
    {
        // Make sure the motor is not already attached
        if (isAttached_)
        {
            THROW_ERROR(
                std::logic_error,
                "Motor already attached to a robot. Please 'detach' method before attaching it.");
        }

        // Make sure the robot still exists
        if (robot.expired())
        {
            THROW_ERROR(std::runtime_error, "Robot pointer expired or unset.");
        }

        // Copy references to the robot and shared data
        robot_ = robot;
        notifyRobot_ = notifyRobot;
        sharedStorage_ = sharedStorage;

        // Get an index
        motorIndex_ = sharedStorage_->num_;

        // Add a value for the motor to the shared data buffer
        sharedStorage_->data_.conservativeResize(sharedStorage_->num_ + 1);
        sharedStorage_->data_.tail<1>().setZero();

        // Add the motor to the shared memory
        sharedStorage_->motors_.push_back(this);
        ++sharedStorage_->num_;

        // Update the flag
        isAttached_ = true;
    }

    void AbstractMotorBase::detach()
    {
        // Delete the part of the shared memory associated with the motor

        if (!isAttached_)
        {
            THROW_ERROR(bad_control_flow, "Motor not attached to any robot.");
        }

        // Remove associated col in the global data buffer
        if (motorIndex_ < sharedStorage_->num_ - 1)
        {
            const Eigen::Index motorShift =
                static_cast<Eigen::Index>(sharedStorage_->num_ - motorIndex_ - 1);
            sharedStorage_->data_.segment(motorIndex_, motorShift) =
                sharedStorage_->data_.tail(motorShift);
        }
        sharedStorage_->data_.conservativeResize(sharedStorage_->num_ - 1);

        // Shift the motor ids
        for (std::size_t i = motorIndex_ + 1; i < sharedStorage_->num_; ++i)
        {
            --sharedStorage_->motors_[i]->motorIndex_;
        }

        // Remove the motor to the shared memory
        sharedStorage_->motors_.erase(std::next(sharedStorage_->motors_.begin(), motorIndex_));
        --sharedStorage_->num_;

        // Clear the references to the robot and shared data
        robot_.reset();
        notifyRobot_ = nullptr;
        sharedStorage_ = nullptr;

        // Unset the motor index
        motorIndex_ = -1;

        // Update the flag
        isAttached_ = false;
    }

    void AbstractMotorBase::resetAll()
    {
        // Make sure the motor is attached to a robot
        if (!isAttached_)
        {
            THROW_ERROR(bad_control_flow, "Motor not attached to any robot.");
        }

        // Make sure the robot still exists
        if (robot_.expired())
        {
            THROW_ERROR(std::runtime_error, "Robot has been deleted. Impossible to reset motors.");
        }

        // Make sure that no simulation is already running
        auto robot = robot_.lock();
        if (robot && robot->getIsLocked())
        {
            THROW_ERROR(bad_control_flow,
                        "Robot already locked, probably because a simulation is running. "
                        "Please stop it before resetting motors.");
        }

        // Clear the shared data buffer
        sharedStorage_->data_.setZero();

        // Update motor scope information
        for (AbstractMotorBase * motor : sharedStorage_->motors_)
        {
            // Refresh proxies that are robot-dependent
            motor->refreshProxies();
        }
    }

    void AbstractMotorBase::setOptions(const GenericConfig & motorOptions)
    {
        // Make sure that no simulation is already running
        auto robot = robot_.lock();
        if (robot && robot->getIsLocked())
        {
            THROW_ERROR(bad_control_flow,
                        "Robot already locked, probably because a simulation is running. "
                        "Please stop it before setting motor options.");
        }

        // Check if the internal buffers must be updated
        bool internalBuffersMustBeUpdated = false;
        if (isInitialized_)
        {
            // Check if armature has changed
            const bool enableArmature = boost::get<bool>(motorOptions.at("enableArmature"));
            internalBuffersMustBeUpdated |= (baseMotorOptions_->enableArmature != enableArmature);
            if (enableArmature)
            {
                const double armature = boost::get<double>(motorOptions.at("armature"));
                internalBuffersMustBeUpdated |=  //
                    std::abs(armature - baseMotorOptions_->armature) > EPS;
            }

            // Check if command limit has changed
            const bool commandLimitFromUrdf =
                boost::get<bool>(motorOptions.at("commandLimitFromUrdf"));
            internalBuffersMustBeUpdated |=
                (baseMotorOptions_->commandLimitFromUrdf != commandLimitFromUrdf);
            if (!commandLimitFromUrdf)
            {
                const double commandLimit = boost::get<double>(motorOptions.at("commandLimit"));
                internalBuffersMustBeUpdated |=
                    std::abs(commandLimit - baseMotorOptions_->commandLimit) > EPS;
            }
        }

        // Update the motor's options
        motorOptionsGeneric_ = motorOptions;
        baseMotorOptions_ = std::make_unique<const AbstractMotorOptions>(motorOptionsGeneric_);

        // Refresh the proxies if the robot is initialized if available
        if (robot)
        {
            if (internalBuffersMustBeUpdated && robot->getIsInitialized() && isAttached_)
            {
                refreshProxies();
            }
        }
    }

    GenericConfig AbstractMotorBase::getOptions() const noexcept
    {
        return motorOptionsGeneric_;
    }

    void AbstractMotorBase::refreshProxies()
    {
        if (!isAttached_)
        {
            THROW_ERROR(bad_control_flow,
                        "Motor not attached to any robot. Impossible to refresh motor proxies.");
        }

        auto robot = robot_.lock();
        if (!robot)
        {
            THROW_ERROR(std::runtime_error,
                        "Robot has been deleted. Impossible to refresh motor proxies.");
        }

        if (!isInitialized_)
        {
            THROW_ERROR(bad_control_flow,
                        "Motor not initialized. Impossible to refresh motor proxies.");
        }

        if (!robot->getIsInitialized())
        {
            THROW_ERROR(bad_control_flow,
                        "Robot not initialized. Impossible to refresh motor proxies.");
        }

        jointIndex_ = ::jiminy::getJointIndex(robot->pinocchioModel_, jointName_);
        jointType_ = getJointTypeFromIndex(robot->pinocchioModel_, jointIndex_);

        // Motors are only supported for linear and rotary joints
        if (jointType_ != JointModelType::LINEAR && jointType_ != JointModelType::ROTARY &&
            jointType_ != JointModelType::ROTARY_UNBOUNDED)
        {
            THROW_ERROR(std::logic_error,
                        "A motor can only be associated with a 1-dof linear or rotary joint.");
        }

        jointPositionIndex_ = getJointPositionFirstIndex(robot->pinocchioModel_, jointName_);
        jointVelocityIndex_ = getJointVelocityFirstIndex(robot->pinocchioModel_, jointName_);

        // Get the motor effort limits from the URDF or the user options.
        if (baseMotorOptions_->commandLimitFromUrdf)
        {
            const Eigen::Index jointVelocityOrigIndex =
                getJointVelocityFirstIndex(robot->pinocchioModelOrig_, jointName_);
            commandLimit_ = robot->pinocchioModelOrig_.effortLimit[jointVelocityOrigIndex] /
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
            notifyRobot_(*this);
        }
    }

    double AbstractMotorBase::get() const
    {
        static double dataEmpty;
        if (isAttached_)
        {
            return sharedStorage_->data_[motorIndex_];
        }
        return dataEmpty;
    }

    double & AbstractMotorBase::data()
    {
        return sharedStorage_->data_[motorIndex_];
    }

    bool AbstractMotorBase::isAttached() const
    {
        return isAttached_;
    }

    const Eigen::VectorXd & AbstractMotorBase::getAll() const
    {
        return sharedStorage_->data_;
    }

    void AbstractMotorBase::setOptionsAll(const GenericConfig & motorOptions)
    {
        // Make sure the motor is attached to a robot
        if (!isAttached_)
        {
            THROW_ERROR(bad_control_flow, "Motor not attached to any robot.");
        }

        for (AbstractMotorBase * motor : sharedStorage_->motors_)
        {
            motor->setOptions(motorOptions);
        }
    }

    bool AbstractMotorBase::getIsInitialized() const
    {
        return isInitialized_;
    }

    const std::string & AbstractMotorBase::getName() const
    {
        return name_;
    }

    std::size_t AbstractMotorBase::getIndex() const
    {
        return motorIndex_;
    }

    const std::string & AbstractMotorBase::getJointName() const
    {
        return jointName_;
    }

    pinocchio::JointIndex AbstractMotorBase::getJointIndex() const
    {
        return jointIndex_;
    }

    JointModelType AbstractMotorBase::getJointType() const
    {
        return jointType_;
    }

    Eigen::Index AbstractMotorBase::getJointPositionIndex() const
    {
        return jointPositionIndex_;
    }

    Eigen::Index AbstractMotorBase::getJointVelocityIndex() const
    {
        return jointVelocityIndex_;
    }

    double AbstractMotorBase::getCommandLimit() const
    {
        return commandLimit_;
    }

    double AbstractMotorBase::getArmature() const
    {
        return armature_;
    }

    void AbstractMotorBase::computeEffortAll(double t,
                                             const Eigen::VectorXd & q,
                                             const Eigen::VectorXd & v,
                                             const Eigen::VectorXd & a,
                                             const Eigen::VectorXd & command)
    {
        // Make sure the motor is attached to a robot
        if (!isAttached_)
        {
            THROW_ERROR(bad_control_flow, "Motor not attached to any robot.");
        }

        // Compute the actual effort of every motor
        for (AbstractMotorBase * motor : sharedStorage_->motors_)
        {
            uint8_t nq_motor;
            if (motor->getJointType() == JointModelType::ROTARY_UNBOUNDED)
            {
                nq_motor = 2;
            }
            else
            {
                nq_motor = 1;
            }
            motor->computeEffort(t,
                                 q.segment(motor->getJointPositionIndex(), nq_motor),
                                 v[motor->getJointVelocityIndex()],
                                 a[motor->getJointVelocityIndex()],
                                 command[motor->getIndex()]);
        }
    }
}
