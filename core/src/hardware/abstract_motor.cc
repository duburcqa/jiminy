#include "jiminy/core/robot/robot.h"
#include "jiminy/core/utilities/pinocchio.h"

#include "jiminy/core/hardware/abstract_motor.h"


namespace jiminy
{
    AbstractMotorBase::AbstractMotorBase(const std::string & name) noexcept :
    name_{name}
    {
        // Initialize options
        motorOptionsGeneric_ = getDefaultMotorOptions();
        setOptions(getOptions());
    }

    AbstractMotorBase::~AbstractMotorBase()
    {
        // Detach the sensor before deleting it if necessary
        if (isAttached_)
        {
            detach();
        }
    }

    void AbstractMotorBase::attach(
        std::weak_ptr<const Robot> robot,
        std::function<void(AbstractMotorBase & /*motor*/, bool /*hasChanged*/)> notifyRobot,
        MotorSharedStorage * sharedStorage)
    {
        // Make sure the motor is not already attached
        if (isAttached_)
        {
            JIMINY_THROW(
                std::logic_error,
                "Motor already attached to a robot. Please 'detach' method before attaching it.");
        }

        // Make sure the robot still exists
        if (robot.expired())
        {
            JIMINY_THROW(std::runtime_error, "Robot pointer expired or unset.");
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
            JIMINY_THROW(bad_control_flow, "Motor not attached to any robot.");
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
            JIMINY_THROW(bad_control_flow, "Motor not attached to any robot.");
        }

        // Make sure all the motors are attached to a robot and initialized
        for (AbstractMotorBase * motor : sharedStorage_->motors_)
        {
            if (!motor->isAttached_)
            {
                JIMINY_THROW(
                    bad_control_flow, "Motor '", motor->name_, "' not attached to any robot.");
            }
            if (!motor->isInitialized_)
            {
                JIMINY_THROW(bad_control_flow, "Motor '", motor->name_, "' not initialized.");
            }
        }

        // Make sure the robot still exists
        if (robot_.expired())
        {
            JIMINY_THROW(std::runtime_error,
                         "Robot has been deleted. Impossible to reset motors.");
        }

        // Make sure that no simulation is already running
        auto robot = robot_.lock();
        if (robot && robot->getIsLocked())
        {
            JIMINY_THROW(bad_control_flow,
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
            JIMINY_THROW(bad_control_flow,
                         "Robot already locked, probably because a simulation is running. "
                         "Please stop it before setting motor options.");
        }

        // Check if the internal buffers must be updated
        if (isInitialized_)
        {
            // Check if armature has changed
            const bool enableArmature = boost::get<bool>(motorOptions.at("enableArmature"));
            mustNotifyRobot_ |= (baseMotorOptions_->enableArmature != enableArmature);
            if (enableArmature)
            {
                const double armature = boost::get<double>(motorOptions.at("armature"));
                mustNotifyRobot_ |=  //
                    std::abs(armature - baseMotorOptions_->armature) > EPS;
            }

            // Check if backlash has changed
            const bool enableBacklash = boost::get<bool>(motorOptions.at("enableBacklash"));
            mustNotifyRobot_ |= (baseMotorOptions_->enableBacklash != enableBacklash);
            if (enableBacklash)
            {
                const double backlash = boost::get<double>(motorOptions.at("backlash"));
                mustNotifyRobot_ |=  //
                    std::abs(backlash - baseMotorOptions_->backlash) > EPS;
            }

            // Check if command limit has changed
            const bool commandLimitFromUrdf =
                boost::get<bool>(motorOptions.at("commandLimitFromUrdf"));
            mustNotifyRobot_ |= (baseMotorOptions_->commandLimitFromUrdf != commandLimitFromUrdf);
            if (!commandLimitFromUrdf)
            {
                const double commandLimit = boost::get<double>(motorOptions.at("commandLimit"));
                mustNotifyRobot_ |= std::abs(commandLimit - baseMotorOptions_->commandLimit) > EPS;
            }
        }

        // Update class-specific "strongly typed" accessor for fast and convenient access
        baseMotorOptions_ = std::make_unique<const AbstractMotorOptions>(motorOptions);

        // Update inherited polymorphic accessor
        deepUpdate(motorOptionsGeneric_, motorOptions);

        // Refresh the proxies if the robot is initialized if available
        if (robot)
        {
            if (mustNotifyRobot_ && robot->getIsInitialized() && isAttached_)
            {
                refreshProxies();
            }
        }
    }

    const GenericConfig & AbstractMotorBase::getOptions() const noexcept
    {
        return motorOptionsGeneric_;
    }

    void AbstractMotorBase::refreshProxies()
    {
        if (!isAttached_)
        {
            JIMINY_THROW(bad_control_flow,
                         "Motor not attached to any robot. Impossible to refresh motor proxies.");
        }

        auto robot = robot_.lock();
        if (!robot)
        {
            JIMINY_THROW(std::runtime_error,
                         "Robot has been deleted. Impossible to refresh motor proxies.");
        }

        if (!isInitialized_)
        {
            JIMINY_THROW(bad_control_flow,
                         "Motor not initialized. Impossible to refresh motor proxies.");
        }

        if (!robot->getIsInitialized())
        {
            JIMINY_THROW(bad_control_flow,
                         "Robot not initialized. Impossible to refresh motor proxies.");
        }

        jointIndex_ = ::jiminy::getJointIndex(robot->pinocchioModel_, jointName_);
        jointType_ = getJointTypeFromIndex(robot->pinocchioModel_, jointIndex_);

        // Motors are only supported for linear and rotary joints
        if (jointType_ != JointModelType::LINEAR && jointType_ != JointModelType::ROTARY &&
            jointType_ != JointModelType::ROTARY_UNBOUNDED)
        {
            JIMINY_THROW(std::logic_error,
                         "A motor can only be associated with a 1-dof linear or rotary joint.");
        }

        jointPositionIndex_ = getJointPositionFirstIndex(robot->pinocchioModel_, jointName_);
        jointVelocityIndex_ = getJointVelocityFirstIndex(robot->pinocchioModel_, jointName_);

        // Get the motor effort limits from the URDF or the user options.
        if (baseMotorOptions_->commandLimitFromUrdf)
        {
            const Eigen::Index mechanicalJointVelocityIndex =
                getJointVelocityFirstIndex(robot->pinocchioModelTh_, jointName_);
            commandLimit_ = robot->pinocchioModelTh_.effortLimit[mechanicalJointVelocityIndex] /
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

        // Get the transmission backlash
        if (baseMotorOptions_->enableBacklash)
        {
            backlash_ = baseMotorOptions_->backlash;
        }
        else
        {
            backlash_ = 0.0;
        }

        // Propagate the user-defined motor inertia at Pinocchio model level
        if (notifyRobot_)
        {
            const bool mustNotifyRobot = mustNotifyRobot_;
            mustNotifyRobot_ = false;
            notifyRobot_(*this, mustNotifyRobot);
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

    const Eigen::VectorXd & AbstractMotorBase::getAll() const
    {
        return sharedStorage_->data_;
    }

    void AbstractMotorBase::setOptionsAll(const GenericConfig & motorOptions)
    {
        // Make sure the motor is attached to a robot
        if (!isAttached_)
        {
            JIMINY_THROW(bad_control_flow, "Motor not attached to any robot.");
        }

        for (AbstractMotorBase * motor : sharedStorage_->motors_)
        {
            motor->setOptions(motorOptions);
        }
    }

    bool AbstractMotorBase::getIsAttached() const
    {
        return isAttached_;
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

    double AbstractMotorBase::getBacklash() const
    {
        return backlash_;
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
            JIMINY_THROW(bad_control_flow, "Motor not attached to any robot.");
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
