#include "jiminy/core/robot/robot.h"
#include "jiminy/core/utilities/pinocchio.h"

#include "jiminy/core/hardware/abstract_motor.h"


namespace jiminy
{
    AbstractMotorBase::AbstractMotorBase(const std::string & name) :
    name_{name}
    {
        // Make sure that the motor name is not empty
        if (name_.empty())
        {
            JIMINY_THROW(std::logic_error, "Motor name must not be empty.");
        }

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

        /* Make sure that the mechanical reduction ratio is larger than 1.0.
           While technically it is possible to have a reduction ratio smaller than 1.0, it is
           extremely unlikely in practice as motors are supposed to spin fast to operate maximum
           efficiency. It is explicitly forbidden here because it is error prone as the user may
           assume the ratio to be inverted. */
        const double mechanicalReduction =
            boost::get<double>(motorOptions.at("mechanicalReduction"));
        if (mechanicalReduction < 1.0 - EPS)
        {
            JIMINY_THROW(std::invalid_argument,
                         "The mechanical reduction must be larger than 1.0.");
        }

        /* Physically, it must be possible to deduce the joint position from the motor position.
           For this condition to be satisfied, the joint position after doing exactly one turn on
           motor side must remain the same (ignoring turns if any). This implies that the inverted
           reduction ratio must be an integer. */
        if (jointType_ == JointModelType::ROTARY_UNBOUNDED)
        {
            if (abs(mechanicalReduction - 1.0) < EPS)
            {
                JIMINY_THROW(std::runtime_error,
                             "The mechanical reduction must be equal to 1.0 for motors actuating "
                             "unbounded revolute joints.");
            }
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

            // Check if mechanical reduction ratio has changed
            mustNotifyRobot_ |= abs(baseMotorOptions_->mechanicalReduction - mechanicalReduction) >
                                EPS;

            // Check if velocity limit has changed
            const bool velocityLimitFromUrdf =
                boost::get<bool>(motorOptions.at("velocityLimitFromUrdf"));
            mustNotifyRobot_ |=
                (baseMotorOptions_->velocityLimitFromUrdf != velocityLimitFromUrdf);
            if (!velocityLimitFromUrdf)
            {
                const double velocityLimit = boost::get<double>(motorOptions.at("velocityLimit"));
                mustNotifyRobot_ |= std::abs(velocityLimit - baseMotorOptions_->velocityLimit) >
                                    EPS;
            }

            // Check if effort limit has changed
            const bool effortLimitFromUrdf =
                boost::get<bool>(motorOptions.at("effortLimitFromUrdf"));
            mustNotifyRobot_ |= (baseMotorOptions_->effortLimitFromUrdf != effortLimitFromUrdf);
            if (!effortLimitFromUrdf)
            {
                const double effortLimit = boost::get<double>(motorOptions.at("effortLimit"));
                mustNotifyRobot_ |= std::abs(effortLimit - baseMotorOptions_->effortLimit) > EPS;
            }
        }

        // Update class-specific "strongly typed" accessor for fast and convenient access
        baseMotorOptions_ = std::make_unique<const AbstractMotorOptions>(motorOptions);

        // Update inherited polymorphic accessor
        deepUpdate(motorOptionsGeneric_, motorOptions);

        // Refresh the proxies if the attached robot must be notified if any
        if (robot && isAttached_ && mustNotifyRobot_)
        {
            refreshProxies();
        }
    }

    const GenericConfig & AbstractMotorBase::getOptions() const noexcept
    {
        return motorOptionsGeneric_;
    }

    void AbstractMotorBase::refreshProxies()
    {
        auto robot = robot_.lock();
        if (!robot)
        {
            JIMINY_THROW(std::runtime_error,
                         "Robot has been deleted. Impossible to refresh motor proxies.");
        }

        if (!isAttached_)
        {
            JIMINY_THROW(bad_control_flow,
                         "Motor not attached to any robot. Impossible to refresh motor proxies.");
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

        if (robot->getIsLocked())
        {
            JIMINY_THROW(bad_control_flow,
                         "Robot already locked, probably because a simulation is running. "
                         "Please stop it before refreshing motor options.");
        }

        // Define proxy for convenience
        const double mechanicalReduction = baseMotorOptions_->mechanicalReduction;
        const pinocchio::Model & model = robot->pinocchioModel_;

        // Get joint index and type
        jointIndex_ = ::jiminy::getJointIndex(model, jointName_);
        jointType_ = getJointTypeFromIndex(model, jointIndex_);

        // Motors are only supported for linear and rotary joints
        if (jointType_ != JointModelType::LINEAR && jointType_ != JointModelType::ROTARY &&
            jointType_ != JointModelType::ROTARY_UNBOUNDED)
        {
            JIMINY_THROW(std::logic_error,
                         "A motor can only be associated with a 1-dof linear or rotary joint.");
        }

        // Deduce the motor position limits for the joint position limits
        if (jointType_ == JointModelType::ROTARY_UNBOUNDED)
        {
            positionLimitLower_ = -INF;
            positionLimitUpper_ = +INF;
        }
        else
        {
            const Eigen::Index jointPositionIndex = model.joints[jointIndex_].idx_q();
            positionLimitLower_ =
                model.lowerPositionLimit[jointPositionIndex] * mechanicalReduction;
            positionLimitUpper_ =
                model.upperPositionLimit[jointPositionIndex] * mechanicalReduction;
        }

        // Get the motor effort limits on motor side from the URDF or the user options
        if (baseMotorOptions_->enableEffortLimit)
        {
            if (baseMotorOptions_->effortLimitFromUrdf)
            {
                const Eigen::Index mechanicalJointVelocityIndex =
                    getJointVelocityFirstIndex(robot->pinocchioModelTh_, jointName_);
                effortLimit_ = robot->pinocchioModelTh_.effortLimit[mechanicalJointVelocityIndex] /
                               mechanicalReduction;
            }
            else
            {
                effortLimit_ = baseMotorOptions_->effortLimit;
            }
        }
        else
        {
            effortLimit_ = INF;
        }

        // Get the motor velocity limits on motor side from the URDF or the user options
        if (baseMotorOptions_->velocityLimitFromUrdf)
        {
            const Eigen::Index mechanicalJointVelocityIndex =
                getJointVelocityFirstIndex(robot->pinocchioModelTh_, jointName_);
            velocityLimit_ = robot->pinocchioModelTh_.velocityLimit[mechanicalJointVelocityIndex] *
                             mechanicalReduction;
        }
        else
        {
            velocityLimit_ = baseMotorOptions_->velocityLimit;
        }

        // Get the rotor inertia on joint side
        if (baseMotorOptions_->enableArmature)
        {
            armature_ = baseMotorOptions_->armature * std::pow(mechanicalReduction, 2);
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

    double AbstractMotorBase::getPositionLimitLower() const
    {
        return positionLimitLower_;
    }

    double AbstractMotorBase::getPositionLimitUpper() const
    {
        return positionLimitUpper_;
    }

    double AbstractMotorBase::getVelocityLimit() const
    {
        return velocityLimit_;
    }

    double AbstractMotorBase::getEffortLimit() const
    {
        return effortLimit_;
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

        // Make sure that the parent robot has not been deleted
        auto robot = robot_.lock();
        if (!robot)
        {
            JIMINY_THROW(std::runtime_error,
                         "Robot has been deleted. Impossible to compute motor efforts.");
        }

        // Compute the actual effort of every motor
        for (AbstractMotorBase * motor : sharedStorage_->motors_)
        {
            const pinocchio::JointIndex jointIndex = motor->getJointIndex();
            const pinocchio::JointModel & jmodel = robot->pinocchioModel_.joints[jointIndex];
            const Eigen::Index jointVelocityIndex = jmodel.idx_v();
            motor->computeEffort(t,
                                 jmodel.jointConfigSelector(q),
                                 v[jointVelocityIndex],
                                 a[jointVelocityIndex],
                                 command[motor->getIndex()]);
        }
    }
}
