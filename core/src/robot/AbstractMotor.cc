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
    jointPositionIdx_(-1),
    jointVelocityIdx_(-1),
    torqueLimit_(0.0),
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
        for (int32_t i = motorIdx_ + 1; i < sharedHolder_->num_; i++)
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
        // Clear the data buffer
        sharedHolder_->data_.setZero();

        // Refresh proxies that are robot-dependent
        refreshProxies();
    }

    hresult_t AbstractMotorBase::setOptions(configHolder_t const & motorOptions)
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

        // Refresh the proxies if the robot is initialized
        if (isAttached_ && internalBuffersMustBeUpdated && robot_->getIsInitialized())
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
            ::jiminy::getJointPositionIdx(robot_->pncModel_, jointName_, jointPositionIdx_);
            ::jiminy::getJointVelocityIdx(robot_->pncModel_, jointName_, jointVelocityIdx_);

            // Get the motor torque limits from the URDF or the user options.
            if (baseMotorOptions_->torqueLimitFromUrdf)
            {
                torqueLimit_ = robot_->pncModel_.effortLimit[jointVelocityIdx_];
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

    hresult_t AbstractMotorBase::computeEffortAll(float64_t                   const & t,
                                                  Eigen::Ref<vectorN_t const> const & q,
                                                  Eigen::Ref<vectorN_t const> const & v,
                                                  Eigen::Ref<vectorN_t const> const & a,
                                                  vectorN_t                   const & uCommand)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        // Compute the motors' output
        for (AbstractMotorBase * motor : sharedHolder_->motors_)
        {
            if (returnCode == hresult_t::SUCCESS)
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
