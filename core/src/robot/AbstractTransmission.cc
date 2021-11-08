#include "jiminy/core/robot/Robot.h"
#include "jiminy/core/Macros.h"

#include "jiminy/core/utilities/Pinocchio.h"
#include "jiminy/core/robot/AbstractTransmission.h"


namespace jiminy
{
    AbstractTransmissionBase::AbstractTransmissionBase(std::string const & name) :
    baseTransmissionOptions_(nullptr),
    transmissionOptionsHolder_(),
    isInitialized_(false),
    isAttached_(false),
    robot_(),
    notifyRobot_(),
    name_(name),
    transmissionIdx_(-1),
    jointNames_(),
    jointModelIndices_(-1),
    jointTypes_(joint_t::NONE),
    jointPositionIndices_(-1),
    jointVelocityIndices_(-1),
    motorNames_(),
    {
        // Initialize the options
        setOptions(getDefaultTransmissionOptions());
    }

    AbstractTransmissionBase::~AbstractTransmissionBase(void)
    {
        // Detach the transmission before deleting it if necessary
        if (isAttached_)
        {
            detach();
        }
    }

    hresult_t AbstractTransmissionBase::initialize(void)
    {
        // Populate jointPositionIndices_
        std::vector<int32_t> jointPositionIndices;
        returnCode = hresult_t::SUCCESS;
        for (std::string const & jointName : jointNames_)
        {
            std::vector<int32_t> jointPositionIdx;
            if (!robot->model.existJointName(jointName))
            {
                PRINT_ERROR("Joint '", jointName, "' not found in robot model.");
                return hresult_t::ERROR_BAD_INPUT;
            }
            if (returnCode == hresult_t::SUCCESS)
            {
                returnCode = getJointPositionIdx(robot->model, jointName, jointPositionIdx);
            }
            if (returnCode == hresult_t::SUCCESS)
            {
                jointPositionIndices.insert(jointPositionIndices.end(), jointPositionIdx.begin(), jointPositionIdx.end());
            }
        }
        jointPositionSize = jointPositionIndices.size()
        jointPositionIndices_.resize(jointPositionSize);
        for (int32_t i = 0; i <  jointPositionSize; ++i)
        {
            jointPositionIndices_(i) = jointPositionIndices[i];
        }


        // Populate jointVelocityIndices_
        std::vector<int32_t> jointVelocityIndices;
        for (std::string const & jointName : jointNames_)
        {
            std::vector<int32_t> jointVelocityIdx;
            if (!robot->model.existJointName(jointName))
            {
                PRINT_ERROR("Joint '", jointName, "' not found in robot model.");
                return hresult_t::ERROR_BAD_INPUT;
            }
            jointIndex_t const & jointModelIdx = robot->model.getJointId(jointName);
            int32_t const & jointVelocityFirstIdx = robot->model.joints[jointModelIdx].idx_v();
            int32_t const & jointNv = robot->model.joints[jointModelIdx].nv();
            jointVelocityIdx.resize(jointNv);
            std::iota(jointVelocityIdx.begin(), jointVelocityIdx.end(), jointVelocityFirstIdx)
            jointVelocityIndices.insert(jointVelocityIndices.end(), jointVelocityIdx.begin(), jointVelocityIdx.end());
        }
    }

    hresult_t AbstractTransmissionBase::attach(std::weak_ptr<Robot const> robot)
    {
        // Make sure the transmission is not already attached
        if (isAttached_)
        {
            PRINT_ERROR("Transmission already attached to a robot. Please 'detach' method before attaching it.");
            return hresult_t::ERROR_GENERIC;
        }

        // Make sure the robot still exists
        if (robot.expired())
        {
            PRINT_ERROR("Robot pointer expired or unset.");
            return hresult_t::ERROR_GENERIC;
        }
        
        // Make sure the joint is not already attached to a transmission
        std_vector<std::string> actuatedJoints = robot->getActuatedJoints()
        for (std::string const & transmissionJoint : getJointNames())
        {
            auto transmissionJointIt = actuatedJoints.find(transmissionJoint);
            if (transmissionJointIt != actuatedJoints.end())
            {
                PRINT_ERROR("Joint already attached to another transmission");
                return hresult_t::ERROR_GENERIC;
            }
        }

        // Copy references to the robot and shared data
        robot_ = robot;

        // Update the actuated joints
        robot_->updateActuatedJoints(jointNames_)

        // Update the flag
        isAttached_ = true;

        return hresult_t::SUCCESS;
    }

    hresult_t AbstractTransmissionBase::detach(void)
    {
        // Delete the part of the shared memory associated with the transmission

        if (!isAttached_)
        {
            PRINT_ERROR("Transmission not attached to any robot.");
            return hresult_t::ERROR_GENERIC;
        }

        // Remove associated col in the global data buffer
        if (transmissionIdx_ < sharedHolder_->num_ - 1)
        {
            int32_t transmissionShift = sharedHolder_->num_ - transmissionIdx_ - 1;
            sharedHolder_->data_.segment(transmissionIdx_, transmissionShift) =
                sharedHolder_->data_.segment(transmissionIdx_ + 1, transmissionShift).eval();  // eval to avoid aliasing
        }
        sharedHolder_->data_.conservativeResize(sharedHolder_->num_ - 1);

        // Shift the transmission ids
        for (int32_t i = transmissionIdx_ + 1; i < sharedHolder_->num_; ++i)
        {
            --sharedHolder_->transmissions_[i]->transmissionIdx_;
        }

        // Remove the transmission to the shared memory
        sharedHolder_->transmissions_.erase(sharedHolder_->transmissions_.begin() + transmissionIdx_);
        --sharedHolder_->num_;

        // Clear the references to the robot and shared data
        robot_.reset();
        sharedHolder_ = nullptr;

        // Unset the Id
        transmissionIdx_ = -1;

        // Update the flag
        isAttached_ = false;

        return hresult_t::SUCCESS;
    }

    hresult_t AbstractTransmissionBase::resetAll(void)
    {
        // Make sure the transmission is attached to a robot
        if (!isAttached_)
        {
            PRINT_ERROR("Transmission not attached to any robot.");
            return hresult_t::ERROR_GENERIC;
        }

        // Make sure the robot still exists
        if (robot_.expired())
        {
            PRINT_ERROR("Robot has been deleted. Impossible to reset the transmissions.");
            return hresult_t::ERROR_GENERIC;
        }

        // Clear the shared data buffer
        sharedHolder_->data_.setZero();

        // Update transmission scope information
        for (AbstractTransmissionBase * transmission : sharedHolder_->transmissions_)
        {
            // Refresh proxies that are robot-dependent
            transmission->refreshProxies();
        }

        return hresult_t::SUCCESS;
    }

    hresult_t AbstractTransmissionBase::setOptions(configHolder_t const & transmissionOptions)
    {
        // Check if the internal buffers must be updated
        bool_t internalBuffersMustBeUpdated = false;
        if (isInitialized_)
        {
            // Check if reduction ratio has changed
            float64_t const & mechanicalReduction = boost::get<float64_t>(transmissionOptions.at("mechanicalReduction"));
            internalBuffersMustBeUpdated |= (baseTransmissionOptions_->mechanicalReduction != mechanicalReduction);
            if (mechanicalReduction)
            {
                float64_t const & mechanicalReduction = boost::get<float64_t>(transmissionOptions.at("mechanicalReduction"));
                internalBuffersMustBeUpdated |= std::abs(armature - baseTransmissionOptions_->armature) > EPS;
            }
        }

        // Update the transmission's options
        transmissionOptionsHolder_ = transmissionOptions;
        baseTransmissionOptions_ = std::make_unique<abstractTransmissionOptions_t const>(transmissionOptionsHolder_);

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

    configHolder_t AbstractTransmissionBase::getOptions(void) const
    {
        return transmissionOptionsHolder_;
    }

    hresult_t AbstractTransmissionBase::refreshProxies(void)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (!isAttached_)
        {
            PRINT_ERROR("Transmission not attached to any robot. Impossible to refresh proxies.");
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
                PRINT_ERROR("Transmission not initialized. Impossible to refresh proxies.");
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

        for (i = 0; i < jointName_.size(); i++)
        {
            if (returnCode == hresult_t::SUCCESS)
            {
                returnCode = ::jiminy::getJointModelIdx(robot->pncModel_, jointName_[i], jointModelIdx_[i]);
            }
        }
        for (i = 0; i < jointName_.size(); i++)
        {
            if (returnCode == hresult_t::SUCCESS)
            {
                returnCode = getJointTypeFromIdx(robot->pncModel_, jointModelIdx_[i], jointType_[i]);
            }
        }

        for (i = 0; i < jointName_.size(); i++)
        {        
            if (returnCode == hresult_t::SUCCESS)
            {
                // Transmissions are only supported for linear and rotary joints
                if (jointType_[i] != joint_t::LINEAR && jointType_[i] != joint_t::ROTARY && jointType_[i] != joint_t::ROTARY_UNBOUNDED)
                {
                    PRINT_ERROR("A transmission can only be associated with a 1-dof linear or rotary joint.");
                    returnCode = hresult_t::ERROR_BAD_INPUT;
                }
            }
        }

        for (i = 0; i < jointName_.size(); i++)
        {       
            if (returnCode == hresult_t::SUCCESS)
            {
                ::jiminy::getJointPositionIdx(robot->pncModel_, jointName_[i], jointPositionIdx_[i]);
                ::jiminy::getJointVelocityIdx(robot->pncModel_, jointName_[i], jointVelocityIdx_[i]);

                // Get the rotor inertia
                if (baseTransmissionOptions_->enableArmature)
                {
                    armature_ = baseTransmissionOptions_->armature;
                }
                else
                {
                    armature_ = 0.0;
                }

                // Propagate the user-defined transmission inertia at Pinocchio model level
                if (notifyRobot_)
                {
                    returnCode = notifyRobot_(*this);
                }
            }
        }

        return returnCode;
    }

    hresult_t AbstractTransmissionBase::setOptionsAll(configHolder_t const & transmissionOptions)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        // Make sure the transmission is attached to a robot
        if (!isAttached_)
        {
            PRINT_ERROR("Transmission not attached to any robot.");
            returnCode = hresult_t::ERROR_GENERIC;
        }

        for (AbstractTransmissionBase * transmission : sharedHolder_->transmissions_)
        {
            if (returnCode == hresult_t::SUCCESS)
            {
                returnCode = transmission->setOptions(transmissionOptions);
            }
        }

        return returnCode;
    }

    bool_t const & AbstractTransmissionBase::getIsInitialized(void) const
    {
        return isInitialized_;
    }

    std::string const & AbstractTransmissionBase::getName(void) const
    {
        return name_;
    }

    int32_t const & AbstractTransmissionBase::getIdx(void) const
    {
        return transmissionIdx_;
    }

    std::vector<std>::string const & AbstractTransmissionBase::getJointNames(void) const
    {
        return jointNames_;
    }

    std::vector<jointIndex_t> const & AbstractTransmissionBase::getJointModelIndices(void) const
    {
        jointIndex_t jointModelIdx;
        for (std::string const & jointName : jointNames_)
        {
            returnCode = ::jiminy::getJointModelIdx(robot->pncModel_, jointName, jointModelIdx);
            if (returnCode == hresult_t::SUCCESS)
            {
                jointModelIndices_.push_back(jointModelIdx);
            }
        }
        return jointModelIndices_;
    }

    std::vector<joint_t> const & AbstractTransmissionBase::getJointTypes(void) const
    {
        jointModelIndices = getJointModelIndices();
        for (jointIndex_t const & idx : jointModelIndices)
        {
            joint_t jointType;
            getJointTypeFromIdx(robot->pncModel, idx, jointType); 
            jointTypes_.push_back(jointType);
        }
        return jointTypes_;
    }

    vectorN_t const & AbstractTransmissionBase::getJointPositionIndices(void) 
    {
        return jointPositionIndices_;
    }


    vectorN_t const & AbstractTransmissionBase::getJointVelocityIndices(void)
    {

        return jointVelocityIndices_;
    }

    std::vector<std>::string const & AbstractTransmissionBase::getMotorNames(void) const
    {
        return motorName_;
    }

    hresult_t AbstractTransmissionBase::computeForward(float64_t const & t,
                                                       vectorN_t & q,
                                                       vectorN_t & v,
                                                       vectorN_t & a,
                                                       vectorN_t & uJoint)
    {
        auto qMotors = q.segment<>(jointPositionIdx_, );
        auto vMotors = v.segment<>(jointVelocityIdx_, );
        computeTransform(qMotors, vMotors);
        q.noalias() = forwardTransform_ * motors_->getPosition();
        v.noalias() = forwardTransform_ * motors_->getVelocity();
        a.noalias() = forwardTransform_ * motors_->getAcceleration();
        uJoint.noalias() = forwardTransform_ * motors_->getEffort();
    }   

    hresult_t AbstractTransmissionBase::computeBackward(float64_t const & t,
                                                        vectorN_t const & q,
                                                        vectorN_t const & v,
                                                        vectorN_t const & a,
                                                        vectorN_t const & uJoint)
    {
        computeInverseTransform(q, v);
        motors_->q = backwardTransform_ * q;
        motors_->v = backwardTransform_ * v;
        motors_->a = backwardTransform_ * a;
        motors_->u = backwardTransform_ * uJoint;
    }                                                   
}
