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
    name_(name),
    transmissionIdx_(-1),
    jointNames_(),
    jointModelIndices_(-1),
    jointTypes_(),
    jointPositionIndices_(-1),
    jointVelocityIndices_(-1),
    motorNames_()
    {
        // Initialize the options
        setOptions(transmissionOptionsHolder_);
    }

    AbstractTransmissionBase::~AbstractTransmissionBase(void)
    {
        // Detach the transmission before deleting it if necessary
        if (isAttached_)
        {
            detach();
        }
    }

    hresult_t AbstractTransmissionBase::initialize(std::vector<std::string> const & jointNames,
                                                   std::vector<std::string> const & motorNames)
    {
        // Copy reference to joint and motors names
        hresult_t returnCode = hresult_t::SUCCESS;
        jointNames_ = jointNames;
        motorNames_ = motorNames;
        isInitialized_ = true;

        returnCode = refreshProxies();
        if (returnCode != hresult_t::SUCCESS)
        {
            jointNames_.clear();
            motorNames_.clear();
            isInitialized_ = false;
        }

        auto robot = robot_.lock();

        // TODO move this stuff to refresh Proxies ?
        // Populate motorIndices_
        std::weak_ptr<AbstractMotorBase const> motor;
        for (std::string const & motorName : motorNames)
        {
            returnCode = robot->getMotor(motorName, motor);
            auto motorTemp = motor.lock();
            int32_t idx = motorTemp->getIdx();
            motorIndices_.push_back(idx);
        }

        // Populate jointPositionIndices_
        std::vector<int32_t> jointPositionIndices;
        hresult_t returnCode = hresult_t::SUCCESS;
        for (std::string const & jointName : jointNames_)
        {
            std::vector<int32_t> jointPositionIdx;
            if (!robot->pncModel_.existJointName(jointName))
            {
                PRINT_ERROR("Joint '", jointName, "' not found in robot model.");
                return hresult_t::ERROR_BAD_INPUT;
            }
            if (returnCode == hresult_t::SUCCESS)
            {
                returnCode = getJointPositionIdx(robot->pncModel_, jointName, jointPositionIdx);
            }
            if (returnCode == hresult_t::SUCCESS)
            {
                jointPositionIndices.insert(jointPositionIndices.end(), jointPositionIdx.begin(), jointPositionIdx.end());
            }
        }
        int32_t jointPositionSize = jointPositionIndices.size();
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
            if (!robot->pncModel_.existJointName(jointName))
            {
                PRINT_ERROR("Joint '", jointName, "' not found in robot model.");
                return hresult_t::ERROR_BAD_INPUT;
            }
            jointIndex_t const & jointModelIdx = robot->pncModel_.getJointId(jointName);
            int32_t const & jointVelocityFirstIdx = robot->pncModel_.joints[jointModelIdx].idx_v();
            int32_t const & jointNv = robot->pncModel_.joints[jointModelIdx].nv();
            jointVelocityIdx.resize(jointNv);
            std::iota(jointVelocityIdx.begin(), jointVelocityIdx.end(), jointVelocityFirstIdx);
            jointVelocityIndices.insert(jointVelocityIndices.end(), jointVelocityIdx.begin(), jointVelocityIdx.end());
        }

        int32_t jointVelocitySize = jointVelocityIndices.size();
        jointVelocityIndices_.resize(jointVelocitySize);
        for (int32_t i = 0; i <  jointVelocitySize; ++i)
        {
            jointVelocityIndices_(i) = jointVelocityIndices[i];
        }
        return returnCode;
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

        // TODO Make sure the joint is not already attached to a transmission
        // WARNING at this point it is still not know which joint or motor the transmision connects
        // auto robotTemp = robot.lock();
        // std::vector<std::string> actuatedJointNames = robotTemp->getActuatedJointNames();
        // for (std::string const & transmissionJoint : getJointNames())
        // {
        //     auto transmissionJointIt = std::find(actuatedJointNames.begin(), actuatedJointNames.end(), transmissionJoint);
        //     if (transmissionJointIt != actuatedJointNames.end())
        //     {
        //         PRINT_ERROR("Joint already attached to another transmission");
        //         return hresult_t::ERROR_GENERIC;
        //     }
        // }

        // Copy references to the robot and shared data
        robot_ = robot;

        // Update the flag
        isAttached_ = true;

        return hresult_t::SUCCESS;
    }

    hresult_t AbstractTransmissionBase::detach(void)
    {
        if (!isAttached_)
        {
            PRINT_ERROR("Transmission not attached to any robot.");
            return hresult_t::ERROR_GENERIC;
        }

        // Clear the references to the robot
        robot_.reset();

        // Unset the Id
        transmissionIdx_ = -1;

        // Delete motor and joint references

        // Update the flag
        isAttached_ = false;

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

        for (int i = 0; i < jointNames_.size(); i++)
        {
            if (returnCode == hresult_t::SUCCESS)
            {
                returnCode = ::jiminy::getJointModelIdx(robot->pncModel_, jointNames_[i], jointModelIndices_[i]);
            }
        }
        for (int i = 0; i < jointNames_.size(); i++)
        {
            if (returnCode == hresult_t::SUCCESS)
            {
                returnCode = getJointTypeFromIdx(robot->pncModel_, jointModelIndices_[i], jointTypes_[i]);
            }
        }

        for (int i = 0; i < jointNames_.size(); i++)
        {
            if (returnCode == hresult_t::SUCCESS)
            {
                // Transmissions are only supported for linear and rotary joints
                if (jointTypes_[i] != joint_t::LINEAR && jointTypes_[i] != joint_t::ROTARY && jointTypes_[i] != joint_t::ROTARY_UNBOUNDED)
                {
                    PRINT_ERROR("A transmission can only be associated with a 1-dof linear or rotary joint.");
                    returnCode = hresult_t::ERROR_BAD_INPUT;
                }
            }
        }

        for (int i = 0; i < jointNames_.size(); i++)
        {
            if (returnCode == hresult_t::SUCCESS)
            {
                ::jiminy::getJointPositionIdx(robot->pncModel_, jointNames_[i], jointPositionIndices_[i]);
                ::jiminy::getJointVelocityIdx(robot->pncModel_, jointNames_[i], jointVelocityIndices_[i]);
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

    std::vector<std::string> const & AbstractTransmissionBase::getJointNames(void) const
    {
        return jointNames_;
    }

    std::vector<jointIndex_t> const & AbstractTransmissionBase::getJointModelIndices(void) const
    {
        return jointModelIndices_;
    }

    std::vector<joint_t> const & AbstractTransmissionBase::getJointTypes(void) const
    {
        return jointTypes_;
    }

    vectorN_t const & AbstractTransmissionBase::getJointPositionIndices(void) const
    {
        return jointPositionIndices_;
    }


    vectorN_t const & AbstractTransmissionBase::getJointVelocityIndices(void) const
    {

        return jointVelocityIndices_;
    }

    std::vector<std::string> const & AbstractTransmissionBase::getMotorNames(void) const
    {
        return motorNames_;
    }

    std::vector<std::string> const & getMotorIndices(void) const
    {
        // TODO create and populate
        return motorIndices_;
    }

    matrixN_t const & getForwardTransform(void) const
    {
        return forwardTransform_;
    }

    matrixN_t const & getInverseTransform(void) const
    {
        return backwardTransform_;
    }

    hresult_t AbstractTransmissionBase::computeForward(float64_t const & t,
                                                       vectorN_t & q,
                                                       vectorN_t & v,
                                                       vectorN_t & a,
                                                       vectorN_t & uJoint)
    {
        // Extract motor configuration and velocity from all motors attached
        // to the robot for this transmission
        auto qMotors = q.segment<>(jointPositionIndices_, );
        auto vMotors = v.segment<>(jointVelocityIndices_, );

        // Compute the transmission effect based on the current configuration
        computeTransform(qMotors, vMotors);

        // Apply transformation from motor to joint level
        auto motors = motors_.lock();
        q.noalias() = forwardTransform_ * motors->getPosition();
        v.noalias() = forwardTransform_ * motors->getVelocity();
        a.noalias() = forwardTransform_ * motors->getAcceleration();
        uJoint.noalias() = forwardTransform_ * motors->getEffort();
    }

    hresult_t AbstractTransmissionBase::computeBackward(float64_t const & t,
                                                        vectorN_t const & q,
                                                        vectorN_t const & v,
                                                        vectorN_t const & a,
                                                        vectorN_t const & uJoint)
    {
        computeInverseTransform(q, v);
        auto motors = motors_.lock();
        motors->q = backwardTransform_ * q;
        motors->v = backwardTransform_ * v;
        motors->a = backwardTransform_ * a;
        motors->u = backwardTransform_ * uJoint;
    }
}
