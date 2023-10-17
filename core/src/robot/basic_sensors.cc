#include <algorithm>

#include "pinocchio/spatial/explog.hpp"    // `pinocchio::exp3`
#include "pinocchio/spatial/se3.hpp"       // `pinocchio::SE3`
#include "pinocchio/spatial/force.hpp"     // `pinocchio::Force  `
#include "pinocchio/spatial/motion.hpp"    // `pinocchio::Motion`
#include "pinocchio/algorithm/frames.hpp"  // `pinocchio::getFrameVelocity`, `pinocchio::getFrameAcceleration`

#include "jiminy/core/robot/robot.h"
#include "jiminy/core/utilities/pinocchio.h"
#include "jiminy/core/utilities/random.h"

#include "jiminy/core/robot/basic_sensors.h"


#define GET_ROBOT_AND_CHECK_SENSOR_INTEGRITY() \
    if (!isAttached_) \
    { \
        PRINT_ERROR("Sensor not attached to any robot. Impossible to refresh proxies."); \
        returnCode = hresult_t::ERROR_INIT_FAILED; \
    } \
     \
    auto robot = robot_.lock(); \
    if (returnCode == hresult_t::SUCCESS) \
    { \
        if (!robot) \
        { \
            PRINT_ERROR("Robot has been deleted. Impossible to refresh proxies."); \
            returnCode = hresult_t::ERROR_GENERIC; \
        } \
    } \
     \
    if (returnCode == hresult_t::SUCCESS) \
    { \
        if (!robot->getIsInitialized())  \
        { \
            PRINT_ERROR("Robot not initialized. Impossible to refresh proxies."); \
            returnCode = hresult_t::ERROR_INIT_FAILED; \
        } \
    } \
     \
    if (returnCode == hresult_t::SUCCESS) \
    { \
        if (!isInitialized_) \
        { \
            PRINT_ERROR("Sensor not initialized. Impossible to refresh proxies."); \
            returnCode = hresult_t::ERROR_INIT_FAILED; \
        } \
    }


#define GET_ROBOT_IF_INITIALIZED() \
    if (!isInitialized_) \
    { \
        PRINT_ERROR("Sensor not initialized. Impossible to update sensor."); \
        return hresult_t::ERROR_INIT_FAILED; \
    } \
     \
    auto robot = robot_.lock();


namespace jiminy
{
    // ===================== ImuSensor =========================

    template<>
    std::string const AbstractSensorTpl<ImuSensor>::type_("ImuSensor");
    template<>
    bool_t const AbstractSensorTpl<ImuSensor>::areFieldnamesGrouped_(false);
    template<>
    std::vector<std::string> const AbstractSensorTpl<ImuSensor>::fieldnames_(
        {"Gyrox", "Gyroy", "Gyroz", "Accelx", "Accely", "Accelz"});

    ImuSensor::ImuSensor(std::string const & name) :
    AbstractSensorTpl(name),
    frameName_(),
    frameIdx_(0),
    sensorRotationBiasInv_(matrix3_t::Identity())
    {
        // Empty on purpose
    }

    hresult_t ImuSensor::initialize(std::string const & frameName)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        frameName_ = frameName;
        isInitialized_ = true;
        returnCode = refreshProxies();

        if (returnCode != hresult_t::SUCCESS)
        {
            frameName_.clear();
            isInitialized_ = false;
        }

        return returnCode;
    }

    hresult_t ImuSensor::setOptions(configHolder_t const & sensorOptions)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        // Check that bias / std is of the correct size
        vectorN_t const & bias = boost::get<vectorN_t>(sensorOptions.at("bias"));
        vectorN_t const & noiseStd = boost::get<vectorN_t>(sensorOptions.at("noiseStd"));
        if (bias.size() && bias.size() != 9)
        {
            PRINT_ERROR("Wrong bias vector. It must contain 9 values:\n"
                        "  - the first three are the angle-axis representation of a rotation bias applied to all sensor signal.\n"
                        "  - the next six are respectively gyroscope and accelerometer additive bias.");
            returnCode = hresult_t::ERROR_BAD_INPUT;
        }
        if (noiseStd.size() && noiseStd.size() != 6)
        {
            PRINT_ERROR("Wrong noise std vector. It must contain 6 values corresponding respectively to gyroscope and accelerometer additive bias.");
            returnCode = hresult_t::ERROR_BAD_INPUT;
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            returnCode = AbstractSensorTpl<ImuSensor>::setOptions(sensorOptions);
        }

        return returnCode;
    }

    hresult_t ImuSensor::refreshProxies(void)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        GET_ROBOT_AND_CHECK_SENSOR_INTEGRITY()

        if (returnCode == hresult_t::SUCCESS)
        {
            returnCode = ::jiminy::getFrameIdx(robot->pncModel_, frameName_, frameIdx_);
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            if (baseSensorOptions_->bias.size())
            {
                // Convert first three components of bias to quaternion
                sensorRotationBiasInv_ = pinocchio::exp3(- baseSensorOptions_->bias.head<3>());
            }
            else
            {
                sensorRotationBiasInv_ = matrix3_t::Identity();
            }
        }

        return returnCode;
    }

    std::string const & ImuSensor::getFrameName(void) const
    {
        return frameName_;
    }

    frameIndex_t const & ImuSensor::getFrameIdx(void) const
    {
        return frameIdx_;
    }

    hresult_t ImuSensor::set(float64_t     const & /* t */,
                             vectorN_t     const & /* q */,
                             vectorN_t     const & /* v */,
                             vectorN_t     const & /* a */,
                             vectorN_t     const & /* uMotor */,
                             forceVector_t const & /* fExternal */)
    {
        GET_ROBOT_IF_INITIALIZED()

        // Compute gyroscope signal
        pinocchio::Motion const velocity = pinocchio::getFrameVelocity(
            robot->pncModel_, robot->pncData_, frameIdx_, pinocchio::LOCAL);
        data().head<3>() = velocity.angular();

        // Compute accelerometer signal
        pinocchio::Motion const acceleration = pinocchio::getFrameClassicalAcceleration(
            robot->pncModel_, robot->pncData_, frameIdx_, pinocchio::LOCAL);

        // Accelerometer signal is sensor classical (not spatial !) linear acceleration minus gravity
        matrix3_t const & rot = robot->pncData_.oMf[frameIdx_].rotation();
        data().tail<3>() = acceleration.linear() - rot.transpose() * robot->pncModel_.gravity.linear();

        return hresult_t::SUCCESS;
    }

    void ImuSensor::measureData(void)
    {
        // Add measurement white noise
        if (baseSensorOptions_->noiseStd.size())
        {
            // Accel + gyroscope: simply apply additive noise
            get() += randVectorNormal(baseSensorOptions_->noiseStd);
        }

        // Add measurement bias
        if (baseSensorOptions_->bias.size())
        {
            // Accel + gyroscope: simply add additive bias
            get() += baseSensorOptions_->bias.tail<6>();

            /* Apply the same bias to the accelerometer / gyroscope output.
               We interpret quaternion bias as angle-axis representation of a
               sensor rotation bias R_b, such that w_R_sensor = w_R_imu R_b. */
            get().head<3>() = sensorRotationBiasInv_ * get().head<3>();
            get().tail<3>() = sensorRotationBiasInv_ * get().tail<3>();
        }
    }

    // ===================== ContactSensor =========================

    template<>
    std::string const AbstractSensorTpl<ContactSensor>::type_("ContactSensor");
    template<>
    bool_t const AbstractSensorTpl<ContactSensor>::areFieldnamesGrouped_(false);
    template<>
    std::vector<std::string> const AbstractSensorTpl<ContactSensor>::fieldnames_({"FX", "FY", "FZ"});

    ContactSensor::ContactSensor(std::string const & name) :
    AbstractSensorTpl(name),
    frameName_(),
    frameIdx_(0)
    {
        // Empty on purpose
    }

    hresult_t ContactSensor::initialize(std::string const & frameName)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        frameName_ = frameName;
        isInitialized_ = true;
        returnCode = refreshProxies();

        if (returnCode != hresult_t::SUCCESS)
        {
            frameName_.clear();
            isInitialized_ = false;
        }

        return returnCode;
    }

    hresult_t ContactSensor::refreshProxies(void)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        GET_ROBOT_AND_CHECK_SENSOR_INTEGRITY()

        if (returnCode == hresult_t::SUCCESS)
        {
            std::vector<std::string> const & contactFramesNames = robot->getContactFramesNames();
            auto contactFrameNameIt = std::find(contactFramesNames.begin(), contactFramesNames.end(), frameName_);
            if (contactFrameNameIt == contactFramesNames.end())
            {
                PRINT_ERROR("Sensor frame not associated with any contact point of the robot. Impossible to refresh proxies.");
                returnCode = hresult_t::ERROR_BAD_INPUT;
            }
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            returnCode = ::jiminy::getFrameIdx(robot->pncModel_, frameName_, frameIdx_);
        }

        return returnCode;
    }

    std::string const & ContactSensor::getFrameName(void) const
    {
        return frameName_;
    }

    frameIndex_t const & ContactSensor::getFrameIdx(void) const
    {
        return frameIdx_;
    }

    hresult_t ContactSensor::set(float64_t     const & /* t */,
                                 vectorN_t     const & /* q */,
                                 vectorN_t     const & /* v */,
                                 vectorN_t     const & /* a */,
                                 vectorN_t     const & /* uMotor */,
                                 forceVector_t const & /* fExternal */)
    {
        GET_ROBOT_IF_INITIALIZED()

        std::vector<frameIndex_t> const & contactFramesIdx = robot->getContactFramesIdx();
        auto it = std::find(contactFramesIdx.begin(), contactFramesIdx.end(), frameIdx_);
        data() = robot->contactForces_[std::distance(contactFramesIdx.begin(), it)].linear();

        return hresult_t::SUCCESS;
    }

    // ===================== ForceSensor =========================

    template<>
    std::string const AbstractSensorTpl<ForceSensor>::type_("ForceSensor");
    template<>
    bool_t const AbstractSensorTpl<ForceSensor>::areFieldnamesGrouped_(false);
    template<>
    std::vector<std::string> const AbstractSensorTpl<ForceSensor>::fieldnames_({"FX", "FY", "FZ", "MX", "MY", "MZ"});

    ForceSensor::ForceSensor(std::string const & name) :
    AbstractSensorTpl(name),
    frameName_(),
    frameIdx_(0),
    parentJointIdx_(0),
    f_()
    {
        // Empty on purpose
    }

    hresult_t ForceSensor::initialize(std::string const & frameName)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        frameName_ = frameName;
        isInitialized_ = true;
        returnCode = refreshProxies();

        if (returnCode != hresult_t::SUCCESS)
        {
            frameName_.clear();
            isInitialized_ = false;
        }

        return returnCode;
    }

    hresult_t ForceSensor::refreshProxies(void)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        GET_ROBOT_AND_CHECK_SENSOR_INTEGRITY()

        if (returnCode == hresult_t::SUCCESS)
        {
            returnCode = ::jiminy::getFrameIdx(robot->pncModel_, frameName_, frameIdx_);
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            parentJointIdx_ = robot->pncModel_.frames[frameIdx_].parent;  // 'parent' returns the parent joint
        }

        return returnCode;
    }

    std::string const & ForceSensor::getFrameName(void) const
    {
        return frameName_;
    }

    frameIndex_t const & ForceSensor::getFrameIdx(void) const
    {
        return frameIdx_;
    }

    jointIndex_t ForceSensor::getJointIdx(void) const
    {
        return parentJointIdx_;
    }

    hresult_t ForceSensor::set(float64_t     const & /* t */,
                               vectorN_t     const & /* q */,
                               vectorN_t     const & /* v */,
                               vectorN_t     const & /* a */,
                               vectorN_t     const & /* uMotor */,
                               forceVector_t const & fExternal)
    {
        // Returns the force applied on parent body in frame

        GET_ROBOT_IF_INITIALIZED()

        // Get the sum of external forces applied on parent joint
        jointIndex_t const & i = parentJointIdx_;
        pinocchio::Force const & fJoint = fExternal[i];

        // Transform the force from joint frame to sensor frame
        pinocchio::SE3 const & framePlacement = robot->pncModel_.frames[frameIdx_].placement;
        f_ = framePlacement.actInv(fJoint);
        data() = f_.toVector();

        return hresult_t::SUCCESS;
    }

    // ===================== EncoderSensor =========================

    template<>
    std::string const AbstractSensorTpl<EncoderSensor>::type_("EncoderSensor");
    template<>
    bool_t const AbstractSensorTpl<EncoderSensor>::areFieldnamesGrouped_(true);
    template<>
    std::vector<std::string> const AbstractSensorTpl<EncoderSensor>::fieldnames_({"Q", "V"});

    EncoderSensor::EncoderSensor(std::string const & name) :
    AbstractSensorTpl(name),
    jointName_(),
    jointIdx_(0),
    jointType_(joint_t::NONE)
    {
        // Empty on purpose
    }

    hresult_t EncoderSensor::initialize(std::string const & jointName)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        jointName_ = jointName;
        isInitialized_ = true;
        returnCode = refreshProxies();

        if (returnCode != hresult_t::SUCCESS)
        {
            jointName_.clear();
            isInitialized_ = false;
        }

        return returnCode;
    }

    hresult_t EncoderSensor::refreshProxies(void)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        GET_ROBOT_AND_CHECK_SENSOR_INTEGRITY()

        if (returnCode == hresult_t::SUCCESS)
        {
            if (!robot->pncModel_.existJointName(jointName_))
            {
                PRINT_ERROR("Sensor attached to a joint that does not exist.");
                returnCode = hresult_t::ERROR_INIT_FAILED;
            }
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            jointIdx_ = robot->pncModel_.getJointId(jointName_);
            getJointTypeFromIdx(robot->pncModel_, jointIdx_, jointType_);

            // Motors are only supported for linear and rotary joints
            if (jointType_ != joint_t::LINEAR && jointType_ != joint_t::ROTARY && jointType_ != joint_t::ROTARY_UNBOUNDED)
            {
                PRINT_ERROR("An encoder sensor can only be associated with a 1-dof linear or rotary joint.");
                returnCode =  hresult_t::ERROR_BAD_INPUT;
            }
        }

        return returnCode;
    }

    std::string const & EncoderSensor::getJointName(void) const
    {
        return jointName_;
    }

    jointIndex_t const & EncoderSensor::getJointIdx(void) const
    {
        return jointIdx_;
    }

    joint_t const & EncoderSensor::getJointType(void) const
    {
        return jointType_;
    }

    hresult_t EncoderSensor::set(float64_t     const & /* t */,
                                 vectorN_t     const & q,
                                 vectorN_t     const & v,
                                 vectorN_t     const & /* a */,
                                 vectorN_t     const & /* uMotor */,
                                 forceVector_t const & /* fExternal */)
    {
        GET_ROBOT_IF_INITIALIZED()

        auto const & joint = robot->pncModel_.joints[jointIdx_];
        int32_t const & jointPositionIdx = joint.idx_q();
        int32_t const & jointVelocityIdx = joint.idx_v();
        if (jointType_ == joint_t::ROTARY_UNBOUNDED)
        {
            float64_t const & cosTheta = q[jointPositionIdx];
            float64_t const & sinTheta = q[jointPositionIdx + 1];
            data()[0] = std::atan2(sinTheta, cosTheta);
        }
        else
        {
            data()[0] = q[jointPositionIdx];
        }
        data()[1] = v[jointVelocityIdx];

        return hresult_t::SUCCESS;
    }

    // ===================== EffortSensor =========================

    template<>
    std::string const AbstractSensorTpl<EffortSensor>::type_("EffortSensor");
    template<>
    bool_t const AbstractSensorTpl<EffortSensor>::areFieldnamesGrouped_(true);
    template<>
    std::vector<std::string> const AbstractSensorTpl<EffortSensor>::fieldnames_({"U"});

    EffortSensor::EffortSensor(std::string const & name) :
    AbstractSensorTpl(name),
    motorName_(),
    motorIdx_(-1)
    {
        // Empty on purpose
    }

    hresult_t EffortSensor::initialize(std::string const & motorName)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        motorName_ = motorName;
        isInitialized_ = true;
        returnCode = refreshProxies();

        if (returnCode != hresult_t::SUCCESS)
        {
            motorName_.clear();
            isInitialized_ = false;
        }

        return returnCode;
    }

    hresult_t EffortSensor::refreshProxies(void)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        GET_ROBOT_AND_CHECK_SENSOR_INTEGRITY()

        std::weak_ptr<AbstractMotorBase const> motor;
        if (returnCode == hresult_t::SUCCESS)
        {
            returnCode = robot->getMotor(motorName_, motor);
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            motorIdx_ = motor.lock()->getIdx();
        }

        return returnCode;
    }

    std::string const & EffortSensor::getMotorName(void) const
    {
        return motorName_;
    }

    std::size_t const & EffortSensor::getMotorIdx(void) const
    {
        return motorIdx_;
    }

    hresult_t EffortSensor::set(float64_t     const & /* t */,
                                vectorN_t     const & /* q */,
                                vectorN_t     const & /* v */,
                                vectorN_t     const & /* a */,
                                vectorN_t     const & uMotor,
                                forceVector_t const & /* fExternal */)
    {
        if (!isInitialized_)
        {
            PRINT_ERROR("Sensor not initialized. Impossible to set sensor data.");
            return hresult_t::ERROR_INIT_FAILED;
        }

        data()[0] = uMotor[motorIdx_];

        return hresult_t::SUCCESS;
    }
}
