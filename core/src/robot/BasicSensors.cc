#include <algorithm>

#include "pinocchio/spatial/explog.hpp"
#include "pinocchio/multibody/model.hpp"
#include "pinocchio/algorithm/frames.hpp"

#include "jiminy/core/robot/Robot.h"
#include "jiminy/core/Macros.h"

#include "jiminy/core/robot/BasicSensors.h"


namespace jiminy
{
    // ===================== ImuSensor =========================

    template<>
    std::string const AbstractSensorTpl<ImuSensor>::type_("ImuSensor");
    template<>
    bool_t const AbstractSensorTpl<ImuSensor>::areFieldnamesGrouped_(false);
    template<>
    std::vector<std::string> const AbstractSensorTpl<ImuSensor>::fieldNames_(
        {"Quatx", "Quaty", "Quatz", "Quatw", "Gyrox", "Gyroy", "Gyroz", "Accelx", "Accely", "Accelz"});

    ImuSensor::ImuSensor(std::string const & name) :
    AbstractSensorTpl(name),
    frameName_(),
    frameIdx_(0),
    sensorRotationBias_()
    {
        // Empty.
    }

    hresult_t ImuSensor::initialize(std::string const & frameName)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (!isAttached_)
        {
            PRINT_ERROR("Sensor not attached to any robot. Impossible to initialize it.");
            returnCode = hresult_t::ERROR_GENERIC;
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            frameName_ = frameName;
            isInitialized_ = true;
            returnCode = refreshProxies();
        }

        if (returnCode != hresult_t::SUCCESS)
        {
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
        if ((bias.size() && bias.size() != 9) || (noiseStd.size() && noiseStd.size() != 9))
        {
            PRINT_ERROR("Wrong bias or std vector size. Bias vector should contain 9 coordinates:\n"
                        "  - the first three are the angle-axis representation of a rotation bias applied to all sensor signal.\n"
                        "  - the next six are respectively gyroscope and accelerometer additive bias.");
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

        if (!robot_->getIsInitialized())
        {
            PRINT_ERROR("Robot not initialized. Impossible to refresh model-dependent proxies.");
            returnCode = hresult_t::ERROR_INIT_FAILED;
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            if (!isInitialized_)
            {
                PRINT_ERROR("Sensor not initialized. Impossible to refresh model-dependent proxies.");
                returnCode = hresult_t::ERROR_INIT_FAILED;
            }
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            returnCode = ::jiminy::getFrameIdx(robot_->pncModel_, frameName_, frameIdx_);
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            if (baseSensorOptions_->bias.size())
            {
                // Convert first three components of bias to quaternion
                sensorRotationBias_ = quaternion_t(pinocchio::exp3(baseSensorOptions_->bias.head<3>()));
            }
            else
            {
                sensorRotationBias_ = quaternion_t(0.0, 0.0, 0.0, 1.0);
            }
        }

        return returnCode;
    }

    std::string const & ImuSensor::getFrameName(void) const
    {
        return frameName_;
    }

    int32_t const & ImuSensor::getFrameIdx(void) const
    {
        return frameIdx_;
    }

    hresult_t ImuSensor::set(float64_t const & t,
                             vectorN_t const & q,
                             vectorN_t const & v,
                             vectorN_t const & a,
                             vectorN_t const & uMotor)
    {
        if (!isInitialized_)
        {
            PRINT_ERROR("Sensor not initialized. Impossible to set sensor data.");
            return hresult_t::ERROR_INIT_FAILED;
        }

        // Compute quaternion
        matrix3_t const & rot = robot_->pncData_.oMf[frameIdx_].rotation();
        quaternion_t const quat(rot); // Convert a rotation matrix to a quaternion
        data().head<4>() = quat.coeffs(); // (x,y,z,w)

        // Compute gyroscope signal
        pinocchio::Motion const velocity = pinocchio::getFrameVelocity(robot_->pncModel_, robot_->pncData_, frameIdx_);
        data().segment<3>(4) = velocity.angular();

        // Compute accelerometer signal
        pinocchio::Motion const acceleration = pinocchio::getFrameAcceleration(robot_->pncModel_, robot_->pncData_, frameIdx_);

        // Accelerometer signal is sensor linear acceleration (not spatial acceleration !) minus gravity
        data().tail<3>() = acceleration.linear() +
                           velocity.angular().cross(velocity.linear()) -
                           quat.conjugate() * robot_->pncModel_.gravity.linear();

        return hresult_t::SUCCESS;
    }

    void ImuSensor::skewMeasurement(void)
    {
        // Add bias
        if (baseSensorOptions_->bias.size())
        {
            // Accel + gyroscope: simply add additive bias
            get().tail<6>() += baseSensorOptions_->bias.tail<6>();

            // Quaternion: interpret bias as angle-axis representation of a
            // sensor rotation bias R_b, such that w_R_sensor = w_R_imu R_b.
            get().head<4>() = (quaternion_t(get().head<4>()) * sensorRotationBias_).coeffs();

            // Apply the same bias to the accelerometer / gyroscope output.
            get().segment<3>(4) = sensorRotationBias_.conjugate() * get().segment<3>(4);
            get().tail<3>() = sensorRotationBias_.conjugate() * get().tail<3>();
        }

        // Add white noise
        if (baseSensorOptions_->noiseStd.size())
        {
            /* Quaternion: interpret noise as a random rotation vector applied
               as an extra bias to the right, i.e. w_R_sensor = w_R_imu R_noise.
               Note that R_noise = exp3(gaussian(noiseStd)): this means the
               rotation vector follows a gaussian probability law, but doesn't
               say much in general about the rotation. However in practice we
               expect the standard deviation to be small, and thus the
               approximation to be valid. */
            vector3_t const randAxis = randVectorNormal(baseSensorOptions_->noiseStd.head<3>());
            get().head<4>() = (quaternion_t(get().head<4>()) *
                               quaternion_t(pinocchio::exp3(randAxis))).coeffs();

            // Accel + gyroscope: simply apply additive noise.
            get().tail<6>() += randVectorNormal(baseSensorOptions_->noiseStd.tail<6>());
        }

    }

    // ===================== ContactSensor =========================

    template<>
    std::string const AbstractSensorTpl<ContactSensor>::type_("ContactSensor");
    template<>
    bool_t const AbstractSensorTpl<ContactSensor>::areFieldnamesGrouped_(false);
    template<>
    std::vector<std::string> const AbstractSensorTpl<ContactSensor>::fieldNames_({"FX", "FY", "FZ"});

    ContactSensor::ContactSensor(std::string const & name) :
    AbstractSensorTpl(name),
    frameName_(),
    frameIdx_(0)
    {
        // Empty.
    }

    hresult_t ContactSensor::initialize(std::string const & frameName)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (!isAttached_)
        {
            PRINT_ERROR("Sensor not attached to any robot. Impossible to initialize it.");
            returnCode = hresult_t::ERROR_GENERIC;
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            frameName_ = frameName;
            isInitialized_ = true;
            returnCode = refreshProxies();
        }

        if (returnCode != hresult_t::SUCCESS)
        {
            isInitialized_ = false;
        }

        return returnCode;
    }

    hresult_t ContactSensor::refreshProxies(void)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (!robot_->getIsInitialized())
        {
            PRINT_ERROR("Robot not initialized. Impossible to refresh model-dependent proxies.");
            returnCode = hresult_t::ERROR_INIT_FAILED;
        }

        std::vector<std::string> const & contactFramesNames = robot_->getContactFramesNames();
        auto contactFrameNameIt = std::find(contactFramesNames.begin(), contactFramesNames.end(), frameName_);
        if (contactFrameNameIt == contactFramesNames.end())
        {
            PRINT_ERROR("Sensor frame not associated with any contact point of the robot. Impossible to refresh model-dependent proxies.");
            return hresult_t::ERROR_BAD_INPUT;
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            if (!isInitialized_)
            {
                PRINT_ERROR("Sensor not initialized. Impossible to refresh model-dependent proxies.");
                returnCode = hresult_t::ERROR_INIT_FAILED;
            }
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            returnCode = ::jiminy::getFrameIdx(robot_->pncModel_, frameName_, frameIdx_);
        }

        return returnCode;
    }

    std::string const & ContactSensor::getFrameName(void) const
    {
        return frameName_;
    }

    int32_t const & ContactSensor::getFrameIdx(void) const
    {
        return frameIdx_;
    }

    hresult_t ContactSensor::set(float64_t const & t,
                                 vectorN_t const & q,
                                 vectorN_t const & v,
                                 vectorN_t const & a,
                                 vectorN_t const & uMotor)
    {
        if (!isInitialized_)
        {
            PRINT_ERROR("Sensor not initialized. Impossible to set sensor data.");
            return hresult_t::ERROR_INIT_FAILED;
        }

        std::vector<int32_t> const & contactFramesIdx = robot_->getContactFramesIdx();
        std::vector<int32_t>::const_iterator it = std::find(contactFramesIdx.begin(), contactFramesIdx.end(), frameIdx_);
        data() = robot_->contactForces_[std::distance(contactFramesIdx.begin(), it)].linear();

        return hresult_t::SUCCESS;
    }

    // ===================== ForceSensor =========================

    template<>
    std::string const AbstractSensorTpl<ForceSensor>::type_("ForceSensor");
    template<>
    bool_t const AbstractSensorTpl<ForceSensor>::areFieldnamesGrouped_(false);
    template<>
    std::vector<std::string> const AbstractSensorTpl<ForceSensor>::fieldNames_({"FX", "FY", "FZ", "MX", "MY", "MZ"});

    ForceSensor::ForceSensor(std::string const & name) :
    AbstractSensorTpl(name),
    frameName_(),
    frameIdx_(0),
    parentJointIdx_(0)
    {
        // Empty.
    }

    hresult_t ForceSensor::initialize(std::string const & frameName)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (!isAttached_)
        {
            PRINT_ERROR("Sensor not attached to any robot. Impossible to initialize it.");
            returnCode = hresult_t::ERROR_GENERIC;
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            frameName_ = frameName;
            isInitialized_ = true;
            returnCode = refreshProxies();
        }

        if (returnCode != hresult_t::SUCCESS)
        {
            isInitialized_ = false;
        }

        return returnCode;
    }

    hresult_t ForceSensor::refreshProxies(void)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (!robot_->getIsInitialized())
        {
            PRINT_ERROR("Robot not initialized. Impossible to refresh model-dependent proxies.");
            returnCode = hresult_t::ERROR_INIT_FAILED;
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            if (!isInitialized_)
            {
                PRINT_ERROR("Sensor not initialized. Impossible to refresh model-dependent proxies.");
                returnCode = hresult_t::ERROR_INIT_FAILED;
            }
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            returnCode = ::jiminy::getFrameIdx(robot_->pncModel_, frameName_, frameIdx_);
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            parentJointIdx_ = robot_->pncModel_.frames[frameIdx_].parent;  // parent always returns the parent joint
        }

        return returnCode;
    }

    std::string const & ForceSensor::getFrameName(void) const
    {
        return frameName_;
    }

    int32_t const & ForceSensor::getFrameIdx(void) const
    {
        return frameIdx_;
    }

    int32_t ForceSensor::getJointIdx(void) const
    {
        return parentJointIdx_;
    }

    hresult_t ForceSensor::set(float64_t const & t,
                               vectorN_t const & q,
                               vectorN_t const & v,
                               vectorN_t const & a,
                               vectorN_t const & uMotor)
    {
        // Returns the force applied at frame location, in the local frame of the parent joint

        if (!isInitialized_)
        {
            PRINT_ERROR("Sensor not initialized. Impossible to set sensor data.");
            return hresult_t::ERROR_INIT_FAILED;
        }

        pinocchio::SE3 const & framePlacement = robot_->pncModel_.frames[frameIdx_].placement;
        pinocchio::Force f = framePlacement.actInv(robot_->pncData_.f[parentJointIdx_]);  // f is in the local frame of the joint
        data() = f.toVector();

        return hresult_t::SUCCESS;
    }

    // ===================== EncoderSensor =========================

    template<>
    std::string const AbstractSensorTpl<EncoderSensor>::type_("EncoderSensor");
    template<>
    bool_t const AbstractSensorTpl<EncoderSensor>::areFieldnamesGrouped_(true);
    template<>
    std::vector<std::string> const AbstractSensorTpl<EncoderSensor>::fieldNames_({"Q", "V"});

    EncoderSensor::EncoderSensor(std::string const & name) :
    AbstractSensorTpl(name),
    jointName_(),
    jointIdx_(0),
    jointType_(joint_t::NONE)
    {
        // Empty.
    }

    hresult_t EncoderSensor::initialize(std::string const & jointName)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (!isAttached_)
        {
            PRINT_ERROR("Sensor not attached to any robot. Impossible to initialize it.");
            returnCode = hresult_t::ERROR_GENERIC;
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            jointName_ = jointName;
            isInitialized_ = true;
            returnCode = refreshProxies();
        }

        if (returnCode != hresult_t::SUCCESS)
        {
            isInitialized_ = false;
        }

        return returnCode;
    }

    hresult_t EncoderSensor::refreshProxies(void)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (!robot_->getIsInitialized())
        {
            PRINT_ERROR("Robot not initialized. Impossible to refresh model-dependent proxies.");
            returnCode = hresult_t::ERROR_INIT_FAILED;
        }

        if (!isInitialized_)
        {
            PRINT_ERROR("Sensor not initialized. Impossible to refresh model-dependent proxies.");
            returnCode = hresult_t::ERROR_INIT_FAILED;
        }

        if (!robot_->pncModel_.existJointName(jointName_))
        {
            PRINT_ERROR("Sensor attached to a joint that does not exist.");
            returnCode = hresult_t::ERROR_INIT_FAILED;
        }

        jointIdx_ = robot_->pncModel_.getJointId(jointName_);

        getJointTypeFromIdx(robot_->pncModel_, jointIdx_, jointType_);  // No need to catch the return flag, since the joint type will be NONE anyway

        // Motors are only supported for linear and rotary joints
        if (jointType_ != joint_t::LINEAR && jointType_ != joint_t::ROTARY && jointType_ != joint_t::ROTARY_UNBOUNDED)
        {
            PRINT_ERROR("An encoder sensor can only be associated with a 1-dof linear or rotary joint.");
            returnCode =  hresult_t::ERROR_BAD_INPUT;
        }

        return returnCode;
    }

    std::string const & EncoderSensor::getJointName(void) const
    {
        return jointName_;
    }

    int32_t const & EncoderSensor::getJointIdx(void) const
    {
        return jointIdx_;
    }

    joint_t const & EncoderSensor::getJointType(void) const
    {
        return jointType_;
    }

    hresult_t EncoderSensor::set(float64_t const & t,
                                 vectorN_t const & q,
                                 vectorN_t const & v,
                                 vectorN_t const & a,
                                 vectorN_t const & uMotor)
    {
        if (!isInitialized_)
        {
            PRINT_ERROR("Sensor not initialized. Impossible to set sensor data.");
            return hresult_t::ERROR_INIT_FAILED;
        }

        auto const & joint = robot_->pncModel_.joints[jointIdx_];
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
    std::vector<std::string> const AbstractSensorTpl<EffortSensor>::fieldNames_({"U"});

    EffortSensor::EffortSensor(std::string const & name) :
    AbstractSensorTpl(name),
    motorName_(),
    motorIdx_(0)
    {
        // Empty.
    }

    hresult_t EffortSensor::initialize(std::string const & motorName)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (!isAttached_)
        {
            PRINT_ERROR("Sensor not attached to any robot. Impossible to initialize it.");
            returnCode = hresult_t::ERROR_GENERIC;
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            motorName_ = motorName;
            isInitialized_ = true;
            returnCode = refreshProxies();
        }

        if (returnCode != hresult_t::SUCCESS)
        {
            isInitialized_ = false;
        }

        return returnCode;
    }

    hresult_t EffortSensor::refreshProxies(void)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (!robot_->getIsInitialized())
        {
            PRINT_ERROR("Robot not initialized. Impossible to refresh model-dependent proxies.");
            returnCode = hresult_t::ERROR_INIT_FAILED;
        }

        if (!isInitialized_)
        {
            PRINT_ERROR("Sensor not initialized. Impossible to refresh model-dependent proxies.");
            returnCode = hresult_t::ERROR_INIT_FAILED;
        }

        AbstractMotorBase const * motor;
        if (returnCode == hresult_t::SUCCESS)
        {
            returnCode = robot_->getMotor(motorName_, motor);
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            motorIdx_ = motor->getIdx();
        }

        return returnCode;
    }

    std::string const & EffortSensor::getMotorName(void) const
    {
        return motorName_;
    }

    int32_t const & EffortSensor::getMotorIdx(void) const
    {
        return motorIdx_;
    }

    hresult_t EffortSensor::set(float64_t const & t,
                                vectorN_t const & q,
                                vectorN_t const & v,
                                vectorN_t const & a,
                                vectorN_t const & uMotor)
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
