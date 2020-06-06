#include <algorithm>

#include "pinocchio/spatial/explog.hpp"
#include "pinocchio/multibody/model.hpp"
#include "pinocchio/algorithm/frames.hpp"

#include "jiminy/core/robot/Robot.h"
#include "jiminy/core/Utilities.h"

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
    frameIdx_(0)
    {
        // Empty.
    }

    hresult_t ImuSensor::initialize(std::string const & frameName)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (!isAttached_)
        {
            std::cout << "Error - ImuSensor::initialize - Sensor not attached to any robot. Impossible to initialize it." << std::endl;
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

    hresult_t ImuSensor::refreshProxies(void)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (!robot_->getIsInitialized())
        {
            std::cout << "Error - ImuSensor::refreshProxies - Robot not initialized. Impossible to refresh model-dependent proxies." << std::endl;
            returnCode = hresult_t::ERROR_INIT_FAILED;
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            if (!isInitialized_)
            {
                std::cout << "Error - ImuSensor::refreshProxies - Sensor not initialized. Impossible to refresh model-dependent proxies." << std::endl;
                returnCode = hresult_t::ERROR_INIT_FAILED;
            }
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            returnCode = ::jiminy::getFrameIdx(robot_->pncModel_, frameName_, frameIdx_);
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

    hresult_t ImuSensor::set(float64_t                   const & t,
                             Eigen::Ref<vectorN_t const> const & q,
                             Eigen::Ref<vectorN_t const> const & v,
                             Eigen::Ref<vectorN_t const> const & a,
                             vectorN_t                   const & uMotor)
    {
        if (!isInitialized_)
        {
            std::cout << "Error - ImuSensor::set - Sensor not initialized. Impossible to set sensor data." << std::endl;
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
            /* Quaternion: interpret bias as angle-axis representation of a
               sensor rotation bias R_b, such that w_R_sensor = w_R_imu R_b. */
            get().head<4>() = (quaternion_t(get().head<4>()) *
                               quaternion_t(pinocchio::exp3(baseSensorOptions_->bias.head<3>()))).coeffs();

            // Accel + gyroscope: simply add additive bias.
            get().tail<6>() += baseSensorOptions_->bias.tail<6>();
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
            get().head<4>() = (quaternion_t(get().head<4>()) *
                               quaternion_t(pinocchio::exp3(randVectorNormal(baseSensorOptions_->noiseStd.head<3>())))).coeffs();

            // Accel + gyroscope: simply apply additive noise.
            get().tail<6>() += randVectorNormal(baseSensorOptions_->noiseStd.tail<6>());
        }

    }

    // ===================== ForceSensor =========================

    template<>
    std::string const AbstractSensorTpl<ForceSensor>::type_("ForceSensor");
    template<>
    bool_t const AbstractSensorTpl<ForceSensor>::areFieldnamesGrouped_(false);
    template<>
    std::vector<std::string> const AbstractSensorTpl<ForceSensor>::fieldNames_({"FX", "FY", "FZ"});

    ForceSensor::ForceSensor(std::string const & name) :
    AbstractSensorTpl(name),
    frameName_(),
    frameIdx_(0)
    {
        // Empty.
    }

    hresult_t ForceSensor::initialize(std::string const & frameName)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (!isAttached_)
        {
            std::cout << "Error - ForceSensor::initialize - Sensor not attached to any robot. Impossible to initialize it." << std::endl;
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
            std::cout << "Error - ForceSensor::refreshProxies - Robot not initialized. Impossible to refresh model-dependent proxies." << std::endl;
            returnCode = hresult_t::ERROR_INIT_FAILED;
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            if (!isInitialized_)
            {
                std::cout << "Error - ForceSensor::refreshProxies - Sensor not initialized. Impossible to refresh model-dependent proxies." << std::endl;
                returnCode = hresult_t::ERROR_INIT_FAILED;
            }
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            returnCode = ::jiminy::getFrameIdx(robot_->pncModel_, frameName_, frameIdx_);
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

    hresult_t ForceSensor::set(float64_t                   const & t,
                               Eigen::Ref<vectorN_t const> const & q,
                               Eigen::Ref<vectorN_t const> const & v,
                               Eigen::Ref<vectorN_t const> const & a,
                               vectorN_t                   const & uMotor)
    {
        if (!isInitialized_)
        {
            std::cout << "Error - ForceSensor::set - Sensor not initialized. Impossible to set sensor data." << std::endl;
            return hresult_t::ERROR_INIT_FAILED;
        }

        std::vector<int32_t> const & contactFramesIdx = robot_->getContactFramesIdx();
        std::vector<int32_t>::const_iterator it = std::find(contactFramesIdx.begin(), contactFramesIdx.end(), frameIdx_);
        data() = robot_->contactForces_[std::distance(contactFramesIdx.begin(), it)].linear();

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
    jointPositionIdx_(0),
    jointVelocityIdx_(0)
    {
        // Empty.
    }

    hresult_t EncoderSensor::initialize(std::string const & jointName)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (!isAttached_)
        {
            std::cout << "Error - EncoderSensor::initialize - Sensor not attached to any robot. Impossible to initialize it." << std::endl;
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
            std::cout << "Error - EncoderSensor::refreshProxies - Robot not initialized. Impossible to refresh model-dependent proxies." << std::endl;
            returnCode = hresult_t::ERROR_INIT_FAILED;
        }

        if (!isInitialized_)
        {
            std::cout << "Error - EncoderSensor::refreshProxies - Sensor not initialized. Impossible to refresh model-dependent proxies." << std::endl;
            returnCode = hresult_t::ERROR_INIT_FAILED;
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            returnCode = ::jiminy::getJointPositionIdx(robot_->pncModel_, jointName_, jointPositionIdx_);
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            ::jiminy::getJointVelocityIdx(robot_->pncModel_, jointName_, jointVelocityIdx_);
        }

        return returnCode;
    }

    std::string const & EncoderSensor::getJointName(void) const
    {
        return jointName_;
    }

    int32_t const & EncoderSensor::getJointPositionIdx(void)  const
    {

        return jointPositionIdx_;
    }
    int32_t const & EncoderSensor::getJointVelocityIdx(void)  const
    {
        return jointVelocityIdx_;
    }

    hresult_t EncoderSensor::set(float64_t                   const & t,
                                 Eigen::Ref<vectorN_t const> const & q,
                                 Eigen::Ref<vectorN_t const> const & v,
                                 Eigen::Ref<vectorN_t const> const & a,
                                 vectorN_t                   const & uMotor)
    {
        if (!isInitialized_)
        {
            std::cout << "Error - EncoderSensor::set - Sensor not initialized. Impossible to set sensor data." << std::endl;
            return hresult_t::ERROR_INIT_FAILED;
        }

        data()[0] = q[jointPositionIdx_];
        data()[1] = v[jointVelocityIdx_];

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
            std::cout << "Error - EffortSensor::initialize - Sensor not attached to any robot. Impossible to initialize it." << std::endl;
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
            std::cout << "Error - EffortSensor::refreshProxies - Robot not initialized. Impossible to refresh model-dependent proxies." << std::endl;
            returnCode = hresult_t::ERROR_INIT_FAILED;
        }

        if (!isInitialized_)
        {
            std::cout << "Error - EffortSensor::refreshProxies - Sensor not initialized. Impossible to refresh model-dependent proxies." << std::endl;
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

    hresult_t EffortSensor::set(float64_t                   const & t,
                                Eigen::Ref<vectorN_t const> const & q,
                                Eigen::Ref<vectorN_t const> const & v,
                                Eigen::Ref<vectorN_t const> const & a,
                                vectorN_t                   const & uMotor)
    {
        if (!isInitialized_)
        {
            std::cout << "Error - EffortSensor::set - Sensor not initialized. Impossible to set sensor data." << std::endl;
            return hresult_t::ERROR_INIT_FAILED;
        }

        data()[0] = uMotor[motorIdx_];

        return hresult_t::SUCCESS;
    }
}
