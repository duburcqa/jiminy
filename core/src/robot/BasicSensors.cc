#include <algorithm>

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
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            returnCode = refreshProxies();
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
            returnCode = getFrameIdx(robot_->pncModel_, frameName_, frameIdx_);
        }

        return returnCode;
    }

    std::string const & ImuSensor::getFrameName(void) const
    {
        return frameName_;
    }

    hresult_t ImuSensor::set(float64_t const & t,
                             vectorN_t const & q,
                             vectorN_t const & v,
                             vectorN_t const & a,
                             vectorN_t const & uMotor)
    {
        if (!isInitialized_)
        {
            std::cout << "Error - ImuSensor::set - Sensor not initialized. Impossible to set sensor data." << std::endl;
            return hresult_t::ERROR_INIT_FAILED;
        }

        matrix3_t const & rot = robot_->pncData_.oMf[frameIdx_].rotation();
        quaternion_t const quat(rot); // Convert a rotation matrix to a quaternion
        data().head<4>() = quat.coeffs(); // (x,y,z,w)
        pinocchio::Motion const gyroIMU = pinocchio::getFrameVelocity(robot_->pncModel_, robot_->pncData_, frameIdx_);
        data().segment<3>(4) = gyroIMU.angular();
        pinocchio::Motion const acceleration = pinocchio::getFrameAcceleration(robot_->pncModel_, robot_->pncData_, frameIdx_);
        data().tail<3>() = acceleration.linear() + quat.conjugate() * robot_->pncModel_.gravity.linear();

        return hresult_t::SUCCESS;
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
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            returnCode = refreshProxies();
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
            returnCode = getFrameIdx(robot_->pncModel_, frameName_, frameIdx_);
        }

        return returnCode;
    }

    std::string const & ForceSensor::getFrameName(void) const
    {
        return frameName_;
    }

    hresult_t ForceSensor::set(float64_t const & t,
                               vectorN_t const & q,
                               vectorN_t const & v,
                               vectorN_t const & a,
                               vectorN_t const & uMotor)
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
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            returnCode = refreshProxies();
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
            returnCode = getJointPositionIdx(robot_->pncModel_, jointName_, jointPositionIdx_);
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            getJointVelocityIdx(robot_->pncModel_, jointName_, jointVelocityIdx_);
        }

        return returnCode;
    }

    std::string const & EncoderSensor::getJointName(void) const
    {
        return jointName_;
    }

    hresult_t EncoderSensor::set(float64_t const & t,
                                 vectorN_t const & q,
                                 vectorN_t const & v,
                                 vectorN_t const & a,
                                 vectorN_t const & uMotor)
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

    // ===================== TorqueSensor =========================

    template<>
    std::string const AbstractSensorTpl<TorqueSensor>::type_("TorqueSensor");
    template<>
    bool_t const AbstractSensorTpl<TorqueSensor>::areFieldnamesGrouped_(true);
    template<>
    std::vector<std::string> const AbstractSensorTpl<TorqueSensor>::fieldNames_({"U"});

    TorqueSensor::TorqueSensor(std::string const & name) :
    AbstractSensorTpl(name),
    motorName_(),
    motorIdx_(0)
    {
        // Empty.
    }

    hresult_t TorqueSensor::initialize(std::string const & motorName)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (!isAttached_)
        {
            std::cout << "Error - TorqueSensor::initialize - Sensor not attached to any robot. Impossible to initialize it." << std::endl;
            returnCode = hresult_t::ERROR_GENERIC;
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            motorName_ = motorName;
            isInitialized_ = true;
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            returnCode = refreshProxies();
        }

        return returnCode;
    }

    hresult_t TorqueSensor::refreshProxies(void)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (!robot_->getIsInitialized())
        {
            std::cout << "Error - TorqueSensor::refreshProxies - Robot not initialized. Impossible to refresh model-dependent proxies." << std::endl;
            returnCode = hresult_t::ERROR_INIT_FAILED;
        }

        if (!isInitialized_)
        {
            std::cout << "Error - TorqueSensor::refreshProxies - Sensor not initialized. Impossible to refresh model-dependent proxies." << std::endl;
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

    std::string const & TorqueSensor::getMotorName(void) const
    {
        return motorName_;
    }

    hresult_t TorqueSensor::set(float64_t const & t,
                                vectorN_t const & q,
                                vectorN_t const & v,
                                vectorN_t const & a,
                                vectorN_t const & uMotor)
    {
        if (!isInitialized_)
        {
            std::cout << "Error - TorqueSensor::set - Sensor not initialized. Impossible to set sensor data." << std::endl;
            return hresult_t::ERROR_INIT_FAILED;
        }

        data()[0] = uMotor[motorIdx_];

        return hresult_t::SUCCESS;
    }
}
