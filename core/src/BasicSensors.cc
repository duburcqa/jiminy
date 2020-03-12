#include <algorithm>

#include "pinocchio/multibody/model.hpp"
#include "pinocchio/algorithm/frames.hpp"

#include "jiminy/core/Model.h"
#include "jiminy/core/BasicMotors.h"
#include "jiminy/core/BasicSensors.h"
#include "jiminy/core/Utilities.h"


namespace jiminy
{
    // ===================== ImuSensor =========================

    template<>
    std::string const AbstractSensorTpl<ImuSensor>::type_("ImuSensor");
    template<>
    bool_t const AbstractSensorTpl<ImuSensor>::areFieldNamesGrouped_(false);
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

    result_t ImuSensor::initialize(std::string const & frameName)
    {
        result_t returnCode = result_t::SUCCESS;

        if (!isAttached_)
        {
            std::cout << "Error - ImuSensor::initialize - Sensor not attached to any model. Impossible to initialize it." << std::endl;
            returnCode = result_t::ERROR_GENERIC;
        }

        if (returnCode == result_t::SUCCESS)
        {
            frameName_ = frameName;
            isInitialized_ = true;
        }

        if (returnCode == result_t::SUCCESS)
        {
            returnCode = refreshProxies();
        }

        return returnCode;
    }

    result_t ImuSensor::refreshProxies(void)
    {
        result_t returnCode = result_t::SUCCESS;

        if (!model_->getIsInitialized())
        {
            std::cout << "Error - ImuSensor::refreshProxies - Model not initialized. Impossible to refresh model-dependent proxies." << std::endl;
            returnCode = result_t::ERROR_INIT_FAILED;
        }

        if (returnCode == result_t::SUCCESS)
        {
            if (!isInitialized_)
            {
                std::cout << "Error - ImuSensor::refreshProxies - Sensor not initialized. Impossible to refresh model-dependent proxies." << std::endl;
                returnCode = result_t::ERROR_INIT_FAILED;
            }
        }

        if (returnCode == result_t::SUCCESS)
        {
            returnCode = getFrameIdx(model_->pncModel_, frameName_, frameIdx_);
        }

        return returnCode;
    }

    std::string const & ImuSensor::getFrameName(void) const
    {
        return frameName_;
    }

    result_t ImuSensor::set(float64_t const & t,
                            vectorN_t const & q,
                            vectorN_t const & v,
                            vectorN_t const & a,
                            vectorN_t const & uMotor)
    {
        if (!isInitialized_)
        {
            std::cout << "Error - ImuSensor::set - Sensor not initialized. Impossible to set sensor data." << std::endl;
            return result_t::ERROR_INIT_FAILED;
        }

        matrix3_t const & rot = model_->pncData_.oMf[frameIdx_].rotation();
        quaternion_t const quat(rot); // Convert a rotation matrix to a quaternion
        data().head<4>() = quat.coeffs(); // (x,y,z,w)
        pinocchio::Motion const gyroIMU = pinocchio::getFrameVelocity(model_->pncModel_, model_->pncData_, frameIdx_);
        data().segment<3>(4) = gyroIMU.angular();
        pinocchio::Motion const acceleration = pinocchio::getFrameAcceleration(model_->pncModel_, model_->pncData_, frameIdx_);
        data().tail<3>() = acceleration.linear() + quat.conjugate() * model_->pncModel_.gravity.linear();

        return result_t::SUCCESS;
    }

    // ===================== ForceSensor =========================

    template<>
    std::string const AbstractSensorTpl<ForceSensor>::type_("ForceSensor");
    template<>
    bool_t const AbstractSensorTpl<ForceSensor>::areFieldNamesGrouped_(false);
    template<>
    std::vector<std::string> const AbstractSensorTpl<ForceSensor>::fieldNames_({"FX", "FY", "FZ"});

    ForceSensor::ForceSensor(std::string const & name) :
    AbstractSensorTpl(name),
    frameName_(),
    frameIdx_(0)
    {
        // Empty.
    }

    result_t ForceSensor::initialize(std::string const & frameName)
    {
        result_t returnCode = result_t::SUCCESS;

        if (!isAttached_)
        {
            std::cout << "Error - ForceSensor::initialize - Sensor not attached to any model. Impossible to initialize it." << std::endl;
            returnCode = result_t::ERROR_GENERIC;
        }

        if (returnCode == result_t::SUCCESS)
        {
            frameName_ = frameName;
            isInitialized_ = true;
        }

        if (returnCode == result_t::SUCCESS)
        {
            returnCode = refreshProxies();
        }

        return returnCode;
    }

    result_t ForceSensor::refreshProxies(void)
    {
        result_t returnCode = result_t::SUCCESS;

        if (!model_->getIsInitialized())
        {
            std::cout << "Error - ForceSensor::refreshProxies - Model not initialized. Impossible to refresh model-dependent proxies." << std::endl;
            returnCode = result_t::ERROR_INIT_FAILED;
        }

        if (returnCode == result_t::SUCCESS)
        {
            if (!isInitialized_)
            {
                std::cout << "Error - ForceSensor::refreshProxies - Sensor not initialized. Impossible to refresh model-dependent proxies." << std::endl;
                returnCode = result_t::ERROR_INIT_FAILED;
            }
        }

        if (returnCode == result_t::SUCCESS)
        {
            returnCode = getFrameIdx(model_->pncModel_, frameName_, frameIdx_);
        }

        return returnCode;
    }

    std::string const & ForceSensor::getFrameName(void) const
    {
        return frameName_;
    }

    result_t ForceSensor::set(float64_t const & t,
                              vectorN_t const & q,
                              vectorN_t const & v,
                              vectorN_t const & a,
                              vectorN_t const & uMotor)
    {
        if (!isInitialized_)
        {
            std::cout << "Error - ForceSensor::set - Sensor not initialized. Impossible to set sensor data." << std::endl;
            return result_t::ERROR_INIT_FAILED;
        }

        std::vector<int32_t> const & contactFramesIdx = model_->getContactFramesIdx();
        std::vector<int32_t>::const_iterator it = std::find(contactFramesIdx.begin(), contactFramesIdx.end(), frameIdx_);
        data() = model_->contactForces_[std::distance(contactFramesIdx.begin(), it)].linear();

        return result_t::SUCCESS;
    }

    // ===================== EncoderSensor =========================

    template<>
    std::string const AbstractSensorTpl<EncoderSensor>::type_("EncoderSensor");
    template<>
    bool_t const AbstractSensorTpl<EncoderSensor>::areFieldNamesGrouped_(true);
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

    result_t EncoderSensor::initialize(std::string const & jointName)
    {
        result_t returnCode = result_t::SUCCESS;

        if (!isAttached_)
        {
            std::cout << "Error - EncoderSensor::initialize - Sensor not attached to any model. Impossible to initialize it." << std::endl;
            returnCode = result_t::ERROR_GENERIC;
        }

        if (returnCode == result_t::SUCCESS)
        {
            jointName_ = jointName;
            isInitialized_ = true;
        }

        if (returnCode == result_t::SUCCESS)
        {
            returnCode = refreshProxies();
        }

        return returnCode;
    }

    result_t EncoderSensor::refreshProxies(void)
    {
        result_t returnCode = result_t::SUCCESS;

        if (!model_->getIsInitialized())
        {
            std::cout << "Error - EncoderSensor::refreshProxies - Model not initialized. Impossible to refresh model-dependent proxies." << std::endl;
            returnCode = result_t::ERROR_INIT_FAILED;
        }

        if (!isInitialized_)
        {
            std::cout << "Error - EncoderSensor::refreshProxies - Sensor not initialized. Impossible to refresh model-dependent proxies." << std::endl;
            returnCode = result_t::ERROR_INIT_FAILED;
        }

        if (returnCode == result_t::SUCCESS)
        {
            returnCode = getJointPositionIdx(model_->pncModel_, jointName_, jointPositionIdx_);
        }

        if (returnCode == result_t::SUCCESS)
        {
            getJointVelocityIdx(model_->pncModel_, jointName_, jointVelocityIdx_);
        }

        return returnCode;
    }

    std::string const & EncoderSensor::getJointName(void) const
    {
        return jointName_;
    }

    result_t EncoderSensor::set(float64_t const & t,
                                vectorN_t const & q,
                                vectorN_t const & v,
                                vectorN_t const & a,
                                vectorN_t const & uMotor)
    {
        if (!isInitialized_)
        {
            std::cout << "Error - EncoderSensor::set - Sensor not initialized. Impossible to set sensor data." << std::endl;
            return result_t::ERROR_INIT_FAILED;
        }

        data()[0] = q[jointPositionIdx_];
        data()[1] = v[jointVelocityIdx_];

        return result_t::SUCCESS;
    }

    // ===================== TorqueSensor =========================

    template<>
    std::string const AbstractSensorTpl<TorqueSensor>::type_("TorqueSensor");
    template<>
    bool_t const AbstractSensorTpl<TorqueSensor>::areFieldNamesGrouped_(true);
    template<>
    std::vector<std::string> const AbstractSensorTpl<TorqueSensor>::fieldNames_({"U"});

    TorqueSensor::TorqueSensor(std::string const & name) :
    AbstractSensorTpl(name),
    motorName_(),
    motorIdx_(0)
    {
        // Empty.
    }

    result_t TorqueSensor::initialize(std::string const & motorName)
    {
        result_t returnCode = result_t::SUCCESS;

        if (!isAttached_)
        {
            std::cout << "Error - TorqueSensor::initialize - Sensor not attached to any model. Impossible to initialize it." << std::endl;
            returnCode = result_t::ERROR_GENERIC;
        }

        if (returnCode == result_t::SUCCESS)
        {
            motorName_ = motorName;
            isInitialized_ = true;
        }

        if (returnCode == result_t::SUCCESS)
        {
            returnCode = refreshProxies();
        }

        return returnCode;
    }

    result_t TorqueSensor::refreshProxies(void)
    {
        result_t returnCode = result_t::SUCCESS;

        if (!model_->getIsInitialized())
        {
            std::cout << "Error - TorqueSensor::refreshProxies - Model not initialized. Impossible to refresh model-dependent proxies." << std::endl;
            returnCode = result_t::ERROR_INIT_FAILED;
        }

        if (!isInitialized_)
        {
            std::cout << "Error - TorqueSensor::refreshProxies - Sensor not initialized. Impossible to refresh model-dependent proxies." << std::endl;
            returnCode = result_t::ERROR_INIT_FAILED;
        }

        std::shared_ptr<AbstractMotorBase const> motor;
        if (returnCode == result_t::SUCCESS)
        {
            returnCode = model_->getMotor(motorName_, motor);
        }

        if (returnCode == result_t::SUCCESS)
        {
            motorIdx_ = motor->getId();
        }

        return returnCode;
    }

    std::string const & TorqueSensor::getMotorName(void) const
    {
        return motorName_;
    }

    result_t TorqueSensor::set(float64_t const & t,
                               vectorN_t const & q,
                               vectorN_t const & v,
                               vectorN_t const & a,
                               vectorN_t const & uMotor)
    {
        if (!isInitialized_)
        {
            std::cout << "Error - TorqueSensor::set - Sensor not initialized. Impossible to set sensor data." << std::endl;
            return result_t::ERROR_INIT_FAILED;
        }

        data()[0] = uMotor[motorIdx_];

        return result_t::SUCCESS;
    }
}
