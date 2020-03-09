#include <algorithm>

#include "pinocchio/multibody/model.hpp"
#include "pinocchio/algorithm/frames.hpp"

#include "jiminy/core/Model.h"
#include "jiminy/core/Utilities.h"
#include "jiminy/core/BasicSensors.h"


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

    ImuSensor::ImuSensor(Model       const & model,
                         std::shared_ptr<SensorSharedDataHolder_t> const & sharedHolder,
                         std::string const & name) :
    AbstractSensorTpl(model, sharedHolder, name),
    frameName_(),
    frameIdx_()
    {
        // Empty.
    }

    result_t ImuSensor::initialize(std::string const & frameName)
    {
        result_t returnCode = result_t::SUCCESS;

        frameName_ = frameName;
        returnCode = refreshProxies();

        if (returnCode == result_t::SUCCESS)
        {
            isInitialized_ = true;
        }

        return returnCode;
    }

    result_t ImuSensor::refreshProxies(void)
    {
        result_t returnCode = result_t::SUCCESS;

        if (model_->getIsInitialized())
        {
            std::cout << "Error - ImuSensor::refreshProxies - Model not initialized. Impossible to refresh model-dependent proxies." << std::endl;
            returnCode = result_t::ERROR_INIT_FAILED;
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
                            vectorN_t const & u)
    {
        if (!isInitialized_)
        {
            std::cout << "Error - ImuSensor::set - Sensor not initialized. Impossible to set sensor data." << std::endl;
            return result_t::ERROR_INIT_FAILED;
        }

        matrix3_t const & rot = model_->pncData_.oMf[frameIdx_].rotation();
        quaternion_t const quat(rot); // Convert a rotation matrix to a quaternion
        data().head(4) = quat.coeffs(); // (x,y,z,w)
        pinocchio::Motion const gyroIMU = pinocchio::getFrameVelocity(model_->pncModel_, model_->pncData_, frameIdx_);
        data().segment<3>(4) = gyroIMU.angular();
        pinocchio::Motion const acceleration = pinocchio::getFrameAcceleration(model_->pncModel_, model_->pncData_, frameIdx_);
        data().tail(3) = acceleration.linear() + quat.conjugate() * model_->pncModel_.gravity.linear();

        return result_t::SUCCESS;
    }

    // ===================== ForceSensor =========================

    template<>
    std::string const AbstractSensorTpl<ForceSensor>::type_("ForceSensor");
    template<>
    bool_t const AbstractSensorTpl<ForceSensor>::areFieldNamesGrouped_(false);
    template<>
    std::vector<std::string> const AbstractSensorTpl<ForceSensor>::fieldNames_({"FX", "FY", "FZ"});

    ForceSensor::ForceSensor(Model       const & model,
                             std::shared_ptr<SensorSharedDataHolder_t> const & sharedHolder,
                             std::string const & name) :
    AbstractSensorTpl(model, sharedHolder, name),
    frameName_(),
    frameIdx_()
    {
        // Empty.
    }

    result_t ForceSensor::initialize(std::string const & frameName)
    {
        result_t returnCode = result_t::SUCCESS;

        frameName_ = frameName;
        returnCode = refreshProxies();

        if (returnCode == result_t::SUCCESS)
        {
            isInitialized_ = true;
        }

        return returnCode;
    }

    result_t ForceSensor::refreshProxies(void)
    {
        result_t returnCode = result_t::SUCCESS;

        if (model_->getIsInitialized())
        {
            std::cout << "Error - ForceSensor::refreshProxies - Model not initialized. Impossible to refresh model-dependent proxies." << std::endl;
            returnCode = result_t::ERROR_INIT_FAILED;
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
                              vectorN_t const & u)
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

    EncoderSensor::EncoderSensor(Model       const & model,
                                 std::shared_ptr<SensorSharedDataHolder_t> const & sharedHolder,
                                 std::string const & name) :
    AbstractSensorTpl(model, sharedHolder, name),
    jointName_(),
    jointPositionIdx_(),
    jointVelocityIdx_()
    {
        // Empty.
    }

    result_t EncoderSensor::initialize(std::string const & jointName)
    {
        result_t returnCode = result_t::SUCCESS;

        jointName_ = jointName;
        returnCode = refreshProxies();

        if (returnCode == result_t::SUCCESS)
        {
            isInitialized_ = true;
        }

        return returnCode;
    }

    result_t EncoderSensor::refreshProxies(void)
    {
        result_t returnCode = result_t::SUCCESS;

        if (model_->getIsInitialized())
        {
            std::cout << "Error - EncoderSensor::refreshProxies - Model not initialized. Impossible to refresh model-dependent proxies." << std::endl;
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
                                vectorN_t const & u)
    {
        if (!isInitialized_)
        {
            std::cout << "Error - EncoderSensor::set - Sensor not initialized. Impossible to set sensor data." << std::endl;
            return result_t::ERROR_INIT_FAILED;
        }

        data().head(1) = q.segment<1>(jointPositionIdx_);
        data().tail(1) = v.segment<1>(jointVelocityIdx_);

        return result_t::SUCCESS;
    }
}
