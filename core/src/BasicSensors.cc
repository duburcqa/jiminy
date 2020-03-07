#include <algorithm>

#include "pinocchio/multibody/model.hpp"
#include "pinocchio/algorithm/frames.hpp"

#include "jiminy/core/Utilities.h"
#include "jiminy/core/Model.h"
#include "jiminy/core/BasicSensors.h"


namespace jiminy
{
    // ===================== ImuSensor =========================

    template<>
    std::string const AbstractSensorTpl<ImuSensor>::type_("ImuSensor");
    template<>
    bool const AbstractSensorTpl<ImuSensor>::areFieldNamesGrouped_(false);
    template<>
    std::vector<std::string> const AbstractSensorTpl<ImuSensor>::fieldNames_(
        {"Quatx", "Quaty", "Quatz", "Quatw", "Gyrox", "Gyroy", "Gyroz", "Accelx", "Accely", "Accelz"});

    ImuSensor::ImuSensor(Model       const & model,
                         std::shared_ptr<SensorSharedDataHolder_t> const & dataHolder,
                         std::string const & name) :
    AbstractSensorTpl(model, dataHolder, name),
    frameName_(),
    frameIdx_()
    {
        // Empty.
    }

    result_t ImuSensor::initialize(std::string const & frameName)
    {
        result_t returnCode = result_t::SUCCESS;

        frameName_ = frameName;
        returnCode = getFrameIdx(model_->pncModel_, frameName_, frameIdx_);

        if (returnCode == result_t::SUCCESS)
        {
            isInitialized_ = true;
        }

        return returnCode;
    }

    void ImuSensor::reset(void)
    {
        AbstractSensorTpl<ImuSensor>::reset();
        getFrameIdx(model_->pncModel_, frameName_, frameIdx_);
    }

    std::string ImuSensor::getFrameName(void) const
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
    bool const AbstractSensorTpl<ForceSensor>::areFieldNamesGrouped_(false);
    template<>
    std::vector<std::string> const AbstractSensorTpl<ForceSensor>::fieldNames_({"FX", "FY", "FZ"});

    ForceSensor::ForceSensor(Model       const & model,
                             std::shared_ptr<SensorSharedDataHolder_t> const & dataHolder,
                             std::string const & name) :
    AbstractSensorTpl(model, dataHolder, name),
    frameName_(),
    frameIdx_()
    {
        // Empty.
    }

    result_t ForceSensor::initialize(std::string const & frameName)
    {
        result_t returnCode = result_t::SUCCESS;

        frameName_ = frameName;
        returnCode = getFrameIdx(model_->pncModel_, frameName_, frameIdx_);

        if (returnCode == result_t::SUCCESS)
        {
            isInitialized_ = true;
        }

        return returnCode;
    }

    void ForceSensor::reset(void)
    {
        AbstractSensorTpl<ForceSensor>::reset();
        getFrameIdx(model_->pncModel_, frameName_, frameIdx_);
    }

    std::string ForceSensor::getFrameName(void) const
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
    bool const AbstractSensorTpl<EncoderSensor>::areFieldNamesGrouped_(true);
    template<>
    std::vector<std::string> const AbstractSensorTpl<EncoderSensor>::fieldNames_({"Q", "V"});

    EncoderSensor::EncoderSensor(Model       const & model,
                                 std::shared_ptr<SensorSharedDataHolder_t> const & dataHolder,
                                 std::string const & name) :
    AbstractSensorTpl(model, dataHolder, name),
    motorName_(),
    motorPositionIdx_(),
    motorVelocityIdx_()
    {
        // Empty.
    }

    result_t EncoderSensor::initialize(std::string const & motorName)
    {
        result_t returnCode = result_t::SUCCESS;

        motorName_ = motorName;
        returnCode = getJointPositionIdx(model_->pncModel_, motorName_, motorPositionIdx_);
        if (returnCode == result_t::SUCCESS)
        {
            returnCode = getJointVelocityIdx(model_->pncModel_, motorName_, motorVelocityIdx_);
        }

        if (returnCode == result_t::SUCCESS)
        {
            isInitialized_ = true;
        }

        return returnCode;
    }

    void EncoderSensor::reset(void)
    {
        AbstractSensorTpl<EncoderSensor>::reset();
        getJointPositionIdx(model_->pncModel_, motorName_, motorPositionIdx_);
        getJointVelocityIdx(model_->pncModel_, motorName_, motorVelocityIdx_);
    }

    std::string EncoderSensor::getMotorName(void) const
    {
        return motorName_;
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

        data().head(1) = q.segment<1>(motorPositionIdx_);
        data().tail(1) = v.segment<1>(motorVelocityIdx_);

        return result_t::SUCCESS;
    }
}
