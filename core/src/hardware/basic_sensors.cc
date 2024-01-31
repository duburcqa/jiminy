#include <algorithm>

#include "pinocchio/spatial/explog.hpp"  // `pinocchio::exp3`
#include "pinocchio/spatial/se3.hpp"     // `pinocchio::SE3`
#include "pinocchio/spatial/force.hpp"   // `pinocchio::Force  `
#include "pinocchio/spatial/motion.hpp"  // `pinocchio::Motion`
#include "pinocchio/algorithm/frames.hpp"  // `pinocchio::getFrameVelocity`, `pinocchio::getFrameAcceleration`

#include "jiminy/core/robot/robot.h"
#include "jiminy/core/hardware/abstract_motor.h"
#include "jiminy/core/utilities/pinocchio.h"
#include "jiminy/core/utilities/random.h"

#include "jiminy/core/hardware/basic_sensors.h"


#define GET_ROBOT_AND_CHECK_SENSOR_INTEGRITY()                                           \
    if (!isAttached_)                                                                    \
    {                                                                                    \
        PRINT_ERROR("Sensor not attached to any robot. Impossible to refresh proxies."); \
        returnCode = hresult_t::ERROR_INIT_FAILED;                                       \
    }                                                                                    \
                                                                                         \
    auto robot = robot_.lock();                                                          \
    if (returnCode == hresult_t::SUCCESS)                                                \
    {                                                                                    \
        if (!robot)                                                                      \
        {                                                                                \
            PRINT_ERROR("Robot has been deleted. Impossible to refresh proxies.");       \
            returnCode = hresult_t::ERROR_GENERIC;                                       \
        }                                                                                \
    }                                                                                    \
                                                                                         \
    if (returnCode == hresult_t::SUCCESS)                                                \
    {                                                                                    \
        if (!robot->getIsInitialized())                                                  \
        {                                                                                \
            PRINT_ERROR("Robot not initialized. Impossible to refresh proxies.");        \
            returnCode = hresult_t::ERROR_INIT_FAILED;                                   \
        }                                                                                \
    }                                                                                    \
                                                                                         \
    if (returnCode == hresult_t::SUCCESS)                                                \
    {                                                                                    \
        if (!isInitialized_)                                                             \
        {                                                                                \
            PRINT_ERROR("Sensor not initialized. Impossible to refresh proxies.");       \
            returnCode = hresult_t::ERROR_INIT_FAILED;                                   \
        }                                                                                \
    }


#define GET_ROBOT_IF_INITIALIZED()                                           \
    if (!isInitialized_)                                                     \
    {                                                                        \
        PRINT_ERROR("Sensor not initialized. Impossible to update sensor."); \
        return hresult_t::ERROR_INIT_FAILED;                                 \
    }                                                                        \
                                                                             \
    auto robot = robot_.lock();


namespace jiminy
{
    // ===================== ImuSensor =========================

    template<>
    const std::string AbstractSensorTpl<ImuSensor>::type_{"ImuSensor"};
    template<>
    const std::vector<std::string> AbstractSensorTpl<ImuSensor>::fieldnames_{
        "Gyrox", "Gyroy", "Gyroz", "Accelx", "Accely", "Accelz"};
    template<>
    const bool AbstractSensorTpl<ImuSensor>::areFieldnamesGrouped_{false};

    hresult_t ImuSensor::initialize(const std::string & frameName)
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

    hresult_t ImuSensor::setOptions(const GenericConfig & sensorOptions)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        // Check that bias / std is of the correct size
        const Eigen::VectorXd & bias = boost::get<Eigen::VectorXd>(sensorOptions.at("bias"));
        const Eigen::VectorXd & noiseStd =
            boost::get<Eigen::VectorXd>(sensorOptions.at("noiseStd"));
        if (bias.size() && bias.size() != 9)
        {
            PRINT_ERROR(
                "Wrong bias vector. It must contain 9 values:\n"
                "  - the first three are the angle-axis representation of a rotation bias applied "
                "to all sensor signal.\n"
                "  - the next six are respectively gyroscope and accelerometer additive bias.");
            returnCode = hresult_t::ERROR_BAD_INPUT;
        }
        if (noiseStd.size() && noiseStd.size() != 6)
        {
            PRINT_ERROR("Wrong noise std vector. It must contain 6 values corresponding "
                        "respectively to gyroscope and accelerometer additive bias.");
            returnCode = hresult_t::ERROR_BAD_INPUT;
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            returnCode = AbstractSensorTpl<ImuSensor>::setOptions(sensorOptions);
        }

        return returnCode;
    }

    hresult_t ImuSensor::refreshProxies()
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        GET_ROBOT_AND_CHECK_SENSOR_INTEGRITY()

        if (returnCode == hresult_t::SUCCESS)
        {
            returnCode = ::jiminy::getFrameIndex(robot->pinocchioModel_, frameName_, frameIndex_);
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            if (baseSensorOptions_->bias.size())
            {
                // Convert first three components of bias to quaternion
                sensorRotationBiasInv_ = pinocchio::exp3(-baseSensorOptions_->bias.head<3>());
            }
            else
            {
                sensorRotationBiasInv_ = Eigen::Matrix3d::Identity();
            }
        }

        return returnCode;
    }

    const std::string & ImuSensor::getFrameName() const
    {
        return frameName_;
    }

    pinocchio::FrameIndex ImuSensor::getFrameIndex() const
    {
        return frameIndex_;
    }

    hresult_t ImuSensor::set(double /* t */,
                             const Eigen::VectorXd & /* q */,
                             const Eigen::VectorXd & /* v */,
                             const Eigen::VectorXd & /* a */,
                             const Eigen::VectorXd & /* uMotor */,
                             const ForceVector & /* fExternal */)
    {
        GET_ROBOT_IF_INITIALIZED()

        // Compute gyroscope signal
        const pinocchio::Motion velocity = pinocchio::getFrameVelocity(
            robot->pinocchioModel_, robot->pinocchioData_, frameIndex_, pinocchio::LOCAL);
        data().head<3>() = velocity.angular();

        // Compute accelerometer signal
        const pinocchio::Motion acceleration = pinocchio::getFrameClassicalAcceleration(
            robot->pinocchioModel_, robot->pinocchioData_, frameIndex_, pinocchio::LOCAL);

        // Accelerometer measures the classical (not spatial !) linear acceleration minus gravity
        const Eigen::Matrix3d & rot = robot->pinocchioData_.oMf[frameIndex_].rotation();
        data().tail<3>() =
            acceleration.linear() - rot.transpose() * robot->pinocchioModel_.gravity.linear();

        return hresult_t::SUCCESS;
    }

    void ImuSensor::measureData()
    {
        // Add measurement white noise
        if (baseSensorOptions_->noiseStd.size())
        {
            get() += normal(generator_, 0.0F, baseSensorOptions_->noiseStd.cast<float>())
                         .cast<double>();
        }

        // Add measurement bias
        if (baseSensorOptions_->bias.size())
        {
            // Accel + gyroscope: simply add additive bias
            get() += baseSensorOptions_->bias.tail<6>();

            /* Apply the same bias to the accelerometer / gyroscope output.
               Quaternion bias is interpreted as angle-axis representation of a sensor rotation
               bias R_b, such that w_R_sensor = w_R_imu R_b. */
            get().head<3>() = sensorRotationBiasInv_ * get().head<3>();
            get().tail<3>() = sensorRotationBiasInv_ * get().tail<3>();
        }
    }

    // ===================== ContactSensor =========================

    template<>
    const std::string AbstractSensorTpl<ContactSensor>::type_{"ContactSensor"};
    template<>
    const std::vector<std::string> AbstractSensorTpl<ContactSensor>::fieldnames_{"FX", "FY", "FZ"};
    template<>
    const bool AbstractSensorTpl<ContactSensor>::areFieldnamesGrouped_{false};

    hresult_t ContactSensor::initialize(const std::string & frameName)
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

    hresult_t ContactSensor::refreshProxies()
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        GET_ROBOT_AND_CHECK_SENSOR_INTEGRITY()

        if (returnCode == hresult_t::SUCCESS)
        {
            const std::vector<std::string> & contactFrameNames = robot->getContactFrameNames();
            auto contactFrameNameIt =
                std::find(contactFrameNames.begin(), contactFrameNames.end(), frameName_);
            if (contactFrameNameIt == contactFrameNames.end())
            {
                PRINT_ERROR("Sensor frame not associated with any contact point of the robot. "
                            "Impossible to refresh proxies.");
                returnCode = hresult_t::ERROR_BAD_INPUT;
            }
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            returnCode = ::jiminy::getFrameIndex(robot->pinocchioModel_, frameName_, frameIndex_);
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            const std::vector<pinocchio::FrameIndex> & contactFrameIndices =
                robot->getContactFrameIndices();
            auto contactFrameIndexIt =
                std::find(contactFrameIndices.begin(), contactFrameIndices.end(), frameIndex_);
            contactIndex_ = std::distance(contactFrameIndices.begin(), contactFrameIndexIt);
        }

        return returnCode;
    }

    const std::string & ContactSensor::getFrameName() const
    {
        return frameName_;
    }

    pinocchio::FrameIndex ContactSensor::getFrameIndex() const
    {
        return frameIndex_;
    }

    hresult_t ContactSensor::set(double /* t */,
                                 const Eigen::VectorXd & /* q */,
                                 const Eigen::VectorXd & /* v */,
                                 const Eigen::VectorXd & /* a */,
                                 const Eigen::VectorXd & /* uMotor */,
                                 const ForceVector & /* fExternal */)
    {
        GET_ROBOT_IF_INITIALIZED()

        data() = robot->contactForces_[contactIndex_].linear();

        return hresult_t::SUCCESS;
    }

    // ===================== ForceSensor =========================

    template<>
    const std::string AbstractSensorTpl<ForceSensor>::type_{"ForceSensor"};
    template<>
    const std::vector<std::string> AbstractSensorTpl<ForceSensor>::fieldnames_{
        "FX", "FY", "FZ", "MX", "MY", "MZ"};
    template<>
    const bool AbstractSensorTpl<ForceSensor>::areFieldnamesGrouped_{false};

    hresult_t ForceSensor::initialize(const std::string & frameName)
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

    hresult_t ForceSensor::refreshProxies()
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        GET_ROBOT_AND_CHECK_SENSOR_INTEGRITY()

        if (returnCode == hresult_t::SUCCESS)
        {
            returnCode = ::jiminy::getFrameIndex(robot->pinocchioModel_, frameName_, frameIndex_);
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            // 'parent' returns the parent joint
            parentJointIndex_ = robot->pinocchioModel_.frames[frameIndex_].parent;
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            contactIndexPlacementPairs_.clear();
            const pinocchio::Frame & frameRef = robot->pinocchioModel_.frames[frameIndex_];
            const std::vector<pinocchio::FrameIndex> & contactFrameIndices =
                robot->getContactFrameIndices();
            for (uint32_t contactIndex = 0; contactIndex < contactFrameIndices.size();
                 ++contactIndex)
            {
                pinocchio::FrameIndex contactFrameIndex = contactFrameIndices[contactIndex];
                const pinocchio::Frame & contactFrame =
                    robot->pinocchioModel_.frames[contactFrameIndex];
                if (parentJointIndex_ == contactFrame.parent)
                {
                    const pinocchio::SE3 contactPlacementRel =
                        frameRef.placement.actInv(contactFrame.placement);
                    contactIndexPlacementPairs_.emplace_back(contactIndex,
                                                             std::move(contactPlacementRel));
                }
            }
        }

        return returnCode;
    }

    const std::string & ForceSensor::getFrameName() const
    {
        return frameName_;
    }

    pinocchio::FrameIndex ForceSensor::getFrameIndex() const
    {
        return frameIndex_;
    }

    pinocchio::JointIndex ForceSensor::getJointIndex() const
    {
        return parentJointIndex_;
    }

    hresult_t ForceSensor::set(double /* t */,
                               const Eigen::VectorXd & /* q */,
                               const Eigen::VectorXd & /* v */,
                               const Eigen::VectorXd & /* a */,
                               const Eigen::VectorXd & /* uMotor */,
                               const ForceVector & /* fExternal */)
    {
        // Returns the force applied on parent body in frame

        GET_ROBOT_IF_INITIALIZED()

        // Compute the sum of all contact forces applied on parent joint
        data().setZero();
        for (const auto & [contactIndex, contactPlacement] : contactIndexPlacementPairs_)
        {
            // Must transform the force from contact frame to sensor frame
            f_ = contactPlacement.act(robot->contactForces_[contactIndex]);
            data() += f_.toVector();
        }

        return hresult_t::SUCCESS;
    }

    // ===================== EncoderSensor =========================

    template<>
    const std::string AbstractSensorTpl<EncoderSensor>::type_{"EncoderSensor"};
    template<>
    const std::vector<std::string> AbstractSensorTpl<EncoderSensor>::fieldnames_{"Q", "V"};
    template<>
    const bool AbstractSensorTpl<EncoderSensor>::areFieldnamesGrouped_{true};

    hresult_t EncoderSensor::initialize(const std::string & jointName)
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

    hresult_t EncoderSensor::refreshProxies()
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        GET_ROBOT_AND_CHECK_SENSOR_INTEGRITY()

        if (returnCode == hresult_t::SUCCESS)
        {
            if (!robot->pinocchioModel_.existJointName(jointName_))
            {
                PRINT_ERROR("Sensor attached to a joint that does not exist.");
                returnCode = hresult_t::ERROR_INIT_FAILED;
            }
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            jointIndex_ = robot->pinocchioModel_.getJointId(jointName_);
            getJointTypeFromIndex(robot->pinocchioModel_, jointIndex_, jointType_);

            // Motors are only supported for linear and rotary joints
            if (jointType_ != JointModelType::LINEAR && jointType_ != JointModelType::ROTARY &&
                jointType_ != JointModelType::ROTARY_UNBOUNDED)
            {
                PRINT_ERROR("An encoder sensor can only be associated with a 1-dof linear or "
                            "rotary joint.");
                returnCode = hresult_t::ERROR_BAD_INPUT;
            }
        }

        return returnCode;
    }

    const std::string & EncoderSensor::getJointName() const
    {
        return jointName_;
    }

    pinocchio::JointIndex EncoderSensor::getJointIndex() const
    {
        return jointIndex_;
    }

    JointModelType EncoderSensor::getJointType() const
    {
        return jointType_;
    }

    hresult_t EncoderSensor::set(double /* t */,
                                 const Eigen::VectorXd & q,
                                 const Eigen::VectorXd & v,
                                 const Eigen::VectorXd & /* a */,
                                 const Eigen::VectorXd & /* uMotor */,
                                 const ForceVector & /* fExternal */)
    {
        GET_ROBOT_IF_INITIALIZED()

        const auto & joint = robot->pinocchioModel_.joints[jointIndex_];
        const Eigen::Index jointPositionIndex = joint.idx_q();
        const Eigen::Index jointVelocityIndex = joint.idx_v();
        if (jointType_ == JointModelType::ROTARY_UNBOUNDED)
        {
            const double cosTheta = q[jointPositionIndex];
            const double sinTheta = q[jointPositionIndex + 1];
            data()[0] = std::atan2(sinTheta, cosTheta);
        }
        else
        {
            data()[0] = q[jointPositionIndex];
        }
        data()[1] = v[jointVelocityIndex];

        return hresult_t::SUCCESS;
    }

    // ===================== EffortSensor =========================

    template<>
    const std::string AbstractSensorTpl<EffortSensor>::type_{"EffortSensor"};
    template<>
    const std::vector<std::string> AbstractSensorTpl<EffortSensor>::fieldnames_{"U"};
    template<>
    const bool AbstractSensorTpl<EffortSensor>::areFieldnamesGrouped_{true};

    hresult_t EffortSensor::initialize(const std::string & motorName)
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

    hresult_t EffortSensor::refreshProxies()
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        GET_ROBOT_AND_CHECK_SENSOR_INTEGRITY()

        std::weak_ptr<const AbstractMotorBase> motor;
        if (returnCode == hresult_t::SUCCESS)
        {
            returnCode = robot->getMotor(motorName_, motor);
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            motorIndex_ = motor.lock()->getIndex();
        }

        return returnCode;
    }

    const std::string & EffortSensor::getMotorName() const
    {
        return motorName_;
    }

    std::size_t EffortSensor::getMotorIndex() const
    {
        return motorIndex_;
    }

    hresult_t EffortSensor::set(double /* t */,
                                const Eigen::VectorXd & /* q */,
                                const Eigen::VectorXd & /* v */,
                                const Eigen::VectorXd & /* a */,
                                const Eigen::VectorXd & uMotor,
                                const ForceVector & /* fExternal */)
    {
        if (!isInitialized_)
        {
            PRINT_ERROR("Sensor not initialized. Impossible to set sensor data.");
            return hresult_t::ERROR_INIT_FAILED;
        }

        data()[0] = uMotor[motorIndex_];

        return hresult_t::SUCCESS;
    }
}
