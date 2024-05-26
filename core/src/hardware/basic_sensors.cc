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


#define GET_ROBOT_AND_CHECK_SENSOR_INTEGRITY()                                                   \
    if (!isAttached_)                                                                            \
    {                                                                                            \
        JIMINY_THROW(bad_control_flow,                                                           \
                     "Sensor not attached to any robot. Impossible to refresh sensor proxies."); \
    }                                                                                            \
                                                                                                 \
    auto robot = robot_.lock();                                                                  \
    if (!robot)                                                                                  \
    {                                                                                            \
        JIMINY_THROW(bad_control_flow,                                                           \
                     "Robot has been deleted. Impossible to refresh sensor proxies.");           \
    }                                                                                            \
                                                                                                 \
    if (!robot->getIsInitialized())                                                              \
    {                                                                                            \
        JIMINY_THROW(bad_control_flow,                                                           \
                     "Robot not initialized. Impossible to refresh sensor proxies.");            \
    }                                                                                            \
                                                                                                 \
    if (!isInitialized_)                                                                         \
    {                                                                                            \
        JIMINY_THROW(bad_control_flow,                                                           \
                     "Sensor not initialized. Impossible to refresh sensor proxies.");           \
    }                                                                                            \
                                                                                                 \
    if (robot->getIsLocked())                                                                    \
    {                                                                                            \
        JIMINY_THROW(bad_control_flow,                                                           \
                     "Robot already locked, probably because a simulation is running. "          \
                     "Please stop it before refreshing sensor proxies.");                        \
    }

#define GET_ROBOT_IF_INITIALIZED()                                                              \
    if (!isInitialized_)                                                                        \
    {                                                                                           \
        JIMINY_THROW(bad_control_flow, "Sensor not initialized. Impossible to update sensor."); \
    }                                                                                           \
                                                                                                \
    auto robot = robot_.lock();


namespace jiminy
{
    // ===================== ImuSensor =========================

    template<>
    const std::string AbstractSensorTpl<ImuSensor>::type_{"ImuSensor"};
    template<>
    const std::vector<std::string> AbstractSensorTpl<ImuSensor>::fieldnames_{
        "GyroX", "GyroY", "GyroZ", "AccelX", "AccelY", "AccelZ"};

    void ImuSensor::initialize(const std::string & frameName)
    {
        // Update frame name
        frameName_ = frameName;
        isInitialized_ = true;

        // Try refreshing proxies if possible, restore internals before throwing exception if not
        try
        {
            refreshProxies();
        }
        catch (...)
        {
            frameName_.clear();
            isInitialized_ = false;
            throw;
        }
    }

    void ImuSensor::setOptions(const GenericConfig & sensorOptions)
    {
        // Check that bias / std is of the correct size
        const Eigen::VectorXd & bias = boost::get<Eigen::VectorXd>(sensorOptions.at("bias"));
        const Eigen::VectorXd & noiseStd =
            boost::get<Eigen::VectorXd>(sensorOptions.at("noiseStd"));
        if (bias.size() && bias.size() != 9)
        {
            JIMINY_THROW(
                std::invalid_argument,
                "Wrong bias vector. It must contain 9 values:\n"
                "  - the first three are the angle-axis representation of a rotation bias applied "
                "to all sensor signal.\n"
                "  - the next six are respectively gyroscope and accelerometer additive bias.");
        }
        if (noiseStd.size() && static_cast<std::size_t>(noiseStd.size()) != getSize())
        {
            JIMINY_THROW(
                std::invalid_argument,
                "Wrong noise std vector. It must contain 6 values corresponding respectively to "
                "gyroscope and accelerometer additive noise.");
        }

        // Set options now that sanity checks were made
        AbstractSensorTpl<ImuSensor>::setOptions(sensorOptions);
    }

    void ImuSensor::refreshProxies()
    {
        GET_ROBOT_AND_CHECK_SENSOR_INTEGRITY()

        frameIndex_ = ::jiminy::getFrameIndex(robot->pinocchioModel_, frameName_);

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

    const std::string & ImuSensor::getFrameName() const
    {
        return frameName_;
    }

    pinocchio::FrameIndex ImuSensor::getFrameIndex() const
    {
        return frameIndex_;
    }

    void ImuSensor::set(double /* t */,
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

    void ContactSensor::initialize(const std::string & frameName)
    {
        // Update frame name
        frameName_ = frameName;
        isInitialized_ = true;

        // Try refreshing proxies if possible, restore internals before throwing exception if not
        try
        {
            refreshProxies();
        }
        catch (...)
        {
            frameName_.clear();
            isInitialized_ = false;
            throw;
        }
    }

    void ContactSensor::setOptions(const GenericConfig & sensorOptions)
    {
        // Check that bias / std is of the correct size
        const Eigen::VectorXd & bias = boost::get<Eigen::VectorXd>(sensorOptions.at("bias"));
        const Eigen::VectorXd & noiseStd =
            boost::get<Eigen::VectorXd>(sensorOptions.at("noiseStd"));
        if (bias.size() && static_cast<std::size_t>(bias.size()) != getSize())
        {
            JIMINY_THROW(std::invalid_argument, "Wrong bias vector size.");
        }
        if (noiseStd.size() && static_cast<std::size_t>(noiseStd.size()) != getSize())
        {
            JIMINY_THROW(std::invalid_argument, "Wrong noise std vector size.");
        }

        // Set options now that sanity checks were made
        AbstractSensorTpl<ContactSensor>::setOptions(sensorOptions);
    }

    void ContactSensor::refreshProxies()
    {
        GET_ROBOT_AND_CHECK_SENSOR_INTEGRITY()

        const std::vector<std::string> & contactFrameNames = robot->getContactFrameNames();
        auto contactFrameNameIt =
            std::find(contactFrameNames.begin(), contactFrameNames.end(), frameName_);
        if (contactFrameNameIt == contactFrameNames.end())
        {
            JIMINY_THROW(std::logic_error,
                         "Sensor frame not associated with any contact point of the robot. "
                         "Impossible to refresh sensor proxies.");
        }

        frameIndex_ = ::jiminy::getFrameIndex(robot->pinocchioModel_, frameName_);

        const std::vector<pinocchio::FrameIndex> & contactFrameIndices =
            robot->getContactFrameIndices();
        auto contactFrameIndexIt =
            std::find(contactFrameIndices.begin(), contactFrameIndices.end(), frameIndex_);
        contactIndex_ = std::distance(contactFrameIndices.begin(), contactFrameIndexIt);
    }

    const std::string & ContactSensor::getFrameName() const
    {
        return frameName_;
    }

    pinocchio::FrameIndex ContactSensor::getFrameIndex() const
    {
        return frameIndex_;
    }

    void ContactSensor::set(double /* t */,
                            const Eigen::VectorXd & /* q */,
                            const Eigen::VectorXd & /* v */,
                            const Eigen::VectorXd & /* a */,
                            const Eigen::VectorXd & /* uMotor */,
                            const ForceVector & /* fExternal */)
    {
        GET_ROBOT_IF_INITIALIZED()

        data() = robot->contactForces_[contactIndex_].linear();
    }

    // ===================== ForceSensor =========================

    template<>
    const std::string AbstractSensorTpl<ForceSensor>::type_{"ForceSensor"};
    template<>
    const std::vector<std::string> AbstractSensorTpl<ForceSensor>::fieldnames_{
        "FX", "FY", "FZ", "MX", "MY", "MZ"};

    void ForceSensor::initialize(const std::string & frameName)
    {
        // Update frame name
        frameName_ = frameName;
        isInitialized_ = true;

        // Try refreshing proxies if possible, restore internals before throwing exception if not
        try
        {
            refreshProxies();
        }
        catch (...)
        {
            frameName_.clear();
            isInitialized_ = false;
            throw;
        }
    }

    void ForceSensor::setOptions(const GenericConfig & sensorOptions)
    {
        // Check that bias / std is of the correct size
        const Eigen::VectorXd & bias = boost::get<Eigen::VectorXd>(sensorOptions.at("bias"));
        const Eigen::VectorXd & noiseStd =
            boost::get<Eigen::VectorXd>(sensorOptions.at("noiseStd"));
        if (bias.size() && static_cast<std::size_t>(bias.size()) != getSize())
        {
            JIMINY_THROW(std::invalid_argument, "Wrong bias vector size.");
        }
        if (noiseStd.size() && static_cast<std::size_t>(noiseStd.size()) != getSize())
        {
            JIMINY_THROW(std::invalid_argument, "Wrong noise std vector size.");
        }

        // Set options now that sanity checks were made
        AbstractSensorTpl<ForceSensor>::setOptions(sensorOptions);
    }

    void ForceSensor::refreshProxies()
    {
        GET_ROBOT_AND_CHECK_SENSOR_INTEGRITY()

        frameIndex_ = ::jiminy::getFrameIndex(robot->pinocchioModel_, frameName_);

        // 'parent' returns the parent joint
        parentJointIndex_ = robot->pinocchioModel_.frames[frameIndex_].parent;

        contactIndexPlacementPairs_.clear();
        const pinocchio::Frame & frameRef = robot->pinocchioModel_.frames[frameIndex_];
        const std::vector<pinocchio::FrameIndex> & contactFrameIndices =
            robot->getContactFrameIndices();
        for (uint32_t contactIndex = 0; contactIndex < contactFrameIndices.size(); ++contactIndex)
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

    void ForceSensor::set(double /* t */,
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
    }

    // ===================== EncoderSensor =========================

    template<>
    const std::string AbstractSensorTpl<EncoderSensor>::type_{"EncoderSensor"};
    template<>
    const std::vector<std::string> AbstractSensorTpl<EncoderSensor>::fieldnames_{"Q", "V"};

    void EncoderSensor::initialize(const std::string & motorOrJointName, bool isJointSide)
    {
        // Update motor name
        isJointSide_ = isJointSide;
        if (isJointSide_)
        {
            jointName_ = motorOrJointName;
        }
        else
        {
            motorName_ = motorOrJointName;
        }
        isInitialized_ = true;

        // Try refreshing proxies if possible, restore internals before throwing exception if not
        try
        {
            refreshProxies();
        }
        catch (...)
        {
            motorName_.clear();
            isInitialized_ = false;
            throw;
        }
    }

    void EncoderSensor::setOptions(const GenericConfig & sensorOptions)
    {
        // Check that bias / std is of the correct size
        const Eigen::VectorXd & bias = boost::get<Eigen::VectorXd>(sensorOptions.at("bias"));
        const Eigen::VectorXd & noiseStd =
            boost::get<Eigen::VectorXd>(sensorOptions.at("noiseStd"));
        if (bias.size() && static_cast<std::size_t>(bias.size()) != getSize())
        {
            JIMINY_THROW(std::invalid_argument, "Wrong bias vector size.");
        }
        if (noiseStd.size() && static_cast<std::size_t>(noiseStd.size()) != getSize())
        {
            JIMINY_THROW(std::invalid_argument, "Wrong noise std vector size.");
        }

        // Set options now that sanity checks were made
        AbstractSensorTpl<EncoderSensor>::setOptions(sensorOptions);
    }

    void EncoderSensor::refreshProxies()
    {
        GET_ROBOT_AND_CHECK_SENSOR_INTEGRITY()

        // Refresh proxies that depends on whether the encoder is on the joint or motor side
        if (isJointSide_)
        {
            motorIndex_ = robot->nmotors();
            jointIndex_ = ::jiminy::getJointIndex(robot->pinocchioModel_, jointName_);
            mechanicalReduction_ = 1.0;
        }
        else
        {
            std::shared_ptr<const AbstractMotorBase> motor = robot->getMotor(motorName_).lock();
            motorIndex_ = motor->getIndex();
            jointIndex_ = motor->getJointIndex();
            jointName_ = robot->pinocchioModel_.names[jointIndex_];
            mechanicalReduction_ = motor->baseMotorOptions_->mechanicalReduction;
        }

        // Motors are only supported for linear and rotary joints
        const auto & jmodel = robot->pinocchioModel_.joints[jointIndex_];
        jointType_ = getJointType(jmodel);
        if (jointType_ != JointModelType::LINEAR && jointType_ != JointModelType::ROTARY &&
            jointType_ != JointModelType::ROTARY_UNBOUNDED)
        {
            JIMINY_THROW(
                std::runtime_error,
                "Encoder sensors can only be associated with a 1-dof linear or rotary joint.");
        }

        /* Tracking motor turns is not supported for now. As a result, the mechanical reduction
           ratio must be equal exactly to 1.0 for revolute unbounded joints. */
        if (!isJointSide_ && jointType_ == JointModelType::ROTARY_UNBOUNDED &&
            std::abs(mechanicalReduction_ - 1.0) > EPS)
        {
            JIMINY_THROW(
                std::runtime_error,
                "Encoder sensors on motor side do not supported motors attached to revolute "
                "unbounded joints while having a mechanical reduction different from 1.0.");
        }

        // Refresh joint position and velocity indices
        jointPositionIndex_ = jmodel.idx_q();
        joinVelocityIndex_ = jmodel.idx_v();
    }

    const std::string & EncoderSensor::getMotorName() const
    {
        return motorName_;
    }

    std::size_t EncoderSensor::getMotorIndex() const
    {
        return motorIndex_;
    }

    const std::string & EncoderSensor::getJointName() const
    {
        return jointName_;
    }

    pinocchio::JointIndex EncoderSensor::getJointIndex() const
    {
        return jointIndex_;
    }

    void EncoderSensor::set(double /* t */,
                            const Eigen::VectorXd & q,
                            const Eigen::VectorXd & v,
                            const Eigen::VectorXd & /* a */,
                            const Eigen::VectorXd & /* uMotor */,
                            const ForceVector & /* fExternal */)
    {
        // Compute the joint position and velocity
        double jointPosition;
        if (jointType_ == JointModelType::ROTARY_UNBOUNDED)
        {
            const double cosTheta = q[jointPositionIndex_];
            const double sinTheta = q[jointPositionIndex_ + 1];
            jointPosition = std::atan2(sinTheta, cosTheta);
        }
        else
        {
            jointPosition = q[jointPositionIndex_];
        }
        const double jointVelocity = v[joinVelocityIndex_];

        // Compute encoder data depending on whether the encoder is on joint or motor side
        if (isJointSide_)
        {
            data() << jointPosition, jointVelocity;
        }
        else
        {
            data() << jointPosition * mechanicalReduction_, jointVelocity * mechanicalReduction_;
        }
    }

    // ===================== EffortSensor =========================

    template<>
    const std::string AbstractSensorTpl<EffortSensor>::type_{"EffortSensor"};
    template<>
    const std::vector<std::string> AbstractSensorTpl<EffortSensor>::fieldnames_{"U"};

    void EffortSensor::initialize(const std::string & motorName)
    {
        // Update motor name
        motorName_ = motorName;
        isInitialized_ = true;

        // Try refreshing proxies if possible, restore internals before throwing exception if not
        try
        {
            refreshProxies();
        }
        catch (...)
        {
            motorName_.clear();
            isInitialized_ = false;
            throw;
        }
    }

    void EffortSensor::setOptions(const GenericConfig & sensorOptions)
    {
        // Check that bias / std is of the correct size
        const Eigen::VectorXd & bias = boost::get<Eigen::VectorXd>(sensorOptions.at("bias"));
        const Eigen::VectorXd & noiseStd =
            boost::get<Eigen::VectorXd>(sensorOptions.at("noiseStd"));
        if (bias.size() && static_cast<std::size_t>(bias.size()) != getSize())
        {
            JIMINY_THROW(std::invalid_argument, "Wrong bias vector size.");
        }
        if (noiseStd.size() && static_cast<std::size_t>(noiseStd.size()) != getSize())
        {
            JIMINY_THROW(std::invalid_argument, "Wrong noise std vector size.");
        }

        // Set options now that sanity checks were made
        AbstractSensorTpl<EffortSensor>::setOptions(sensorOptions);
    }

    void EffortSensor::refreshProxies()
    {
        GET_ROBOT_AND_CHECK_SENSOR_INTEGRITY()

        std::weak_ptr<const AbstractMotorBase> motor = robot->getMotor(motorName_);
        motorIndex_ = motor.lock()->getIndex();
    }

    const std::string & EffortSensor::getMotorName() const
    {
        return motorName_;
    }

    std::size_t EffortSensor::getMotorIndex() const
    {
        return motorIndex_;
    }

    void EffortSensor::set(double /* t */,
                           const Eigen::VectorXd & /* q */,
                           const Eigen::VectorXd & /* v */,
                           const Eigen::VectorXd & /* a */,
                           const Eigen::VectorXd & uMotor,
                           const ForceVector & /* fExternal */)
    {
        if (!isInitialized_)
        {
            JIMINY_THROW(bad_control_flow,
                         "Sensor not initialized. Impossible to set sensor data.");
        }

        data()[0] = uMotor[motorIndex_];
    }
}
