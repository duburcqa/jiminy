/// \brief Generic interface for any motor.
///
/// \details Any motor must inherit from this base class and implement its virtual methods.
///
/// \remark Each motor added to a Jiminy Robot is down-casted as an instance of AbstractMotorBase
///         and polymorphism is used to call the actual implementations.

#ifndef JIMINY_ABSTRACT_MOTOR_H
#define JIMINY_ABSTRACT_MOTOR_H

#include <memory>

#include "jiminy/core/fwd.h"


namespace jiminy
{
    class Robot;
    class AbstractMotorBase;

    /// \brief Structure holding the data for every motor.
    ///
    /// \details This structure enables to optimize the efficiency of data storage by gathering the
    ///          state of every motor.
    struct MotorSharedStorage
    {
        /// \brief Buffer storing the current motor efforts on joint and motor sides respectively.
        Eigen::MatrixX2d data_;
        /// \brief Vector of pointers to the motors.
        std::vector<AbstractMotorBase *> motors_;
        /// \brief Number of motors
        std::size_t num_;
    };

    class JIMINY_DLLAPI AbstractMotorBase
    {
        /* AKA AbstractSensorBase */
        friend Robot;

    public:
        /// \brief Dictionary gathering the configuration options shared between motors.
        virtual GenericConfig getDefaultMotorOptions()
        {
            GenericConfig config;
            config["mechanicalReduction"] = 1.0;
            config["velocityLimitFromUrdf"] = true;
            config["velocityLimit"] = 0.0;
            config["effortLimitFromUrdf"] = true;
            config["effortLimit"] = 0.0;
            config["enableArmature"] = false;
            config["armature"] = 0.0;
            config["enableBacklash"] = false;
            config["backlash"] = 0.0;

            return config;
        };

        struct AbstractMotorOptions
        {
            /// \brief Mechanical reduction ratio of transmission (joint/motor), usually >= 1.0.
            const double mechanicalReduction;
            const bool velocityLimitFromUrdf;
            const double velocityLimit;
            const bool effortLimitFromUrdf;
            const double effortLimit;
            const bool enableArmature;
            const double armature;
            const bool enableBacklash;
            const double backlash;

            AbstractMotorOptions(const GenericConfig & options) :
            mechanicalReduction(boost::get<double>(options.at("mechanicalReduction"))),
            velocityLimitFromUrdf(boost::get<bool>(options.at("velocityLimitFromUrdf"))),
            velocityLimit(boost::get<double>(options.at("velocityLimit"))),
            effortLimitFromUrdf(boost::get<bool>(options.at("effortLimitFromUrdf"))),
            effortLimit(boost::get<double>(options.at("effortLimit"))),
            enableArmature(boost::get<bool>(options.at("enableArmature"))),
            armature(boost::get<double>(options.at("armature"))),
            enableBacklash(boost::get<bool>(options.at("enableBacklash"))),
            backlash(boost::get<double>(options.at("backlash")))
            {
            }
        };

    public:
        JIMINY_DISABLE_COPY(AbstractMotorBase)

    public:
        /// \param[in] name Name of the motor.
        explicit AbstractMotorBase(const std::string & name);
        virtual ~AbstractMotorBase();

        /// \brief Refresh the proxies.
        ///
        /// \remark This method is not intended to be called manually. The Robot to which the motor
        ///         is added is taking care of it when its own `refresh` method is called.
        virtual void refreshProxies();

        /// \brief Reset the internal state of the motors.
        ///
        /// \details This method resets the internal state of the motor.
        ///
        /// \remark  This method is not intended to be called manually. The Robot to which the
        ///          motor is added is taking care of it when its own `reset` method is called.
        virtual void resetAll();

        /// \brief Configuration options of the motor.
        const GenericConfig & getOptions() const noexcept;

        /// \brief Actual effort of the motor at the current time.
        std::tuple<double, double> get() const;

        /// \brief Actual effort of all the motors at the current time.
        std::tuple<Eigen::Ref<const Eigen::VectorXd>, Eigen::Ref<const Eigen::VectorXd>>
        getAll() const;

        /// \brief Set the configuration options of the motor.
        ///
        /// \param[in] motorOptions Dictionary with the parameters of the motor.
        virtual void setOptions(const GenericConfig & motorOptions);

        /// \brief Set the same configuration options for all motors of same type as current one.
        ///
        /// \param[in] motorOptions Dictionary with the parameters used for any motor.
        void setOptionsAll(const GenericConfig & motorOptions);

        /// \brief Whether the motor has been attached to a robot.
        bool getIsAttached() const;

        /// \brief Whether the motor has been initialized.
        bool getIsInitialized() const;

        /// \brief Name of the motor.
        const std::string & getName() const;

        /// \brief Index of the motor.
        std::size_t getIndex() const;

        /// \brief Name of the joint associated with the motor.
        const std::string & getJointName() const;

        /// \brief Index of the joint associated with the motor in the kinematic tree.
        pinocchio::JointIndex getJointIndex() const;

        /// \brief Maximum position of the actuated joint translated on motor side.
        double getPositionLimitLower() const;

        /// \brief Minimum position of the actuated joint translated on motor side.
        double getPositionLimitUpper() const;

        /// \brief Maximum velocity of the motor.
        double getVelocityLimit() const;

        /// \brief Maximum effort of the motor.
        double getEffortLimit() const;

        /// \brief Rotor inertia of the motor on joint side.
        double getArmature() const;

        /// \brief Backlash of the transmission on joint side.
        double getBacklash() const;

        /// \brief Request the motor to update its actual effort based of the input data.
        ///
        /// \details It assumes that the internal state of the robot is consistent with the input
        ///          arguments.
        ///
        /// \param[in] t Current time.
        /// \param[in] q Current configuration of the motor.
        /// \param[in] v Current velocity of the motor.
        /// \param[in] a Current acceleration of the motor.
        /// \param[in] command Current command effort of the motor.
        virtual void computeEffort(double t,
                                   const Eigen::VectorBlock<const Eigen::VectorXd> & q,
                                   double v,
                                   double a,
                                   double command) = 0; /* copy on purpose */

        /// \brief Request every motors to update their actual effort based of the input data.
        ///
        /// \details It assumes that the internal state of the robot is consistent with the input
        ///          arguments.
        ///
        /// \remark This method is not intended to be called manually. The Robot to which the motor
        ///         is added is taking care of it while updating the state of the motors.
        ///
        /// \param[in] t Current time.
        /// \param[in] q Current configuration vector of the robot.
        /// \param[in] v Current velocity vector of the robot.
        /// \param[in] a Current acceleration vector of the robot.
        /// \param[in] command Current command effort vector of the robot.
        void computeEffortAll(double t,
                              const Eigen::VectorXd & q,
                              const Eigen::VectorXd & v,
                              const Eigen::VectorXd & a,
                              const Eigen::VectorXd & command);

    protected:
        /// \brief Reference to the last data buffer corresponding to the true effort of the motor.
        std::tuple<double &, double &> data();

    private:
        /// \brief Attach the sensor to a robot
        ///
        /// \details This method must be called before initializing the sensor.
        void attach(
            std::weak_ptr<const Robot> robot,
            std::function<void(AbstractMotorBase & /*motor*/, bool /*hasChanged*/)> notifyRobot,
            MotorSharedStorage * sharedStorage);

        /// \brief Detach the sensor from the robot.
        void detach();

    public:
        /// \brief Structure with the parameters of the motor.
        std::unique_ptr<const AbstractMotorOptions> baseMotorOptions_{nullptr};

    protected:
        /// \brief Dictionary with the parameters of the motor.
        GenericConfig motorOptionsGeneric_{};
        /// \brief Flag to determine whether the controller has been initialized or not.
        bool isInitialized_{false};
        /// \brief Robot for which the command and internal dynamics.
        std::weak_ptr<const Robot> robot_{};
        /// \brief Index of the motor in the measurement buffer.
        std::size_t motorIndex_{0};
        std::string jointName_{};
        pinocchio::JointIndex jointIndex_{0};
        JointModelType jointType_{JointModelType::UNSUPPORTED};
        double positionLimitLower_{};
        double positionLimitUpper_{};
        double velocityLimit_{0.0};
        double effortLimit_{0.0};
        double armature_{0.0};
        double backlash_{0.0};

    private:
        /// \brief Name of the motor.
        std::string name_;
        /// \brief Flag to determine whether the motor is attached to a robot.
        bool isAttached_{false};
        /// \brief Whether the robot must be notified.
        bool mustNotifyRobot_{false};
        /// \brief Notify the robot that the configuration of the motor have changed.
        std::function<void(AbstractMotorBase &, bool)> notifyRobot_{};
        /// \brief Shared data between every motors associated with the robot.
        MotorSharedStorage * sharedStorage_{nullptr};
    };
}

#endif  // JIMINY_ABSTRACT_MOTOR_H
