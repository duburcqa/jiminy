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
    struct MotorSharedDataHolder_t
    {
        /// \brief Buffer storing the current true motor efforts.
        Eigen::VectorXd data_;
        /// \brief Vector of pointers to the motors.
        std::vector<AbstractMotorBase *> motors_;
        /// \brief Number of motors
        std::size_t num_;
    };

    class JIMINY_DLLAPI AbstractMotorBase : public std::enable_shared_from_this<AbstractMotorBase>
    {
        /* AKA AbstractSensorBase */
        friend Robot;

    public:
        /// \brief Dictionary gathering the configuration options shared between motors.
        virtual GenericConfig getDefaultMotorOptions()
        {
            GenericConfig config;
            config["mechanicalReduction"] = 1.0;
            config["enableCommandLimit"] = true;
            config["commandLimitFromUrdf"] = true;
            config["commandLimit"] = 0.0;
            config["enableArmature"] = false;
            config["armature"] = 0.0;

            return config;
        };

        struct abstractMotorOptions_t
        {
            /// \brief Mechanical reduction ratio of transmission (joint/motor), usually >= 1.0.
            const double mechanicalReduction;
            const bool enableCommandLimit;
            const bool commandLimitFromUrdf;
            const double commandLimit;
            const bool enableArmature;
            const double armature;

            abstractMotorOptions_t(const GenericConfig & options) :
            mechanicalReduction(boost::get<double>(options.at("mechanicalReduction"))),
            enableCommandLimit(boost::get<bool>(options.at("enableCommandLimit"))),
            commandLimitFromUrdf(boost::get<bool>(options.at("commandLimitFromUrdf"))),
            commandLimit(boost::get<double>(options.at("commandLimit"))),
            enableArmature(boost::get<bool>(options.at("enableArmature"))),
            armature(boost::get<double>(options.at("armature")))
            {
            }
        };

    public:
        DISABLE_COPY(AbstractMotorBase)

    public:
        /// \param[in] name Name of the motor.
        explicit AbstractMotorBase(const std::string & name) noexcept;
        virtual ~AbstractMotorBase();

        /// \brief Refresh the proxies.
        ///
        /// \remark This method is not intended to be called manually. The Robot to which the motor
        ///         is added is taking care of it when its own `refresh` method is called.
        virtual hresult_t refreshProxies();

        /// \brief Reset the internal state of the motors.
        ///
        /// \details This method resets the internal state of the motor.
        ///
        /// \remark  This method is not intended to be called manually. The Robot to which the
        ///          motor is added is taking care of it when its own `reset` method is called.
        virtual hresult_t resetAll();

        /// \brief Configuration options of the motor.
        GenericConfig getOptions() const noexcept;

        /// \brief Actual effort of the motor at the current time.
        double get() const;

        /// \brief Actual effort of all the motors at the current time.
        const Eigen::VectorXd & getAll() const;

        /// \brief Set the configuration options of the motor.
        ///
        /// \param[in] motorOptions Dictionary with the parameters of the motor.
        virtual hresult_t setOptions(const GenericConfig & motorOptions);

        /// \brief Set the same configuration options for all motors of same type as current one.
        ///
        /// \param[in] motorOptions Dictionary with the parameters used for any motor.
        hresult_t setOptionsAll(const GenericConfig & motorOptions);

        /// \brief Whether the motor has been initialized.
        bool getIsInitialized() const;

        /// \brief Name of the motor.
        const std::string & getName() const;

        /// \brief Index of the motor.
        std::size_t getIdx() const;

        /// \brief Name of the joint associated with the motor.
        const std::string & getJointName() const;

        /// \brief Index of the joint associated with the motor in the kinematic tree.
        pinocchio::JointIndex getJointModelIdx() const;

        /// \brief Type of joint associated with the motor.
        JointModelType getJointType() const;

        /// \brief Index of the joint associated with the motor in configuration vector.
        Eigen::Index getJointPositionIdx() const;

        /// \brief Index of the joint associated with the motor in the velocity vector.
        Eigen::Index getJointVelocityIdx() const;

        /// \brief Maximum effort of the motor.
        double getCommandLimit() const;

        /// \brief Rotor inertia of the motor.
        double getArmature() const;

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
        virtual hresult_t computeEffort(double t,
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
        ///
        /// \return Return code to determine whether the execution of the method was successful.
        hresult_t computeEffortAll(double t,
                                   const Eigen::VectorXd & q,
                                   const Eigen::VectorXd & v,
                                   const Eigen::VectorXd & a,
                                   const Eigen::VectorXd & command);

    protected:
        /// \brief Reference to the last data buffer corresponding to the true effort of the motor.
        double & data();

    private:
        /// \brief Attach the sensor to a robot
        ///
        /// \details This method must be called before initializing the sensor.
        hresult_t attach(std::weak_ptr<const Robot> robot,
                         std::function<hresult_t(AbstractMotorBase & /*motor*/)> notifyRobot,
                         MotorSharedDataHolder_t * sharedHolder);

        /// \brief Detach the sensor from the robot.
        hresult_t detach();

    public:
        /// \brief Structure with the parameters of the motor.
        std::unique_ptr<const abstractMotorOptions_t> baseMotorOptions_{nullptr};

    protected:
        /// \brief Dictionary with the parameters of the motor.
        GenericConfig motorOptionsHolder_{};
        /// \brief Flag to determine whether the controller has been initialized or not.
        bool isInitialized_{false};
        /// \brief Flag to determine whether the motor is attached to a robot.
        bool isAttached_{false};
        /// \brief Robot for which the command and internal dynamics.
        std::weak_ptr<const Robot> robot_{};
        /// \brief Notify the robot that the configuration of the sensors have changed.
        std::function<hresult_t(AbstractMotorBase &)> notifyRobot_{};
        /// \brief Name of the motor.
        std::string name_;
        /// \brief Index of the motor in the measurement buffer.
        std::size_t motorIdx_{0};
        std::string jointName_{};
        pinocchio::JointIndex jointModelIdx_{0};
        JointModelType jointType_{JointModelType::UNSUPPORTED};
        Eigen::Index jointPositionIdx_{0};
        Eigen::Index jointVelocityIdx_{0};
        double commandLimit_{0.0};
        double armature_{0.0};

    private:
        /// \brief Shared data between every motors associated with the robot.
        MotorSharedDataHolder_t * sharedHolder_{nullptr};
    };
}

#endif  // JIMINY_ABSTRACT_MOTOR_H
