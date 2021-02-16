///////////////////////////////////////////////////////////////////////////////////////////////
///
/// \brief      Generic interface for any motor.
///
/// \details    Any motor must inherit from this base class and implement its virtual methods.
///
/// \remark     Each motor added to a Jiminy Robot is downcasted as an instance of
///             AbstractMotorBase and polymorphism is used to call the actual implementations.
///
///////////////////////////////////////////////////////////////////////////////////////////////

#ifndef JIMINY_ABSTRACT_MOTOR_H
#define JIMINY_ABSTRACT_MOTOR_H

#include <memory>

#include "jiminy/core/Macros.h"
#include "jiminy/core/Types.h"


namespace jiminy
{
    class Robot;

    class AbstractMotorBase;

    ///////////////////////////////////////////////////////////////////////////////////////////////
    /// \brief      Structure holding the data for every motor.
    ///
    /// \details    This structure enables to optimize the efficiency of data storage by gathering
    ///             the state of every motor.
    ///////////////////////////////////////////////////////////////////////////////////////////////
    struct MotorSharedDataHolder_t
    {
        MotorSharedDataHolder_t(void) :
        data_(),
        motors_(),
        num_(0)
        {
            // Empty.
        };

        ~MotorSharedDataHolder_t(void) = default;

        vectorN_t data_;                           ///< Buffer with current actual motor effort
        std::vector<AbstractMotorBase *> motors_;  ///< Vector of pointers to the motors.
        int32_t num_;                              ///< Number of motors
    };

    class AbstractMotorBase: public std::enable_shared_from_this<AbstractMotorBase>
    {
        /* AKA AbstractSensorBase */
        friend Robot;

    public:
        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Dictionary gathering the configuration options shared between motors
        ///////////////////////////////////////////////////////////////////////////////////////////////
        virtual configHolder_t getDefaultMotorOptions(void)
        {
            configHolder_t config;
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
            float64_t const mechanicalReduction;        ///< Mechanical reduction ratio of the transmission (joint / motor, usually >= 1.0
            bool_t    const enableCommandLimit;
            bool_t    const commandLimitFromUrdf;
            float64_t const commandLimit;
            bool_t    const enableArmature;
            float64_t const armature;

            abstractMotorOptions_t(configHolder_t const & options) :
            mechanicalReduction(boost::get<float64_t>(options.at("mechanicalReduction"))),
            enableCommandLimit(boost::get<bool_t>(options.at("enableCommandLimit"))),
            commandLimitFromUrdf(boost::get<bool_t>(options.at("commandLimitFromUrdf"))),
            commandLimit(boost::get<float64_t>(options.at("commandLimit"))),
            enableArmature(boost::get<bool_t>(options.at("enableArmature"))),
            armature(boost::get<float64_t>(options.at("armature")))
            {
                // Empty.
            }
        };

    public:
        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Forbid the copy of the class
        ///////////////////////////////////////////////////////////////////////////////////////////////
        AbstractMotorBase(AbstractMotorBase const & abstractMotor) = delete;
        AbstractMotorBase & operator = (AbstractMotorBase const & other) = delete;

        auto shared_from_this() { return shared_from(this); }
        auto shared_from_this() const { return shared_from(this); }

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Constructor
        ///
        /// \param[in]  robot   Robot
        /// \param[in]  name    Name of the motor
        ///////////////////////////////////////////////////////////////////////////////////////////////
        AbstractMotorBase(std::string const & name);
        virtual ~AbstractMotorBase(void);

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief    Refresh the proxies.
        ///
        /// \remark   This method is not intended to be called manually. The Robot to which the
        ///           motor is added is taking care of it when its own `refresh` method is called.
        ///////////////////////////////////////////////////////////////////////////////////////////////
        virtual hresult_t refreshProxies(void);

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief    Reset the internal state of the motors.
        ///
        /// \details  This method resets the internal state of the motor.
        ///
        /// \remark   This method is not intended to be called manually. The Robot to which the
        ///           motor is added is taking care of it when its own `reset` method is called.
        ///////////////////////////////////////////////////////////////////////////////////////////////
        virtual hresult_t resetAll(void);

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Get the configuration options of the motor.
        ///
        /// \return     Dictionary with the parameters of the motor
        ///////////////////////////////////////////////////////////////////////////////////////////////
        configHolder_t getOptions(void) const;

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Get the actual effort of the motor at the current time.
        ///////////////////////////////////////////////////////////////////////////////////////////////
        float64_t const & get(void) const;

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Get the actual effort of all the motors at the current time.
        ///////////////////////////////////////////////////////////////////////////////////////////////
        vectorN_t const & getAll(void) const;

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Set the configuration options of the motor.
        ///
        /// \param[in]  motorOptions   Dictionary with the parameters of the motor
        ///////////////////////////////////////////////////////////////////////////////////////////////
        virtual hresult_t setOptions(configHolder_t const & motorOptions);

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Set the same configuration options for every motors.
        ///
        /// \param[in]  motorOptions   Dictionary with the parameters used for any motor
        ///////////////////////////////////////////////////////////////////////////////////////////////
        hresult_t setOptionsAll(configHolder_t const & motorOptions);

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Get isInitialized_.
        ///
        /// \details    It is a flag used to determine if the motor has been initialized.
        ///////////////////////////////////////////////////////////////////////////////////////////////
        bool_t const & getIsInitialized(void) const;

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Get name_.
        ///
        /// \details    It is the name of the motor.
        ///////////////////////////////////////////////////////////////////////////////////////////////
        std::string const & getName(void) const;

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Get motorIdx_.
        ///
        /// \details    It is the index of the motor.
        ///////////////////////////////////////////////////////////////////////////////////////////////
        int32_t const & getIdx(void) const;

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Get jointName_.
        ///
        /// \details    It is the name of the joint associated with the motor.
        ///////////////////////////////////////////////////////////////////////////////////////////////
        std::string const & getJointName(void) const;

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Get jointModelIdx_.
        ///
        /// \details    It is the index of the joint associated with the motor in the kinematic tree.
        ///////////////////////////////////////////////////////////////////////////////////////////////
        int32_t const & getJointModelIdx(void) const;

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Get jointType_.
        ///
        /// \details    It is the type of joint associated with the motor.
        ///////////////////////////////////////////////////////////////////////////////////////////////
        joint_t const & getJointType(void) const;

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Get jointPositionIdx_.
        ///
        /// \details    It is the index of the joint associated with the motor in the configuration vector.
        ///////////////////////////////////////////////////////////////////////////////////////////////
        int32_t const & getJointPositionIdx(void) const;

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Get jointVelocityIdx_.
        ///
        /// \details    It is the index of the joint associated with the motor in the velocity vector.
        ///////////////////////////////////////////////////////////////////////////////////////////////
        int32_t const & getJointVelocityIdx(void) const;

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Get commandLimit_.
        ///
        /// \details    It is the maximum effort of the motor.
        ///////////////////////////////////////////////////////////////////////////////////////////////
        float64_t const & getCommandLimit(void) const;

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Get armature_.
        ///
        /// \details    It is the rotor inertia of the motor.
        ///////////////////////////////////////////////////////////////////////////////////////////////
        float64_t const & getArmature(void) const;

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Request the motor to update its actual effort based of the input data.
        ///
        /// \details    It assumes that the internal state of the robot is consistent with the
        ///             input arguments.
        ///
        /// \param[in]  t        Current time.
        /// \param[in]  q        Current configuration of the motor.
        /// \param[in]  v        Current velocity of the motor.
        /// \param[in]  a        Current acceleration of the motor.
        /// \param[in]  command  Current command effort of the motor.
        ///
        ///////////////////////////////////////////////////////////////////////////////////////////////
        virtual hresult_t computeEffort(float64_t const & t,
                                        Eigen::VectorBlock<vectorN_t const> const & q,
                                        float64_t const & v,
                                        float64_t const & a,
                                        float64_t command) = 0;  /* copy on purpose */

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Request every motors to update their actual effort based of the input data.
        ///
        /// \details    It assumes that the internal state of the robot is consistent with the
        ///             input arguments.
        ///
        /// \remark     This method is not intended to be called manually. The Robot to which the
        ///             motor is added is taking care of it while updating the state of the motors.
        ///
        /// \param[in]  t        Current time.
        /// \param[in]  q        Current configuration vector of the robot.
        /// \param[in]  v        Current velocity vector of the robot.
        /// \param[in]  a        Current acceleration vector of the robot.
        /// \param[in]  command  Current command effort vector of the robot.
        ///
        /// \return     Return code to determine whether the execution of the method was successful.
        ///////////////////////////////////////////////////////////////////////////////////////////////
        hresult_t computeEffortAll(float64_t const & t,
                                   vectorN_t const & q,
                                   vectorN_t const & v,
                                   vectorN_t const & a,
                                   vectorN_t const & command);

    protected:
        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Get a reference to the last data buffer corresponding to the actual effort
        ///             of the motor.
        ///////////////////////////////////////////////////////////////////////////////////////////////
        float64_t & data(void);

    private:
        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief    Attach the sensor to a robot
        ///
        /// \details  This method must be called before initializing the sensor.
        ///////////////////////////////////////////////////////////////////////////////////////////////
        hresult_t attach(std::weak_ptr<Robot const> robot,
                         MotorSharedDataHolder_t * sharedHolder);

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief    Detach the sensor from the robot
        ///////////////////////////////////////////////////////////////////////////////////////////////
        hresult_t detach(void);

    public:
        std::unique_ptr<abstractMotorOptions_t const> baseMotorOptions_;  ///< Structure with the parameters of the motor

    protected:
        configHolder_t motorOptionsHolder_;         ///< Dictionary with the parameters of the motor
        bool_t isInitialized_;                      ///< Flag to determine whether the controller has been initialized or not
        bool_t isAttached_;                         ///< Flag to determine whether the motor is attached to a robot
        std::weak_ptr<Robot const> robot_;          ///< Robot for which the command and internal dynamics
        std::string name_;                          ///< Name of the motor
        int32_t motorIdx_;                          ///< Index of the motor in the measurement buffer
        std::string jointName_;
        int32_t jointModelIdx_;
        joint_t jointType_;
        int32_t jointPositionIdx_;
        int32_t jointVelocityIdx_;
        float64_t commandLimit_;
        float64_t armature_;

    private:
        MotorSharedDataHolder_t * sharedHolder_;  ///< Shared data between every motors associated with the robot
    };
}

#endif //end of JIMINY_ABSTRACT_MOTOR_H
