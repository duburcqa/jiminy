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
        position_(),
        velocity_(),
        acceleration_(),
        effort_(),
        motors_(),
        num_(0)
        {
            // Empty.
        };

        ~MotorSharedDataHolder_t(void) = default;

        vectorN_t position_;
        vectorN_t velocity_;
        vectorN_t acceleration_;
        vectorN_t effort_;

        std::vector<AbstractMotorBase *> motors_;  ///< Vector of pointers to the motors.
        int32_t num_;                              ///< Number of motors
    };

    class AbstractMotorBase : public std::enable_shared_from_this<AbstractMotorBase>
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
            config["enableCommandLimit"] = true;
            config["commandLimitFromUrdf"] = true;
            config["commandLimit"] = 0.0;
            config["enableArmature"] = false;
            config["armature"] = 0.0;

            return config;
        };

        struct abstractMotorOptions_t
        {
            bool_t    const enableCommandLimit;
            bool_t    const commandLimitFromUrdf;
            float64_t const commandLimit;
            bool_t    const enableArmature;
            float64_t const armature;

            abstractMotorOptions_t(configHolder_t const & options) :
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
        /// \brief      Get the actual position of the motor at the current time.
        ///////////////////////////////////////////////////////////////////////////////////////////////
        float64_t const & getPosition(void) const;

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Get the actual velocity of the motor at the current time.
        ///////////////////////////////////////////////////////////////////////////////////////////////
        float64_t const & getVelocity(void) const;

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Get the actual acc of the motor at the current time.
        ///////////////////////////////////////////////////////////////////////////////////////////////
        float64_t const & getAcceleration(void) const;

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Get the actual effort of the motor at the current time.
        ///////////////////////////////////////////////////////////////////////////////////////////////
        float64_t const & getEffort(void) const;

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Get the actual position of all the motor at the current time.
        ///////////////////////////////////////////////////////////////////////////////////////////////
        vectorN_t const & getPositionAll(void) const;

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Get the actual velocity of all the motor at the current time.
        ///////////////////////////////////////////////////////////////////////////////////////////////
        vectorN_t const & getVelocityAll(void) const;

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Get the actual acc of all the motor at the current time.
        ///////////////////////////////////////////////////////////////////////////////////////////////
        vectorN_t const & getAccelerationAll(void) const;

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Get the actual effort of all the motor at the current time.
        ///////////////////////////////////////////////////////////////////////////////////////////////
        vectorN_t const & getEffortAll(void) const;

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
        /// \brief      Get name_.
        ///
        /// \details    It is the name of the motor.
        ///////////////////////////////////////////////////////////////////////////////////////////////
        std::string const & getName(void) const;

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
        /// \param[in]  command  Current command effort of the motor.
        ///
        ///////////////////////////////////////////////////////////////////////////////////////////////
        virtual hresult_t computeEffort(float64_t command) = 0;  /* copy on purpose */

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Request every motors to update their actual effort based of the input data.
        ///
        /// \details    It assumes that the internal state of the robot is consistent with the
        ///             input arguments.
        ///
        /// \remark     This method is not intended to be called manually. The Robot to which the
        ///             motor is added is taking care of it while updating the state of the motors.
        ///
        /// \param[in]  command  Current command effort vector of the robot.
        ///
        /// \return     Return code to determine whether the execution of the method was successful.
        ///////////////////////////////////////////////////////////////////////////////////////////////
        hresult_t computeEffortAll(vectorN_t const & command);

    protected:
        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Get a reference to the last data buffer corresponding to the actual position
        ///             of the motor.
        ///////////////////////////////////////////////////////////////////////////////////////////////
        float64_t & q(void);

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Get a reference to the last data buffer corresponding to the actual velocity
        ///             of the motor.
        ///////////////////////////////////////////////////////////////////////////////////////////////
        float64_t & v(void);

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Get a reference to the last data buffer corresponding to the actual acc
        ///             of the motor.
        ///////////////////////////////////////////////////////////////////////////////////////////////
        float64_t & a(void);

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Get a reference to the last data buffer corresponding to the actual effort
        ///             of the motor.
        ///////////////////////////////////////////////////////////////////////////////////////////////
        float64_t & u(void);             

    private:
        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief    Attach the motor to a robot
        ///
        /// \details  This method must be called before initializing the motor.
        ///////////////////////////////////////////////////////////////////////////////////////////////
        hresult_t attach(std::weak_ptr<Robot const> robot,
                         std::function<hresult_t(AbstractMotorBase & /* motor */)> notifyRobot,
                         MotorSharedDataHolder_t * sharedHolder);

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief    Detach the motor from the robot
        ///////////////////////////////////////////////////////////////////////////////////////////////
        hresult_t detach(void);

    public:
        std::unique_ptr<abstractMotorOptions_t const> baseMotorOptions_;  ///< Structure with the parameters of the motor

    protected:
        configHolder_t motorOptionsHolder_;                          ///< Dictionary with the parameters of the motor
        bool_t isAttached_;                                          ///< Flag to determine whether the motor is attached to a robot
        std::weak_ptr<Robot const> robot_;                           ///< Robot for which the command and internal dynamics
        std::function<hresult_t(AbstractMotorBase &)> notifyRobot_;  ///< Notify the robot that the configuration of the motors have changed
        std::string name_;                                           ///< Name of the motor
        int32_t motorIdx_;                                           ///< Index of the motor in the measurement buffer
        float64_t commandLimit_;
        float64_t armature_; 

    private:
        MotorSharedDataHolder_t * sharedHolder_;  ///< Shared data between every motors associated with the robot
    };
}

#endif //end of JIMINY_ABSTRACT_MOTOR_H
