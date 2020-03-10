///////////////////////////////////////////////////////////////////////////////////////////////
///
/// \brief      Generic interface for any motor.
///
/// \details    Any motor must inherit from this base class and implement its virtual methods.
///
/// \remarks    Each motor added to a Jiminy Model is downcasted as an instance of
///             AbstractMotorBase and polymorphism is used to call the actual implementations.
///
///////////////////////////////////////////////////////////////////////////////////////////////

#ifndef JIMINY_ABSTRACT_MOTOR_H
#define JIMINY_ABSTRACT_MOTOR_H

#include "jiminy/core/Types.h"


namespace jiminy
{
    class Model;

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

        vectorN_t data_;                            ///< Buffer with current actual motor torque
        std::vector<AbstractMotorBase *> motors_;   ///< Vector of pointers to the motors
        uint8_t num_;                               ///< Number of motors
    };

    class AbstractMotorBase
    {
    public:
        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Dictionary gathering the configuration options shared between motors
        ///////////////////////////////////////////////////////////////////////////////////////////////
        virtual configHolder_t getDefaultOptions(void)
        {
            configHolder_t config;
            config["enableTorqueLimit"] = true;
            config["torqueLimitFromUrdf"] = true;
            config["torqueLimit"] = 0.0;
            config["enableMotorInertia"] = false;
            config["motorInertia"] = 0.0;

            return config;
        };

        struct abstractMotorOptions_t
        {
            bool_t    const enableMotorInertia;
            float64_t const motorInertia;
            bool_t    const enableTorqueLimit;
            bool_t    const torqueLimitFromUrdf;
            float64_t const torqueLimit;

            abstractMotorOptions_t(configHolder_t const & options) :
            enableMotorInertia(boost::get<bool_t>(options.at("enableMotorInertia"))),
            motorInertia(boost::get<float64_t>(options.at("motorInertia"))),
            enableTorqueLimit(boost::get<bool_t>(options.at("enableTorqueLimit"))),
            torqueLimitFromUrdf(boost::get<bool_t>(options.at("torqueLimitFromUrdf"))),
            torqueLimit(boost::get<float64_t>(options.at("torqueLimit")))
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

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Constructor
        ///
        /// \param[in]  model   Model of the system
        /// \param[in]  name    Name of the motor
        ///////////////////////////////////////////////////////////////////////////////////////////////
        AbstractMotorBase(Model       const & model,
                          std::shared_ptr<MotorSharedDataHolder_t> const & sharedHolder,
                          std::string const & name);

        virtual ~AbstractMotorBase(void);

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief    Plug the motor on a given joint of the model.
        ///////////////////////////////////////////////////////////////////////////////////////////////
        result_t initialize(std::string const & jointName);

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief    Refresh the proxies.
        ///
        /// \remark   This method is not intended to be called manually. The Model to which the
        ///           motor is added is taking care of it when its own `refresh` method is called.
        ///////////////////////////////////////////////////////////////////////////////////////////////
        virtual result_t refreshProxies(void);

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief    Reset the internal state of the motor.
        ///
        /// \details  This method resets the internal state of the motor.
        ///
        /// \remark   This method is not intended to be called manually. The Model to which the
        ///           motor is added is taking care of it when its own `reset` method is called.
        ///////////////////////////////////////////////////////////////////////////////////////////////
        virtual void reset(void);

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Get the configuration options of the motor.
        ///
        /// \return     Dictionary with the parameters of the motor
        ///////////////////////////////////////////////////////////////////////////////////////////////
        configHolder_t getOptions(void) const;

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Get the actual torque of the motor at the current time.
        ///////////////////////////////////////////////////////////////////////////////////////////////
        float64_t const & get(void) const;

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Get the actual torque of all the motors at the current time.
        ///////////////////////////////////////////////////////////////////////////////////////////////
        vectorN_t const & getAll(void) const;

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Set the configuration options of the motor.
        ///
        /// \param[in]  motorOptions   Dictionary with the parameters of the motor
        ///////////////////////////////////////////////////////////////////////////////////////////////
        virtual result_t setOptions(configHolder_t const & motorOptions);

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Set the same configuration options for every motors.
        ///
        /// \param[in]  motorOptions   Dictionary with the parameters used for any motor
        ///////////////////////////////////////////////////////////////////////////////////////////////
        result_t setOptionsAll(configHolder_t const & motorOptions);

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
        /// \brief      Get motorId_.
        ///
        /// \details    It is the index of the motor.
        ///////////////////////////////////////////////////////////////////////////////////////////////
        uint8_t const & getId(void) const;

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
        /// \brief      Get torqueLimit_.
        ///
        /// \details    It is the maximum torque of the motor.
        ///////////////////////////////////////////////////////////////////////////////////////////////
        float64_t const & getTorqueLimit(void) const;

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Request the motor to update its actual torque based of the input data.
        ///
        /// \details    It assumes that the internal state of the model is consistent with the
        ///             input arguments.
        ///
        /// \param[in]  t       Current time
        /// \param[in]  q       Current configuration of the motor
        /// \param[in]  v       Current velocity of the motor
        /// \param[in]  a       Current acceleration of the motor
        /// \param[in]  u       Current command torque of the motor
        ///
        ///////////////////////////////////////////////////////////////////////////////////////////////
        virtual result_t computeEffort(float64_t const & t,
                                       float64_t const & q,
                                       float64_t const & v,
                                       float64_t const & a,
                                       float64_t const & uCommand) = 0;

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Request every motors to update their actual torque based of the input data.
        ///
        /// \details    It assumes that the internal state of the model is consistent with the
        ///             input arguments.
        ///
        /// \remark     This method is not intended to be called manually. The Model to which the
        ///             motor is added is taking care of it while updating the state of the motors.
        ///
        /// \param[in]  t       Current time
        /// \param[in]  q       Current configuration vector
        /// \param[in]  v       Current velocity vector
        /// \param[in]  a       Current acceleration vector
        /// \param[in]  u       Current command torque vector
        ///
        /// \return     Return code to determine whether the execution of the method was successful.
        ///////////////////////////////////////////////////////////////////////////////////////////////
        result_t computeAllEffort(float64_t const & t,
                                  vectorN_t const & q,
                                  vectorN_t const & v,
                                  vectorN_t const & a,
                                  vectorN_t const & uCommand);

        void clearDataBuffer(void);

    protected:
        float64_t & data(void);

    public:
        std::unique_ptr<abstractMotorOptions_t const> baseMotorOptions_; ///< Structure with the parameters of the motor

    protected:
        configHolder_t motorOptionsHolder_;         ///< Dictionary with the parameters of the motor
        bool_t isInitialized_;                      ///< Flag to determine whether the controller has been initialized or not
        Model const * model_;                       ///< Model of the system for which the command and internal dynamics

    private:
        std::shared_ptr<MotorSharedDataHolder_t> sharedHolder_;    ///< Shared data between every motors associated with the model
        std::string name_;                          ///< Name of the motor
        uint8_t motorId_;                           ///< Index of the motor in the measurement buffer

        std::string jointName_;
        int32_t jointModelIdx_;
        int32_t jointPositionIdx_;
        int32_t jointVelocityIdx_;
        float64_t torqueLimit_;
    };
}

#endif //end of JIMINY_ABSTRACT_MOTOR_H
