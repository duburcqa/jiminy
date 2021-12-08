///////////////////////////////////////////////////////////////////////////////////////////////
///
/// \brief      Generic interface for any transmission.
///
/// \details    Any transmission must inherit from this base class and implement its virtual methods.
///
/// \remark     Each transmission added to a Jiminy Robot is downcasted as an instance of
///             AbstractTransmissionBase and polymorphism is used to call the actual implementations.
///
///////////////////////////////////////////////////////////////////////////////////////////////

#ifndef JIMINY_ABSTRACT_TRANSMISSION_H
#define JIMINY_ABSTRACT_TRANSMISSION_H

#include <memory>

#include "jiminy/core/Macros.h"
#include "jiminy/core/Types.h"


namespace jiminy
{
    class Robot;

    class AbstractTransmissionBase : public std::enable_shared_from_this<AbstractTransmissionBase>
    {
        /* AKA AbstractSensorBase */
        friend Robot;

    public:
        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Dictionary gathering the configuration options shared between transmissions
        ///////////////////////////////////////////////////////////////////////////////////////////////
        virtual configHolder_t getDefaultTransmissionOptions(void)
        {
            configHolder_t config;
            config["mechanicalReduction"] = 0.0;
            return config;
        };

        struct abstractTransmissionOptions_t
        {
            float64_t const mechanicalReduction;

            abstractTransmissionOptions_t(configHolder_t const & options) :
            mechanicalReduction(boost::get<float64_t>(options.at("mechanicalReduction")))
            {
                // Empty.
            }
        };

    public:
        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Forbid the copy of the class
        ///////////////////////////////////////////////////////////////////////////////////////////////
        AbstractTransmissionBase(AbstractTransmissionBase const & abstractTransmission) = delete;
        AbstractTransmissionBase & operator = (AbstractTransmissionBase const & other) = delete;

        auto shared_from_this() { return shared_from(this); }
        auto shared_from_this() const { return shared_from(this); }

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Constructor
        ///
        /// \param[in]  robot   Robot
        /// \param[in]  name    Name of the transmission
        ///////////////////////////////////////////////////////////////////////////////////////////////
        AbstractTransmissionBase(std::string const & name);
        virtual ~AbstractTransmissionBase(void);

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief    Initialize
        ///
        /// \remark   Initialize the transmission with the names of connected motors and actuated joits.
        ///////////////////////////////////////////////////////////////////////////////////////////////
        virtual hresult_t initialize(std::vector<std::string> const & jointName,
                                     std::vector<std::string> const & motorName);

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief    Refresh the proxies.
        ///
        /// \remark   This method is not intended to be called manually. The Robot to which the
        ///           transmission is added is taking care of it when its own `refresh` method is called.
        ///////////////////////////////////////////////////////////////////////////////////////////////
        virtual hresult_t refreshProxies(void);

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief    Reset the internal state of the transmissions.
        ///
        /// \details  This method resets the internal state of the transmission.
        ///
        /// \remark   This method is not intended to be called manually. The Robot to which the
        ///           transmission is added is taking care of it when its own `reset` method is called.
        ///////////////////////////////////////////////////////////////////////////////////////////////
        virtual hresult_t resetAll(void);

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Get the configuration options of the transmission.
        ///
        /// \return     Dictionary with the parameters of the transmission
        ///////////////////////////////////////////////////////////////////////////////////////////////
        configHolder_t getOptions(void) const;

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Get the actual state of the transmission at the current time.
        ///////////////////////////////////////////////////////////////////////////////////////////////
        float64_t const & get(void) const;

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Set the configuration options of the transmission.
        ///
        /// \param[in]  transmissionOptions   Dictionary with the parameters of the transmission
        ///////////////////////////////////////////////////////////////////////////////////////////////
        virtual hresult_t setOptions(configHolder_t const & transmissionOptions);

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Set the same configuration options for every transmissions.
        ///
        /// \param[in]  transmissionOptions   Dictionary with the parameters used for any transmission
        ///////////////////////////////////////////////////////////////////////////////////////////////
        hresult_t setOptionsAll(configHolder_t const & transmissionOptions);

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Get isInitialized_.
        ///
        /// \details    It is a flag used to determine if the transmission has been initialized.
        ///////////////////////////////////////////////////////////////////////////////////////////////
        bool_t const & getIsInitialized(void) const;

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Get name_.
        ///
        /// \details    It is the name of the transmission.
        ///////////////////////////////////////////////////////////////////////////////////////////////
        std::string const & getName(void) const;

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Get transmissionIdx_.
        ///
        /// \details    It is the index of the transmission.
        ///////////////////////////////////////////////////////////////////////////////////////////////
        int32_t const & getIdx(void) const;

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Get jointName_.
        ///
        /// \details    It is the name of the joints associated with the transmission.
        ///////////////////////////////////////////////////////////////////////////////////////////////
        std::vector<std::string> const & getJointNames(void) const;

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Get jointModelIdx_.
        ///
        /// \details    It is the index of the joints associated with the transmission in the kinematic tree.
        ///////////////////////////////////////////////////////////////////////////////////////////////
        std::vector<jointIndex_t >const & getJointModelIndices(void) const;

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Get jointType_.
        ///
        /// \details    It is the type of joints associated with the transmission.
        ///////////////////////////////////////////////////////////////////////////////////////////////
        std::vector<joint_t> const & getJointTypes(void) const;

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Get jointPositionIdx_.
        ///
        /// \details    It is the index of the joints associated with the transmission in the configuration vector.
        ///////////////////////////////////////////////////////////////////////////////////////////////
        vectorN_t const & getJointPositionIndices(void) const;

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Get jointVelocityIdx_.
        ///
        /// \details    It is the index of the joints associated with the transmission in the velocity vector.
        ///////////////////////////////////////////////////////////////////////////////////////////////
        vectorN_t const & getJointVelocityIndices(void) const;

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Get motorName_.
        ///
        /// \details    It is the name of the motors associated with the transmission.
        ///////////////////////////////////////////////////////////////////////////////////////////////
        std::vector<std::string> const & getMotorNames(void) const;

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Compute forward transmission.
        ///
        /// \details    Compute forward transmission from motor to joint.
        ///////////////////////////////////////////////////////////////////////////////////////////////
        virtual hresult_t computeForward(float64_t const & t,
                                 vectorN_t & q,
                                 vectorN_t & v,
                                 vectorN_t & a,
                                 vectorN_t & uJoint) final;

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Compute backward transmission.
        ///
        /// \details    Compute backward transmission from joint to motor.
        ///////////////////////////////////////////////////////////////////////////////////////////////
        virtual hresult_t computeBackward(float64_t const & t,
                                  vectorN_t const & q,
                                  vectorN_t const & v,
                                  vectorN_t const & a,
                                  vectorN_t const & uJoint) final;

    protected:
        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Request the transmission to update its actual state based of the input data.
        ///
        /// \details    It assumes that the internal state of the robot is consistent with the
        ///             input arguments.
        ///
        /// \param[in]  q        Current configuration of the motors.
        /// \param[in]  v        Current velocity of the motors.
        ///////////////////////////////////////////////////////////////////////////////////////////////
        virtual void computeTransform(Eigen::VectorBlock<vectorN_t const> const & q,
                                      Eigen::VectorBlock<vectorN_t const> const & v) = 0;  /* copy on purpose */

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Request the transmission to update its actual state based of the input data.
        ///
        /// \details    It assumes that the internal state of the robot is consistent with the
        ///             input arguments.
        ///
        /// \param[in]  q        Current configuration of the motors.
        /// \param[in]  v        Current velocity of the motors.
        ///////////////////////////////////////////////////////////////////////////////////////////////
        virtual void computeInverseTransform(Eigen::VectorBlock<vectorN_t const> const & q,
                                             Eigen::VectorBlock<vectorN_t const> const & v) = 0;  /* copy on purpose */


        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Compute energy dissipation in the transmission.
        ///
        ///////////////////////////////////////////////////////////////////////////////////////////////
        virtual computeEffortTransmission(void) = 0;

    private:
        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief    Attach the transmission to a robot
        ///
        /// \details  This method must be called before initializing the transmission.
        ///////////////////////////////////////////////////////////////////////////////////////////////
        hresult_t attach(std::weak_ptr<Robot const> robot);

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief    Detach the transmission from the robot
        ///////////////////////////////////////////////////////////////////////////////////////////////
        hresult_t detach(void);

    public:
        std::unique_ptr<abstractTransmissionOptions_t const> baseTransmissionOptions_;  ///< Structure with the parameters of the transmission

    protected:
        configHolder_t transmissionOptionsHolder_;                   ///< Dictionary with the parameters of the transmission
        bool_t isInitialized_;                                       ///< Flag to determine whether the transmission has been initialized or not
        bool_t isAttached_;                                          ///< Flag to determine whether the transmission is attached to a robot
        std::weak_ptr<Robot const> robot_;                           ///< Robot for which the command and internal dynamics
        std::string name_;                                           ///< Name of the transmission
        int32_t transmissionIdx_;                                           ///< Index of the transmission in the transmission buffer
        std::vector<std::string> jointNames_;
        std::vector<jointIndex_t> jointModelIndices_;
        std::vector<joint_t> jointTypes_;
        vectorN_t jointPositionIndices_;
        vectorN_t jointVelocityIndices_;
        std::vector<std::string> motorNames_;
        std::vector<std::weak_ptr<AbstractMotorBase> > motors_;
        matrixN_t forwardTransform_;
        matrixN_t backwardTransform_;

    };
}

#endif //end of JIMINY_ABSTRACT_TRANSMISSION_H
