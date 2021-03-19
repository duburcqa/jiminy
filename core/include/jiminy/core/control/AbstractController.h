#ifndef JIMINY_ABSTRACT_CONTROLLER_H
#define JIMINY_ABSTRACT_CONTROLLER_H

#include "jiminy/core/telemetry/TelemetrySender.h"
#include "jiminy/core/Types.h"
#include "jiminy/core/Macros.h"


namespace jiminy
{
    std::string const CONTROLLER_TELEMETRY_NAMESPACE("HighLevelController");  ///< Namespace of the telemetry object

    class TelemetryData;
    class Robot;
    class Engine;

    ///////////////////////////////////////////////////////////////////////////////////////////////
    ///
    /// \brief          Generic interface for any controller.
    ///
    ///                 Any controller must inherit from this base class and implement its virtual
    ///                 methods.
    ///
    ///                 The controller used to initialize a Jiminy Engine is downcasted as an
    ///                 instance of AbstractController and polymorphism is used to call the actual
    ///                 implementations.
    ///
    //////////////////////////////////////////////////////////////////////////////////////////////
    class AbstractController
    {
    public:
        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Dictionary gathering the configuration options shared between controllers
        ///////////////////////////////////////////////////////////////////////////////////////////////
        virtual configHolder_t getDefaultControllerOptions()
        {
            configHolder_t config;
            config["telemetryEnable"] = true;

            return config;
        };

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Structure with the configuration options shared between controllers
        ///////////////////////////////////////////////////////////////////////////////////////////////
        struct controllerOptions_t
        {
            bool_t const telemetryEnable;     ///< Flag used to enable the telemetry of the controller

            controllerOptions_t(configHolder_t const & options) :
            telemetryEnable(boost::get<bool_t>(options.at("telemetryEnable")))
            {
                // Empty.
            }
        };

    public:
        // Forbid the copy of the class
        AbstractController(AbstractController const & controller) = delete;
        AbstractController & operator = (AbstractController const & controller) = delete;

    public:
        AbstractController(void);
        virtual ~AbstractController(void) = default;

        ///////////////////////////////////////////////////////////////////////////////////////////////
        ///
        /// \brief      Set the parameters of the controller.
        ///
        /// \param[in]  robot   Robot
        ///
        /// \return     Return code to determine whether the execution of the method was successful.
        ///
        ///////////////////////////////////////////////////////////////////////////////////////////////
        virtual hresult_t initialize(std::weak_ptr<Robot const> robot);

        ///////////////////////////////////////////////////////////////////////////////////////////////
        ///
        /// \brief      Dynamically registered a Eigen Vector to the telemetry.
        ///
        /// \details    Internally, all it does is to store a reference to the variable, then it logs
        ///             its value periodically. There is no update mechanism what so ever nor safety
        ///             check. So the user has to take care of the life span of the variable, and to
        ///             update it manually whenever it is necessary to do so.
        ///
        /// \param[in]  fieldnames      Name of each element of the variable. It will appear in the header of the log.
        /// \param[in]  values          Eigen vector to add to the telemetry. It accepts non-contiguous temporary.
        ///
        /// \return     Return code to determine whether the execution of the method was successful.
        ///
        ///////////////////////////////////////////////////////////////////////////////////////////////
        hresult_t registerVariable(std::vector<std::string> const & fieldnames,
                                   Eigen::Ref<vectorN_t, 0, Eigen::InnerStride<> > values);  // Make a "copy" to support both vectorN_t and reference

        ///////////////////////////////////////////////////////////////////////////////////////////////
        ///
        /// \brief      Dynamically registered a float64 to the telemetry.
        ///
        /// \details    Internally, all it does is to store a reference to the variable, then it logs
        ///             its value periodically. There is no update mechanism what so ever nor safety
        ///             check. So the user has to take care of the life span of the variable, and to
        ///             update it manually whenever it is necessary to do so.
        ///
        /// \param[in]  fieldnames      Name of the variable. It will appear in the header of the log.
        /// \param[in]  values          Variable to add to the telemetry
        ///
        /// \return     Return code to determine whether the execution of the method was successful.
        ///
        ///////////////////////////////////////////////////////////////////////////////////////////////
        hresult_t registerVariable(std::string const & fieldName,
                                   float64_t   const & value);

        ///////////////////////////////////////////////////////////////////////////////////////////////
        ///
        /// \brief      Register a constant float64 to the telemetry.
        ///
        /// \param[in]  fieldnames      Name of the variable.
        /// \param[in]  values          Variable to add to the telemetry
        ///
        /// \return     Return code to determine whether the execution of the method was successful.
        ///
        ///////////////////////////////////////////////////////////////////////////////////////////////
        template<typename T>
        hresult_t registerConstant(std::string const & fieldName,
                                   T           const & value);

        ///////////////////////////////////////////////////////////////////////////////////////////////
        ///
        /// \brief      Remove all variables dynamically registered to the telemetry.
        ///
        /// \details    Note that one must reset Jiminy Engine for this to take effect.
        ///
        ///////////////////////////////////////////////////////////////////////////////////////////////
        void removeEntries(void);

        ///////////////////////////////////////////////////////////////////////////////////////////////
        ///
        /// \brief      Compute the command.
        ///
        /// \details    It assumes that the robot internal state (including sensors) is consistent
        ///             with other input arguments. It fetches the sensor data automatically.
        ///
        /// \param[in]  t        Current time
        /// \param[in]  q        Current configuration vector
        /// \param[in]  v        Current velocity vector
        /// \param[out] command  Output effort vector
        ///
        /// \return     Return code to determine whether the execution of the method was successful.
        ///
        ///////////////////////////////////////////////////////////////////////////////////////////////
        virtual hresult_t computeCommand(float64_t const & t,
                                         vectorN_t const & q,
                                         vectorN_t const & v,
                                         vectorN_t       & command) = 0;

        ///////////////////////////////////////////////////////////////////////////////////////////////
        ///
        /// \brief      Emulate internal dynamics of the system at are not included in the
        ///             physics engine.
        ///
        /// \param[in]  t        Current time
        /// \param[in]  q        Current configuration vector
        /// \param[in]  v        Current velocity vector
        /// \param[in]  uCustom  Output effort vector
        ///
        /// \return     Return code to determine whether the execution of the method was successful.
        ///
        ///////////////////////////////////////////////////////////////////////////////////////////////
        virtual hresult_t internalDynamics(float64_t const & t,
                                           vectorN_t const & q,
                                           vectorN_t const & v,
                                           vectorN_t       & uCustom) = 0;

        ///////////////////////////////////////////////////////////////////////////////////////////////
        ///
        /// \brief      Get the configuration options of the controller.
        ///
        /// \return     Dictionary with the parameters of the controller
        ///
        ///////////////////////////////////////////////////////////////////////////////////////////////
        configHolder_t getOptions(void) const;

        ///////////////////////////////////////////////////////////////////////////////////////////////
        ///
        /// \brief      Set the configuration options of the controller.
        ///
        /// \details    Note that one must reset Jiminy Engine for this to take effect.
        ///
        /// \param[in]  ctrlOptions   Dictionary with the parameters of the controller
        ///
        /// \return     Return code to determine whether the execution of the method was successful.
        ///
        ///////////////////////////////////////////////////////////////////////////////////////////////
        void setOptions(configHolder_t const & ctrlOptions);

        ///////////////////////////////////////////////////////////////////////////////////////////////
        ///
        /// \brief      Configure the telemetry of the controller.
        ///
        /// \details    This method connects the controller-specific telemetry sender to a given
        ///             telemetry data (which is unique for a given robot), so that it is
        ///             later possible to register the variables that one want to monitor. Finally,
        ///             the telemetry recoder logs every registered variables at each timestep in a
        ///             memory buffer.
        ///
        /// \remark     This method is not intended to be called manually. The Engine is taking care
        ///             of it before flushing the telemetry data at the end of each simulation steps.
        ///
        /// \param[in]  telemetryData       Shared pointer to the robot-wide telemetry data object
        ///
        /// \return     Return code to determine whether the execution of the method was successful.
        ///
        ///////////////////////////////////////////////////////////////////////////////////////////////
        virtual hresult_t configureTelemetry(std::shared_ptr<TelemetryData> telemetryData,
                                             std::string const & objectPrefixName = "");

        ///////////////////////////////////////////////////////////////////////////////////////////////
        ///
        /// \brief      Update the internal buffers of the telemetry associated with variables
        ///             monitored by the controller.
        /// \details    As the main entry point for a user to log extra variables, the engine also
        ///             passes the current state of the robot to enable logging of custom state-related variables.
        ///
        ///
        /// \param[in]  t Current time.
        /// \param[in]  q Current position.
        /// \param[in]  v Current velocity.
        ///
        /// \remark     This method is not intended to be called manually. The Engine is taking care
        ///             of it before flushing the telemetry data at the end of each simulation steps.
        ///
        ///////////////////////////////////////////////////////////////////////////////////////////////
        virtual void updateTelemetry(void);

        ///////////////////////////////////////////////////////////////////////////////////////////////
        ///
        /// \brief      Reset the internal state of the controller.
        ///
        /// \details    Note that it resets the configuration of the telemetry.
        ///
        /// \remark     This method is not intended to be called manually. The Engine is taking care
        ///             of it when its own `reset` method is called.
        ///
        /// \param[in]  resetDynamicTelemetry   Whether or not variables dynamically registered to the
        ///                                     telemetry must be removed.
        ///                                     Optional: False by default
        ///
        ///////////////////////////////////////////////////////////////////////////////////////////////
        virtual hresult_t reset(bool_t const & resetDynamicTelemetry = false);

        ///////////////////////////////////////////////////////////////////////////////////////////////
        ///
        /// \brief      Get isInitialized_.
        ///
        /// \details    It is a flag used to determine if the controller has been initialized.
        ///
        /// \remark     Note that a controller can be considered initialized even if its telemetry is
        ///             not properly configured. If not, it is the only to do before being ready to
        ///             use.
        ///
        ///////////////////////////////////////////////////////////////////////////////////////////////
        bool_t const & getIsInitialized(void) const;

        ///////////////////////////////////////////////////////////////////////////////////////////////
        ///
        /// \brief      Get isTelemetryConfigured_.
        ///
        /// \details    It is a flag used to determine if the telemetry of the controller has been
        ///             initialized.
        ///
        ///////////////////////////////////////////////////////////////////////////////////////////////
        bool_t const & getIsTelemetryConfigured(void) const;

    public:
        std::unique_ptr<controllerOptions_t const> baseControllerOptions_;    ///< Structure with the parameters of the controller
        std::weak_ptr<Robot const> robot_;                                    ///< Robot for which to compute the command and internal dynamics must be computed
        sensorsDataMap_t sensorsData_;

    protected:
        bool_t isInitialized_;                  ///< Flag to determine whether the controller has been initialized or not
        bool_t isTelemetryConfigured_;          ///< Flag to determine whether the telemetry of the controller has been initialized or not
        configHolder_t ctrlOptionsHolder_;      ///< Dictionary with the parameters of the controller
        TelemetrySender telemetrySender_;       ///< Telemetry sender of the controller used to register and update telemetry variables

    private:
        static_map_t<std::string, float64_t const *> registeredVariables_;    ///< Vector of dynamically registered telemetry variables
        static_map_t<std::string, std::string> registeredConstants_;          ///< Vector of dynamically registered telemetry constants
    };
}

#include "AbstractController.tpp"

#endif //end of JIMINY_ABSTRACT_CONTROLLER_H
