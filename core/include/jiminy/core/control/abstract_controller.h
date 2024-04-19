#ifndef JIMINY_ABSTRACT_CONTROLLER_H
#define JIMINY_ABSTRACT_CONTROLLER_H

#include <variant>

#include "jiminy/core/fwd.h"
#include "jiminy/core/hardware/fwd.h"
#include "jiminy/core/hardware/abstract_sensor.h"


namespace jiminy
{
    /// \brief Namespace of the telemetry object.
    inline constexpr std::string_view CONTROLLER_TELEMETRY_NAMESPACE{"controller"};

    class TelemetrySender;
    class TelemetryData;
    class Robot;

    /// \brief Generic interface for any controller.
    ///
    /// \details Any controller must inherit from this base class and implement its virtual
    ///          methods. Polymorphism is used to call the actual implementations.
    class JIMINY_DLLAPI AbstractController
    {
    public:
        /// \brief Dictionary gathering the configuration options shared between controllers.
        virtual GenericConfig getDefaultControllerOptions()
        {
            GenericConfig config;
            config["telemetryEnable"] = true;

            return config;
        };

        /// \brief Structure with the configuration options shared between controllers.
        struct ControllerOptions
        {
            /// \brief Flag used to enable the telemetry of the controller.
            const bool telemetryEnable;

            ControllerOptions(const GenericConfig & options) :
            telemetryEnable(boost::get<bool>(options.at("telemetryEnable")))
            {
            }
        };

    public:
        JIMINY_DISABLE_COPY(AbstractController)

    public:
        explicit AbstractController() noexcept;
        virtual ~AbstractController();

        /// \brief Initialize the internal state of the controller for a given robot.
        ///
        /// \details This method can be called multiple times with different robots. The internal
        ///          state of the controller will be systematically overwritten.
        ///
        /// \param[in] robot Robot
        virtual void initialize(std::weak_ptr<const Robot> robot);

        /// \brief Register a constant (so-called invariant) to the telemetry.
        ///
        /// \details The user is responsible to convert it as a byte array (eg `std::string`),
        ///          either using `toString` for arithmetic types or `saveToBinary` complex types.
        ///
        /// \param[in] name Name of the constant.
        /// \param[in] value Constant to add to the telemetry.
        void registerConstant(const std::string_view & name, const std::string & value);

        /// \brief Dynamically registered a scalar variable to the telemetry. It is the main entry
        ///        point for a user to log custom variables.
        ///
        /// \details Internally, all it does is to store a reference to the variable, then it logs
        ///          its value periodically. There is no update mechanism what so ever nor safety
        ///          check. The user has to take care of the life span of the variable, and to
        ///          update it manually whenever it is necessary to do so.
        ///
        /// \param[in] name Name of the variable. It will appear in the header of the log.
        /// \param[in] values Variable to add to the telemetry.
        template<typename T>
        void registerVariable(const std::string_view & name, const T & value);

        /// \brief Dynamically registered a Eigen Vector to the telemetry.
        ///
        /// \param[in] fieldnames Name of each element of the variable. It will appear in the
        ///                       header of the log.
        /// \param[in] values Eigen vector to add to the telemetry. It accepts non-contiguous
        ///                   temporary.
        void registerVariable(
            const std::vector<std::string> & fieldnames,
            const Eigen::Ref<VectorX<double>, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> &
                values);
        void registerVariable(
            const std::vector<std::string> & fieldnames,
            const Eigen::Ref<VectorX<int64_t>, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> &
                values);

        /// \brief Remove all variables dynamically registered to the telemetry.
        ///
        /// \details Note that one must reset Jiminy Engine for this to take effect.
        void removeEntries();

        /// \brief Compute the command.
        ///
        /// \details It assumes that the robot internal state (including sensors) is consistent
        ///          with other input arguments. It fetches the sensor data automatically.
        ///
        /// \param[in] t Current time.
        /// \param[in] q Current configuration vector.
        /// \param[in] v Current velocity vector.
        /// \param[out] command Output effort vector.
        virtual void computeCommand(double t,
                                    const Eigen::VectorXd & q,
                                    const Eigen::VectorXd & v,
                                    Eigen::VectorXd & command) = 0;

        /// \brief Emulate custom phenomenon that are part of the internal dynamics of the system
        ///        but not included in the physics engine.
        ///
        /// \param[in] t Current time.
        /// \param[in] q Current configuration vector.
        /// \param[in] v Current velocity vector.
        /// \param[in] uCustom Output effort vector.
        virtual void internalDynamics(double t,
                                      const Eigen::VectorXd & q,
                                      const Eigen::VectorXd & v,
                                      Eigen::VectorXd & uCustom) = 0;

        /// \brief Set the configuration options of the controller.
        ///
        /// \details Note that one must reset Jiminy Engine for this to take effect.
        ///
        /// \param[in] controllerOptions Dictionary with the parameters of the controller.
        void setOptions(const GenericConfig & controllerOptions);

        /// \brief Dictionary with the parameters of the controller.
        const GenericConfig & getOptions() const noexcept;

        /// \brief Configure the telemetry of the controller.
        ///
        /// \details This method connects the controller-specific telemetry sender to a given
        ///          telemetry data (which is unique for a given robot), so that it is later
        ///          possible to register the variables that one want to monitor. Finally, the
        ///          telemetry recorder logs every registered variables at each timestep in a
        ///          memory buffer.
        ///
        /// \remark This method is not intended to be called manually. The Engine is taking care of
        ///         it before flushing the telemetry data at the end of each simulation steps.
        ///
        /// \param[in] telemetryData Shared pointer to the robot-wide telemetry data object
        virtual void configureTelemetry(std::shared_ptr<TelemetryData> telemetryData,
                                        const std::string & prefix = {});

        /// \brief Update the internal buffers of the telemetry associated with variables monitored
        ///        by the controller.
        /// \details As the main entry point for a user to log extra variables, the engine also
        ///          passes the current state of the robot to enable logging of custom state-
        ///          related variables.
        ///
        /// \param[in] t Current time.
        /// \param[in] q Current position.
        /// \param[in] v Current velocity.
        ///
        /// \remark This method is not intended to be called manually. The Engine is taking care
        ///         of it before flushing the telemetry data at the end of each simulation steps.
        virtual void updateTelemetry();

        /// \brief Reset the internal state of the controller.
        ///
        /// \details Note that it resets the configuration of the telemetry.
        ///
        /// \remarks This method is not intended to be called manually. The Engine is taking care
        ///          of it when its own `reset` method is called.
        ///
        /// \param[in] resetDynamicTelemetry Whether variables dynamically registered to the
        ///                                  telemetry must be removed. Optional: False by default.
        virtual void reset(bool resetDynamicTelemetry = false);

        /// \brief Whether the controller has been initialized.
        ///
        /// \remark Note that a controller can be considered initialized even if its telemetry is
        ///         not properly configured. If not, it must be done before being ready to use.
        bool getIsInitialized() const;

        /// \brief Whether the telemetry of the controller has been initialized.
        bool getIsTelemetryConfigured() const;

    public:
        /// \brief Structure with the parameters of the controller.
        std::unique_ptr<const ControllerOptions> baseControllerOptions_{nullptr};
        /// \brief Robot for which to compute the command and internal dynamics must be computed.
        std::weak_ptr<const Robot> robot_{};
        SensorMeasurementTree sensorMeasurements_{};

    protected:
        /// \brief Flag to determine whether the controller has been initialized or not.
        bool isInitialized_{false};
        /// \brief Flag to determine whether the telemetry of the controller has been initialized.
        bool isTelemetryConfigured_{false};
        /// \brief Dictionary with the parameters of the controller.
        GenericConfig controllerOptionsGeneric_{};
        /// \brief Telemetry sender used to register and update telemetry variables.
        std::unique_ptr<TelemetrySender> telemetrySender_;

    private:
        /// \brief Vector of dynamically registered telemetry variables.
        static_map_t<std::string, std::variant<const double *, const int64_t *>>
            variableRegistry_{};
        /// \brief Vector of dynamically registered telemetry constants.
        static_map_t<std::string, std::string> constantRegistry_{};
    };
}

#include "jiminy/core/control/abstract_controller.hxx"

#endif  // end of JIMINY_ABSTRACT_CONTROLLER_H
