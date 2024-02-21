#include "pinocchio/algorithm/joint-configuration.hpp"  // pinocchio::neutral

#include "jiminy/core/telemetry/telemetry_sender.h"
#include "jiminy/core/robot/robot.h"

#include "jiminy/core/control/abstract_controller.h"


namespace jiminy
{
    AbstractController::AbstractController() noexcept :
    telemetrySender_{std::make_unique<TelemetrySender>()}
    {
        // Clarify that the base implementation is called
        AbstractController::setOptions(getDefaultControllerOptions());
    }

    AbstractController::~AbstractController() = default;

    void AbstractController::initialize(std::weak_ptr<const Robot> robotIn)
    {
        /* Note that it is not possible to reinitialize a controller for a different robot, because
           otherwise, it would be necessary to check consistency with system at engine level when
           calling reset. */

        // Make sure the robot is valid
        auto robot = robotIn.lock();
        if (!robot)
        {
            THROW_ERROR(bad_control_flow, "Robot pointer expired or unset.");
        }

        if (!robot->getIsInitialized())
        {
            THROW_ERROR(bad_control_flow, "Robot not initialized.");
        }

        // Make sure that the controller is not already bound to another robot
        if (isInitialized_)
        {
            auto robotOld = robot_.lock();
            if (robotOld && robotOld.get() != robot.get())
            {
                auto controllerOld = robotOld->getController().lock();
                if (controllerOld && controllerOld.get() == this)
                {
                    THROW_ERROR(bad_control_flow,
                                "Controller already bound to another robot. Please unbind it "
                                "first before re-initializing it.");
                }
            }
        }

        // Backup robot
        robot_ = robotIn;

        /* Set initialization flag to true temporarily to enable calling 'reset', 'computeCommand'
           and 'internalDynamics' methods. */
        isInitialized_ = true;

        // Reset the controller completely
        reset(true);

        // Make sure that calling command and internal dynamics is not raising an exception
        double t = 0.0;
        const Eigen::VectorXd q = pinocchio::neutral(robot->pinocchioModel_);
        const Eigen::VectorXd v = Eigen::VectorXd::Zero(robot->nv());
        Eigen::VectorXd command = Eigen::VectorXd(robot->nmotors());
        Eigen::VectorXd uCustom = Eigen::VectorXd(robot->nv());
        try
        {
            computeCommand(t, q, v, command);
        }
        catch (const std::exception & e)
        {
            isInitialized_ = false;
            robot_.reset();
            sensorMeasurements_.clear();
            THROW_ERROR(
                std::invalid_argument,
                "Something is wrong, probably because of 'commandFun'.\nRaised from exception: ",
                e.what());
        }
        if (command.size() != robot->nmotors())
        {
            THROW_ERROR(std::invalid_argument,
                        "'computeCommand' returns command with wrong size.");
        }
        internalDynamics(t, q, v, uCustom);
        if (uCustom.size() != robot->nv())
        {
            THROW_ERROR(std::invalid_argument,
                        "'internalDynamics' returns command with wrong size.");
        }
    }

    void AbstractController::reset(bool resetDynamicTelemetry)
    {
        if (!isInitialized_)
        {
            THROW_ERROR(bad_control_flow, "The controller is not initialized.");
        }

        // Reset the telemetry buffer of dynamically registered quantities
        if (resetDynamicTelemetry)
        {
            removeEntries();
        }

        // Make sure the robot still exists
        auto robot = robot_.lock();
        if (!robot)
        {
            THROW_ERROR(bad_control_flow, "Robot pointer expired or unset.");
        }

        /* Refresh the sensor data proxy.
           Note that it is necessary to do so since sensors may have been added or removed. */
        sensorMeasurements_ = robot->getSensorMeasurements();

        // Update the telemetry flag
        isTelemetryConfigured_ = false;
    }

    void AbstractController::configureTelemetry(std::shared_ptr<TelemetryData> telemetryData,
                                                const std::string & prefix)
    {
        if (!isInitialized_)
        {
            THROW_ERROR(bad_control_flow, "Controller not initialized.");
        }

        if (!isTelemetryConfigured_ && baseControllerOptions_->telemetryEnable)
        {
            if (!telemetryData)
            {
                THROW_ERROR(bad_control_flow,
                            "Telemetry not initialized. Impossible to log controller data.");
            }

            std::string telemetryName{CONTROLLER_TELEMETRY_NAMESPACE};
            if (!prefix.empty())
            {
                telemetryName =
                    addCircumfix(telemetryName, prefix, {}, TELEMETRY_FIELDNAME_DELIMITER);
            }
            telemetrySender_->configure(telemetryData, telemetryName);

            for (const auto & [constantName, constantValue] : constantRegistry_)
            {
                telemetrySender_->registerConstant(constantName, constantValue);
            }
            for (const auto & [variableNameIn, variableValuePtr] : variableRegistry_)
            {
                // FIXME: Remove explicit `name` capture when moving to C++20
                std::visit([&, &variableName = variableNameIn](auto && arg)
                           { telemetrySender_->registerVariable(variableName, arg); },
                           variableValuePtr);
            }

            isTelemetryConfigured_ = true;
        }
    }

    template<typename Scalar>
    void registerVariableImpl(
        static_map_t<std::string, std::variant<const double *, const int64_t *>> &
            registeredVariables,
        bool isTelemetryConfigured,
        const std::vector<std::string> & fieldnames,
        const Eigen::Ref<VectorX<Scalar>, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> &
            values)
    {
        if (isTelemetryConfigured)
        {
            THROW_ERROR(bad_control_flow,
                        "Telemetry already initialized. Impossible to register new variables.");
        }

        std::vector<std::string>::const_iterator fieldIt = fieldnames.begin();
        for (std::size_t i = 0; fieldIt != fieldnames.end(); ++fieldIt, ++i)
        {
            // Check in local cache before.
            auto variableIt = std::find_if(registeredVariables.begin(),
                                           registeredVariables.end(),
                                           [&fieldIt](const auto & element)
                                           { return element.first == *fieldIt; });
            if (variableIt != registeredVariables.end())
            {
                THROW_ERROR(lookup_error, "Variable already registered.");
            }
            registeredVariables.emplace_back(*fieldIt, &values[i]);
        }
    }

    void AbstractController::registerVariable(
        const std::vector<std::string> & fieldnames,
        const Eigen::Ref<VectorX<double>, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> &
            values)
    {
        return registerVariableImpl<double>(
            variableRegistry_, isTelemetryConfigured_, fieldnames, values);
    }

    void AbstractController::registerVariable(
        const std::vector<std::string> & fieldnames,
        const Eigen::Ref<VectorX<int64_t>, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> &
            values)
    {
        return registerVariableImpl<int64_t>(
            variableRegistry_, isTelemetryConfigured_, fieldnames, values);
    }

    void AbstractController::removeEntries()
    {
        variableRegistry_.clear();
        constantRegistry_.clear();
    }

    void AbstractController::updateTelemetry()
    {
        if (isTelemetryConfigured_)
        {
            telemetrySender_->updateValues();
        }
    }

    GenericConfig AbstractController::getOptions() const noexcept
    {
        return controllerOptionsGeneric_;
    }

    void AbstractController::setOptions(const GenericConfig & controllerOptions)
    {
        controllerOptionsGeneric_ = controllerOptions;
        baseControllerOptions_ =
            std::make_unique<const ControllerOptions>(controllerOptionsGeneric_);
    }

    bool AbstractController::getIsInitialized() const
    {
        return isInitialized_;
    }

    bool AbstractController::getIsTelemetryConfigured() const
    {
        return isTelemetryConfigured_;
    }
}
