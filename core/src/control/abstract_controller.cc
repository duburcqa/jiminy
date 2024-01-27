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

    hresult_t AbstractController::initialize(std::weak_ptr<const Robot> robotIn)
    {
        /* Note that it is not possible to reinitialize a controller for a different robot, because
           otherwise, it would be necessary to check consistency with system at engine level when
           calling reset. */

        // Make sure the robot is valid
        auto robot = robotIn.lock();
        if (!robot)
        {
            PRINT_ERROR("Robot pointer expired or unset.");
            return hresult_t::ERROR_GENERIC;
        }

        if (!robot->getIsInitialized())
        {
            PRINT_ERROR("The robot is not initialized.");
            return hresult_t::ERROR_INIT_FAILED;
        }

        // Backup robot
        robot_ = robotIn;

        /* Set initialization flag to true temporarily to enable calling 'reset', 'computeCommand'
           and 'internalDynamics' methods. */
        isInitialized_ = true;

        // Reset the controller completely
        reset(true);  // Cannot fail at this point

        try
        {
            double t = 0.0;
            const Eigen::VectorXd q = pinocchio::neutral(robot->pncModel_);
            const Eigen::VectorXd v = Eigen::VectorXd::Zero(robot->nv());
            Eigen::VectorXd command = Eigen::VectorXd(robot->getMotorsNames().size());
            Eigen::VectorXd uCustom = Eigen::VectorXd(robot->nv());
            hresult_t returnCode = computeCommand(t, q, v, command);
            if (returnCode == hresult_t::SUCCESS)
            {
                if (static_cast<std::size_t>(command.size()) != robot->getMotorsNames().size())
                {
                    PRINT_ERROR("'computeCommand' returns command with wrong size.");
                    return hresult_t::ERROR_BAD_INPUT;
                }

                internalDynamics(t, q, v, uCustom);
                if (uCustom.size() != robot->nv())
                {
                    PRINT_ERROR("'internalDynamics' returns command with wrong size.");
                    return hresult_t::ERROR_BAD_INPUT;
                }
            }
            return returnCode;
        }
        catch (const std::exception & e)
        {
            isInitialized_ = false;
            robot_.reset();
            sensorsData_.clear();
            PRINT_ERROR(
                "Something is wrong, probably because of 'commandFct'.\nRaised from exception: ",
                e.what());
            return hresult_t::ERROR_GENERIC;
        }

        return hresult_t::SUCCESS;
    }

    hresult_t AbstractController::reset(bool resetDynamicTelemetry)
    {
        if (!isInitialized_)
        {
            PRINT_ERROR("The controller is not initialized.");
            return hresult_t::ERROR_INIT_FAILED;
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
            PRINT_ERROR("Robot pointer expired or unset.");
            return hresult_t::ERROR_GENERIC;
        }

        /* Refresh the sensor data proxy.
           Note that it is necessary to do so since sensors may have been added or removed. */
        sensorsData_ = robot->getSensorsData();

        // Update the telemetry flag
        isTelemetryConfigured_ = false;

        return hresult_t::SUCCESS;
    }

    hresult_t AbstractController::configureTelemetry(std::shared_ptr<TelemetryData> telemetryData,
                                                     const std::string & objectPrefixName)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (!isInitialized_)
        {
            PRINT_ERROR("The controller is not initialized.");
            returnCode = hresult_t::ERROR_INIT_FAILED;
        }

        if (!isTelemetryConfigured_ && baseControllerOptions_->telemetryEnable)
        {
            if (telemetryData)
            {
                std::string objectName{CONTROLLER_TELEMETRY_NAMESPACE};
                if (!objectPrefixName.empty())
                {
                    objectName = addCircumfix(
                        objectName, objectPrefixName, {}, TELEMETRY_FIELDNAME_DELIMITER);
                }
                telemetrySender_->configureObject(telemetryData, objectName);
                for (const auto & [name, valuePtr] : registeredVariables_)
                {
                    if (returnCode == hresult_t::SUCCESS)
                    {
                        // FIXME: Remove explicit `name` capture when moving to C++20
                        std::visit([&, &name = name](auto && arg)
                                   { telemetrySender_->registerVariable(name, arg); },
                                   valuePtr);
                    }
                }
                for (const auto & [name, value] : registeredConstants_)
                {
                    if (returnCode == hresult_t::SUCCESS)
                    {
                        returnCode = telemetrySender_->registerConstant(name, value);
                    }
                }
                if (returnCode == hresult_t::SUCCESS)
                {
                    isTelemetryConfigured_ = true;
                }
            }
            else
            {
                PRINT_ERROR("Telemetry not initialized. Impossible to log controller data.");
                returnCode = hresult_t::ERROR_INIT_FAILED;
            }
        }

        return returnCode;
    }

    template<typename Scalar>
    hresult_t registerVariableImpl(
        static_map_t<std::string, std::variant<const double *, const int64_t *>> &
            registeredVariables,
        bool isTelemetryConfigured,
        const std::vector<std::string> & fieldnames,
        const Eigen::Ref<VectorX<Scalar>, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> &
            values)
    {
        if (isTelemetryConfigured)
        {
            PRINT_ERROR("Telemetry already initialized. Impossible to register new variables.");
            return hresult_t::ERROR_INIT_FAILED;
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
                PRINT_ERROR("Variable already registered.");
                return hresult_t::ERROR_BAD_INPUT;
            }
            registeredVariables.emplace_back(*fieldIt, &values[i]);
        }

        return hresult_t::SUCCESS;
    }

    hresult_t AbstractController::registerVariable(
        const std::vector<std::string> & fieldnames,
        const Eigen::Ref<VectorX<double>, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> &
            values)
    {
        return registerVariableImpl<double>(
            registeredVariables_, isTelemetryConfigured_, fieldnames, values);
    }

    hresult_t AbstractController::registerVariable(
        const std::vector<std::string> & fieldnames,
        const Eigen::Ref<VectorX<int64_t>, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> &
            values)
    {
        return registerVariableImpl<int64_t>(
            registeredVariables_, isTelemetryConfigured_, fieldnames, values);
    }

    void AbstractController::removeEntries()
    {
        registeredVariables_.clear();
        registeredConstants_.clear();
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
        return ctrlOptionsHolder_;
    }

    hresult_t AbstractController::setOptions(const GenericConfig & ctrlOptions)
    {
        ctrlOptionsHolder_ = ctrlOptions;
        baseControllerOptions_ = std::make_unique<const controllerOptions_t>(ctrlOptionsHolder_);
        return hresult_t::SUCCESS;
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
