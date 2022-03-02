#include <iostream>

#include "pinocchio/algorithm/joint-configuration.hpp"  // pinocchio::neutral

#include "jiminy/core/robot/Robot.h"
#include "jiminy/core/Constants.h"

#include "jiminy/core/control/AbstractController.h"


namespace jiminy
{
    AbstractController::AbstractController(void) :
    baseControllerOptions_(nullptr),
    robot_(),
    sensorsData_(),
    isInitialized_(false),
    isTelemetryConfigured_(false),
    ctrlOptionsHolder_(),
    telemetrySender_(),
    registeredVariables_(),
    registeredConstants_()
    {
        AbstractController::setOptions(getDefaultControllerOptions());  // Clarify that the base implementation is called
    }

    hresult_t AbstractController::initialize(std::weak_ptr<Robot const> robotIn)
    {
        /* Note that it is not possible to reinitialize a controller for a different robot,
           because otherwise, it would be necessary to check consistency with system at
           engine level when calling reset. */

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

        /* Set initialization flag to true temporarily to enable calling
           'reset', 'computeCommand' and 'internalDynamics' methods. */
        isInitialized_ = true;

        // Reset the controller completely
        reset(true);  // It cannot fail at this point

        try
        {
            float64_t t = 0.0;
            vectorN_t q = pinocchio::neutral(robot->pncModel_);
            vectorN_t v = vectorN_t::Zero(robot->nv());
            vectorN_t command = vectorN_t::Zero(robot->getMotorsNames().size());
            vectorN_t uCustom = vectorN_t::Zero(robot->nv());
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
        catch (std::exception const & e)
        {
            isInitialized_ = false;
            robot_.reset();
            sensorsData_.clear();
            PRINT_ERROR("Something is wrong, probably because of 'commandFct'.\n"
                        "Raised from exception: ", e.what());
            return hresult_t::ERROR_GENERIC;
        }

        return hresult_t::SUCCESS;
    }

    hresult_t AbstractController::reset(bool_t const & resetDynamicTelemetry)
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
                                                     std::string const & objectPrefixName)
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
                std::string objectName = CONTROLLER_TELEMETRY_NAMESPACE;
                if (!objectPrefixName.empty())
                {
                    objectName = objectPrefixName + TELEMETRY_FIELDNAME_DELIMITER + objectName;
                }
                telemetrySender_.configureObject(telemetryData, objectName);
                for (std::pair<std::string, float64_t const *> const & registeredVariable : registeredVariables_)
                {
                    if (returnCode == hresult_t::SUCCESS)
                    {
                        returnCode = telemetrySender_.registerVariable(registeredVariable.first,
                                                                       *registeredVariable.second);
                    }
                }
                for (std::pair<std::string, std::string> const & registeredConstant : registeredConstants_)
                {
                    if (returnCode == hresult_t::SUCCESS)
                    {
                        returnCode = telemetrySender_.registerConstant(registeredConstant.first,
                                                                       registeredConstant.second);
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

    hresult_t AbstractController::registerVariable(std::vector<std::string> const & fieldnames,
                                                   Eigen::Ref<vectorN_t, 0, Eigen::InnerStride<> > values)
    {
        // Delayed variable registration (Taken into account by 'configureTelemetry')

        if (isTelemetryConfigured_)
        {
            PRINT_ERROR("Telemetry already initialized. Impossible to register new variables.");
            return hresult_t::ERROR_INIT_FAILED;
        }

        std::vector<std::string>::const_iterator fieldIt = fieldnames.begin();
        for (std::size_t i=0; fieldIt != fieldnames.end(); ++fieldIt, ++i)
        {
            // Check in local cache before.
            auto variableIt = std::find_if(registeredVariables_.begin(),
                                           registeredVariables_.end(),
                                           [&fieldIt](auto const & element)
                                           {
                                               return element.first == *fieldIt;
                                           });
            if (variableIt != registeredVariables_.end())
            {
                PRINT_ERROR("Variable already registered.");
                return hresult_t::ERROR_BAD_INPUT;
            }
            registeredVariables_.emplace_back(*fieldIt, &values[i]);
        }

        return hresult_t::SUCCESS;
    }

    hresult_t AbstractController::registerVariable(std::string const & fieldName,
                                                   float64_t   const & value)
    {
        // Delayed variable registration (Taken into account by 'configureTelemetry')

        if (isTelemetryConfigured_)
        {
            PRINT_ERROR("Telemetry already initialized. Impossible to register new variables.");
            return hresult_t::ERROR_INIT_FAILED;
        }

        // Check in local cache before.
        auto variableIt = std::find_if(registeredVariables_.begin(),
                                        registeredVariables_.end(),
                                        [&fieldName](auto const & element)
                                        {
                                            return element.first == fieldName;
                                        });
        if (variableIt != registeredVariables_.end())
        {
            PRINT_ERROR("Variable already registered.");
            return hresult_t::ERROR_BAD_INPUT;
        }
        registeredVariables_.emplace_back(fieldName, &value);

        return hresult_t::SUCCESS;
    }

    void AbstractController::removeEntries(void)
    {
        registeredVariables_.clear();
        registeredConstants_.clear();
    }

    void AbstractController::updateTelemetry(void)
    {
        if (isTelemetryConfigured_)
        {
            for (std::pair<std::string, float64_t const *> const & registeredVariable : registeredVariables_)
            {
                telemetrySender_.updateValue(registeredVariable.first, *registeredVariable.second);
            }
        }
    }

    configHolder_t AbstractController::getOptions(void) const
    {
        return ctrlOptionsHolder_;
    }

    void AbstractController::setOptions(configHolder_t const & ctrlOptions)
    {
        ctrlOptionsHolder_ = ctrlOptions;
        baseControllerOptions_ = std::make_unique<controllerOptions_t const>(ctrlOptionsHolder_);
    }

    bool_t const & AbstractController::getIsInitialized(void) const
    {
        return isInitialized_;
    }

    bool_t const & AbstractController::getIsTelemetryConfigured(void) const
    {
        return isTelemetryConfigured_;
    }
}
