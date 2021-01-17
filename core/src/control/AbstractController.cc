#include <iostream>

#include "pinocchio/algorithm/joint-configuration.hpp"

#include "jiminy/core/robot/Robot.h"
#include "jiminy/core/Constants.h"

#include "jiminy/core/control/AbstractController.h"


namespace jiminy
{
    AbstractController::AbstractController(void) :
    baseControllerOptions_(nullptr),
    robot_(nullptr),
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

    hresult_t AbstractController::initialize(Robot const * robot)
    {
        if (!robot->getIsInitialized())
        {
            PRINT_ERROR("The robot is not initialized.");
            return hresult_t::ERROR_INIT_FAILED;
        }

        // Backup robot
        robot_ = robot;

        // Reset the controller completely
        reset(true);

        try
        {
            // isInitialized_ must be true to execute the 'computeCommand' and 'internalDynamics' methods
            isInitialized_ = true;
            float64_t t = 0.0;
            vectorN_t q = pinocchio::neutral(robot->pncModel_);
            vectorN_t v = vectorN_t::Zero(robot_->nv());
            vectorN_t uCommand = vectorN_t::Zero(robot_->getMotorsNames().size());
            vectorN_t uInternal = vectorN_t::Zero(robot_->nv());
            hresult_t returnCode = computeCommand(t, q, v, uCommand);
            if (returnCode == hresult_t::SUCCESS)
            {
                if (uCommand.size() != (int32_t) robot_->getMotorsNames().size())
                {
                    PRINT_ERROR("'computeCommand' returns command with wrong size.");
                    return hresult_t::ERROR_BAD_INPUT;
                }

                internalDynamics(t, q, v, uInternal);
                if (uInternal.size() != robot_->nv())
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
            PRINT_ERROR("Something is wrong, probably because of 'commandFct'.\n"
                        "Raised from exception: ", e.what());
            return hresult_t::ERROR_GENERIC;
        }
    }

    void AbstractController::reset(bool_t const & resetDynamicTelemetry)
    {
        // Reset the telemetry buffer of dynamically registered quantities
        if (resetDynamicTelemetry)
        {
            removeEntries();
        }

        /* Refresh the sensor data proxy.
           Note that it is necessary to do so since sensors may have been added ore removed. */
        sensorsData_ = robot_->getSensorsData();

        // Update the telemetry flag
        isTelemetryConfigured_ = false;
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
                telemetrySender_.configureObject(std::move(telemetryData), objectName);
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
                                                   Eigen::Ref<vectorN_t>            values)
    {
        // Delayed variable registration (Taken into account by 'configureTelemetry')

        if (isTelemetryConfigured_)
        {
            PRINT_ERROR("Telemetry already initialized. Impossible to register new variables.");
            return hresult_t::ERROR_INIT_FAILED;
        }

        std::vector<std::string>::const_iterator fieldIt = fieldnames.begin();
        for (uint32_t i=0; fieldIt != fieldnames.end(); ++fieldIt, ++i)
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
            registeredVariables_.emplace_back(*fieldIt, values.data() + i);
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
