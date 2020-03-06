#include <iostream>

#include "jiminy/core/Utilities.h"
#include "jiminy/core/Model.h"
#include "jiminy/core/AbstractController.h"


namespace jiminy
{
    AbstractController::AbstractController(void) :
    ctrlOptions_(nullptr),
    model_(nullptr),
    isInitialized_(false),
    isTelemetryConfigured_(false),
    ctrlOptionsHolder_(),
    telemetrySender_(),
    registeredVariables_(),
    registeredConstants_()
    {
        AbstractController::setOptions(getDefaultOptions()); // Clarify that the base implementation is called
    }

    AbstractController::~AbstractController(void)
    {
        // Empty.
    }


    result_t AbstractController::initialize(std::shared_ptr<Model const> const & model)
    {
        if (!model->getIsInitialized())
        {
            std::cout << "Error - AbstractController::initialize - The model is not initialized." << std::endl;
            return result_t::ERROR_INIT_FAILED;
        }

        model_ = model;

        try
        {
            // isInitialized_ must be true to execute the 'computeCommand' and 'internalDynamics' methods
            isInitialized_ = true;
            float64_t t = 0;
            vectorN_t q = vectorN_t::Zero(model_->nq());
            vectorN_t v = vectorN_t::Zero(model_->nv());
            vectorN_t uCommand = vectorN_t::Zero(model_->getMotorsNames().size());
            vectorN_t uInternal = vectorN_t::Zero(model_->nv());
            result_t returnCode = computeCommand(t, q, v, uCommand);
            if (returnCode == result_t::SUCCESS)
            {
                if(uCommand.size() != (int32_t) model_->getMotorsNames().size())
                {
                    std::cout << "Error - AbstractController::initialize - 'computeCommand' returns command with wrong size." << std::endl;
                    return result_t::ERROR_BAD_INPUT;
                }

                internalDynamics(t, q, v, uInternal);
                if(uInternal.size() != model_->nv())
                {
                    std::cout << "Error - AbstractController::initialize - 'internalDynamics' returns command with wrong size." << std::endl;
                    return result_t::ERROR_BAD_INPUT;
                }
            }
            return returnCode;
        }
        catch (std::exception& e)
        {
            isInitialized_ = false;
            std::cout << "Error - AbstractController::initialize - Something is wrong, probably because of 'commandFct'." << std::endl;
            return result_t::ERROR_GENERIC;
        }
    }

    void AbstractController::reset(bool const & resetDynamicTelemetry)
    {
        // Reset the telemetry buffer of dynamically registered quantities
        if (resetDynamicTelemetry)
        {
            removeEntries();
        }

        isTelemetryConfigured_ = false;
    }

    result_t AbstractController::configureTelemetry(std::shared_ptr<TelemetryData> const & telemetryData)
    {
        result_t returnCode = result_t::SUCCESS;

        if (!isInitialized_)
        {
            std::cout << "Error - AbstractController::configureTelemetry - The controller is not initialized." << std::endl;
            returnCode = result_t::ERROR_INIT_FAILED;
        }

        if (!isTelemetryConfigured_ && ctrlOptions_->telemetryEnable)
        {
            if (telemetryData)
            {
                telemetrySender_.configureObject(telemetryData, CONTROLLER_OBJECT_NAME);
                for (std::pair<std::string, float64_t const *> const & registeredVariable : registeredVariables_)
                {
                    if (returnCode == result_t::SUCCESS)
                    {
                        returnCode = telemetrySender_.registerVariable(registeredVariable.first,
                                                                       *registeredVariable.second);
                    }
                }
                for (std::pair<std::string, std::string> const & registeredConstant : registeredConstants_)
                {
                    if (returnCode == result_t::SUCCESS)
                    {
                        returnCode = telemetrySender_.registerConstant(registeredConstant.first,
                                                                       registeredConstant.second);
                    }
                }
                if (returnCode == result_t::SUCCESS)
                {
                    isTelemetryConfigured_ = true;
                }
            }
            else
            {
                std::cout << "Error - AbstractController::configureTelemetry - Telemetry not initialized. Impossible to log controller data." << std::endl;
                returnCode = result_t::ERROR_INIT_FAILED;
            }
        }

        return returnCode;
    }

    result_t AbstractController::registerVariable(std::vector<std::string> const & fieldNames,
                                                  Eigen::Ref<vectorN_t>            values)
    {
        // Delayed variable registration (Taken into account by 'configureTelemetry')

        if (isTelemetryConfigured_)
        {
            std::cout << "Error - AbstractController::registerVariable - Telemetry already initialized. Impossible to register new variables." << std::endl;
            return result_t::ERROR_INIT_FAILED;
        }

        std::vector<std::string>::const_iterator fieldIt = fieldNames.begin();
        for (uint32_t i=0; fieldIt != fieldNames.end(); ++fieldIt, ++i)
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
                std::cout << "Error - AbstractController::registerVariable - Variable already registered." << std::endl;
                return result_t::ERROR_BAD_INPUT;
            }
            registeredVariables_.emplace_back(*fieldIt, values.data() + i);
        }

        return result_t::SUCCESS;
    }

    result_t AbstractController::registerVariable(std::string const & fieldName,
                                                  float64_t   const & value)
    {
        // Delayed variable registration (Taken into account by 'configureTelemetry')

        if (isTelemetryConfigured_)
        {
            std::cout << "Error - AbstractController::registerVariable - Telemetry already initialized. Impossible to register new variables." << std::endl;
            return result_t::ERROR_INIT_FAILED;
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
            std::cout << "Error - AbstractController::registerVariable - Variable already registered." << std::endl;
            return result_t::ERROR_BAD_INPUT;
        }
        registeredVariables_.emplace_back(fieldName, &value);

        return result_t::SUCCESS;
    }

    void AbstractController::removeEntries(void)
    {
        registeredVariables_.clear();
        registeredConstants_.clear();
    }

    void AbstractController::updateTelemetry(float64_t const& t,
                                             vectorN_t const& q,
                                             vectorN_t const& v)
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
        ctrlOptions_ = std::make_unique<controllerOptions_t const>(ctrlOptionsHolder_);
    }

    bool AbstractController::getIsInitialized(void) const
    {
        return isInitialized_;
    }

    bool AbstractController::getIsTelemetryConfigured(void) const
    {
        return isTelemetryConfigured_;
    }
}
