#include <algorithm>

#include "jiminy/core/utilities/Helpers.h"

#include "jiminy/core/robot/BasicTransmissions.h"


namespace jiminy
{
    SimpleTransmission::SimpleTransmission(std::string const & name) :
    AbstractTransmissionBase(name),
    transmissionOptions_(nullptr)
    {
        /* AbstractTransmissionBase constructor calls the base implementations of
           the virtual methods since the derived class is not available at
           this point. Thus it must be called explicitly in the constructor. */
        setOptions(getDefaultTransmissionOptions());
    }

    hresult_t SimpleTransmission::initialize(std::string const & jointName, std::string const & motorName)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        jointName_ = jointName;
        motorName_ = motorName;
        isInitialized_ = true;
        returnCode = refreshProxies();

        if (returnCode != hresult_t::SUCCESS)
        {
            jointName_.clear();
            motorName_.clear();
            isInitialized_ = false;
        }

        return returnCode;
    }

    hresult_t SimpleTransmission::setOptions(configHolder_t const & transmissionOptions)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        returnCode = AbstractTransmissionBase::setOptions(transmissionOptions);

        // Check if the friction parameters make sense
        if (returnCode == hresult_t::SUCCESS)
        {
            // Make sure the user-defined position limit has the right dimension
            if (boost::get<float64_t>(transmissionOptions.at("frictionViscousPositive")) > 0.0)
            {
                PRINT_ERROR("'frictionViscousPositive' must be negative.");
                returnCode = hresult_t::ERROR_BAD_INPUT;
            }
            if (boost::get<float64_t>(transmissionOptions.at("frictionViscousNegative")) > 0.0)
            {
                PRINT_ERROR("'frictionViscousNegative' must be negative.");
                returnCode = hresult_t::ERROR_BAD_INPUT;
            }
            if (boost::get<float64_t>(transmissionOptions.at("frictionDryPositive")) > 0.0)
            {
                PRINT_ERROR("'frictionDryPositive' must be negative.");
                returnCode = hresult_t::ERROR_BAD_INPUT;
            }
            if (boost::get<float64_t>(transmissionOptions.at("frictionDryNegative")) > 0.0)
            {
                PRINT_ERROR("'frictionDryNegative' must be negative.");
                returnCode = hresult_t::ERROR_BAD_INPUT;
            }
            if (boost::get<float64_t>(transmissionOptions.at("frictionDrySlope")) < 0.0)
            {
                PRINT_ERROR("'frictionDrySlope' must be positive.");
                returnCode = hresult_t::ERROR_BAD_INPUT;
            }
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            transmissionOptions_ = std::make_unique<transmissionOptions_t const>(transmissionOptions);
        }

        return returnCode;
    }

    float64_t SimpleTransmission::computeTransform(float64_t const & /* t */,
                                                   Eigen::VectorBlock<vectorN_t> q,
                                                   float64_t v,
                                                   float64_t const & /* a */,
                                                   float64_t command)
    {
        if (!isInitialized_)
        {
            PRINT_ERROR("Transmission not initialized. Impossible to compute actual transmission effort.");
            return hresult_t::ERROR_INIT_FAILED;
        }

        return transmissionOptions_->mechanicalReduction;
    }

    float64_t SimpleTransmission::computeInverseTransform(float64_t const & /* t */,
                                                          Eigen::VectorBlock<vectorN_t> /* q */,
                                                          float64_t /* v */,
                                                          float64_t const & /* a */,
                                                          float64_t /* command */)
    {
        if (!isInitialized_)
        {
            PRINT_ERROR("Transmission not initialized. Impossible to compute actual transmission effort.");
            return hresult_t::ERROR_INIT_FAILED;
        }
        return transmissionOptions_->mechanicalReduction;
    }
}
