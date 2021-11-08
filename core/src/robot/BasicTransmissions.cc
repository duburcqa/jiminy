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

    hresult_t SimpleTransmission::initialize(std::string const & jointNames, std::string const & motorNames)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        jointNames_ = jointNames;
        motorNames_ = motorNames;
        isInitialized_ = true;

        AbstractTransmissionBase::initialize()

        returnCode = refreshProxies();
        if (returnCode != hresult_t::SUCCESS)
        {
            jointNames_.clear();
            motorNames_.clear();
            isInitialized_ = false;
        }

        return returnCode;
    }

    hresult_t SimpleTransmission::setOptions(configHolder_t const & transmissionOptions)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        returnCode = AbstractTransmissionBase::setOptions(transmissionOptions);

        if (returnCode == hresult_t::SUCCESS)
        {
            transmissionOptions_ = std::make_unique<transmissionOptions_t const>(transmissionOptions);
        }

        return returnCode;
    }

    float64_t SimpleTransmission::computeTransform(Eigen::VectorBlock<vectorN_t> /* q */,
                                                   Eigen::VectorBlock<vectorN_t> /* v */))
    {
        if (!isInitialized_)
        {
            PRINT_ERROR("Transmission not initialized. Impossible to compute actual transmission effort.");
            return hresult_t::ERROR_INIT_FAILED;
        }

        return transmissionOptions_->mechanicalReduction;
    }

    float64_t SimpleTransmission::computeInverseTransform(Eigen::VectorBlock<vectorN_t> /* q */,
                                                          Eigen::VectorBlock<vectorN_t> /* v */))
    {
        if (!isInitialized_)
        {
            PRINT_ERROR("Transmission not initialized. Impossible to compute actual transmission effort.");
            return hresult_t::ERROR_INIT_FAILED;
        }
        return transmissionOptions_-> 1.0 / mechanicalReduction;
    }
}
