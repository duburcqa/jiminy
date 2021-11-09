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
        setOptions(getOptions());
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
        return  1.0 / transmissionOptions_->mechanicalReduction;
    }
}
