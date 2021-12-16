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

    void SimpleTransmission::computeTransform(Eigen::VectorBlock<vectorN_t const> const & /*q*/,
                                              Eigen::VectorBlock<vectorN_t const> const & /*v*/,
                                              matrixN_t &out)
    {
        if (!isInitialized_)
        {
            PRINT_ERROR("Transmission not initialized. Impossible to compute transformation of transmission.");
        }
    }

    void SimpleTransmission::computeInverseTransform(Eigen::VectorBlock<vectorN_t const> const & /*q*/,
                                                     Eigen::VectorBlock<vectorN_t const> const & /*v*/,
                                                     matrixN_t &out)
    {
        if (!isInitialized_)
        {
            PRINT_ERROR("Transmission not initialized. Impossible to compute transformation of transmission.");
        }
    }

    void computeEffortTransmission(void)
    {
        // TODO
    }
}
