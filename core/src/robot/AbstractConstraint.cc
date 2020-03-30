#include "jiminy/core/robot/Robot.h"
#include "jiminy/core/Utilities.h"

#include "jiminy/core/robot/AbstractConstraint.h"


namespace jiminy
{
    AbstractConstraint::AbstractConstraint() :
    model_(nullptr),
    isInitialized_(false)
    {
        // Empty on purpose
    }

    AbstractConstraint::~AbstractConstraint()
    {
        // Empty on purpose
    }

    hresult_t AbstractConstraint::initialize(Model *model)
    {
        model_ = model;
        isInitialized_ = true;
        return hresult_t::SUCCESS;
    }

    hresult_t AbstractConstraint::refreshProxies()
    {
        return hresult_t::SUCCESS;
    }

    matrixN_t AbstractConstraint::getJacobian(vectorN_t const & q) const
    {
        return matrixN_t::Zero(0, model_->pncModel_.nv);
    }

    vectorN_t AbstractConstraint::getDrift(vectorN_t const & q, vectorN_t const & v) const
    {
        return vectorN_t::Zero(0);
    }
}
