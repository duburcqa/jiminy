#include "jiminy/core/robot/Robot.h"
#include "jiminy/core/Utilities.h"

#include "jiminy/core/robot/AbstractConstraint.h"


namespace jiminy
{
    AbstractConstraint::AbstractConstraint() :
    model_(nullptr),
    isAttached_(false),
    jacobian_(matrixN_t::Zero(0,0)),
    drift_(vectorN_t::Zero(0))
    {
        // Empty on purpose
    }

    matrixN_t const & AbstractConstraint::getJacobian(vectorN_t const & q)
    {
        return jacobian_;
    }

    vectorN_t const & AbstractConstraint::getDrift(vectorN_t const & q, vectorN_t const & v)
    {
        return drift_;
    }


    hresult_t AbstractConstraint::attach(Model const * model)
    {
        model_ = model;
        isAttached_ = true;
        return hresult_t::SUCCESS;
    }

    void AbstractConstraint::detach()
    {
        model_ = nullptr;
        isAttached_ = false;
    }

    hresult_t AbstractConstraint::refreshProxies()
    {
        return hresult_t::SUCCESS;
    }
}
