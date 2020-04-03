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

    matrixN_t const & AbstractConstraint::getJacobian(Eigen::Ref<vectorN_t const> const & q)
    {
        return jacobian_;
    }

    vectorN_t const & AbstractConstraint::getDrift(Eigen::Ref<vectorN_t const> const & q,
                                                   Eigen::Ref<vectorN_t const> const & v)
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
