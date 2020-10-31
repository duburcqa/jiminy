#include "jiminy/core/robot/Robot.h"
#include "jiminy/core/Utilities.h"

#include "jiminy/core/robot/AbstractConstraint.h"


namespace jiminy
{
    AbstractConstraint::AbstractConstraint() :
    model_(nullptr),
    isAttached_(false),
    jacobian_(),
    drift_()
    {
        // Empty on purpose
    }

    matrixN_t const & AbstractConstraint::getJacobian(vectorN_t const & q)
    {
        return jacobian_;
    }

    vectorN_t const & AbstractConstraint::getDrift(vectorN_t const & q,
                                                   vectorN_t const & v)
    {
        return drift_;
    }


    hresult_t AbstractConstraint::attach(Model const * model)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (isAttached_)
        {
            PRINT_ERROR("Constraint already attached to a robot.")
            return hresult_t::ERROR_GENERIC;
        }

        model_ = model;

        // Refresh proxies: this checks for the existence of frameName_ in model_.
        returnCode = refreshProxies();
        if (returnCode == hresult_t::SUCCESS)
        {
            isAttached_ = true;
        }

        return returnCode;
    }

    void AbstractConstraint::detach(void)
    {
        model_ = nullptr;
        isAttached_ = false;
    }
}
