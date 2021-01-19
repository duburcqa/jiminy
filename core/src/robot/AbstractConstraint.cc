#include "jiminy/core/robot/Robot.h"
#include "jiminy/core/Macros.h"

#include "jiminy/core/robot/AbstractConstraint.h"


namespace jiminy
{
    AbstractConstraint::AbstractConstraint(void) :
    model_(),
    isAttached_(false),
    jacobian_(),
    drift_()
    {
        // Empty on purpose
    }

    AbstractConstraint::~AbstractConstraint(void)
    {
        // Detach the constraint before deleting it if necessary
        if (isAttached_)
        {
            detach();
        }
    }

    hresult_t AbstractConstraint::attach(std::weak_ptr<Model const> model)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (isAttached_)
        {
            PRINT_ERROR("Constraint already attached to a model.");
            returnCode = hresult_t::ERROR_GENERIC;
        }

        // Make sure the model still exists
        if (model.expired())
        {
            PRINT_ERROR("Model pointer expired or unset.");
            return hresult_t::ERROR_GENERIC;
        }

        model_ = model;
        isAttached_ = true;

        return returnCode;
    }

    void AbstractConstraint::detach(void)
    {
        model_.reset();
        isAttached_ = false;
    }

    uint32_t AbstractConstraint::getDim(void) const
    {
        return jacobian_.rows();
    }

    matrixN_t const & AbstractConstraint::getJacobian(void) const
    {
        return jacobian_;
    }

    vectorN_t const & AbstractConstraint::getDrift(void) const
    {
        return drift_;
    }
}
