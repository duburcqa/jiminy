#include "jiminy/core/robot/Robot.h"
#include "jiminy/core/Macros.h"

#include "jiminy/core/robot/AbstractConstraint.h"


namespace jiminy
{
    AbstractConstraintBase::AbstractConstraintBase(void) :
    model_(),
    isAttached_(false),
    isEnabled_(true),
    jacobian_(),
    drift_()
    {
        // Empty on purpose
    }

    AbstractConstraintBase::~AbstractConstraintBase(void)
    {
        // Detach the constraint before deleting it if necessary
        if (isAttached_)
        {
            detach();
        }
    }

    hresult_t AbstractConstraintBase::attach(std::weak_ptr<Model const> model)
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

    void AbstractConstraintBase::detach(void)
    {
        model_.reset();
        isAttached_ = false;
    }

    void AbstractConstraintBase::enable(void)
    {
        isEnabled_ = true;
    }

    void AbstractConstraintBase::disable(void)
    {
        isEnabled_ = false;
    }

    bool_t const & AbstractConstraintBase::getIsEnabled(void) const
    {
        return isEnabled_;
    }

    uint32_t AbstractConstraintBase::getDim(void) const
    {
        return drift_.size();
    }

    matrixN_t const & AbstractConstraintBase::getJacobian(void) const
    {
        return jacobian_;
    }

    vectorN_t const & AbstractConstraintBase::getDrift(void) const
    {
        return drift_;
    }
}
