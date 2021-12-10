#include "jiminy/core/robot/Robot.h"
#include "jiminy/core/Macros.h"

#include "jiminy/core/constraints/AbstractConstraint.h"


namespace jiminy
{
    AbstractConstraintBase::AbstractConstraintBase(void) :
    lambda_(),
    model_(),
    isAttached_(false),
    isEnabled_(true),
    kp_(0.0),
    kd_(0.0),
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
        lambda_.setZero();
        isEnabled_ = false;
    }

    bool_t const & AbstractConstraintBase::getIsEnabled(void) const
    {
        return isEnabled_;
    }

    hresult_t AbstractConstraintBase::setBaumgarteFreq(float64_t const & freq)
    {
        if (freq < 0.0)
        {
            PRINT_ERROR("The natural frequency must be positive.");
            return hresult_t::ERROR_GENERIC;
        }

        // Critically damped position/velocity gains
        float64_t const omega = 2.0 * M_PI * freq;
        kp_ = omega * omega;
        kd_ = 2.0 * omega;

        return hresult_t::SUCCESS;
    }

    float64_t AbstractConstraintBase::getBaumgarteFreq(void) const
    {
        return kd_ / (4.0 * M_PI);
    }

    uint64_t AbstractConstraintBase::getDim(void) const
    {
        return static_cast<uint64_t>(drift_.size());
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
