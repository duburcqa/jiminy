#include "jiminy/core/robot/Robot.h"
#include "jiminy/core/Macros.h"

#include "jiminy/core/constraints/AbstractConstraint.h"


namespace jiminy
{
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
        // Make sure the constraint is not already attached
        if (isAttached_)
        {
            PRINT_ERROR("Constraint already attached to a model.");
            return hresult_t::ERROR_GENERIC;
        }

        // Make sure the model still exists
        if (model.expired())
        {
            PRINT_ERROR("Model pointer expired or unset.");
            return hresult_t::ERROR_GENERIC;
        }

        // Consider the constraint is attached at this point
        model_ = model;
        isAttached_ = true;

        // Enable constraint by default
        isEnabled_ = true;

        return hresult_t::SUCCESS;
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

    hresult_t AbstractConstraintBase::setBaumgartePositionGain(float64_t const & kp)
    {
        if (kp < 0.0)
        {
            PRINT_ERROR("The position gain must be positive.");
            return hresult_t::ERROR_GENERIC;
        }
        kp_ = kp;
        return hresult_t::SUCCESS;
    }

    float64_t AbstractConstraintBase::getBaumgartePositionGain(void) const
    {
        return kp_;
    }

    hresult_t AbstractConstraintBase::setBaumgarteVelocityGain(float64_t const & kd)
    {
        if (kd < 0.0)
        {
            PRINT_ERROR("The velocity gain must be positive.");
            return hresult_t::ERROR_GENERIC;
        }
        kd_ = kd;
        return hresult_t::SUCCESS;
    }

    float64_t AbstractConstraintBase::getBaumgarteVelocityGain(void) const
    {
        return kd_;
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
        float64_t zeta = kd_ / 2.0;
        if (zeta < std::sqrt(kp_))
        {
            zeta = std::max(zeta, std::sqrt(kp_ - std::pow(zeta, 2)));
        }
        return zeta / (2.0 * M_PI);
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
