#include "jiminy/core/robot/robot.h"

#include "jiminy/core/constraints/abstract_constraint.h"


namespace jiminy
{
    AbstractConstraintBase::~AbstractConstraintBase()
    {
        // Detach the constraint before deleting it if necessary
        if (isAttached_)
        {
            detach();
        }
    }

    hresult_t AbstractConstraintBase::attach(std::weak_ptr<const Model> model)
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

    void AbstractConstraintBase::detach()
    {
        model_.reset();
        isAttached_ = false;
    }

    void AbstractConstraintBase::enable()
    {
        isEnabled_ = true;
    }

    void AbstractConstraintBase::disable()
    {
        lambda_.setZero();
        isEnabled_ = false;
    }

    const bool_t & AbstractConstraintBase::getIsEnabled() const
    {
        return isEnabled_;
    }

    hresult_t AbstractConstraintBase::setBaumgartePositionGain(const float64_t & kp)
    {
        if (kp < 0.0)
        {
            PRINT_ERROR("The position gain must be positive.");
            return hresult_t::ERROR_GENERIC;
        }
        kp_ = kp;
        return hresult_t::SUCCESS;
    }

    float64_t AbstractConstraintBase::getBaumgartePositionGain() const
    {
        return kp_;
    }

    hresult_t AbstractConstraintBase::setBaumgarteVelocityGain(const float64_t & kd)
    {
        if (kd < 0.0)
        {
            PRINT_ERROR("The velocity gain must be positive.");
            return hresult_t::ERROR_GENERIC;
        }
        kd_ = kd;
        return hresult_t::SUCCESS;
    }

    float64_t AbstractConstraintBase::getBaumgarteVelocityGain() const
    {
        return kd_;
    }

    hresult_t AbstractConstraintBase::setBaumgarteFreq(const float64_t & freq)
    {
        if (freq < 0.0)
        {
            PRINT_ERROR("The natural frequency must be positive.");
            return hresult_t::ERROR_GENERIC;
        }

        // Critically damped position/velocity gains
        const float64_t omega = 2.0 * M_PI * freq;
        kp_ = omega * omega;
        kd_ = 2.0 * omega;

        return hresult_t::SUCCESS;
    }

    float64_t AbstractConstraintBase::getBaumgarteFreq() const
    {
        float64_t zeta = kd_ / 2.0;
        if (zeta < std::sqrt(kp_))
        {
            zeta = std::max(zeta, std::sqrt(kp_ - std::pow(zeta, 2)));
        }
        return zeta / (2.0 * M_PI);
    }

    uint64_t AbstractConstraintBase::getDim() const
    {
        return static_cast<uint64_t>(drift_.size());
    }

    const Eigen::MatrixXd & AbstractConstraintBase::getJacobian() const
    {
        return jacobian_;
    }

    const Eigen::VectorXd & AbstractConstraintBase::getDrift() const
    {
        return drift_;
    }
}
