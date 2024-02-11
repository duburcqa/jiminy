#include "jiminy/core/robot/model.h"

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

    void AbstractConstraintBase::attach(std::weak_ptr<const Model> model)
    {
        // Make sure the constraint is not already attached
        if (isAttached_)
        {
            THROW_ERROR(bad_control_flow, "Constraint already attached to a model.");
        }

        // Make sure the model still exists
        if (model.expired())
        {
            THROW_ERROR(bad_control_flow, "Model pointer expired or unset.");
        }

        // Consider the constraint is attached at this point
        model_ = model;
        isAttached_ = true;

        // Enable constraint by default
        isEnabled_ = true;
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

    bool AbstractConstraintBase::getIsEnabled() const
    {
        return isEnabled_;
    }

    void AbstractConstraintBase::setBaumgartePositionGain(double kp)
    {
        if (kp < 0.0)
        {
            THROW_ERROR(std::invalid_argument, "Position gain must be positive.");
        }
        kp_ = kp;
    }

    double AbstractConstraintBase::getBaumgartePositionGain() const
    {
        return kp_;
    }

    void AbstractConstraintBase::setBaumgarteVelocityGain(double kd)
    {
        if (kd < 0.0)
        {
            THROW_ERROR(std::invalid_argument, "Velocity gain must be positive.");
        }
        kd_ = kd;
    }

    double AbstractConstraintBase::getBaumgarteVelocityGain() const
    {
        return kd_;
    }

    void AbstractConstraintBase::setBaumgarteFreq(double freq)
    {
        if (freq < 0.0)
        {
            THROW_ERROR(std::invalid_argument, "Natural frequency must be positive.");
        }

        // Critically damped position/velocity gains
        const double omega = 2.0 * M_PI * freq;
        kp_ = omega * omega;
        kd_ = 2.0 * omega;
    }

    double AbstractConstraintBase::getBaumgarteFreq() const
    {
        double zeta = kd_ / 2.0;
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
