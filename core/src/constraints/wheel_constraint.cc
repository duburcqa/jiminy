#include "pinocchio/algorithm/frames.hpp"  // `pinocchio::getFrameVelocity`, `pinocchio::getFrameAcceleration`

#include "jiminy/core/robot/model.h"
#include "jiminy/core/utilities/pinocchio.h"

#include "jiminy/core/constraints/wheel_constraint.h"


namespace jiminy
{
    template<>
    const std::string AbstractConstraintTpl<WheelConstraint>::type_("WheelConstraint");

    WheelConstraint::WheelConstraint(const std::string & frameName,
                                     double wheelRadius,
                                     const Eigen::Vector3d & groundNormal,
                                     const Eigen::Vector3d & wheelAxis) noexcept :
    AbstractConstraintTpl(),
    frameName_{frameName},
    radius_{wheelRadius},
    normal_{groundNormal.normalized()},
    axis_{wheelAxis.normalized()}
    {
    }

    const std::string & WheelConstraint::getFrameName() const noexcept
    {
        return frameName_;
    }

    pinocchio::FrameIndex WheelConstraint::getFrameIndex() const noexcept
    {
        return frameIndex_;
    }

    void WheelConstraint::setReferenceTransform(const pinocchio::SE3 & transformRef) noexcept
    {
        transformRef_ = transformRef;
    }

    const pinocchio::SE3 & WheelConstraint::getReferenceTransform() const noexcept
    {
        return transformRef_;
    }

    void WheelConstraint::reset(const Eigen::VectorXd & /* q */, const Eigen::VectorXd & /* v */)
    {
        // Make sure the model still exists
        auto model = model_.lock();
        if (!model)
        {
            THROW_ERROR(bad_control_flow, "Model pointer expired or unset.");
        }

        // Get frame index
        frameIndex_ = ::jiminy::getFrameIndex(model->pinocchioModel_, frameName_);

        // Initialize frames jacobians buffers
        frameJacobian_.setZero(6, model->pinocchioModel_.nv);

        // Initialize jacobian, drift and multipliers
        jacobian_.setZero(3, model->pinocchioModel_.nv);
        drift_.setZero(3);
        lambda_.setZero(3);

        // Get the current frame position and use it as reference
        transformRef_ = model->pinocchioData_.oMf[frameIndex_];
    }

    void WheelConstraint::computeJacobianAndDrift(const Eigen::VectorXd & /* q */,
                                                  const Eigen::VectorXd & /* v */)
    {
        if (!isAttached_)
        {
            THROW_ERROR(bad_control_flow, "Constraint not attached to a model.");
        }

        // Assuming the model still exists
        auto model = model_.lock();

        // Compute ground normal in local frame
        const pinocchio::SE3 & framePose = model->pinocchioData_.oMf[frameIndex_];
        const Eigen::Vector3d axis = framePose.rotation() * axis_;
        const Eigen::Vector3d x = axis.cross(normal_).cross(axis);
        const double xNorm = x.norm();
        const Eigen::Vector3d y = x / xNorm;
        pinocchio::alphaSkew(radius_, y, skewRadius_);

        // Compute position error
        auto positionRel = framePose.translation() - transformRef_.translation();
        const double deltaPosition = (positionRel + radius_ * (normal_ - y)).dot(normal_);

        // Compute frame jacobian in local frame
        getFrameJacobian(model->pinocchioModel_,
                         model->pinocchioData_,
                         frameIndex_,
                         pinocchio::LOCAL_WORLD_ALIGNED,
                         frameJacobian_);

        // Contact point is at -radius_ x in local frame: compute corresponding jacobian
        jacobian_ = frameJacobian_.topRows(3);
        jacobian_.noalias() += skewRadius_ * frameJacobian_.bottomRows(3);

        // Compute ground normal derivative
        const pinocchio::Motion frameVelocity = getFrameVelocity(model->pinocchioModel_,
                                                                 model->pinocchioData_,
                                                                 frameIndex_,
                                                                 pinocchio::LOCAL_WORLD_ALIGNED);
        const Eigen::Vector3d & omega = frameVelocity.angular();

        const Eigen::Vector3d daxis_ = omega.cross(axis);
        const Eigen::Vector3d dx =
            daxis_.cross(normal_).cross(axis) + axis.cross(normal_).cross(daxis_);
        const Eigen::Vector3d z = dx / xNorm;
        const Eigen::Vector3d dy = z - y.dot(z) * y;

        Eigen::Vector3d velocity = frameVelocity.linear();
        velocity.noalias() += skewRadius_ * omega;

        // Compute frame drift in local frame
        pinocchio::Motion frameAcceleration = getFrameAcceleration(model->pinocchioModel_,
                                                                   model->pinocchioData_,
                                                                   frameIndex_,
                                                                   pinocchio::LOCAL_WORLD_ALIGNED);
        frameAcceleration.linear() += omega.cross(frameVelocity.linear());

        /* Compute total drift.
           Note that the cross product is (very) slightly slower than the matrix product by the
           skew matrix. */
        drift_ = frameAcceleration.linear();
        drift_.noalias() += skewRadius_ * frameAcceleration.angular();
        pinocchio::alphaSkew(radius_, dy, dskewRadius_);
        drift_.noalias() += dskewRadius_ * omega;

        // Add Baumgarte stabilization drift
        drift_ += kp_ * deltaPosition * normal_ + kd_ * velocity;
    }
}
