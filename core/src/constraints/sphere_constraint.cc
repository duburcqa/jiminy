#include "pinocchio/algorithm/frames.hpp"  // `pinocchio::getFrameVelocity`, `pinocchio::getFrameAcceleration`

#include "jiminy/core/robot/model.h"
#include "jiminy/core/utilities/pinocchio.h"

#include "jiminy/core/constraints/sphere_constraint.h"


namespace jiminy
{
    template<>
    const std::string AbstractConstraintTpl<SphereConstraint>::type_{"SphereConstraint"};

    SphereConstraint::SphereConstraint(const std::string & frameName,
                                       double sphereRadius,
                                       const Eigen::Vector3d & groundNormal) noexcept :
    AbstractConstraintTpl(),
    frameName_{frameName},
    radius_{sphereRadius},
    normal_{groundNormal.normalized()}
    {
    }

    const std::string & SphereConstraint::getFrameName() const noexcept
    {
        return frameName_;
    }

    pinocchio::FrameIndex SphereConstraint::getFrameIndex() const noexcept
    {
        return frameIndex_;
    }

    double SphereConstraint::getRadius() const noexcept
    {
        return radius_;
    }
    const Eigen::Vector3d & SphereConstraint::getNormal() const noexcept
    {
        return normal_;
    }

    void SphereConstraint::setReferenceTransform(const pinocchio::SE3 & transformRef) noexcept
    {
        transformRef_ = transformRef;
    }

    const pinocchio::SE3 & SphereConstraint::getReferenceTransform() const noexcept
    {
        return transformRef_;
    }

    void SphereConstraint::reset(const Eigen::VectorXd & /* q */, const Eigen::VectorXd & /* v */)
    {
        // Make sure the model still exists
        auto model = model_.lock();
        if (!model)
        {
            JIMINY_THROW(bad_control_flow, "Model pointer expired or unset.");
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

    void SphereConstraint::computeJacobianAndDrift(const Eigen::VectorXd & /* q */,
                                                   const Eigen::VectorXd & /* v */)
    {
        if (!isAttached_)
        {
            JIMINY_THROW(bad_control_flow, "Constraint not attached to a model.");
        }

        // Assuming the model still exists
        auto model = model_.lock();

        // Compute frame jacobian in local frame
        getFrameJacobian(model->pinocchioModel_,
                         model->pinocchioData_,
                         frameIndex_,
                         pinocchio::LOCAL_WORLD_ALIGNED,
                         frameJacobian_);

        // Contact point is at - radius_ * normal_: compute corresponding jacobian
        jacobian_ = frameJacobian_.topRows(3);
        if (radius_ > EPS)
        {
            jacobian_.noalias() += skewRadius_ * frameJacobian_.bottomRows(3);
        }

        // Compute position error
        const pinocchio::SE3 & framePose = model->pinocchioData_.oMf[frameIndex_];
        auto positionRel = framePose.translation() - transformRef_.translation();
        const double deltaPosition = positionRel.dot(normal_);

        // Compute velocity error
        const pinocchio::Motion frameVelocity = getFrameVelocity(model->pinocchioModel_,
                                                                 model->pinocchioData_,
                                                                 frameIndex_,
                                                                 pinocchio::LOCAL_WORLD_ALIGNED);
        Eigen::Vector3d velocity = frameVelocity.linear();
        velocity.noalias() += skewRadius_ * frameVelocity.angular();

        // Compute frame drift in local frame
        pinocchio::Motion driftLocal = getFrameAcceleration(model->pinocchioModel_,
                                                            model->pinocchioData_,
                                                            frameIndex_,
                                                            pinocchio::LOCAL_WORLD_ALIGNED);
        driftLocal.linear() += frameVelocity.angular().cross(frameVelocity.linear());

        // Compute total drift
        drift_ = driftLocal.linear();
        if (radius_ > EPS)
        {
            drift_.noalias() += skewRadius_ * driftLocal.angular();
        }

        // Add Baumgarte stabilization drift
        drift_ += kp_ * deltaPosition * normal_ + kd_ * velocity;
    }
}
