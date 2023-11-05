#include "pinocchio/algorithm/frames.hpp"  // `pinocchio::getFrameVelocity`, `pinocchio::getFrameAcceleration`

#include "jiminy/core/robot/model.h"
#include "jiminy/core/utilities/pinocchio.h"

#include "jiminy/core/constraints/wheel_constraint.h"


namespace jiminy
{
    template<>
    const std::string AbstractConstraintTpl<WheelConstraint>::type_("WheelConstraint");

    WheelConstraint::WheelConstraint(const std::string & frameName,
                                     const float64_t & wheelRadius,
                                     const Eigen::Vector3d & groundNormal,
                                     const Eigen::Vector3d & wheelAxis) :
    AbstractConstraintTpl(),
    frameName_(frameName),
    frameIdx_(0),
    radius_(wheelRadius),
    normal_(groundNormal.normalized()),
    axis_(wheelAxis.normalized()),
    skewRadius_(),
    dskewRadius_(),
    transformRef_(),
    frameJacobian_()
    {
    }

    const std::string & WheelConstraint::getFrameName() const
    {
        return frameName_;
    }

    const frameIndex_t & WheelConstraint::getFrameIdx() const
    {
        return frameIdx_;
    }

    void WheelConstraint::setReferenceTransform(const pinocchio::SE3 & transformRef)
    {
        transformRef_ = transformRef;
    }

    const pinocchio::SE3 & WheelConstraint::getReferenceTransform() const
    {
        return transformRef_;
    }

    hresult_t WheelConstraint::reset(const Eigen::VectorXd & /* q */,
                                     const Eigen::VectorXd & /* v */)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        // Make sure the model still exists
        auto model = model_.lock();
        if (!model)
        {
            PRINT_ERROR("Model pointer expired or unset.");
            returnCode = hresult_t::ERROR_GENERIC;
        }

        // Get frame index
        if (returnCode == hresult_t::SUCCESS)
        {
            returnCode = ::jiminy::getFrameIdx(model->pncModel_, frameName_, frameIdx_);
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            // Initialize frames jacobians buffers
            frameJacobian_.setZero(6, model->pncModel_.nv);

            // Initialize jacobian, drift and multipliers
            jacobian_.setZero(3, model->pncModel_.nv);
            drift_.setZero(3);
            lambda_.setZero(3);

            // Get the current frame position and use it as reference
            transformRef_ = model->pncData_.oMf[frameIdx_];
        }

        return returnCode;
    }

    hresult_t WheelConstraint::computeJacobianAndDrift(const Eigen::VectorXd & /* q */,
                                                       const Eigen::VectorXd & /* v */)
    {
        if (!isAttached_)
        {
            PRINT_ERROR("Constraint not attached to a model.");
            return hresult_t::ERROR_GENERIC;
        }

        // Assuming the model still exists
        auto model = model_.lock();

        // Compute ground normal in local frame
        const pinocchio::SE3 & framePose = model->pncData_.oMf[frameIdx_];
        const Eigen::Vector3d axis = framePose.rotation() * axis_;
        const Eigen::Vector3d x = axis.cross(normal_).cross(axis);
        const float64_t xNorm = x.norm();
        const Eigen::Vector3d y = x / xNorm;
        pinocchio::alphaSkew(radius_, y, skewRadius_);

        // Compute position error
        const float64_t deltaPosition =
            (framePose.translation() - transformRef_.translation() + radius_ * (normal_ - y))
                .dot(normal_);

        // Compute frame jacobian in local frame
        getFrameJacobian(model->pncModel_,
                         model->pncData_,
                         frameIdx_,
                         pinocchio::LOCAL_WORLD_ALIGNED,
                         frameJacobian_);

        // Contact point is at -radius_ x in local frame: compute corresponding jacobian
        jacobian_ = frameJacobian_.topRows(3);
        jacobian_.noalias() += skewRadius_ * frameJacobian_.bottomRows(3);

        // Compute ground normal derivative
        const pinocchio::Motion frameVelocity = getFrameVelocity(
            model->pncModel_, model->pncData_, frameIdx_, pinocchio::LOCAL_WORLD_ALIGNED);
        const Eigen::Vector3d & omega = frameVelocity.angular();

        const Eigen::Vector3d daxis_ = omega.cross(axis);
        const Eigen::Vector3d dx =
            daxis_.cross(normal_).cross(axis) + axis.cross(normal_).cross(daxis_);
        const Eigen::Vector3d z = dx / xNorm;
        const Eigen::Vector3d dy = z - y.dot(z) * y;

        Eigen::Vector3d velocity = frameVelocity.linear();
        velocity.noalias() += skewRadius_ * omega;

        // Compute frame drift in local frame
        pinocchio::Motion frameAcceleration = getFrameAcceleration(
            model->pncModel_, model->pncData_, frameIdx_, pinocchio::LOCAL_WORLD_ALIGNED);
        frameAcceleration.linear() += omega.cross(frameVelocity.linear());

        // Compute total drift
        pinocchio::alphaSkew(radius_, dy, dskewRadius_);
        drift_ = frameAcceleration.linear() + skewRadius_ * frameAcceleration.angular() +
                 dskewRadius_ * omega;

        // Add Baumgarte stabilization drift
        drift_ += kp_ * deltaPosition * normal_ + kd_ * velocity;

        return hresult_t::SUCCESS;
    }
}
