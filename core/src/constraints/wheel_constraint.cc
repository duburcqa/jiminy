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
                                     const vector3_t & groundNormal,
                                     const vector3_t & wheelAxis) :
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

    const std::string & WheelConstraint::getFrameName(void) const
    {
        return frameName_;
    }

    const frameIndex_t & WheelConstraint::getFrameIdx(void) const
    {
        return frameIdx_;
    }

    void WheelConstraint::setReferenceTransform(const pinocchio::SE3 & transformRef)
    {
        transformRef_ = transformRef;
    }

    const pinocchio::SE3 & WheelConstraint::getReferenceTransform(void) const
    {
        return transformRef_;
    }

    hresult_t WheelConstraint::reset(const vectorN_t & /* q */, const vectorN_t & /* v */)
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

    hresult_t WheelConstraint::computeJacobianAndDrift(const vectorN_t & /* q */,
                                                       const vectorN_t & /* v */)
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
        const vector3_t axis = framePose.rotation() * axis_;
        const vector3_t x = axis.cross(normal_).cross(axis);
        const float64_t xNormInv = 1.0 / x.norm();
        const vector3_t y = x * xNormInv;
        pinocchio::alphaSkew(radius_, y, skewRadius_);

        // Compute position error
        auto positionRel = framePose.translation() - transformRef_.translation();
        const float64_t deltaPosition = (positionRel + radius_ * (normal_ - y)).dot(normal_);

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
        const vector3_t & omega = frameVelocity.angular();

        const vector3_t daxis_ = omega.cross(axis);
        const vector3_t dx = daxis_.cross(normal_).cross(axis) + axis.cross(normal_).cross(daxis_);
        const vector3_t z = dx * xNormInv;
        const vector3_t dy = z - y.dot(z) * y;

        vector3_t velocity = frameVelocity.linear();
        velocity.noalias() += skewRadius_ * omega;

        // Compute frame drift in local frame
        pinocchio::Motion frameAcceleration = getFrameAcceleration(
            model->pncModel_, model->pncData_, frameIdx_, pinocchio::LOCAL_WORLD_ALIGNED);
        frameAcceleration.linear() += omega.cross(frameVelocity.linear());

        // Compute total drift
        drift_ = frameAcceleration.linear() + skewRadius_ * frameAcceleration.angular();
        // The cross product is (very) slightly slower than the matrix product by the skew matrix
        pinocchio::alphaSkew(radius_, dy, dskewRadius_);
        drift_.noalias() += dskewRadius_ * omega;

        // Add Baumgarte stabilization drift
        drift_ += kp_ * deltaPosition * normal_ + kd_ * velocity;

        return hresult_t::SUCCESS;
    }
}
