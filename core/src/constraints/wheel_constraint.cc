#include "pinocchio/algorithm/frames.hpp"  // `pinocchio::getFrameVelocity`, `pinocchio::getFrameAcceleration`

#include "jiminy/core/robot/model.h"
#include "jiminy/core/utilities/pinocchio.h"

#include "jiminy/core/constraints/wheel_constraint.h"


namespace jiminy
{
    template<>
    std::string const AbstractConstraintTpl<WheelConstraint>::type_("WheelConstraint");

    WheelConstraint::WheelConstraint(std::string const & frameName,
                                     float64_t   const & wheelRadius,
                                     vector3_t   const & groundNormal,
                                     vector3_t   const & wheelAxis) :
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
        // Empty on purpose
    }

    std::string const & WheelConstraint::getFrameName(void) const
    {
        return frameName_;
    }

    frameIndex_t const & WheelConstraint::getFrameIdx(void) const
    {
        return frameIdx_;
    }

    void WheelConstraint::setReferenceTransform(pinocchio::SE3 const & transformRef)
    {
        transformRef_ = transformRef;
    }

    pinocchio::SE3 const & WheelConstraint::getReferenceTransform(void) const
    {
        return transformRef_;
    }

    hresult_t WheelConstraint::reset(vectorN_t const & /* q */,
                                     vectorN_t const & /* v */)
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

    hresult_t WheelConstraint::computeJacobianAndDrift(vectorN_t const & /* q */,
                                                       vectorN_t const & /* v */)
    {
        if (!isAttached_)
        {
            PRINT_ERROR("Constraint not attached to a model.");
            return hresult_t::ERROR_GENERIC;
        }

        // Assuming the model still exists
        auto model = model_.lock();

        // Compute ground normal in local frame
        pinocchio::SE3 const & framePose = model->pncData_.oMf[frameIdx_];
        vector3_t const axis = framePose.rotation() * axis_;
        vector3_t const x = axis.cross(normal_).cross(axis);
        float64_t const xNorm = x.norm();
        vector3_t const y = x / xNorm;
        pinocchio::alphaSkew(radius_, y, skewRadius_);

        // Compute position error
        float64_t const deltaPosition =
            (framePose.translation() - transformRef_.translation() + radius_ * (normal_ - y)).dot(normal_);

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
        pinocchio::Motion const frameVelocity = getFrameVelocity(model->pncModel_,
                                                                 model->pncData_,
                                                                 frameIdx_,
                                                                 pinocchio::LOCAL_WORLD_ALIGNED);
        vector3_t const & omega = frameVelocity.angular();

        vector3_t const daxis_ = omega.cross(axis);
        vector3_t const dx = daxis_.cross(normal_).cross(axis) + axis.cross(normal_).cross(daxis_);
        vector3_t const z = dx / xNorm;
        vector3_t const dy = z - y.dot(z) * y;

        vector3_t velocity = frameVelocity.linear();
        velocity.noalias() += skewRadius_ * omega;

        // Compute frame drift in local frame
        pinocchio::Motion frameAcceleration = getFrameAcceleration(model->pncModel_,
                                                                   model->pncData_,
                                                                   frameIdx_,
                                                                   pinocchio::LOCAL_WORLD_ALIGNED);
        frameAcceleration.linear() += omega.cross(frameVelocity.linear());

        // Compute total drift
        pinocchio::alphaSkew(radius_, dy, dskewRadius_);
        drift_ = frameAcceleration.linear() +
                 skewRadius_ * frameAcceleration.angular() +
                 dskewRadius_ * omega;

        // Add Baumgarte stabilization drift
        drift_ += kp_ * deltaPosition * normal_ + kd_ * velocity;

        return hresult_t::SUCCESS;
    }
}
