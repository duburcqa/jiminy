#include "pinocchio/algorithm/frames.hpp"  // `pinocchio::getFrameVelocity`, `pinocchio::getFrameAcceleration`

#include "jiminy/core/robot/model.h"
#include "jiminy/core/utilities/pinocchio.h"

#include "jiminy/core/constraints/sphere_constraint.h"


namespace jiminy
{
    template<>
    std::string const AbstractConstraintTpl<SphereConstraint>::type_("SphereConstraint");

    SphereConstraint::SphereConstraint(std::string const & frameName,
                                       float64_t   const & sphereRadius,
                                       vector3_t   const & groundNormal) :
    AbstractConstraintTpl(),
    frameName_(frameName),
    frameIdx_(0),
    radius_(sphereRadius),
    normal_(groundNormal.normalized()),
    skewRadius_(pinocchio::alphaSkew(radius_, normal_)),
    transformRef_(),
    frameJacobian_()
    {
        // Empty on purpose
    }

    std::string const & SphereConstraint::getFrameName(void) const
    {
        return frameName_;
    }

    frameIndex_t const & SphereConstraint::getFrameIdx(void) const
    {
        return frameIdx_;
    }

    void SphereConstraint::setReferenceTransform(pinocchio::SE3 const & transformRef)
    {
        transformRef_ = transformRef;
    }

    pinocchio::SE3 const & SphereConstraint::getReferenceTransform(void) const
    {
        return transformRef_;
    }

    hresult_t SphereConstraint::reset(vectorN_t const & /* q */,
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

    hresult_t SphereConstraint::computeJacobianAndDrift(vectorN_t const & /* q */,
                                                        vectorN_t const & /* v */)
    {
        if (!isAttached_)
        {
            PRINT_ERROR("Constraint not attached to a model.");
            return hresult_t::ERROR_GENERIC;
        }

        // Assuming the model still exists
        auto model = model_.lock();

        // Compute frame jacobian in local frame
        getFrameJacobian(model->pncModel_,
                         model->pncData_,
                         frameIdx_,
                         pinocchio::LOCAL_WORLD_ALIGNED,
                         frameJacobian_);

        // Contact point is at - radius_ * normal_: compute corresponding jacobian
        jacobian_ = frameJacobian_.topRows(3);
        if (radius_ > EPS)
        {
            jacobian_.noalias() += skewRadius_ * frameJacobian_.bottomRows(3);
        }

        // Compute position error
        pinocchio::SE3 const & framePose = model->pncData_.oMf[frameIdx_];
        float64_t const deltaPosition =
            (framePose.translation() - transformRef_.translation()).dot(normal_);

        // Compute velocity error
        pinocchio::Motion const frameVelocity = getFrameVelocity(model->pncModel_,
                                                                 model->pncData_,
                                                                 frameIdx_,
                                                                 pinocchio::LOCAL_WORLD_ALIGNED);
        vector3_t velocity = frameVelocity.linear();
        velocity.noalias() += skewRadius_ * frameVelocity.angular();

        // Compute frame drift in local frame
        pinocchio::Motion driftLocal = getFrameAcceleration(model->pncModel_,
                                                            model->pncData_,
                                                            frameIdx_,
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

        return hresult_t::SUCCESS;
    }
}
