#include "pinocchio/algorithm/frames.hpp"  // `pinocchio::getFrameVelocity`, `pinocchio::getFrameAcceleration`

#include "jiminy/core/robot/Model.h"
#include "jiminy/core/utilities/Pinocchio.h"

#include "jiminy/core/constraints/SphereConstraint.h"


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
    shewRadius_(pinocchio::alphaSkew(radius_, normal_)),
    transformRef_(),
    frameJacobian_()
    {
        // Empty on purpose
    }

    SphereConstraint::~SphereConstraint(void)
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
            jacobian_.noalias() += shewRadius_ * frameJacobian_.bottomRows(3);
        }

        // Compute frame drift in local frame
        pinocchio::Motion const driftLocal = getFrameAcceleration(model->pncModel_,
                                                                  model->pncData_,
                                                                  frameIdx_,
                                                                  pinocchio::LOCAL_WORLD_ALIGNED);

        // Compute total drift
        drift_ = driftLocal.linear();
        if (radius_ > EPS)
        {
            drift_.noalias() += shewRadius_ * driftLocal.angular();
        }

        // Add Baumgarte stabilization drift
        pinocchio::SE3 const & framePose = model->pncData_.oMf[frameIdx_];
        float64_t const deltaPosition =
            (framePose.translation() - transformRef_.translation()).dot(normal_);
        pinocchio::Motion const frameVelocity = getFrameVelocity(model->pncModel_,
                                                                 model->pncData_,
                                                                 frameIdx_,
                                                                 pinocchio::LOCAL_WORLD_ALIGNED);
        float64_t const velocity = frameVelocity.linear().dot(normal_);

        drift_.array() += (kp_ * deltaPosition + kd_ * velocity) * normal_.array();

        return hresult_t::SUCCESS;
    }
}
