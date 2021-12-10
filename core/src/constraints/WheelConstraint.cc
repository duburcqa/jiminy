#include "pinocchio/algorithm/frames.hpp"  // `pinocchio::getFrameVelocity`, `pinocchio::getFrameAcceleration`

#include "jiminy/core/robot/Model.h"
#include "jiminy/core/utilities/Pinocchio.h"

#include "jiminy/core/constraints/WheelConstraint.h"


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
    x3_(),
    skewRadius_(),
    dskewRadius_(),
    transformRef_(),
    frameJacobian_()
    {
        // Empty on purpose
    }

    WheelConstraint::~WheelConstraint(void)
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
        x3_.noalias() = model->pncData_.oMf[frameIdx_].rotation().transpose() * normal_;
        pinocchio::alphaSkew(radius_, x3_, skewRadius_);

        // Compute frame jacobian in local frame
        getFrameJacobian(model->pncModel_,
                         model->pncData_,
                         frameIdx_,
                         pinocchio::LOCAL,
                         frameJacobian_);

        // Contact point is at -radius_ x3 in local frame: compute corresponding jacobian
        jacobian_ = frameJacobian_.topRows(3) +
                    skewRadius_ * frameJacobian_.bottomRows(3);

        // Compute ground normal derivative
        vector3_t const omega = getFrameVelocity(model->pncModel_,
                                                 model->pncData_,
                                                 frameIdx_,
                                                 pinocchio::LOCAL).angular();
        auto dx3_ = - omega.cross(x3_);  // Using auto to not evaluate the expression
        pinocchio::alphaSkew(radius_, dx3_, dskewRadius_);

        // Compute frame drift in local frame
        pinocchio::Motion const driftLocal = getFrameAcceleration(model->pncModel_,
                                                                  model->pncData_,
                                                                  frameIdx_,
                                                                  pinocchio::LOCAL);

        // Compute total drift
        drift_ = driftLocal.linear() +
                 skewRadius_ * driftLocal.angular() + dskewRadius_ * omega;


        return hresult_t::SUCCESS;
    }
}
