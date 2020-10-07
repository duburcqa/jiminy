#include "jiminy/core/robot/Robot.h"
#include "jiminy/core/Utilities.h"

#include "jiminy/core/robot/WheelConstraint.h"


namespace jiminy
{
    WheelConstraint::WheelConstraint(std::string const & frameName,
                                     float64_t   const & wheelRadius,
                                     vector3_t   const & groundNormal,
                                     vector3_t   const & wheelAxis) :
    AbstractConstraint(),
    frameName_(frameName),
    frameIdx_(0),
    radius_(wheelRadius),
    normal_(groundNormal / groundNormal.norm()),
    axis_(wheelAxis / wheelAxis.norm()),
    frameJacobian_(),
    jLas_()
    {
        // Empty on purpose
    }

    WheelConstraint::~WheelConstraint(void)
    {
        // Empty on purpose
    }

    inline matrix3_t skew(vector3_t const & v)
    {
        matrix3_t skew;
        skew <<   0.0, -v(2),  v(1),
                 v(2),   0.0, -v(0),
                -v(1),  v(0),   0.0;
        return skew;
    }

    matrixN_t const & WheelConstraint::getJacobian(vectorN_t const & q)
    {
        if (isAttached_)
        {
            // Compute frame jacobian in local frame
            getFrameJacobian(model_->pncModel_,
                             model_->pncData_,
                             frameIdx_,
                             pinocchio::LOCAL,
                             frameJacobian_);

            // Compute ground normal in local frame
            vector3_t const x3 = model_->pncData_.oMf[frameIdx_].rotation().transpose() * normal_;

            // Contact point is at -radius_ x3 in local frame: compute corresponding jacobian
            jacobian_ = frameJacobian_.topRows(3)
                + radius_ * skew(x3) * frameJacobian_.bottomRows(3);
        }
        else
        {
            jacobian_.setZero();
        }

        return jacobian_;
    }

    vectorN_t const & WheelConstraint::getDrift(vectorN_t const & q,
                                                vectorN_t const & v)
    {
        if (isAttached_)
        {
            // Compute frame drift in local frame.
            pinocchio::Motion const driftLocal = getFrameAcceleration(model_->pncModel_,
                                                                      model_->pncData_,
                                                                      frameIdx_);

            // Compute x3 and its derivative.
            vector3_t const x3 = model_->pncData_.oMf[frameIdx_].rotation().transpose() * normal_;

            vector3_t const omega = getFrameVelocity(model_->pncModel_,
                                                     model_->pncData_,
                                                     frameIdx_).angular();
            vector3_t const dx3 = - omega.cross(x3);

            // Compute total drift.
            drift_ = driftLocal.linear() +
                     radius_ * skew(x3) * driftLocal.angular() +
                     radius_ * skew(dx3) * omega;
        }
        else
        {
            drift_.setZero();
        }

        return drift_;
    }

    hresult_t WheelConstraint::refreshProxies()
    {
        // Resize the jacobian to the model dimension.
        jacobian_.resize(3, model_->pncModel_.nv);
        frameJacobian_.resize(6, model_->pncModel_.nv);
        drift_.resize(3);
        return getFrameIdx(model_->pncModel_, frameName_, frameIdx_);
    }
}
