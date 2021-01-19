#include "jiminy/core/robot/Robot.h"
#include "jiminy/core/Macros.h"

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

    hresult_t WheelConstraint::reset(void)
    {
        // Make sure the model still exists
        auto model = model_.lock();
        if (!model)
        {
            PRINT_ERROR("Model pointer expired or unset.");
            return hresult_t::ERROR_GENERIC;
        }

        // Set jacobian / drift to right dimension
        jacobian_ = matrixN_t::Zero(3, model->pncModel_.nv);
        frameJacobian_ = matrixN_t::Zero(6, model->pncModel_.nv);
        drift_ = vectorN_t::Zero(3);

        return getFrameIdx(model->pncModel_, frameName_, frameIdx_);
    }

    inline matrix3_t skew(vector3_t const & v)
    {
        matrix3_t skew;
        skew <<   0.0, -v(2),  v(1),
                 v(2),   0.0, -v(0),
                -v(1),  v(0),   0.0;
        return skew;
    }

    hresult_t WheelConstraint::computeJacobianAndDrift(vectorN_t const & q,
                                                       vectorN_t const & v)
    {
        if (!isAttached_)
        {
            PRINT_ERROR("Constraint not attached to a model.");
            return hresult_t::ERROR_GENERIC;
        }

        // Assuming the model still exists.
        auto model = model_.lock();

        // Compute ground normal in local frame
        vector3_t const x3 = model->pncData_.oMf[frameIdx_].rotation().transpose() * normal_;

        // Compute frame jacobian in local frame
        getFrameJacobian(model->pncModel_,
                         model->pncData_,
                         frameIdx_,
                         pinocchio::LOCAL,
                         frameJacobian_);

        // Contact point is at -radius_ x3 in local frame: compute corresponding jacobian
        jacobian_ = frameJacobian_.topRows(3)
            + radius_ * skew(x3) * frameJacobian_.bottomRows(3);

        // Compute frame drift in local frame.
        pinocchio::Motion const driftLocal = getFrameAcceleration(model->pncModel_,
                                                                  model->pncData_,
                                                                  frameIdx_,
                                                                  pinocchio::LOCAL);

        // Compute total drift.
        vector3_t const omega = getFrameVelocity(model->pncModel_,
                                                 model->pncData_,
                                                 frameIdx_,
                                                 pinocchio::LOCAL).angular();
        auto dx3 = - omega.cross(x3);  // Using auto to not evaluate the expression

        drift_ = driftLocal.linear() +
                 radius_ * skew(x3) * driftLocal.angular() +
                 radius_ * skew(dx3) * omega;

        return hresult_t::SUCCESS;
    }
}
