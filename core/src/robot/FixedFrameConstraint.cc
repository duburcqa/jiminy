#include "jiminy/core/robot/Robot.h"
#include "jiminy/core/Utilities.h"

#include "jiminy/core/robot/FixedFrameConstraint.h"


namespace jiminy
{
    FixedFrameConstraint::FixedFrameConstraint(std::string const & frameName) :
    AbstractConstraint(),
    frameName_(frameName),
    frameIdx_(0)
    {
        // Empty on purpose
    }

    FixedFrameConstraint::~FixedFrameConstraint(void)
    {
        // Empty on purpose
    }

    matrixN_t const & FixedFrameConstraint::getJacobian(vectorN_t const & q)
    {
        jacobian_.setZero();
        if (isAttached_)
        {
            // Get jacobian in local frame because drift is expressed
            // in local frame by pinocchio.
            getFrameJacobian(model_->pncModel_,
                             model_->pncData_,
                             frameIdx_,
                             pinocchio::LOCAL,
                             jacobian_);
        }
        return jacobian_;
    }

    vectorN_t const & FixedFrameConstraint::getDrift(vectorN_t const & q,
                                                     vectorN_t const & v)
    {
        if (isAttached_)
        {
            drift_ = getFrameAcceleration(model_->pncModel_,
                                          model_->pncData_,
                                          frameIdx_).toVector();
        }
        return drift_;
    }

    hresult_t FixedFrameConstraint::refreshProxies()
    {
        // Set jacobian / drift to right dimension
        jacobian_.resize(6, model_->pncModel_.nv);
        drift_.resize(6);
        return getFrameIdx(model_->pncModel_, frameName_, frameIdx_);
    }
}
