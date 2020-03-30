#include "jiminy/core/robot/Robot.h"
#include "jiminy/core/Utilities.h"

#include "jiminy/core/robot/FixedFrameConstraint.h"


namespace jiminy
{
    FixedFrameConstraint::FixedFrameConstraint(std::string const & frameName) :
    AbstractConstraint(),
    frameName_(frameName),
    frameId_(0)
    {
        // Empty on purpose
    }
    FixedFrameConstraint::~FixedFrameConstraint(void)
    {
        // Empty on purpose
    }

    hresult_t FixedFrameConstraint::initialize(Model *model)
    {
        model_ = model;
        // Refresh proxies: this checks for the existence of frameName_ in model_.
        hresult_t returnCode = refreshProxies();
        isInitialized_ = returnCode == hresult_t::SUCCESS;
        return returnCode;
    }

    hresult_t FixedFrameConstraint::refreshProxies()
    {
        return getFrameIdx(model_->pncModel_, frameName_, frameId_);
    }

    matrixN_t FixedFrameConstraint::getJacobian(vectorN_t const & q) const
    {
        matrixN_t J = matrixN_t::Zero(6, model_->pncModel_.nv);
        if (isInitialized_)
        {
            // Get jacobian in local frame because drift is expressed
            // in local frame by pinocchio.
            getFrameJacobian(model_->pncModel_,
                             model_->pncData_,
                             frameId_,
                             pinocchio::LOCAL,
                             J);
        }
        return J;
    }

    vectorN_t FixedFrameConstraint::getDrift(vectorN_t const & q,
                                             vectorN_t const & v) const
    {
        vectorN_t drift = vectorN_t::Zero(6);
        if (isInitialized_)
        {
            drift = getFrameAcceleration(model_->pncModel_,
                                         model_->pncData_,
                                         frameId_).toVector();
        }
        return drift;
    }
}


