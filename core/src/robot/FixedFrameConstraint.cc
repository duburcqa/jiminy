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

    matrixN_t FixedFrameConstraint::getJacobian(vectorN_t const & q)
    {
        jacobian_.setZero();
        if (isAttached_)
        {
            // Get jacobian in local frame because drift is expressed
            // in local frame by pinocchio.
            getFrameJacobian(model_->pncModel_,
                             model_->pncData_,
                             frameId_,
                             pinocchio::LOCAL,
                             jacobian_);
        }
        return jacobian_;
    }

    vectorN_t FixedFrameConstraint::getDrift(vectorN_t const & q,
                                             vectorN_t const & v)
    {
        if (isAttached_)
        {
            drift_ = getFrameAcceleration(model_->pncModel_,
                                         model_->pncData_,
                                         frameId_).toVector();
        }
        return drift_;
    }

    hresult_t FixedFrameConstraint::attach(Model const * model)
    {
        if (isAttached_)
        {
            std::cout << "Error - FixedFrameConstraint::attach - Constraint already attached to a robot." << std::endl;
            return hresult_t::ERROR_GENERIC;
        }
        model_ = model;
        // Refresh proxies: this checks for the existence of frameName_ in model_.
        hresult_t returnCode = refreshProxies();
        if (returnCode == hresult_t::SUCCESS)
        {
             isAttached_ = true;
             // Set jacobian / drift to right dimension now that we know the model.
             jacobian_.resize(6, model_->pncModel_.nv);
             drift_.resize(6);
        }
        return returnCode;
    }

    hresult_t FixedFrameConstraint::refreshProxies()
    {
        return getFrameIdx(model_->pncModel_, frameName_, frameId_);
    }
}


