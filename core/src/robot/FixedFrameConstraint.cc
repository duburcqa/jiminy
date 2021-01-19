#include "jiminy/core/robot/Robot.h"
#include "jiminy/core/Macros.h"

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

    hresult_t FixedFrameConstraint::reset(void)
    {
        // Make sure the model still exists
        auto model = model_.lock();
        if (!model)
        {
            PRINT_ERROR("Model pointer expired or unset.");
            return hresult_t::ERROR_GENERIC;
        }

        // Set jacobian / drift to right dimension
        jacobian_ = matrixN_t::Zero(6, model->pncModel_.nv);
        drift_ = vectorN_t::Zero(6);

        return getFrameIdx(model->pncModel_, frameName_, frameIdx_);
    }

    hresult_t FixedFrameConstraint::computeJacobianAndDrift(vectorN_t const & q,
                                                            vectorN_t const & v)
    {
        if (!isAttached_)
        {
            PRINT_ERROR("Constraint not attached to a model.");
            return hresult_t::ERROR_GENERIC;
        }

        // Assuming the model still exists.
        auto model = model_.lock();

        // Get jacobian and drift in local frame
        getFrameJacobian(model->pncModel_,
                         model->pncData_,
                         frameIdx_,
                         pinocchio::LOCAL,
                         jacobian_);

        drift_ = getFrameAcceleration(model->pncModel_,
                                      model->pncData_,
                                      frameIdx_,
                                      pinocchio::LOCAL).toVector();

        return hresult_t::SUCCESS;
    }
}
