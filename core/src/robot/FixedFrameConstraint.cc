#include "jiminy/core/robot/Robot.h"
#include "jiminy/core/Macros.h"

#include "jiminy/core/robot/FixedFrameConstraint.h"


namespace jiminy
{
    template<>
    std::string const AbstractConstraintTpl<FixedFrameConstraint>::type_("FixedFrameConstraint");

    FixedFrameConstraint::FixedFrameConstraint(std::string const & frameName,
                                               bool_t const & isTranslationFixed,
                                               bool_t const & isRotationFixed) :
    AbstractConstraintTpl(),
    frameName_(frameName),
    frameIdx_(0),
    isTranslationFixed_(isTranslationFixed),
    isRotationFixed_(isRotationFixed),
    transformRef_(),
    frameJacobian_()
    {
        // Empty on purpose
    }

    FixedFrameConstraint::~FixedFrameConstraint(void)
    {
        // Empty on purpose
    }

    std::string const & FixedFrameConstraint::getFrameName(void) const
    {
        return frameName_;
    }

    int32_t const & FixedFrameConstraint::getFrameIdx(void) const
    {
        return frameIdx_;
    }

    bool_t const & FixedFrameConstraint::getIsTranslationFixed(void) const
    {
        return isTranslationFixed_;
    }

    bool_t const & FixedFrameConstraint::getIsRotationFixed(void) const
    {
        return isRotationFixed_;
    }

    void FixedFrameConstraint::setReferenceTransform(pinocchio::SE3 const & transformRef)
    {
        transformRef_ = transformRef;
    }

    pinocchio::SE3 & FixedFrameConstraint::getReferenceTransform(void)
    {
        return transformRef_;
    }

    hresult_t FixedFrameConstraint::reset(vectorN_t const & /* q */,
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
            // Set jacobian / drift to right dimension
            frameJacobian_ = matrixN_t::Zero(6, model->pncModel_.nv);
            uint32_t dim = 3 * (uint32_t(isTranslationFixed_) + uint32_t(isRotationFixed_));
            jacobian_ = matrixN_t::Zero(dim, model->pncModel_.nv);
            drift_ = vectorN_t::Zero(dim);

            // Get the current frame position and use it as reference
            transformRef_ = model->pncData_.oMf[frameIdx_];
        }

        return returnCode;
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

        // Get jacobian in local frame
        if (isTranslationFixed_ && isRotationFixed_)
        {
            getFrameJacobian(model->pncModel_,
                             model->pncData_,
                             frameIdx_,
                             pinocchio::LOCAL_WORLD_ALIGNED,
                             jacobian_);
        }
        else
        {
            getFrameJacobian(model->pncModel_,
                             model->pncData_,
                             frameIdx_,
                             pinocchio::LOCAL_WORLD_ALIGNED,
                             frameJacobian_);
            if (isTranslationFixed_)
            {
                jacobian_ = frameJacobian_.topRows<3>();
            }
            else
            {
                jacobian_ = frameJacobian_.bottomRows<3>();
            }
        }

        // Get drift in local frame
        if (isTranslationFixed_ && isRotationFixed_)
        {
            drift_ = getFrameAcceleration(model->pncModel_,
                                          model->pncData_,
                                          frameIdx_,
                                          pinocchio::LOCAL_WORLD_ALIGNED).toVector();
        }
        else
        {
            pinocchio::Motion const driftLocal = getFrameAcceleration(model->pncModel_,
                                                                      model->pncData_,
                                                                      frameIdx_,
                                                                      pinocchio::LOCAL_WORLD_ALIGNED);
            if (isTranslationFixed_)
            {
                drift_ = driftLocal.linear();
            }
            else
            {
                drift_ = driftLocal.angular();
            }
        }

        // Add Baumgarte stabilization drift
        pinocchio::Motion const velocity = getFrameVelocity(model->pncModel_,
                                                            model->pncData_,
                                                            frameIdx_,
                                                            pinocchio::LOCAL_WORLD_ALIGNED);
        if (isTranslationFixed_)
        {
            auto deltaPosition = model->pncData_.oMf[frameIdx_].translation() - transformRef_.translation();
            drift_.head<3>() += kp_ * deltaPosition + kd_ * velocity.linear();
        }
        if (isRotationFixed_)
        {
            auto deltaRotation = transformRef_.rotation().transpose() * model->pncData_.oMf[frameIdx_].rotation();
            vectorN_t const axis = pinocchio::log3(deltaRotation);
            drift_.tail<3>() += kp_ * axis + kd_ * velocity.angular();
        }

        return hresult_t::SUCCESS;
    }
}
