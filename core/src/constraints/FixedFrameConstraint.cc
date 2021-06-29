#include "pinocchio/algorithm/frames.hpp"  // `pinocchio::getFrameVelocity`, `pinocchio::getFrameAcceleration`

#include "jiminy/core/robot/Model.h"
#include "jiminy/core/utilities/Pinocchio.h"

#include "jiminy/core/constraints/FixedFrameConstraint.h"


namespace jiminy
{
    template<>
    std::string const AbstractConstraintTpl<FixedFrameConstraint>::type_("FixedFrameConstraint");

    FixedFrameConstraint::FixedFrameConstraint(std::string const & frameName,
                                               Eigen::Matrix<bool_t, 6, 1> const & maskFixed,
                                               pinocchio::ReferenceFrame const & frameRef) :
    AbstractConstraintTpl(),
    frameName_(frameName),
    frameIdx_(0),
    frameRef_(frameRef),
    dofsFixed_(),
    transformRef_(),
    frameJacobian_(),
    frameDrift_()
    {
        dofsFixed_.resize(maskFixed.cast<int32_t>().array().sum());
        uint32_t dofIndex = 0;
        for (int32_t i=0; i < 6; ++i)
        {
            if (maskFixed[i])
            {
                dofsFixed_[dofIndex] = i;
                ++dofIndex;
            }
        }
    }

    FixedFrameConstraint::~FixedFrameConstraint(void)
    {
        // Empty on purpose
    }

    std::string const & FixedFrameConstraint::getFrameName(void) const
    {
        return frameName_;
    }

    frameIndex_t const & FixedFrameConstraint::getFrameIdx(void) const
    {
        return frameIdx_;
    }

    pinocchio::ReferenceFrame const & FixedFrameConstraint::getReferenceFrame(void) const
    {
        return frameRef_;
    }

    std::vector<uint32_t> const & FixedFrameConstraint::getDofsFixed(void) const
    {
        return dofsFixed_;
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
            // Initialize jacobian, drift and multipliers
            frameJacobian_ = matrixN_t::Zero(6, model->pncModel_.nv);
            frameDrift_ = vector6_t::Zero();
            uint64_t const dim = dofsFixed_.size();
            jacobian_ = matrixN_t::Zero(dim, model->pncModel_.nv);
            drift_ = vectorN_t::Zero(dim);
            lambda_ = vectorN_t::Zero(dim);

            // Get the current frame position and use it as reference
            transformRef_ = model->pncData_.oMf[frameIdx_];
        }

        return returnCode;
    }

    hresult_t FixedFrameConstraint::computeJacobianAndDrift(vectorN_t const & /* q */,
                                                            vectorN_t const & /* v */)
    {
        if (!isAttached_)
        {
            PRINT_ERROR("Constraint not attached to a model.");
            return hresult_t::ERROR_GENERIC;
        }

        // Assuming the model still exists.
        auto model = model_.lock();

        // Get jacobian
        getFrameJacobian(model->pncModel_,
                         model->pncData_,
                         frameIdx_,
                         frameRef_,
                         frameJacobian_);

        // Get drift
        frameDrift_ = getFrameAcceleration(model->pncModel_,
                                           model->pncData_,
                                           frameIdx_,
                                           frameRef_).toVector();

        // Add Baumgarte stabilization drift
        if (frameRef_ == pinocchio::LOCAL_WORLD_ALIGNED || frameRef_ == pinocchio::WORLD)
        {
            auto deltaPosition = model->pncData_.oMf[frameIdx_].translation() - transformRef_.translation();
            frameDrift_.head<3>() += kp_ * deltaPosition;
            auto deltaRotation = transformRef_.rotation().transpose() * model->pncData_.oMf[frameIdx_].rotation();
            vectorN_t const axis = pinocchio::log3(deltaRotation);
            frameDrift_.tail<3>() += kp_ * axis;
        }
        vector6_t const velocity = getFrameVelocity(model->pncModel_,
                                                    model->pncData_,
                                                    frameIdx_,
                                                    frameRef_).toVector();
        frameDrift_ += kd_ * velocity;

        // Extract masked jacobian and drift, only containing fixed dofs
        for (uint32_t i=0; i < dofsFixed_.size(); ++i)
        {
            uint32_t const & dofIndex = dofsFixed_[i];
            jacobian_.row(i) = frameJacobian_.row(dofIndex);
            drift_[i] = frameDrift_[dofIndex];
        }

        return hresult_t::SUCCESS;
    }
}
