#include "pinocchio/algorithm/frames.hpp"  // `pinocchio::getFrameVelocity`, `pinocchio::getFrameAcceleration`

#include "jiminy/core/exceptions.h"
#include "jiminy/core/robot/model.h"
#include "jiminy/core/utilities/pinocchio.h"

#include "jiminy/core/constraints/fixed_frame_constraint.h"


namespace jiminy
{
    template<>
    const std::string AbstractConstraintTpl<FixedFrameConstraint>::type_("FixedFrameConstraint");

    FixedFrameConstraint::FixedFrameConstraint(const std::string & frameName,
                                               const Eigen::Matrix<bool_t, 6, 1> & maskFixed) :
    AbstractConstraintTpl(),
    frameName_(frameName),
    frameIdx_(0),
    dofsFixed_(),
    transformRef_(),
    normal_(),
    rotationLocal_(Eigen::Matrix3d::Identity()),
    frameJacobian_(),
    frameDrift_()
    {
        dofsFixed_.clear();
        for (uint32_t i = 0; i < 6; ++i)
        {
            if (maskFixed[i])
            {
                dofsFixed_.push_back(i);
            }
        }
    }

    const std::string & FixedFrameConstraint::getFrameName() const
    {
        return frameName_;
    }

    const pinocchio::FrameIndex & FixedFrameConstraint::getFrameIdx() const
    {
        return frameIdx_;
    }

    const std::vector<uint32_t> & FixedFrameConstraint::getDofsFixed() const
    {
        return dofsFixed_;
    }

    void FixedFrameConstraint::setReferenceTransform(const pinocchio::SE3 & transformRef)
    {
        transformRef_ = transformRef;
    }

    const pinocchio::SE3 & FixedFrameConstraint::getReferenceTransform() const
    {
        return transformRef_;
    }

    void FixedFrameConstraint::setNormal(const Eigen::Vector3d & normal)
    {
        normal_ = normal;
        rotationLocal_.col(2) = normal_;
        rotationLocal_.col(1) = normal_.cross(Eigen::Vector3d::UnitX()).normalized();
        rotationLocal_.col(0) = rotationLocal_.col(1).cross(rotationLocal_.col(2));
    }

    const Eigen::Matrix3d & FixedFrameConstraint::getLocalFrame() const
    {
        return rotationLocal_;
    }

    hresult_t FixedFrameConstraint::reset(const Eigen::VectorXd & /* q */,
                                          const Eigen::VectorXd & /* v */)
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

            // Initialize constraint jacobian, drift and multipliers
            const Eigen::Index dim = static_cast<Eigen::Index>(dofsFixed_.size());
            jacobian_.setZero(dim, model->pncModel_.nv);
            drift_.setZero(dim);
            lambda_.setZero(dim);

            // Get the current frame position and use it as reference
            transformRef_ = model->pncData_.oMf[frameIdx_];

            // Set local frame to world by default
            rotationLocal_.setIdentity();
        }

        return returnCode;
    }

    hresult_t FixedFrameConstraint::computeJacobianAndDrift(const Eigen::VectorXd & /* q */,
                                                            const Eigen::VectorXd & /* v */)
    {
        if (!isAttached_)
        {
            PRINT_ERROR("Constraint not attached to a model.");
            return hresult_t::ERROR_GENERIC;
        }

        // Assuming the model still exists.
        auto model = model_.lock();

        // Get jacobian in local frame
        const pinocchio::SE3 & framePose = model->pncData_.oMf[frameIdx_];
        const pinocchio::SE3 transformLocal(rotationLocal_, framePose.translation());
        const pinocchio::Frame & frame = model->pncModel_.frames[frameIdx_];
        const pinocchio::JointModel & joint = model->pncModel_.joints[frame.parent];
        const int32_t colRef = joint.nv() + joint.idx_v() - 1;
        for (Eigen::DenseIndex j = colRef; j >= 0;
             j = model->pncData_.parents_fromRow[static_cast<std::size_t>(j)])
        {
            const pinocchio::MotionRef<Matrix6Xd::ColXpr> vIn(model->pncData_.J.col(j));
            pinocchio::MotionRef<Matrix6Xd::ColXpr> vOut(frameJacobian_.col(j));
            vOut = transformLocal.actInv(vIn);
        }

        // Compute pose error
        auto deltaPosition = framePose.translation() - transformRef_.translation();
        const Eigen::Vector3d deltaRotation =
            pinocchio::log3(framePose.rotation() * transformRef_.rotation().transpose());

        // Compute frame velocity in local frame
        const pinocchio::Motion velocity = getFrameVelocity(
            model->pncModel_, model->pncData_, frameIdx_, pinocchio::LOCAL_WORLD_ALIGNED);

        /* Get drift in world frame.
           We are actually looking for the classical acceleration here ! */
        frameDrift_ = getFrameAcceleration(
            model->pncModel_, model->pncData_, frameIdx_, pinocchio::LOCAL_WORLD_ALIGNED);
        frameDrift_.linear() += velocity.angular().cross(velocity.linear());

        // Add Baumgarte stabilization to drift in world frame
        frameDrift_.linear() += kp_ * deltaPosition;
        frameDrift_.angular() += kp_ * deltaRotation;
        frameDrift_ += kd_ * velocity;

        // Compute drift in local frame
        frameDrift_.linear() = rotationLocal_.transpose() * frameDrift_.linear();
        frameDrift_.angular() = rotationLocal_.transpose() * frameDrift_.angular();

        // Extract masked jacobian and drift, only containing fixed dofs
        for (uint32_t i = 0; i < dofsFixed_.size(); ++i)
        {
            const uint32_t & dofIndex = dofsFixed_[i];
            jacobian_.row(i) = frameJacobian_.row(dofIndex);
            drift_[i] = frameDrift_.toVector()[dofIndex];
        }

        return hresult_t::SUCCESS;
    }
}
