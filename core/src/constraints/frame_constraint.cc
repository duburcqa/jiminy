#include "pinocchio/algorithm/frames.hpp"  // `pinocchio::getFrameVelocity`, `pinocchio::getFrameAcceleration`

#include "jiminy/core/robot/model.h"
#include "jiminy/core/utilities/pinocchio.h"

#include "jiminy/core/constraints/frame_constraint.h"


namespace jiminy
{
    template<int N>
    std::vector<uint32_t> maskToVector(const std::array<bool, N> & mask)
    {
        std::vector<uint32_t> vec;
        vec.reserve(N);
        for (uint8_t i = 0; i < N; ++i)
        {
            if (mask[i])
            {
                vec.push_back(i);
            }
        }
        return vec;
    }

    template<>
    const std::string AbstractConstraintTpl<FrameConstraint>::type_{"FrameConstraint"};

    FrameConstraint::FrameConstraint(const std::string & frameName,
                                     const std::array<bool, 6> & maskDoFs) noexcept :
    AbstractConstraintTpl(),
    frameName_{frameName},
    dofsFixed_{maskToVector<6>(maskDoFs)}
    {
    }

    const std::string & FrameConstraint::getFrameName() const noexcept
    {
        return frameName_;
    }

    pinocchio::FrameIndex FrameConstraint::getFrameIdx() const noexcept
    {
        return frameIdx_;
    }

    const std::vector<uint32_t> & FrameConstraint::getDofsFixed() const noexcept
    {
        return dofsFixed_;
    }

    void FrameConstraint::setReferenceTransform(const pinocchio::SE3 & transformRef) noexcept
    {
        transformRef_ = transformRef;
    }

    const pinocchio::SE3 & FrameConstraint::getReferenceTransform() const noexcept
    {
        return transformRef_;
    }

    void FrameConstraint::setNormal(const Eigen::Vector3d & normal) noexcept
    {
        normal_ = normal;
        rotationLocal_.col(2) = normal_;
        rotationLocal_.col(1) = normal_.cross(Eigen::Vector3d::UnitX()).normalized();
        rotationLocal_.col(0) = rotationLocal_.col(1).cross(rotationLocal_.col(2));
    }

    const Eigen::Matrix3d & FrameConstraint::getLocalFrame() const noexcept
    {
        return rotationLocal_;
    }

    hresult_t FrameConstraint::reset(const Eigen::VectorXd & /* q */,
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

    hresult_t FrameConstraint::computeJacobianAndDrift(const Eigen::VectorXd & /* q */,
                                                       const Eigen::VectorXd & /* v */)
    {
        if (!isAttached_)
        {
            PRINT_ERROR("Constraint not attached to a model.");
            return hresult_t::ERROR_GENERIC;
        }

        // Assuming the model still exists.
        auto model = model_.lock();

        /* Get jacobian in local frame:

           In general, for the angular part we have:
               p_c = log3(R_f),
               v_c = J_log3 * v_f = J_log3 * J_f * dq,
               a_c = d(J_log3)/dt * v_f + J_log3 * a_f
                   = d(J_log3)/dt * v_f + J_log3 * a_f^0 + (J_log3 * J_f) * ddq
                     where a_f^0 = d(J_f)/dt * dq
           It means:
               jac = J_log3 * J_f ,
               drift = d(J_log3)/dt * v_f + J_log3 * a_f^0
           Yet, we have the identity:
               d(log3)/dt = J_log3 * v_f = v_f
           It follows:
               d(J_log3)/dt * v_f = 0,
               J_log3 * a_f^0 = a_f^0
           Hence, it yields:
               jac = J_f,
               drift = a_f^0

           For reference, see: https://github.com/duburcqa/jiminy/pull/603 */
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
            const uint32_t dofIndex = dofsFixed_[i];
            jacobian_.row(i) = frameJacobian_.row(dofIndex);
            drift_[i] = frameDrift_.toVector()[dofIndex];
        }

        return hresult_t::SUCCESS;
    }
}
