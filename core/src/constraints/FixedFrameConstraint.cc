#include "pinocchio/algorithm/frames.hpp"    // `pinocchio::getFrameVelocity`, `pinocchio::getFrameAcceleration`

#include "jiminy/core/robot/Model.h"
#include "jiminy/core/utilities/Pinocchio.h"

#include "jiminy/core/constraints/FixedFrameConstraint.h"


namespace jiminy
{
    template<>
    std::string const AbstractConstraintTpl<FixedFrameConstraint>::type_("FixedFrameConstraint");

    FixedFrameConstraint::FixedFrameConstraint(std::string const & frameName,
                                               Eigen::Matrix<bool_t, 6, 1> const & maskFixed) :
    AbstractConstraintTpl(),
    frameName_(frameName),
    frameIdx_(0),
    maskFixed_(maskFixed),
    transformRef_(),
    normal_(),
    rotationLocal_(matrix3_t::Identity()),
    frameDrift_()
    {
    }

    std::string const & FixedFrameConstraint::getFrameName(void) const
    {
        return frameName_;
    }

    frameIndex_t const & FixedFrameConstraint::getFrameIdx(void) const
    {
        return frameIdx_;
    }

    Eigen::Matrix<bool_t, 6, 1> const & FixedFrameConstraint::getMaskFixed(void) const
    {
        return maskFixed_;
    }

    void FixedFrameConstraint::setReferenceTransform(pinocchio::SE3 const & transformRef)
    {
        transformRef_ = transformRef;
    }

    pinocchio::SE3 const & FixedFrameConstraint::getReferenceTransform(void) const
    {
        return transformRef_;
    }

    void FixedFrameConstraint::setNormal(vector3_t const & normal)
    {
        normal_ = normal;
        rotationLocal_.col(2) = normal_;
        rotationLocal_.col(1) = normal_.cross(vector3_t::UnitX()).normalized();
        rotationLocal_.col(0) = rotationLocal_.col(1).cross(rotationLocal_.col(2));
    }

    matrix3_t const & FixedFrameConstraint::getLocalFrame(void) const
    {
        return rotationLocal_;
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
            // Initialize constraint jacobian, drift and multipliers
            Eigen::Index const dim = maskFixed_.cast<Eigen::Index>().sum();
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

        // Get frame velocity in local frame
        pinocchio::Motion const velocity = getFrameVelocity(model->pncModel_,
                                                            model->pncData_,
                                                            frameIdx_,
                                                            pinocchio::LOCAL_WORLD_ALIGNED);

        // Get frame classical acceleration in local frame
        pinocchio::Motion accel = getFrameAcceleration(model->pncModel_,
                                                       model->pncData_,
                                                       frameIdx_,
                                                       pinocchio::LOCAL_WORLD_ALIGNED);
        accel.linear() += velocity.angular().cross(velocity.linear());

        // Compute pose error
        pinocchio::SE3 const & framePose = model->pncData_.oMf[frameIdx_];
        auto deltaPosition = framePose.translation() - transformRef_.translation();
        matrix3_t const deltaRotation = framePose.rotation() * transformRef_.rotation().transpose();
        float64_t deltaAngle; vector3_t const deltaLog3 = pinocchio::log3(deltaRotation, deltaAngle);

        // Compute jacobian in local frame
        matrix3_t deltaJlog3; Jlog3(deltaAngle, deltaLog3, deltaJlog3);
        pinocchio::SE3 const transformLocal(rotationLocal_, framePose.translation());
        pinocchio::Frame const & frame = model->pncModel_.frames[frameIdx_];
        pinocchio::JointModel const & joint = model->pncModel_.joints[frame.parent];
        int32_t const colRef = joint.nv() + joint.idx_v() - 1;
        for (Eigen::DenseIndex j=colRef; j>=0; j=model->pncData_.parents_fromRow[static_cast<std::size_t>(j)])
        {
            pinocchio::MotionRef<matrix6N_t::ColXpr> const vIn(model->pncData_.J.col(j));
            Eigen::Index k = 0;
            for (Eigen::Index i = 0; i < 3; ++i)
            {
                if (maskFixed_[i])
                {
                    jacobian_(k++, j) = transformLocal.actInv(vIn).linear()[i];
                }
            }
            for (Eigen::Index i = 0; i < 3; ++i)
            {
                if (maskFixed_[i + 3])
                {
                    jacobian_(k++, j) = deltaJlog3.row(i).dot(transformLocal.actInv(vIn).angular());
                }
            }
        }

        // Compute frame drift in world frame
        frameDrift_.linear() = accel.linear();
        matrix3_t dDeltaJlog3; dJlog3(deltaAngle, deltaLog3, deltaJlog3, velocity.angular(), dDeltaJlog3);
        frameDrift_.angular().noalias() = dDeltaJlog3 * velocity.angular();
        frameDrift_.angular().noalias() += deltaJlog3 * accel.angular();

        // Add Baumgarte stabilization to drift in world frame
        frameDrift_.linear() += kp_ * deltaPosition + kd_ * velocity.linear();
        frameDrift_.angular() += kp_ * deltaLog3;
        frameDrift_.angular().noalias() += kd_ * deltaJlog3 * velocity.angular();

        // Rotate drift in local frame
        Eigen::Index k = 0;
        for (Eigen::Index i = 0; i < 6; ++i)
        {
            if (maskFixed_[i])
            {
                drift_[k++] = rotationLocal_.col(i % 3).dot(i < 3 ? frameDrift_.linear() : frameDrift_.angular());
            }
        }

        return hresult_t::SUCCESS;
    }
}
