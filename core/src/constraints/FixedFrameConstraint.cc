#include "pinocchio/algorithm/frames.hpp"    // `pinocchio::getFrameVelocity`, `pinocchio::getFrameAcceleration`
#include "pinocchio/algorithm/cholesky.hpp"  // `pinocchio::cholesky::`

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
    dofsFixed_(),
    isFixedPositionXY_(maskFixed.head<2>().all()),
    transformRef_(),
    normal_(),
    rotationLocal_(matrix3_t::Identity()),
    frameJacobian_(),
    frameDrift_(),
    UiJt_()
    {
        dofsFixed_.resize(static_cast<std::size_t>(maskFixed.cast<int32_t>().array().sum()));
        uint32_t dofIndex = 0;
        for (uint32_t const & i : std::array<uint32_t, 6>{{2, 1, 0, 3, 4, 5}})
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

    std::vector<uint32_t> const & FixedFrameConstraint::getDofsFixed(void) const
    {
        return dofsFixed_;
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
            // Initialize frames jacobians buffers
            frameJacobian_.setZero(6, model->pncModel_.nv);

            // Initialize constraint jacobian, drift and multipliers
            Eigen::Index const dim = static_cast<Eigen::Index>(dofsFixed_.size());
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

        // Define inverse rotation matrix of local frame
        auto rotInvLocal = rotationLocal_.transpose();

        // Get jacobian in local frame
        getFrameJacobian(model->pncModel_,
                         model->pncData_,
                         frameIdx_,
                         pinocchio::LOCAL_WORLD_ALIGNED,
                         frameJacobian_);

        pinocchio::Frame const & frame = model->pncModel_.frames[frameIdx_];
        pinocchio::JointModel const & joint = model->pncModel_.joints[frame.parent];
        int32_t const colRef = joint.nv() + joint.idx_v() - 1;
        for(Eigen::DenseIndex j=colRef; j>=0; j=model->pncData_.parents_fromRow[static_cast<std::size_t>(j)])
        {
            pinocchio::MotionRef<matrix6N_t::ColXpr> J_col(frameJacobian_.col(j));
            J_col.linear() = rotInvLocal * J_col.linear();
            J_col.angular() = rotInvLocal * J_col.angular();
        }

        // Get drift in world frame
        frameDrift_ = getFrameAcceleration(model->pncModel_,
                                           model->pncData_,
                                           frameIdx_,
                                           pinocchio::LOCAL_WORLD_ALIGNED);

        // Compute pose error
        pinocchio::SE3 const & framePose = model->pncData_.oMf[frameIdx_];
        vector3_t deltaPosition = framePose.translation() - transformRef_.translation();
        matrix3_t const deltaRotation = transformRef_.rotation().transpose() * framePose.rotation();

        // Compute velocity error
        pinocchio::Motion velocity = getFrameVelocity(model->pncModel_,
                                                      model->pncData_,
                                                      frameIdx_,
                                                      pinocchio::LOCAL_WORLD_ALIGNED);

        // Correct bias of PGS solver in tangential plane
        if (isFixedPositionXY_)
        {
            UiJt_ = frameJacobian_.topRows<2>().transpose();
            pinocchio::cholesky::Uiv(model->pncModel_, model->pncData_, UiJt_);
            vector2_t a;
            a << model->pncData_.D[1] * UiJt_.col(0).squaredNorm(),
                 model->pncData_.D[0] * UiJt_.col(1).squaredNorm();
            a /= a.array().maxCoeff();
            velocity.linear().head<2>().array() *= a.array();
            deltaPosition.head<2>().array() *= a.array();
        }

        // Add Baumgarte stabilization to drift in world frame
        frameDrift_.linear() += kp_ * deltaPosition;
        frameDrift_.angular() += kp_ * framePose.rotation() * pinocchio::log3(deltaRotation);
        frameDrift_ += kd_ * velocity;

        // Compute drift in local frame
        frameDrift_.linear() = rotInvLocal * frameDrift_.linear();
        frameDrift_.angular() = rotInvLocal * frameDrift_.angular();

        // Extract masked jacobian and drift, only containing fixed dofs
        for (uint32_t i=0; i < dofsFixed_.size(); ++i)
        {
            uint32_t const & dofIndex = dofsFixed_[i];
            jacobian_.row(i) = frameJacobian_.row(dofIndex);
            drift_[i] = frameDrift_.toVector()[dofIndex];
        }

        return hresult_t::SUCCESS;
    }
}
