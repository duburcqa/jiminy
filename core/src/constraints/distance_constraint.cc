#include "pinocchio/algorithm/frames.hpp"  // `pinocchio::getFrameVelocity`, `pinocchio::getFrameAcceleration`

#include "jiminy/core/robot/model.h"
#include "jiminy/core/utilities/pinocchio.h"

#include "jiminy/core/constraints/distance_constraint.h"


namespace jiminy
{
    template<>
    const std::string AbstractConstraintTpl<DistanceConstraint>::type_{"DistanceConstraint"};

    DistanceConstraint::DistanceConstraint(const std::string & firstFrameName,
                                           const std::string & secondFrameName) noexcept :
    AbstractConstraintTpl(),
    frameNames_{firstFrameName, secondFrameName}
    {
    }

    const std::array<std::string, 2> & DistanceConstraint::getFramesNames() const noexcept
    {
        return frameNames_;
    }

    const std::array<pinocchio::FrameIndex, 2> &
    DistanceConstraint::getFrameIndices() const noexcept
    {
        return frameIndices_;
    }

    void DistanceConstraint::setReferenceDistance(double distanceRef)
    {
        if (distanceRef < 0.0)
        {
            THROW_ERROR(std::invalid_argument, "Reference distance must be positive.");
        }
        distanceRef_ = distanceRef;
    }

    double DistanceConstraint::getReferenceDistance() const noexcept
    {
        return distanceRef_;
    }

    void DistanceConstraint::reset(const Eigen::VectorXd & /* q */,
                                   const Eigen::VectorXd & /* v */)
    {
        // Make sure the model still exists
        auto model = model_.lock();
        if (!model)
        {
            THROW_ERROR(bad_control_flow, "Model pointer expired or unset.");
        }

        // Get frames indices
        for (uint8_t i = 0; i < 2; ++i)
        {
            frameIndices_[i] = ::jiminy::getFrameIndex(model->pinocchioModel_, frameNames_[i]);
        }

        // Initialize frames jacobians buffers
        for (Matrix6Xd & frameJacobian : frameJacobians_)
        {
            frameJacobian.setZero(6, model->pinocchioModel_.nv);
        }

        // Initialize jacobian, drift and multipliers
        jacobian_.setZero(1, model->pinocchioModel_.nv);
        drift_.setZero(1);
        lambda_.setZero(1);

        // Compute the current distance and use it as reference
        const Eigen::Vector3d deltaPosition =
            model->pinocchioData_.oMf[frameIndices_[0]].translation() -
            model->pinocchioData_.oMf[frameIndices_[1]].translation();
        distanceRef_ = deltaPosition.norm();
    }

    void DistanceConstraint::computeJacobianAndDrift(const Eigen::VectorXd & /* q */,
                                                     const Eigen::VectorXd & /* v */)
    {
        if (!isAttached_)
        {
            THROW_ERROR(bad_control_flow, "Constraint not attached to a model.");
        }

        // Assuming model still exists.
        auto model = model_.lock();

        // Compute direction between frames
        const Eigen::Vector3d deltaPosition =
            model->pinocchioData_.oMf[frameIndices_[0]].translation() -
            model->pinocchioData_.oMf[frameIndices_[1]].translation();
        const double deltaPositionNorm = deltaPosition.norm();
        const Eigen::Vector3d direction = deltaPosition / deltaPositionNorm;

        // Compute relative velocity between frames
        std::array<pinocchio::Motion, 2> frameVelocities{};
        frameVelocities[0] = getFrameVelocity(model->pinocchioModel_,
                                              model->pinocchioData_,
                                              frameIndices_[0],
                                              pinocchio::LOCAL_WORLD_ALIGNED);
        frameVelocities[1] = getFrameVelocity(model->pinocchioModel_,
                                              model->pinocchioData_,
                                              frameIndices_[1],
                                              pinocchio::LOCAL_WORLD_ALIGNED);
        Eigen::Vector3d deltaVelocity = frameVelocities[0].linear() - frameVelocities[1].linear();

        // Get jacobian in local frame: J_1 - J_2
        for (uint8_t i = 0; i < 2; ++i)
        {
            getFrameJacobian(model->pinocchioModel_,
                             model->pinocchioData_,
                             frameIndices_[i],
                             pinocchio::LOCAL_WORLD_ALIGNED,
                             frameJacobians_[i]);
        }
        jacobian_.noalias() =
            direction.transpose() * (frameJacobians_[0] - frameJacobians_[1]).topRows<3>();

        // Get drift in local frame
        std::array<pinocchio::Motion, 2> frameAccelerations{};
        for (uint8_t i = 0; i < 2; ++i)
        {
            frameAccelerations[i] = getFrameAcceleration(model->pinocchioModel_,
                                                         model->pinocchioData_,
                                                         frameIndices_[i],
                                                         pinocchio::LOCAL_WORLD_ALIGNED);
            frameAccelerations[i].linear() +=
                frameVelocities[i].angular().cross(frameVelocities[i].linear());
        }
        drift_[0] = direction.dot(frameAccelerations[0].linear() - frameAccelerations[1].linear());

        /* dDir.T * (dp_A - dp_B) =
               [(dp_A - dp_B) ** 2 - (dir.T * (dp_A - dp_B)) ** 2] / norm(p_A - p_B) */
        const double deltaVelocityProj = deltaVelocity.dot(direction);
        drift_[0] +=
            (deltaVelocity.squaredNorm() - std::pow(deltaVelocityProj, 2)) / deltaPositionNorm;

        // Add Baumgarte stabilization drift
        drift_[0] += kp_ * (deltaPositionNorm - distanceRef_) + kd_ * deltaVelocityProj;
    }
}
