#include "pinocchio/algorithm/frames.hpp"  // `pinocchio::getFrameVelocity`, `pinocchio::getFrameAcceleration`

#include "jiminy/core/robot/model.h"
#include "jiminy/core/utilities/pinocchio.h"

#include "jiminy/core/constraints/distance_constraint.h"


namespace jiminy
{
    template<>
    std::string const AbstractConstraintTpl<DistanceConstraint>::type_("DistanceConstraint");

    DistanceConstraint::DistanceConstraint(std::string const & firstFrameName,
                                           std::string const & secondFrameName) :
    AbstractConstraintTpl(),
    framesNames_{firstFrameName, secondFrameName},
    framesIdx_(),
    distanceRef_(0.0),
    firstFrameJacobian_(),
    secondFrameJacobian_()
    {
        // Empty on purpose
    }

    std::vector<std::string> const & DistanceConstraint::getFramesNames(void) const
    {
        return framesNames_;
    }

    std::vector<frameIndex_t> const & DistanceConstraint::getFramesIdx(void) const
    {
        return framesIdx_;
    }

    hresult_t DistanceConstraint::setReferenceDistance(float64_t const & distanceRef)
    {
        if (distanceRef < 0.0)
        {
            PRINT_ERROR("The reference distance must be positive.");
            return hresult_t::ERROR_BAD_INPUT;
        }
        distanceRef_ = distanceRef;

        return hresult_t::SUCCESS;
    }

    float64_t const & DistanceConstraint::getReferenceDistance(void) const
    {
        return distanceRef_;
    }

    hresult_t DistanceConstraint::reset(vectorN_t const & /* q */,
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

        // Get frames indices
        framesIdx_.clear();
        framesIdx_.reserve(framesNames_.size());
        for (std::string const & frameName : framesNames_)
        {
            frameIndex_t frameIdx = 0;
            if (returnCode == hresult_t::SUCCESS)
            {
                returnCode = ::jiminy::getFrameIdx(model->pncModel_, frameName, frameIdx);
            }
            if (returnCode == hresult_t::SUCCESS)
            {
                framesIdx_.emplace_back(frameIdx);
            }
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            // Initialize frames jacobians buffers
            firstFrameJacobian_.setZero(6, model->pncModel_.nv);
            secondFrameJacobian_.setZero(6, model->pncModel_.nv);

            // Initialize jacobian, drift and multipliers
            jacobian_.setZero(1, model->pncModel_.nv);
            drift_.setZero(1);
            lambda_.setZero(1);

            // Compute the current distance and use it as reference
            vector3_t const deltaPosition =
                model->pncData_.oMf[framesIdx_[0]].translation() -
                model->pncData_.oMf[framesIdx_[1]].translation();
            distanceRef_ = deltaPosition.norm();
        }

        return returnCode;
    }

    hresult_t DistanceConstraint::computeJacobianAndDrift(vectorN_t const & /* q */,
                                                          vectorN_t const & /* v */)
    {
        if (!isAttached_)
        {
            PRINT_ERROR("Constraint not attached to a model.");
            return hresult_t::ERROR_GENERIC;
        }

        // Assuming model still exists.
        auto model = model_.lock();

        // Compute direction between frames
        vector3_t const deltaPosition = model->pncData_.oMf[framesIdx_[0]].translation()
                                      - model->pncData_.oMf[framesIdx_[1]].translation();
        float64_t const deltaPositionNorm = deltaPosition.norm();
        vector3_t const direction = deltaPosition / deltaPositionNorm;

        // Compute relative velocity between frames
        pinocchio::Motion const velocity0 = getFrameVelocity(model->pncModel_,
                                                             model->pncData_,
                                                             framesIdx_[0],
                                                             pinocchio::LOCAL_WORLD_ALIGNED);
        pinocchio::Motion const velocity1 = getFrameVelocity(model->pncModel_,
                                                             model->pncData_,
                                                             framesIdx_[1],
                                                             pinocchio::LOCAL_WORLD_ALIGNED);
        vector3_t deltaVelocity = velocity0.linear() - velocity1.linear();

        // Get jacobian in local frame: J_1 - J_2
        getFrameJacobian(model->pncModel_,
                         model->pncData_,
                         framesIdx_[0],
                         pinocchio::LOCAL_WORLD_ALIGNED,
                         firstFrameJacobian_);
        getFrameJacobian(model->pncModel_,
                         model->pncData_,
                         framesIdx_[1],
                         pinocchio::LOCAL_WORLD_ALIGNED,
                         secondFrameJacobian_);
        jacobian_ = direction.transpose() * (
            firstFrameJacobian_.topRows<3>() - secondFrameJacobian_.topRows<3>());

        // Get drift in local frame
        pinocchio::Motion accel0 = getFrameAcceleration(model->pncModel_,
                                                        model->pncData_,
                                                        framesIdx_[0],
                                                        pinocchio::LOCAL_WORLD_ALIGNED);
        accel0.linear() += velocity0.angular().cross(velocity0.linear());
        pinocchio::Motion accel1 = getFrameAcceleration(model->pncModel_,
                                                        model->pncData_,
                                                        framesIdx_[1],
                                                        pinocchio::LOCAL_WORLD_ALIGNED);
        accel1.linear() += velocity1.angular().cross(velocity1.linear());
        drift_[0] = direction.dot(accel0.linear() - accel1.linear());

        // dDir.T * (dp_A - dp_B) = [(dp_A - dp_B) ** 2 - (dir.T * (dp_A - dp_B)) ** 2] / norm(p_A - p_B)
        float64_t const deltaVelocityProj = deltaVelocity.dot(direction);
        drift_[0] += (deltaVelocity.squaredNorm() - std::pow(deltaVelocityProj, 2)) / deltaPositionNorm;

        // Add Baumgarte stabilization drift
        drift_[0] += kp_ * (deltaPositionNorm - distanceRef_) + kd_ * deltaVelocityProj;

        return hresult_t::SUCCESS;
    }
}
