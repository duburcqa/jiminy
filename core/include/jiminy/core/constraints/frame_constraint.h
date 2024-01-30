#ifndef JIMINY_FIXED_FRAME_CONSTRAINT_H
#define JIMINY_FIXED_FRAME_CONSTRAINT_H

#include <memory>

#include "jiminy/core/constraints/abstract_constraint.h"
#include "jiminy/core/fwd.h"
#include "jiminy/core/utilities/helpers.h"

namespace jiminy
{
    class Model;

    /// \brief This class implements the constraint for fixing a given frame wrt
    /// world.
    class JIMINY_DLLAPI FrameConstraint : public AbstractConstraintTpl<FrameConstraint>
    {
    public:
        DISABLE_COPY(FrameConstraint)

        auto shared_from_this() { return shared_from(this); }

    public:
        /// \param[in] frameName Name of the frame on which the constraint is to be
        /// applied.
        explicit FrameConstraint(const std::string & frameName,
                                 const std::array<bool, 6> & maskDoFs = {
                                     {true, true, true, true, true, true}}) noexcept;
        virtual ~FrameConstraint() = default;

        const std::string & getFrameName() const noexcept;
        pinocchio::FrameIndex getFrameIdx() const noexcept;

        const std::vector<uint32_t> & getDofsFixed() const noexcept;

        void setReferenceTransform(const pinocchio::SE3 & transformRef) noexcept;
        const pinocchio::SE3 & getReferenceTransform() const noexcept;

        void setNormal(const Eigen::Vector3d & normal) noexcept;
        const Eigen::Matrix3d & getLocalFrame() const noexcept;

        virtual hresult_t reset(const Eigen::VectorXd & q,
                                const Eigen::VectorXd & v) override final;

        virtual hresult_t computeJacobianAndDrift(const Eigen::VectorXd & q,
                                                  const Eigen::VectorXd & v) override final;

    private:
        /// \brief Name of the frame on which the constraint operates.
        const std::string frameName_;
        /// \brief Corresponding frame index.
        pinocchio::FrameIndex frameIdx_{0};
        /// \brief Degrees of freedom to fix.
        std::vector<uint32_t> dofsFixed_;
        /// \brief Reference pose of the frame to enforce.
        pinocchio::SE3 transformRef_{};
        /// \brief Normal direction locally at the interface.
        Eigen::Vector3d normal_{};
        /// \brief Rotation matrix of the local frame in which to apply masking
        Eigen::Matrix3d rotationLocal_{Eigen::Matrix3d::Identity()};
        /// \brief Stores full frame jacobian in reference frame.
        Matrix6Xd frameJacobian_{};
        /// \brief Stores full frame drift in reference frame.
        pinocchio::Motion frameDrift_{};
    };
}

#endif  // end of JIMINY_ABSTRACT_MOTOR_H
