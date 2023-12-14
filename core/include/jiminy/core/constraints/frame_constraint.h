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
        FrameConstraint(
            const std::string & frameName,
            const std::array<bool_t, 6> & maskDoFs = {{true, true, true, true, true, true}});
        virtual ~FrameConstraint() = default;

        const std::string & getFrameName() const;
        pinocchio::FrameIndex getFrameIdx() const;

        const std::vector<uint32_t> & getDofsFixed() const;

        void setReferenceTransform(const pinocchio::SE3 & transformRef);
        const pinocchio::SE3 & getReferenceTransform() const;

        void setNormal(const Eigen::Vector3d & normal);
        const Eigen::Matrix3d & getLocalFrame() const;

        virtual hresult_t reset(const Eigen::VectorXd & q,
                                const Eigen::VectorXd & v) override final;

        virtual hresult_t computeJacobianAndDrift(const Eigen::VectorXd & q,
                                                  const Eigen::VectorXd & v) override final;

    private:
        /// \brief Name of the frame on which the constraint operates.
        const std::string frameName_;
        /// \brief Corresponding frame index.
        pinocchio::FrameIndex frameIdx_;
        /// \brief Degrees of freedom to fix.
        std::vector<uint32_t> dofsFixed_;
        /// \brief Reference pose of the frame to enforce.
        pinocchio::SE3 transformRef_;
        /// \brief Normal direction locally at the interface.
        Eigen::Vector3d normal_;
        /// \brief Rotation matrix of the local frame in which to apply masking
        Eigen::Matrix3d rotationLocal_;
        /// \brief Stores full frame jacobian in reference frame.
        Matrix6Xd frameJacobian_;
        /// \brief Stores full frame drift in reference frame.
        pinocchio::Motion frameDrift_;
    };
}  // namespace jiminy

#endif  // end of JIMINY_ABSTRACT_MOTOR_H
