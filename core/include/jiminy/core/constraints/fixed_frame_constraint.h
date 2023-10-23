#ifndef JIMINY_FIXED_FRAME_CONSTRAINT_H
#define JIMINY_FIXED_FRAME_CONSTRAINT_H

#include <memory>

#include "jiminy/core/types.h"
#include "jiminy/core/constraints/abstract_constraint.h"


namespace jiminy
{
    class Model;

    /// \brief This class implements the constraint for fixing a given frame wrt world.
    class FixedFrameConstraint : public AbstractConstraintTpl<FixedFrameConstraint>
    {
    public:
        DISABLE_COPY(FixedFrameConstraint)

        auto shared_from_this() { return shared_from(this); }

    public:
        /// \param[in] frameName Name of the frame on which the constraint is to be applied.
        FixedFrameConstraint(const std::string & frameName,
                             const Eigen::Matrix<bool_t, 6, 1> & maskFixed =
                                 Eigen::Matrix<bool_t, 6, 1>::Constant(true));
        virtual ~FixedFrameConstraint() = default;

        const std::string & getFrameName() const;
        const frameIndex_t & getFrameIdx() const;

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
        frameIndex_t frameIdx_;
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
}

#endif  // end of JIMINY_ABSTRACT_MOTOR_H
