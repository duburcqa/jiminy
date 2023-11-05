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
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    public:
        /// Forbid the copy of the class
        FixedFrameConstraint(const FixedFrameConstraint & abstractConstraint) = delete;
        FixedFrameConstraint & operator=(const FixedFrameConstraint & other) = delete;

        auto shared_from_this() { return shared_from(this); }

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

        void setNormal(const vector3_t & normal);
        const matrix3_t & getLocalFrame() const;

        virtual hresult_t reset(const vectorN_t & q, const vectorN_t & v) override final;

        virtual hresult_t computeJacobianAndDrift(const vectorN_t & q,
                                                  const vectorN_t & v) override final;

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
        vector3_t normal_;
        /// \brief Rotation matrix of the local frame in which to apply masking
        matrix3_t rotationLocal_;
        /// \brief Stores full frame jacobian in reference frame.
        matrix6N_t frameJacobian_;
        /// \brief Stores full frame drift in reference frame.
        pinocchio::Motion frameDrift_;
    };
}

#endif  // end of JIMINY_ABSTRACT_MOTOR_H
