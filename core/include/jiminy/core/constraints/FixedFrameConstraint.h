///////////////////////////////////////////////////////////////////////////////////////////////
///
/// \brief      Class representing a fixed frame constraint.
///
/// \details    This class  implements the constraint to have a specified frame fixed (in the world frame).
///
///////////////////////////////////////////////////////////////////////////////////////////////

#ifndef JIMINY_FIXED_FRAME_CONSTRAINT_H
#define JIMINY_FIXED_FRAME_CONSTRAINT_H

#include <memory>

#include "jiminy/core/Types.h"
#include "jiminy/core/constraints/AbstractConstraint.h"


namespace jiminy
{
    class Model;

    class FixedFrameConstraint : public AbstractConstraintTpl<FixedFrameConstraint>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    public:
        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Forbid the copy of the class
        ///////////////////////////////////////////////////////////////////////////////////////////////
        FixedFrameConstraint(FixedFrameConstraint const & abstractConstraint) = delete;
        FixedFrameConstraint & operator = (FixedFrameConstraint const & other) = delete;

        auto shared_from_this() { return shared_from(this); }

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Constructor
        ///
        /// \param[in]  frameName   Name of the frame on which the constraint is to be applied.
        ///////////////////////////////////////////////////////////////////////////////////////////////
        FixedFrameConstraint(std::string const & frameName,
                             Eigen::Matrix<bool_t, 6, 1> const & maskFixed = Eigen::Matrix<bool_t, 6, 1>::Constant(true),
                             pinocchio::ReferenceFrame const & frameRef = pinocchio::LOCAL_WORLD_ALIGNED);
        virtual ~FixedFrameConstraint(void);

        std::string const & getFrameName(void) const;
        frameIndex_t const & getFrameIdx(void) const;

        std::vector<uint32_t> const & getDofsFixed(void) const;

        pinocchio::ReferenceFrame const & getReferenceFrame(void) const;

        void setReferenceTransform(pinocchio::SE3 const & transformRef);
        pinocchio::SE3 & getReferenceTransform(void);

        virtual hresult_t reset(vectorN_t const & q,
                                vectorN_t const & v) override final;

        virtual hresult_t computeJacobianAndDrift(vectorN_t const & q,
                                                  vectorN_t const & v) override final;

    private:
        std::string const frameName_;         ///< Name of the frame on which the constraint operates.
        frameIndex_t frameIdx_;               ///< Corresponding frame index.
        pinocchio::ReferenceFrame frameRef_;  ///< Reference frame.
        std::vector<uint32_t> dofsFixed_;     ///< Degrees of freedom to fix.
        pinocchio::SE3 transformRef_;         ///< Reference pose of the frame to enforce.
        matrixN_t frameJacobian_;             ///< Stores full frame jacobian in reference frame.
        vector6_t frameDrift_;                ///< Stores full frame drift in reference frame.
    };
}

#endif //end of JIMINY_ABSTRACT_MOTOR_H
