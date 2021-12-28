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
                             Eigen::Matrix<bool_t, 6, 1> const & maskFixed = Eigen::Matrix<bool_t, 6, 1>::Constant(true));
        virtual ~FixedFrameConstraint(void);

        std::string const & getFrameName(void) const;
        frameIndex_t const & getFrameIdx(void) const;

        std::vector<uint32_t> const & getDofsFixed(void) const;

        void setReferenceTransform(pinocchio::SE3 const & transformRef);
        pinocchio::SE3 const & getReferenceTransform(void) const;

        void setNormal(vector3_t const & normal);
        matrix3_t const & getLocalFrame(void) const;

        virtual hresult_t reset(vectorN_t const & q,
                                vectorN_t const & v) override final;

        virtual hresult_t computeJacobianAndDrift(vectorN_t const & q,
                                                  vectorN_t const & v) override final;

    private:
        std::string const frameName_;                       ///< Name of the frame on which the constraint operates.
        frameIndex_t frameIdx_;                             ///< Corresponding frame index.
        std::vector<uint32_t> dofsFixed_;                   ///< Degrees of freedom to fix.
        bool_t isFixedPositionXY_;                          ///< Whether or not the frame is fixed for both X and Y translations
        pinocchio::SE3 transformRef_;                       ///< Reference pose of the frame to enforce.
        vector3_t normal_;                                  ///< Normal direction locally at the interface.
        matrix3_t rotationLocal_;                           ///< Rotation matrix of the local frame in which to apply masking
        matrix6N_t frameJacobian_;                          ///< Stores full frame jacobian in reference frame.
        pinocchio::Motion frameDrift_;                      ///< Stores full frame drift in reference frame.
        Eigen::Matrix<float64_t, Eigen::Dynamic, 2> UiJt_;  ///< Used to store intermediary computation to compute diag(J.Minv.Jt)_t
    };
}

#endif //end of JIMINY_ABSTRACT_MOTOR_H
