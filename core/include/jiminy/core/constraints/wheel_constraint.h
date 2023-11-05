#ifndef JIMINY_WHEEL_CONSTRAINT_H
#define JIMINY_WHEEL_CONSTRAINT_H

#include <memory>

#include "jiminy/core/types.h"
#include "jiminy/core/constraints/abstract_constraint.h"


namespace jiminy
{
    class Model;

    /// \brief Class constraining a wheel to roll without slipping on a flat plane.
    ///
    /// \details Given a frame to represent the wheel center, this class constrains it to move
    ///          like it were rolling without slipping on a flat (not necessarily level) surface.
    class WheelConstraint : public AbstractConstraintTpl<WheelConstraint>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    public:
        /// Forbid the copy of the class
        WheelConstraint(const WheelConstraint & abstractConstraint) = delete;
        WheelConstraint & operator=(const WheelConstraint & other) = delete;

        auto shared_from_this() { return shared_from(this); }

        /// \param[in] frameName Name of the frame representing the center of the wheel.
        /// \param[in] wheelRadius Radius of the wheel (in m).
        /// \param[in] groundNormal Normal to the ground in world frame as a unit vector.
        /// \param[in] wheelAxis Axis of the wheel, in the local frame.
        WheelConstraint(const std::string & frameName,
                        const float64_t & wheelRadius,
                        const vector3_t & groundNormal,
                        const vector3_t & wheelAxis);
        virtual ~WheelConstraint() = default;

        const std::string & getFrameName() const;
        const frameIndex_t & getFrameIdx() const;

        void setReferenceTransform(const pinocchio::SE3 & transformRef);
        const pinocchio::SE3 & getReferenceTransform() const;

        virtual hresult_t reset(const vectorN_t & /* q */,
                                const vectorN_t & /* v */) override final;

        virtual hresult_t computeJacobianAndDrift(const vectorN_t & q,
                                                  const vectorN_t & v) override final;

    private:
        /// \brief Name of the frame on which the constraint operates.
        std::string frameName_;
        /// \brief Corresponding frame index.
        frameIndex_t frameIdx_;
        /// \brief Wheel radius.
        float64_t radius_;
        /// \brief Ground normal, world frame.
        vector3_t normal_;
        /// \brief Wheel axis, local frame.
        vector3_t axis_;
        /// \brief Skew matrix of wheel axis, in world frame, scaled by radius.
        matrix3_t skewRadius_;
        /// \brief Derivative of skew matrix of wheel axis, in world frame, scaled by radius.
        matrix3_t dskewRadius_;
        /// \brief Reference pose of the frame to enforce.
        pinocchio::SE3 transformRef_;
        /// \brief Stores full frame jacobian in world.
        matrix6N_t frameJacobian_;
    };
}

#endif  // end of JIMINY_WHEEL_CONSTRAINT_H
