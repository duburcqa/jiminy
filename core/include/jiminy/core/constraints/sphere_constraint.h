#ifndef JIMINY_SPHERE_CONSTRAINT_H
#define JIMINY_SPHERE_CONSTRAINT_H

#include <memory>

#include "jiminy/core/types.h"
#include "jiminy/core/constraints/abstract_constraint.h"


namespace jiminy
{
    class Model;

    /// \brief Class constraining a sphere to roll without slipping on a flat plane.
    ///
    /// \details Given a frame to represent the sphere center, this class constrains it to move
    ///          like it were rolling without slipping on a flat (not necessarily level) surface.
    class SphereConstraint : public AbstractConstraintTpl<SphereConstraint>
    {
    public:
        DISABLE_COPY(SphereConstraint)

        auto shared_from_this() { return shared_from(this); }

    public:
        /// \param[in] frameName Name of the frame representing the center of the sphere.
        /// \param[in] sphereRadius Radius of the sphere (in m).
        /// \param[in] groundNormal Normal to the ground in the world as a unit vector.
        SphereConstraint(const std::string & frameName,
                         const float64_t & sphereRadius,
                         const vector3_t & groundNormal = vector3_t::UnitZ());
        virtual ~SphereConstraint(void) = default;

        const std::string & getFrameName(void) const;
        const frameIndex_t & getFrameIdx(void) const;

        void setReferenceTransform(const pinocchio::SE3 & transformRef);
        const pinocchio::SE3 & getReferenceTransform(void) const;

        virtual hresult_t reset(const vectorN_t & /* q */,
                                const vectorN_t & /* v */) override final;

        virtual hresult_t computeJacobianAndDrift(const vectorN_t & q,
                                                  const vectorN_t & v) override final;

    private:
        /// \brief Name of the frame on which the constraint operates.
        std::string frameName_;
        /// \brief Corresponding frame index.
        frameIndex_t frameIdx_;
        /// \brief Sphere radius.
        float64_t radius_;
        /// \brief Ground normal, world frame.
        vector3_t normal_;
        /// \brief Skew of ground normal, in world frame, scaled by radius.
        matrix3_t skewRadius_;
        /// \brief Reference pose of the frame to enforce.
        pinocchio::SE3 transformRef_;
        /// \brief Stores full frame jacobian in world.
        matrix6N_t frameJacobian_;
    };
}

#endif  // end of JIMINY_SPHERE_CONSTRAINT_H
