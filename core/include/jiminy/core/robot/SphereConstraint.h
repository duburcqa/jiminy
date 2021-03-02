///////////////////////////////////////////////////////////////////////////////////////////////
///
/// \brief      Class constraining a sphere to roll without slipping on a flat plane.
///
/// \details    Given a frame to represent the sphere center, this class constrains it to move
///             like it were rolling without slipping on a flat (not necessarily level) surface.
///
///////////////////////////////////////////////////////////////////////////////////////////////

#ifndef JIMINY_SPHERE_CONSTRAINT_H
#define JIMINY_SPHERE_CONSTRAINT_H

#include <memory>

#include "jiminy/core/Types.h"
#include "jiminy/core/robot/AbstractConstraint.h"


namespace jiminy
{
    class Model;

    class SphereConstraint: public AbstractConstraintTpl<SphereConstraint>
    {

    public:
        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Forbid the copy of the class
        ///////////////////////////////////////////////////////////////////////////////////////////////
        SphereConstraint(SphereConstraint const & abstractConstraint) = delete;
        SphereConstraint & operator = (SphereConstraint const & other) = delete;

        auto shared_from_this() { return shared_from(this); }

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Constructor
        ///
        /// \param[in]  frameName     Name of the frame representing the center of the sphere.
        /// \param[in]  sphereRadius  Radius of the sphere (in m).
        /// \param[in]  groundNormal  Unit vector representing the normal to the ground, in the world frame.
        ///////////////////////////////////////////////////////////////////////////////////////////////
        SphereConstraint(std::string const & frameName,
                         float64_t   const & sphereRadius,
                         vector3_t   const & groundNormal = (vector3_t() << 0.0, 0.0, 1.0).finished());
        virtual ~SphereConstraint(void);

        std::string const & getFrameName(void) const;
        int32_t const & getFrameIdx(void) const;

        void setReferenceTransform(pinocchio::SE3 const & transformRef);
        pinocchio::SE3 & getReferenceTransform(void);

        virtual hresult_t reset(vectorN_t const & /* q */,
                                vectorN_t const & /* v */) override final;

        virtual hresult_t computeJacobianAndDrift(vectorN_t const & q,
                                                  vectorN_t const & v) override final;

    private:
        std::string frameName_;        ///< Name of the frame on which the constraint operates.
        int32_t frameIdx_;             ///< Corresponding frame index.
        float64_t radius_;             ///< Sphere radius.
        vector3_t normal_;             ///< Ground normal, world frame.
        matrix3_t shewRadius_;         ///< Skew of ground normal, in world frame, scaled by radius.
        pinocchio::SE3 transformRef_;  ///< Reference pose of the frame to enforce.
        matrixN_t frameJacobian_;      ///< Stores full frame jacobian in world.
    };
}

#endif //end of JIMINY_SPHERE_CONSTRAINT_H
