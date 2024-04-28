#ifndef JIMINY_SPHERE_CONSTRAINT_H
#define JIMINY_SPHERE_CONSTRAINT_H

#include <memory>

#include "jiminy/core/fwd.h"
#include "jiminy/core/utilities/helpers.h"
#include "jiminy/core/constraints/abstract_constraint.h"


namespace jiminy
{
    class Model;

    class SphereConstraint;
#if defined EXPORT_SYMBOLS || (!defined _WIN32 && !defined __CYGWIN__)
    template<>
    const std::string JIMINY_DLLAPI AbstractConstraintTpl<SphereConstraint>::type_;
#endif
    template class JIMINY_TEMPLATE_INSTANTIATION_DLLAPI AbstractConstraintTpl<SphereConstraint>;

    /// \brief Class constraining a sphere to roll without slipping on a flat plane.
    ///
    /// \details Given a frame to represent the sphere center, this class constrains it to move
    ///          like it were rolling without slipping on a flat (not necessarily level) surface.
    class JIMINY_DLLAPI SphereConstraint : public AbstractConstraintTpl<SphereConstraint>
    {
    public:
        JIMINY_DISABLE_COPY(SphereConstraint)

    public:
        /// \param[in] frameName Name of the frame representing the center of the sphere.
        /// \param[in] sphereRadius Radius of the sphere (in m).
        /// \param[in] groundNormal Normal to the ground in the world as a unit vector.
        SphereConstraint(const std::string & frameName,
                         double sphereRadius,
                         const Eigen::Vector3d & groundNormal = Eigen::Vector3d::UnitZ()) noexcept;
        virtual ~SphereConstraint() = default;

        const std::string & getFrameName() const noexcept;
        pinocchio::FrameIndex getFrameIndex() const noexcept;

        double getRadius() const noexcept;
        const Eigen::Vector3d & getNormal() const noexcept;

        void setReferenceTransform(const pinocchio::SE3 & transformRef) noexcept;
        const pinocchio::SE3 & getReferenceTransform() const noexcept;

        virtual void reset(const Eigen::VectorXd & /* q */,
                           const Eigen::VectorXd & /* v */) override final;

        virtual void computeJacobianAndDrift(const Eigen::VectorXd & q,
                                             const Eigen::VectorXd & v) override final;

    private:
        /// \brief Name of the frame on which the constraint operates.
        std::string frameName_;
        /// \brief Corresponding frame index.
        pinocchio::FrameIndex frameIndex_{0};
        /// \brief Sphere radius.
        double radius_;
        /// \brief Ground normal, world frame.
        Eigen::Vector3d normal_;
        /// \brief Skew of ground normal, in world frame, scaled by radius.
        Eigen::Matrix3d skewRadius_{pinocchio::alphaSkew(radius_, normal_)};
        /// \brief Reference pose of the frame to enforce.
        pinocchio::SE3 transformRef_{};
        /// \brief Stores full frame jacobian in world.
        Matrix6Xd frameJacobian_{};
    };
}

#endif  // end of JIMINY_SPHERE_CONSTRAINT_H
