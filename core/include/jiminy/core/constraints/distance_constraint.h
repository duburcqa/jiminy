#ifndef JIMINY_DISTANCE_CONSTRAINT_H
#define JIMINY_DISTANCE_CONSTRAINT_H

#include <memory>

#include "jiminy/core/fwd.h"
#include "jiminy/core/utilities/helpers.h"
#include "jiminy/core/constraints/abstract_constraint.h"


namespace jiminy
{
    class Model;

    class DistanceConstraint;
#if defined EXPORT_SYMBOLS || (!defined _WIN32 && !defined __CYGWIN__)
    template<>
    const std::string JIMINY_DLLAPI AbstractConstraintTpl<DistanceConstraint>::type_;
#endif
    template class JIMINY_TEMPLATE_INSTANTIATION_DLLAPI AbstractConstraintTpl<DistanceConstraint>;

    class JIMINY_DLLAPI DistanceConstraint : public AbstractConstraintTpl<DistanceConstraint>
    {
    public:
        JIMINY_DISABLE_COPY(DistanceConstraint)

    public:
        explicit DistanceConstraint(const std::string & firstFrameName,
                                    const std::string & secondFrameName) noexcept;
        virtual ~DistanceConstraint() = default;

        const std::array<std::string, 2> & getFrameNames() const noexcept;
        const std::array<pinocchio::FrameIndex, 2> & getFrameIndices() const noexcept;

        void setReferenceDistance(double distanceRef);
        double getReferenceDistance() const noexcept;

        virtual void reset(const Eigen::VectorXd & q, const Eigen::VectorXd & v) override final;

        virtual void computeJacobianAndDrift(const Eigen::VectorXd & q,
                                             const Eigen::VectorXd & v) override final;

    private:
        /// \brief Names of the frames on which the constraint operates.
        std::array<std::string, 2> frameNames_;
        /// \brief Corresponding frames indices.
        std::array<pinocchio::FrameIndex, 2> frameIndices_{0, 0};
        /// \brief Reference Distance between the frames
        double distanceRef_{0.0};
        /// \brief Stores frame jacobians in world.
        std::array<Matrix6Xd, 2> frameJacobians_{};
    };
}

#endif  // end of JIMINY_TRANSMISSION_CONSTRAINT_H
