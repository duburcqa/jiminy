#ifndef JIMINY_DISTANCE_CONSTRAINT_H
#define JIMINY_DISTANCE_CONSTRAINT_H

#include <memory>

#include "jiminy/core/fwd.h"
#include "jiminy/core/utilities/helpers.h"
#include "jiminy/core/constraints/abstract_constraint.h"


namespace jiminy
{
    class Model;

    class JIMINY_DLLAPI DistanceConstraint : public AbstractConstraintTpl<DistanceConstraint>
    {
    public:
        DISABLE_COPY(DistanceConstraint)

        auto shared_from_this() { return shared_from(this); }

    public:
        explicit DistanceConstraint(const std::string & firstFrameName,
                                    const std::string & secondFrameName) noexcept;
        virtual ~DistanceConstraint() = default;

        const std::array<std::string, 2> & getFramesNames() const noexcept;
        const std::array<pinocchio::FrameIndex, 2> & getFramesIdx() const noexcept;

        hresult_t setReferenceDistance(double distanceRef);
        double getReferenceDistance() const noexcept;

        virtual hresult_t reset(const Eigen::VectorXd & q,
                                const Eigen::VectorXd & v) override final;

        virtual hresult_t computeJacobianAndDrift(const Eigen::VectorXd & q,
                                                  const Eigen::VectorXd & v) override final;

    private:
        /// \brief Names of the frames on which the constraint operates.
        std::array<std::string, 2> frameNames_;
        /// \brief Corresponding frames indices.
        std::array<pinocchio::FrameIndex, 2> frameIndices_{{0, 0}};
        /// \brief Reference Distance between the frames
        double distanceRef_{0.0};
        /// \brief Stores frame jacobians in world.
        std::array<Matrix6Xd, 2> frameJacobians_{};
    };
}

#endif  // end of JIMINY_TRANSMISSION_CONSTRAINT_H
