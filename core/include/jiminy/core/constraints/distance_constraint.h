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
        DistanceConstraint(const std::string & firstFrameName,
                           const std::string & secondFrameName);
        virtual ~DistanceConstraint() = default;

        const std::vector<std::string> & getFramesNames() const;
        const std::vector<pinocchio::FrameIndex> & getFramesIdx() const;

        hresult_t setReferenceDistance(float64_t distanceRef);
        float64_t getReferenceDistance() const;

        virtual hresult_t reset(const Eigen::VectorXd & q,
                                const Eigen::VectorXd & v) override final;

        virtual hresult_t computeJacobianAndDrift(const Eigen::VectorXd & q,
                                                  const Eigen::VectorXd & v) override final;

    private:
        /// \brief Names of the frames on which the constraint operates.
        std::vector<std::string> framesNames_;
        /// \brief Corresponding frames indices.
        std::vector<pinocchio::FrameIndex> framesIdx_;
        /// \brief Reference Distance between the frames
        float64_t distanceRef_;
        /// \brief Stores first frame jacobian in world.
        Matrix6Xd firstFrameJacobian_;
        /// \brief Stores second frame jacobian in world.
        Matrix6Xd secondFrameJacobian_;
    };
}

#endif  // end of JIMINY_TRANSMISSION_CONSTRAINT_H
