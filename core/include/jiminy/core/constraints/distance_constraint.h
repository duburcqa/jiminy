#ifndef JIMINY_DISTANCE_CONSTRAINT_H
#define JIMINY_DISTANCE_CONSTRAINT_H

#include <memory>

#include "jiminy/core/types.h"
#include "jiminy/core/constraints/abstract_constraint.h"


namespace jiminy
{
    class Model;

    class DistanceConstraint : public AbstractConstraintTpl<DistanceConstraint>
    {
    public:
        DISABLE_COPY(DistanceConstraint)

        auto shared_from_this() { return shared_from(this); }

    public:
        DistanceConstraint(const std::string & firstFrameName,
                           const std::string & secondFrameName);
        virtual ~DistanceConstraint(void) = default;

        const std::vector<std::string> & getFramesNames(void) const;
        const std::vector<frameIndex_t> & getFramesIdx(void) const;

        hresult_t setReferenceDistance(const float64_t & distanceRef);
        const float64_t & getReferenceDistance(void) const;

        virtual hresult_t reset(const vectorN_t & q, const vectorN_t & v) override final;

        virtual hresult_t computeJacobianAndDrift(const vectorN_t & q,
                                                  const vectorN_t & v) override final;

    private:
        /// \brief Names of the frames on which the constraint operates.
        std::vector<std::string> framesNames_;
        /// \brief Corresponding frames indices.
        std::vector<frameIndex_t> framesIdx_;
        /// \brief Reference Distance between the frames
        float64_t distanceRef_;
        /// \brief Stores first frame jacobian in world.
        matrix6N_t firstFrameJacobian_;
        /// \brief Stores second frame jacobian in world.
        matrix6N_t secondFrameJacobian_;
    };
}

#endif  // end of JIMINY_TRANSMISSION_CONSTRAINT_H
