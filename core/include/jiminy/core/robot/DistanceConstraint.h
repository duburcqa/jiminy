#ifndef JIMINY_DISTANCE_CONSTRAINT_H
#define JIMINY_DISTANCE_CONSTRAINT_H

#include <memory>

#include "jiminy/core/Types.h"
#include "jiminy/core/robot/AbstractConstraint.h"


namespace jiminy
{
    class Model;

    class DistanceConstraint: public AbstractConstraintTpl<DistanceConstraint>
    {

    public:
        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Forbid the copy of the class
        ///////////////////////////////////////////////////////////////////////////////////////////////
        DistanceConstraint(DistanceConstraint const & abstractConstraint) = delete;
        DistanceConstraint & operator = (DistanceConstraint const & other) = delete;

        auto shared_from_this() { return shared_from(this); }

        DistanceConstraint(std::string const & firstFrameName,
                           std::string const & secondFrameName,
                           float64_t const & distanceReference);
        virtual ~DistanceConstraint(void);

        std::vector<std::string> const & getFramesNames(void) const;
        std::vector<int32_t> const & getFramesIdx(void) const;

        float64_t const & getReferenceDistance(void) const;

        virtual hresult_t reset(vectorN_t const & q,
                                vectorN_t const & v) override final;

        virtual hresult_t computeJacobianAndDrift(vectorN_t const & q,
                                                  vectorN_t const & v) override final;

    private:
        std::vector<std::string> framesNames_;    ///< Names of the frames on which the constraint operates.
        std::vector<int32_t> framesIdx_;          ///< Corresponding frames indices.
        float64_t distanceRef_;                   ///< Reference Distance between the frames
        matrixN_t firstFrameJacobian_;            ///< Stores first frame jacobian in world.
        matrixN_t secondFrameJacobian_;           ///< Stores second frame jacobian in world.
    };
}

#endif //end of JIMINY_TRANSMISSION_CONSTRAINT_H
