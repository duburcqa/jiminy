///////////////////////////////////////////////////////////////////////////////////////////////
///
/// \brief      Class constraining a wheel to roll without slipping on a flat plane.
///
/// \details    Given a frame to represent the wheel center, this class constrains it to move
///             like it were rolling without slipping on a flat (not necessarily level) surface.
///
///////////////////////////////////////////////////////////////////////////////////////////////

#ifndef JIMINY_WHEEL_CONSTRAINT_H
#define JIMINY_WHEEL_CONSTRAINT_H

#include <memory>

#include "jiminy/core/Types.h"
#include "jiminy/core/robot/AbstractConstraint.h"


namespace jiminy
{
    class Model;

    class WheelConstraint: public AbstractConstraint
    {

    public:
        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Forbid the copy of the class
        ///////////////////////////////////////////////////////////////////////////////////////////////
        WheelConstraint(WheelConstraint const & abstractMotor) = delete;
        WheelConstraint & operator = (WheelConstraint const & other) = delete;

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Constructor
        ///
        /// \param[in]  frameName   Name of the frame representing the center of the wheel.
        /// \param[in]  wheelRadius Radius of the wheel (in m).
        /// \param[in]  groundNormal Unit vector representing the normal to the ground, in the world frame.
        /// \param[in]  wheelAxis   Axis of the wheel, in the local frame.
        ///////////////////////////////////////////////////////////////////////////////////////////////
        WheelConstraint(std::string const & frameName,
                        float64_t   const & wheelRadius,
                        vector3_t   const & groundNormal,
                        vector3_t   const & wheelAxis);
        virtual ~WheelConstraint(void);

        virtual hresult_t reset(void) override final;

        virtual hresult_t computeJacobianAndDrift(vectorN_t const & q,
                                                  vectorN_t const & v) override final;

    private:
        std::string frameName_;     ///< Name of the frame on which the constraint operates.
        int32_t frameIdx_;          ///< Corresponding frame index.
        float64_t radius_;          ///< Wheel radius.
        vector3_t normal_;          ///< Ground normal, world frame.
        vector3_t axis_;            ///< Wheel axis, local frame.
        matrixN_t frameJacobian_;   ///< Stores full frame jacobian in world.
        matrixN_t jLas_;            ///< Stores full frame jacobian in world.
    };
}

#endif //end of JIMINY_WHEEL_CONSTRAINT_H
