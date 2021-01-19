///////////////////////////////////////////////////////////////////////////////////////////////
///
/// \brief      Class representing a fixed frame constraint.
///
/// \details    This class  implements the constraint to have a specified frame fixed (in the world frame).
///
///////////////////////////////////////////////////////////////////////////////////////////////

#ifndef JIMINY_FIXED_FRAME_CONSTRAINT_H
#define JIMINY_FIXED_FRAME_CONSTRAINT_H

#include <memory>

#include "jiminy/core/Types.h"
#include "jiminy/core/robot/AbstractConstraint.h"


namespace jiminy
{
    class Model;

    class FixedFrameConstraint: public AbstractConstraint
    {

    public:
        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Forbid the copy of the class
        ///////////////////////////////////////////////////////////////////////////////////////////////
        FixedFrameConstraint(FixedFrameConstraint const & abstractMotor) = delete;
        FixedFrameConstraint & operator = (FixedFrameConstraint const & other) = delete;

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Constructor
        ///
        /// \param[in]  frameName   Name of the frame on which the constraint is to be applied.
        ///////////////////////////////////////////////////////////////////////////////////////////////
        FixedFrameConstraint(std::string const & frameName);
        virtual ~FixedFrameConstraint(void);

        virtual hresult_t reset(void) override final;

        virtual hresult_t computeJacobianAndDrift(vectorN_t const & q,
                                                  vectorN_t const & v) override final;

    private:
        std::string const frameName_;     ///< Name of the frame on which the constraint operates.
        int32_t frameIdx_;                ///< Corresponding frame index.
    };
}

#endif //end of JIMINY_ABSTRACT_MOTOR_H
