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

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief    Compute and return the jacobian of the constraint.
        ///
        /// \note     To avoid duplicate kinematic computation, it is assumed that
        ///           computeJointJacobians and framesForwardKinematics has already
        ///           been called on model->pncModel_.
        ///
        /// \param[in] q    Current joint position.
        /// \return         Jacobian of the constraint.
        ///////////////////////////////////////////////////////////////////////////////////////////////
        virtual matrixN_t const & getJacobian(Eigen::Ref<vectorN_t const> const & q) override final;

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief    Compute and return the drift of the constraint.
        ///
        /// \note     To avoid duplicate kinematic computation, it is assumed that forward kinematics
        ///           and jacobian computation has already been done on model->pncModel_.
        ///
        /// \param[in] q    Current joint position.
        /// \param[in] v    Current joint velocity.
        /// \return         Drift of the constraint.
        ///////////////////////////////////////////////////////////////////////////////////////////////
        virtual vectorN_t const & getDrift(Eigen::Ref<vectorN_t const>  const & q,
                                           Eigen::Ref<vectorN_t const>  const & v) override final;

    protected:
        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Link the constraint on the given model, and initialize it.
        ///
        /// \param[in] model    Model on which to apply the constraint.
        /// \return     Error code: attach may fail if:
        ///              - the constraint is already attached.
        ///              - the target frame name does not exist in model.
        ///////////////////////////////////////////////////////////////////////////////////////////////
        virtual hresult_t attach(Model const * model) override final;

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief    Refresh the proxies.
        ///
        /// \remark   This method is not intended to be called manually. The Robot to which the
        ///           motor is added is taking care of it when its own `refresh` method is called.
        ///////////////////////////////////////////////////////////////////////////////////////////////
        virtual hresult_t refreshProxies(void) override final;

    private:
        std::string frameName_;     ///< Name of the frame on which the constraint operates.
        int frameIdx_;              ///< Corresponding frame index.
    };
}

#endif //end of JIMINY_ABSTRACT_MOTOR_H
