///////////////////////////////////////////////////////////////////////////////////////////////
///
/// \brief      Generic interface for a kinematic constraint.
///
/// \details    This class enables the support of kinematics constraints on the system.
///             The most common constraint is to say that a given frame is fixed: this
///             is implemented in FixedFrameConstraint
///
/// \remarks    Each constraint applied to a system is downcasted as an instance of
///             AbstractMotorBase and polymorphism is used to call the actual implementations.
///
///////////////////////////////////////////////////////////////////////////////////////////////

#ifndef JIMINY_ABSTRACT_CONSTRAINT_H
#define JIMINY_ABSTRACT_CONSTRAINT_H

#include <memory>

#include "jiminy/core/Types.h"


namespace jiminy
{
    class Model;

    class AbstractConstraint: public std::enable_shared_from_this<AbstractConstraint>
    {

    public:
        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Forbid the copy of the class
        ///////////////////////////////////////////////////////////////////////////////////////////////
        AbstractConstraint(AbstractConstraint const & abstractMotor) = delete;
        AbstractConstraint & operator = (AbstractConstraint const & other) = delete;

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Constructor
        ///////////////////////////////////////////////////////////////////////////////////////////////
        AbstractConstraint();
        virtual ~AbstractConstraint(void);

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Initialize the constraint on the given model.
        ///
        /// \note       This function is called internally when adding a constraint to a robot:
        ///             there is no need to call it manually.
        /// \param[in] model    Model on which to apply the constraint.
        ///////////////////////////////////////////////////////////////////////////////////////////////
        virtual hresult_t initialize(Model *model);

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief    Refresh the proxies.
        ///
        /// \remark   This method is not intended to be called manually. The Robot to which the
        ///           motor is added is taking care of it when its own `refresh` method is called.
        ///////////////////////////////////////////////////////////////////////////////////////////////
        virtual hresult_t refreshProxies(void);

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
        virtual matrixN_t getJacobian(vectorN_t const & q) const;

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
        virtual vectorN_t getDrift(vectorN_t const & q,
                                   vectorN_t const & v) const;

    protected:
        Model * model_; ///< Model on which the constraint operates.
        bool isInitialized_; ///< Flag to indicate if the constraint has been initialized.
    };
}

#endif //end of JIMINY_ABSTRACT_MOTOR_H
