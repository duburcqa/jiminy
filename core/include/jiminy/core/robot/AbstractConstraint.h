///////////////////////////////////////////////////////////////////////////////////////////////
///
/// \brief      Generic interface for a kinematic constraint.
///
/// \details    This class enables the support of kinematics constraints on the system.
///             The most common constraint is to say that a given frame is fixed: this
///             is implemented in FixedFrameConstraint
///
/// \remark     Each constraint applied to a system is downcasted as an instance of
///             AbstractMotorBase and polymorphism is used to call the actual implementations.
///
///////////////////////////////////////////////////////////////////////////////////////////////

#ifndef JIMINY_ABSTRACT_CONSTRAINT_H
#define JIMINY_ABSTRACT_CONSTRAINT_H

#include <memory>

#include "jiminy/core/Macros.h"
#include "jiminy/core/Types.h"


namespace jiminy
{
    class Model;

    class AbstractConstraintBase: public std::enable_shared_from_this<AbstractConstraintBase>
    {
        // See AbstractSensor for comment on this.
        friend Model;

    public:
        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Forbid the copy of the class
        ///////////////////////////////////////////////////////////////////////////////////////////////
        AbstractConstraintBase(AbstractConstraintBase const & abstractMotor) = delete;
        AbstractConstraintBase & operator = (AbstractConstraintBase const & other) = delete;

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Constructor
        ///////////////////////////////////////////////////////////////////////////////////////////////
        AbstractConstraintBase();
        virtual ~AbstractConstraintBase(void);

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief    Refresh the internal buffers and proxies.
        ///
        /// \remark   This method is not intended to be called manually. The Robot to which the
        ///           constraint is added is taking care of it when its own `reset` method is called.
        ///////////////////////////////////////////////////////////////////////////////////////////////
        virtual hresult_t reset(vectorN_t const & q,
                                vectorN_t const & v) = 0;

        void enable(void);
        void disable(void);
        bool_t const & getIsEnabled(void) const;

        hresult_t setBaumgarteFreq(float64_t const & freq);
        /// \brief    Natural frequency of critically damping position/velocity error correction.
        float64_t getBaumgarteFreq(void) const;

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief    Compute the jacobian and drift of the constraint.
        ///
        /// \note     To avoid duplicate kinematic computation, it is assumed that
        ///           computeJointJacobians and framesForwardKinematics has already
        ///           been called on model->pncModel_.
        ///
        /// \param[in] q    Current joint position.
        /// \param[in] v    Current joint velocity.
        ///////////////////////////////////////////////////////////////////////////////////////////////
        virtual hresult_t computeJacobianAndDrift(vectorN_t const & q,
                                                  vectorN_t const & v) = 0;

        virtual std::string const & getType(void) const = 0;

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief    Return the dimension of the constraint.
        ///////////////////////////////////////////////////////////////////////////////////////////////
        uint32_t getDim(void) const;

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief    Return the jacobian of the constraint.
        ///////////////////////////////////////////////////////////////////////////////////////////////
        matrixN_t const & getJacobian(void) const;

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief    Return the drift of the constraint.
        ///////////////////////////////////////////////////////////////////////////////////////////////
        vectorN_t const & getDrift(void) const;

    private:
        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Link the constraint on the given model, and initialize it.
        ///
        /// \param[in] model    Model on which to apply the constraint.
        /// \return     Error code: attach may fail, including if the constraint is already attached.
        ///////////////////////////////////////////////////////////////////////////////////////////////
        hresult_t attach(std::weak_ptr<Model const> model);

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Detach the constraint from its model.
        ///////////////////////////////////////////////////////////////////////////////////////////////
        void detach(void);

    protected:
        std::weak_ptr<Model const> model_;  ///< Model on which the constraint operates.
        bool_t isAttached_;                 ///< Flag to indicate if the constraint has been attached to a model.
        bool_t isEnabled_;                  ///< Flag to indicate if the constraint is enabled. Handling of this flag is done at Robot level.
        float64_t kp_;                      ///< Position-related baumgarte stabilization gain.
        float64_t kd_;                      ///< Velocity-related baumgarte stabilization gain.
        matrixN_t jacobian_;                ///< Jacobian of the constraint.
        vectorN_t drift_;                   ///< Drift of the constraint.
    };

    template<class T>
    class AbstractConstraintTpl : public AbstractConstraintBase
    {
    public:
        auto shared_from_this() { return shared_from(this); }
        auto shared_from_this() const { return shared_from(this); }

        std::string const & getType(void) const { return type_; }

    public:
        static std::string const type_;
    };
}

#endif //end of JIMINY_ABSTRACT_MOTOR_H
