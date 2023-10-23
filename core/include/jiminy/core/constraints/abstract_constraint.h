#ifndef JIMINY_ABSTRACT_CONSTRAINT_H
#define JIMINY_ABSTRACT_CONSTRAINT_H

#include <memory>

#include "jiminy/core/macros.h"
#include "jiminy/core/types.h"


namespace jiminy
{
    class Model;

    /// \brief Generic interface for kinematic constraints.
    class AbstractConstraintBase : public std::enable_shared_from_this<AbstractConstraintBase>
    {
        // See AbstractSensor for comment on this.
        friend Model;

    public:
        DISABLE_COPY(AbstractConstraintBase)

    public:
        AbstractConstraintBase(void) = default;
        virtual ~AbstractConstraintBase(void);

        /// \brief Refresh the internal buffers and proxies.
        ///
        /// \remark This method is not intended to be called manually. The Robot to which the
        ///         constraint is added is taking care of it when its own `reset` method is called.
        virtual hresult_t reset(const vectorN_t & q, const vectorN_t & v) = 0;

        void enable(void);
        void disable(void);
        const bool_t & getIsEnabled(void) const;

        hresult_t setBaumgartePositionGain(const float64_t & kp);
        float64_t getBaumgartePositionGain(void) const;
        hresult_t setBaumgarteVelocityGain(const float64_t & kd);
        float64_t getBaumgarteVelocityGain(void) const;
        hresult_t setBaumgarteFreq(const float64_t & freq);
        /// \brief Natural frequency of critically damping position/velocity error correction.
        float64_t getBaumgarteFreq(void) const;

        /// \brief Compute the jacobian and drift of the constraint.
        ///
        /// \note To avoid duplicate kinematic computation, it assumes that `computeJointJacobians`
        ///       and `framesForwardKinematics` has already been called on `model->pncModel_`.
        ///
        /// \param[in] q Current joint position.
        /// \param[in] v Current joint velocity.
        virtual hresult_t computeJacobianAndDrift(const vectorN_t & q, const vectorN_t & v) = 0;

        virtual const std::string & getType(void) const = 0;

        /// \brief Dimension of the constraint.
        uint64_t getDim(void) const;

        /// \brief Jacobian of the constraint.
        const matrixN_t & getJacobian(void) const;

        /// \brief Drift of the constraint.
        const vectorN_t & getDrift(void) const;

    private:
        /// \brief Link the constraint on the given model, and initialize it.
        ///
        /// \param[in] model Model on which to apply the constraint.
        ///
        /// \return Error code: attach may fail, including if the constraint is already attached.
        hresult_t attach(std::weak_ptr<const Model> model);

        /// \brief Detach the constraint from its model.
        void detach(void);

    public:
        /// \brief Lambda multipliers.
        vectorN_t lambda_;

    protected:
        /// \brief Model on which the constraint operates.
        std::weak_ptr<const Model> model_;
        /// \brief Flag to indicate whether the constraint has been attached to a model.
        bool_t isAttached_;
        /// \brief Flag to indicate whether the constraint is enabled.
        ///
        /// \remarks Handling of this flag is done at Robot level.
        bool_t isEnabled_;
        /// \brief Position-related baumgarte stabilization gain.
        float64_t kp_;
        /// \brief Velocity-related baumgarte stabilization gain.
        float64_t kd_;
        /// \brief Jacobian of the constraint.
        matrixN_t jacobian_;
        /// \brief Drift of the constraint.
        vectorN_t drift_;
    };

    template<class T>
    class AbstractConstraintTpl : public AbstractConstraintBase
    {
    public:
        auto shared_from_this() { return shared_from(this); }
        auto shared_from_this() const { return shared_from(this); }

        const std::string & getType(void) const { return type_; }

    public:
        static const std::string type_;
    };
}

#endif  // end of JIMINY_ABSTRACT_MOTOR_H
