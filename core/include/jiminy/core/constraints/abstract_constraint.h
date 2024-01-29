#ifndef JIMINY_ABSTRACT_CONSTRAINT_H
#define JIMINY_ABSTRACT_CONSTRAINT_H

#include <memory>

#include "jiminy/core/fwd.h"
#include "jiminy/core/utilities/helpers.h"


namespace jiminy
{
    class Model;

    /// \brief Generic interface for kinematic constraints.
    class JIMINY_DLLAPI AbstractConstraintBase :
    public std::enable_shared_from_this<AbstractConstraintBase>
    {
        // See AbstractSensor for comment on this.
        friend Model;

    public:
        DISABLE_COPY(AbstractConstraintBase)

    public:
        explicit AbstractConstraintBase() = default;
        virtual ~AbstractConstraintBase();

        /// \brief Refresh the internal buffers and proxies.
        ///
        /// \remark This method is not intended to be called manually. The Robot to which the
        ///         constraint is added is taking care of it when its own `reset` method is called.
        virtual hresult_t reset(const Eigen::VectorXd & q, const Eigen::VectorXd & v) = 0;

        void enable();
        void disable();
        bool getIsEnabled() const;

        hresult_t setBaumgartePositionGain(double kp);
        double getBaumgartePositionGain() const;
        hresult_t setBaumgarteVelocityGain(double kd);
        double getBaumgarteVelocityGain() const;
        hresult_t setBaumgarteFreq(double freq);
        /// \brief Natural frequency of critically damping position/velocity error correction.
        double getBaumgarteFreq() const;

        /// \brief Compute the jacobian and drift of the constraint.
        ///
        /// \note To avoid redundant computations, it assumes that `computeJointJacobians` and
        ///       `framesForwardKinematics` has already been called on `model->pinocchioModel_`.
        ///
        /// \param[in] q Current joint position.
        /// \param[in] v Current joint velocity.
        virtual hresult_t computeJacobianAndDrift(const Eigen::VectorXd & q,
                                                  const Eigen::VectorXd & v) = 0;

        virtual const std::string & getType() const = 0;

        /// \brief Dimension of the constraint.
        uint64_t getDim() const;

        /// \brief Jacobian of the constraint.
        const Eigen::MatrixXd & getJacobian() const;

        /// \brief Drift of the constraint.
        const Eigen::VectorXd & getDrift() const;

    private:
        /// \brief Link the constraint on the given model, and initialize it.
        ///
        /// \param[in] model Model on which to apply the constraint.
        ///
        /// \return Error code: attach may fail, including if the constraint is already attached.
        hresult_t attach(std::weak_ptr<const Model> model);

        /// \brief Detach the constraint from its model.
        void detach();

    public:
        /// \brief Lambda multipliers.
        Eigen::VectorXd lambda_{};

    protected:
        /// \brief Model on which the constraint operates.
        std::weak_ptr<const Model> model_{};
        /// \brief Flag to indicate whether the constraint has been attached to a model.
        bool isAttached_{false};
        /// \brief Flag to indicate whether the constraint is enabled.
        ///
        /// \remarks Handling of this flag is done at Robot level.
        bool isEnabled_{false};
        /// \brief Position-related baumgarte stabilization gain.
        double kp_{0.0};
        /// \brief Velocity-related baumgarte stabilization gain.
        double kd_{0.0};
        /// \brief Jacobian of the constraint.
        Eigen::MatrixXd jacobian_{};
        /// \brief Drift of the constraint.
        Eigen::VectorXd drift_{};
    };

    template<class T>
    class JIMINY_DLLAPI AbstractConstraintTpl : public AbstractConstraintBase
    {
    public:
        auto shared_from_this() { return shared_from(this); }
        auto shared_from_this() const { return shared_from(this); }

        const std::string & getType() const { return type_; }

    public:
        static const std::string type_;
    };
}

#endif  // end of JIMINY_ABSTRACT_MOTOR_H
