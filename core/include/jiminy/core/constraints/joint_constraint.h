#ifndef JIMINY_JOINT_CONSTRAINT_H
#define JIMINY_JOINT_CONSTRAINT_H

#include <memory>

#include "jiminy/core/fwd.h"
#include "jiminy/core/utilities/helpers.h"
#include "jiminy/core/constraints/abstract_constraint.h"


namespace jiminy
{
    class Model;

    class JointConstraint;
#if defined EXPORT_SYMBOLS || (!defined _WIN32 && !defined __CYGWIN__)
    template<>
    const std::string JIMINY_DLLAPI AbstractConstraintTpl<JointConstraint>::type_;
#endif
    template class JIMINY_TEMPLATE_INSTANTIATION_DLLAPI AbstractConstraintTpl<JointConstraint>;

    class JIMINY_DLLAPI JointConstraint : public AbstractConstraintTpl<JointConstraint>
    {
    public:
        JIMINY_DISABLE_COPY(JointConstraint)

    public:
        /// \param[in] jointName Name of the joint.
        explicit JointConstraint(const std::string & jointName) noexcept;
        virtual ~JointConstraint() = default;

        const std::string & getJointName() const noexcept;
        pinocchio::JointIndex getJointIndex() const noexcept;

        void setReferenceConfiguration(const Eigen::VectorXd & configurationRef) noexcept;
        const Eigen::VectorXd & getReferenceConfiguration() const noexcept;

        void setRotationDir(bool isReversed) noexcept;
        bool getRotationDir() const noexcept;

        virtual void reset(const Eigen::VectorXd & q, const Eigen::VectorXd & v) override final;

        virtual void computeJacobianAndDrift(const Eigen::VectorXd & q,
                                             const Eigen::VectorXd & v) override final;

    private:
        /// \brief Name of the joint on which the constraint operates.
        std::string jointName_;
        /// \brief Corresponding joint index.
        pinocchio::JointIndex jointIndex_{0};
        /// \brief Reference position of the joint to enforce.
        Eigen::VectorXd configurationRef_{};
        /// \brief Whether to reverse the sign of the constraint.
        bool isReversed_{false};
    };
}

#endif  // end of JIMINY_JOINT_CONSTRAINT_H
