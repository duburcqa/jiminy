#ifndef JIMINY_JOINT_CONSTRAINT_H
#define JIMINY_JOINT_CONSTRAINT_H

#include <memory>

#include "jiminy/core/fwd.h"
#include "jiminy/core/utilities/helpers.h"
#include "jiminy/core/constraints/abstract_constraint.h"


namespace jiminy
{
    class Model;

    class JIMINY_DLLAPI JointConstraint : public AbstractConstraintTpl<JointConstraint>
    {
    public:
        DISABLE_COPY(JointConstraint)

        auto shared_from_this() { return shared_from(this); }

    public:
        /// \param[in] jointName Name of the joint.
        JointConstraint(const std::string & jointName);
        virtual ~JointConstraint() = default;

        const std::string & getJointName() const;
        pinocchio::JointIndex getJointIdx() const;

        void setReferenceConfiguration(const Eigen::VectorXd & configurationRef);
        const Eigen::VectorXd & getReferenceConfiguration() const;

        void setRotationDir(bool isReversed);
        bool getRotationDir();

        virtual hresult_t reset(const Eigen::VectorXd & q,
                                const Eigen::VectorXd & v) override final;

        virtual hresult_t computeJacobianAndDrift(const Eigen::VectorXd & q,
                                                  const Eigen::VectorXd & v) override final;

    private:
        /// \brief Name of the joint on which the constraint operates.
        std::string jointName_;
        /// \brief Corresponding joint index.
        pinocchio::JointIndex jointIdx_;
        /// \brief Reference position of the joint to enforce.
        Eigen::VectorXd configurationRef_;
        /// \brief Whether to reverse the sign of the constraint.
        bool isReversed_;
    };
}

#endif  // end of JIMINY_JOINT_CONSTRAINT_H
