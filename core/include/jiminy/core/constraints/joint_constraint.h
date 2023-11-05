#ifndef JIMINY_JOINT_CONSTRAINT_H
#define JIMINY_JOINT_CONSTRAINT_H

#include <memory>

#include "jiminy/core/types.h"
#include "jiminy/core/constraints/abstract_constraint.h"


namespace jiminy
{
    class Model;

    class JointConstraint : public AbstractConstraintTpl<JointConstraint>
    {
    public:
        DISABLE_COPY(JointConstraint)

        auto shared_from_this() { return shared_from(this); }

    public:
        /// \param[in] jointName Name of the joint.
        JointConstraint(const std::string & jointName);
        virtual ~JointConstraint() = default;

        const std::string & getJointName() const;
        const jointIndex_t & getJointIdx() const;

        void setReferenceConfiguration(const Eigen::VectorXd & configurationRef);
        const Eigen::VectorXd & getReferenceConfiguration() const;

        void setRotationDir(bool_t isReversed);
        const bool_t & getRotationDir();

        virtual hresult_t reset(const Eigen::VectorXd & q,
                                const Eigen::VectorXd & v) override final;

        virtual hresult_t computeJacobianAndDrift(const Eigen::VectorXd & q,
                                                  const Eigen::VectorXd & v) override final;

    private:
        /// \brief Name of the joint on which the constraint operates.
        std::string jointName_;
        /// \brief Corresponding joint index.
        jointIndex_t jointIdx_;
        /// \brief Reference position of the joint to enforce.
        Eigen::VectorXd configurationRef_;
        /// \brief Whether to reverse the sign of the constraint.
        bool_t isReversed_;
    };
}

#endif  // end of JIMINY_JOINT_CONSTRAINT_H
