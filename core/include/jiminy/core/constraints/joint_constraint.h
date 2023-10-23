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
        virtual ~JointConstraint(void) = default;

        const std::string & getJointName(void) const;
        const jointIndex_t & getJointIdx(void) const;

        void setReferenceConfiguration(const vectorN_t & configurationRef);
        const vectorN_t & getReferenceConfiguration(void) const;

        void setRotationDir(bool_t isReversed);
        const bool_t & getRotationDir();

        virtual hresult_t reset(const vectorN_t & q, const vectorN_t & v) override final;

        virtual hresult_t computeJacobianAndDrift(const vectorN_t & q,
                                                  const vectorN_t & v) override final;

    private:
        /// \brief Name of the joint on which the constraint operates.
        std::string jointName_;
        /// \brief Corresponding joint index.
        jointIndex_t jointIdx_;
        /// \brief Reference position of the joint to enforce.
        vectorN_t configurationRef_;
        /// \brief Whether to reverse the sign of the constraint.
        bool_t isReversed_;
    };
}

#endif  // end of JIMINY_JOINT_CONSTRAINT_H