#ifndef JIMINY_JOINT_CONSTRAINT_H
#define JIMINY_JOINT_CONSTRAINT_H

#include <memory>

#include "jiminy/core/Types.h"
#include "jiminy/core/constraints/AbstractConstraint.h"


namespace jiminy
{
    class Model;

    class JointConstraint : public AbstractConstraintTpl<JointConstraint>
    {

    public:
        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Forbid the copy of the class
        ///////////////////////////////////////////////////////////////////////////////////////////////
        JointConstraint(JointConstraint const & abstractConstraint) = delete;
        JointConstraint & operator = (JointConstraint const & other) = delete;

        auto shared_from_this() { return shared_from(this); }

        ///////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief      Constructor
        ///
        /// \param[in]  jointName     Name of the joint.
        ///////////////////////////////////////////////////////////////////////////////////////////////
        JointConstraint(std::string const & jointName);
        virtual ~JointConstraint(void);

        std::string const & getJointName(void) const;
        jointIndex_t const & getJointIdx(void) const;

        void setReferenceConfiguration(vectorN_t const & configurationRef);
        vectorN_t const & getReferenceConfiguration(void) const;

        void setRotationDir(bool_t isReversed);
        bool_t const & getRotationDir();

        virtual hresult_t reset(vectorN_t const & q,
                                vectorN_t const & v) override final;

        virtual hresult_t computeJacobianAndDrift(vectorN_t const & q,
                                                  vectorN_t const & v) override final;

    private:
        std::string jointName_;        ///< Name of the joint on which the constraint operates.
        jointIndex_t jointIdx_;        ///< Corresponding joint index.
        vectorN_t configurationRef_;   ///< Reference position of the joint to enforce.
        bool_t isReversed_;            ///< Whether or not to reverse the sign of the constraint.
    };
}

#endif //end of JIMINY_JOINT_CONSTRAINT_H
