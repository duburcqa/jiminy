#ifndef JIMINY_JOINT_CONSTRAINT_H
#define JIMINY_JOINT_CONSTRAINT_H

#include <memory>

#include "jiminy/core/Types.h"
#include "jiminy/core/robot/AbstractConstraint.h"


namespace jiminy
{
    class Model;

    class JointConstraint: public AbstractConstraintTpl<JointConstraint>
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
        int32_t const & getJointIdx(void) const;

        template<typename DerivedType>
        void setReferenceConfiguration(Eigen::MatrixBase<DerivedType> const & configurationRef)
        {
            configurationRef_ = configurationRef;
        }
        vectorN_t & getReferenceConfiguration(void);

        virtual hresult_t reset(vectorN_t const & q,
                                vectorN_t const & v) override final;

        virtual hresult_t computeJacobianAndDrift(vectorN_t const & q,
                                                  vectorN_t const & v) override final;

    private:
        std::string jointName_;       ///< Name of the joint on which the constraint operates.
        int32_t jointIdx_;            ///< Corresponding joint index.
        vectorN_t configurationRef_;  ///< Reference position of the joint to enforce.
    };
}

#endif //end of JIMINY_JOINT_CONSTRAINT_H
