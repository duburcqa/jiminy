#include "jiminy/core/robot/robot.h"
#include "jiminy/core/macros.h"
#include "jiminy/core/types.h"

#include "jiminy/core/constraints/joint_constraint.h"


namespace jiminy
{
    template<>
    const std::string AbstractConstraintTpl<JointConstraint>::type_("JointConstraint");

    JointConstraint::JointConstraint(const std::string & jointName) :
    AbstractConstraintTpl(),
    jointName_(jointName),
    jointIdx_(0),
    configurationRef_(),
    isReversed_(false)
    {
        // Empty on purpose
    }

    const std::string & JointConstraint::getJointName(void) const
    {
        return jointName_;
    }

    const jointIndex_t & JointConstraint::getJointIdx(void) const
    {
        return jointIdx_;
    }

    void JointConstraint::setReferenceConfiguration(const vectorN_t & configurationRef)
    {
        configurationRef_ = configurationRef;
    }

    const vectorN_t & JointConstraint::getReferenceConfiguration(void) const
    {
        return configurationRef_;
    }

    void JointConstraint::setRotationDir(bool_t isReversed)
    {
        // Update the Jacobian
        if (isReversed_ != isReversed)
        {
            jacobian_ *= -1;
        }

        // Update active dir
        isReversed_ = isReversed;
    }

    const bool_t & JointConstraint::getRotationDir(void)
    {
        return isReversed_;
    }

    hresult_t JointConstraint::reset(const vectorN_t & q, const vectorN_t & /* v */)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        // Make sure the model still exists
        auto model = model_.lock();
        if (!model)
        {
            PRINT_ERROR("Model pointer expired or unset.");
            returnCode = hresult_t::ERROR_GENERIC;
        }

        // Get joint index
        if (returnCode == hresult_t::SUCCESS)
        {
            jointIdx_ = model->pncModel_.getJointId(jointName_);
            if (jointIdx_ == static_cast<uint32_t>(model->pncModel_.njoints))
            {
                PRINT_ERROR("No joint with name '", jointName_, "' in model.");
                returnCode = hresult_t::ERROR_GENERIC;
            }
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            // Get the joint model
            const pinocchio::JointModel & jointModel = model->pncModel_.joints[jointIdx_];

            // Initialize constraint jacobian, drift and multipliers
            jacobian_.setZero(jointModel.nv(), model->pncModel_.nv);
            jacobian_.middleCols(jointModel.idx_v(), jointModel.nv()).setIdentity();
            if (isReversed_)
            {
                jacobian_ *= -1;
            }
            drift_.setZero(jointModel.nv());
            lambda_.setZero(jointModel.nv());

            // Get the current joint position and use it as reference
            configurationRef_ = jointModel.jointConfigSelector(q);
        }

        return returnCode;
    }

    template<typename ConfigVectorIn1, typename ConfigVectorIn2, typename TangentVectorType>
    struct DifferenceStep :
    public pinocchio::fusion::JointUnaryVisitorBase<
        DifferenceStep<ConfigVectorIn1, ConfigVectorIn2, TangentVectorType>>
    {
        typedef boost::fusion::vector<const ConfigVectorIn1 &,
                                      const ConfigVectorIn2 &,
                                      TangentVectorType &,
                                      size_t,
                                      size_t>
            ArgsType;

        template<typename JointModel>
        static std::enable_if_t<!is_pinocchio_joint_composite_v<JointModel>, void>
        algo(const pinocchio::JointModelBase<JointModel> & jmodel,
             const ConfigVectorIn1 & q0,
             const ConfigVectorIn2 & q1,
             TangentVectorType & v,
             size_t qIdx,
             size_t vIdx)
        {
            typename pinocchio::LieGroupMap::template operation<JointModel>::type lgo;
            lgo.difference(q0.segment(qIdx, jmodel.nq()),
                           q1.segment(qIdx, jmodel.nq()),
                           v.segment(vIdx, jmodel.nv()));
        }

        template<typename JointModel>
        static std::enable_if_t<is_pinocchio_joint_composite_v<JointModel>, void>
        algo(const pinocchio::JointModelBase<JointModel> & jmodel,
             const ConfigVectorIn1 & q0,
             const ConfigVectorIn2 & q1,
             TangentVectorType & v,
             size_t qIdx,
             size_t vIdx)
        {
            for (const auto & joint : jmodel.derived().joints)
            {
                algo(joint.derived(), q0, q1, v, qIdx, vIdx);
                qIdx += joint.nq();
                vIdx += joint.nv();
            }
        }
    };

    template<typename ConfigVectorIn1, typename ConfigVectorIn2>
    vectorN_t difference(const pinocchio::JointModel & jmodel,
                         const ConfigVectorIn1 & q0,
                         const ConfigVectorIn2 & q1)
    {
        vectorN_t v(jmodel.nv());
        typedef DifferenceStep<ConfigVectorIn1, ConfigVectorIn2, vectorN_t> Pass;
        Pass::run(jmodel, typename Pass::ArgsType(q0, q1, v, 0, 0));
        return v;
    }

    hresult_t JointConstraint::computeJacobianAndDrift(const vectorN_t & q, const vectorN_t & v)
    {
        if (!isAttached_)
        {
            PRINT_ERROR("Constraint not attached to a model.");
            return hresult_t::ERROR_GENERIC;
        }

        // Assuming the model still exists
        auto model = model_.lock();

        // Get the joint model
        const pinocchio::JointModel & jointModel = model->pncModel_.joints[jointIdx_];

        // Add Baumgarte stabilization drift
        const vectorN_t deltaPosition =
            difference(jointModel, configurationRef_, jointModel.jointConfigSelector(q));
        drift_ = kp_ * deltaPosition + kd_ * jointModel.jointVelocitySelector(v);
        if (isReversed_)
        {
            drift_ *= -1;
        }

        return hresult_t::SUCCESS;
    }
}
