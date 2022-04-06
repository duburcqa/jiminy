#include "jiminy/core/robot/Robot.h"
#include "jiminy/core/Macros.h"
#include "jiminy/core/Types.h"

#include "jiminy/core/constraints/JointConstraint.h"


namespace jiminy
{
    template<>
    std::string const AbstractConstraintTpl<JointConstraint>::type_("JointConstraint");

    JointConstraint::JointConstraint(std::string const & jointName) :
    AbstractConstraintTpl(),
    jointName_(jointName),
    jointIdx_(0),
    configurationRef_(),
    isReversed_(false)
    {
        // Empty on purpose
    }

    JointConstraint::~JointConstraint(void)
    {
        // Empty on purpose
    }

    std::string const & JointConstraint::getJointName(void) const
    {
        return jointName_;
    }

    jointIndex_t const & JointConstraint::getJointIdx(void) const
    {
        return jointIdx_;
    }

    void JointConstraint::setReferenceConfiguration(vectorN_t const & configurationRef)
    {
        configurationRef_ = configurationRef;
    }

    vectorN_t const & JointConstraint::getReferenceConfiguration(void) const
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

    bool_t const & JointConstraint::getRotationDir(void)
    {
        return isReversed_;
    }

    hresult_t JointConstraint::reset(vectorN_t const & q,
                                     vectorN_t const & /* v */)
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
            pinocchio::JointModel const & jointModel = model->pncModel_.joints[jointIdx_];

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
        DifferenceStep<ConfigVectorIn1, ConfigVectorIn2, TangentVectorType> >
    {
        typedef boost::fusion::vector<ConfigVectorIn1 const &,
                                      ConfigVectorIn2 const &,
                                      TangentVectorType &,
                                      size_t,
                                      size_t> ArgsType;

        template<typename JointModel>
        static std::enable_if_t<!is_pinocchio_joint_composite_v<JointModel>, void>
        algo(pinocchio::JointModelBase<JointModel> const & jmodel,
             ConfigVectorIn1 const & q0,
             ConfigVectorIn2 const & q1,
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
        algo(pinocchio::JointModelBase<JointModel> const & jmodel,
             ConfigVectorIn1 const & q0,
             ConfigVectorIn2 const & q1,
             TangentVectorType & v,
             size_t qIdx,
             size_t vIdx)
        {
            for (size_t i = 0; i < jmodel.derived().joints.size(); ++i)
            {
                pinocchio::JointModel const & joint = jmodel.derived().joints[i];
                algo(joint.derived(), q0, q1, v, qIdx, vIdx);
                qIdx += joint.nq();
                vIdx += joint.nv();
            }
        }
    };

    template<typename ConfigVectorIn1, typename ConfigVectorIn2>
    vectorN_t difference(pinocchio::JointModel              const & jmodel,
                         Eigen::MatrixBase<ConfigVectorIn1> const & q0,
                         Eigen::MatrixBase<ConfigVectorIn2> const & q1)
    {
        vectorN_t v(jmodel.nv());
        typedef DifferenceStep<ConfigVectorIn1, ConfigVectorIn2, vectorN_t> Pass;
        Pass::run(jmodel, typename Pass::ArgsType(q0.derived(), q1.derived(), v, 0, 0));
        return v;
    }

    hresult_t JointConstraint::computeJacobianAndDrift(vectorN_t const & q,
                                                       vectorN_t const & v)
    {
        if (!isAttached_)
        {
            PRINT_ERROR("Constraint not attached to a model.");
            return hresult_t::ERROR_GENERIC;
        }

        // Assuming the model still exists
        auto model = model_.lock();

        // Get the joint model
        pinocchio::JointModel const & jointModel = model->pncModel_.joints[jointIdx_];

        // Add Baumgarte stabilization drift
        vectorN_t const deltaPosition = difference(
            jointModel, configurationRef_, jointModel.jointConfigSelector(q));
        drift_ = kp_ * deltaPosition + kd_ * jointModel.jointVelocitySelector(v);
        if (isReversed_)
        {
            drift_ *= -1;
        }

        return hresult_t::SUCCESS;
    }
}
