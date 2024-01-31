#include "jiminy/core/robot/model.h"

#include "jiminy/core/constraints/joint_constraint.h"


namespace jiminy
{
    template<>
    const std::string AbstractConstraintTpl<JointConstraint>::type_{"JointConstraint"};

    JointConstraint::JointConstraint(const std::string & jointName) noexcept :
    AbstractConstraintTpl(),
    jointName_{jointName}
    {
    }

    const std::string & JointConstraint::getJointName() const noexcept
    {
        return jointName_;
    }

    pinocchio::JointIndex JointConstraint::getJointIndex() const noexcept
    {
        return jointIndex_;
    }

    void JointConstraint::setReferenceConfiguration(
        const Eigen::VectorXd & configurationRef) noexcept
    {
        configurationRef_ = configurationRef;
    }

    const Eigen::VectorXd & JointConstraint::getReferenceConfiguration() const noexcept
    {
        return configurationRef_;
    }

    void JointConstraint::setRotationDir(bool isReversed) noexcept
    {
        // Update the Jacobian
        if (isReversed_ != isReversed)
        {
            jacobian_ *= -1;
        }

        // Update active dir
        isReversed_ = isReversed;
    }

    bool JointConstraint::getRotationDir() noexcept
    {
        return isReversed_;
    }

    hresult_t JointConstraint::reset(const Eigen::VectorXd & q, const Eigen::VectorXd & /* v */)
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
            jointIndex_ = model->pinocchioModel_.getJointId(jointName_);
            if (static_cast<int>(jointIndex_) == model->pinocchioModel_.njoints)
            {
                PRINT_ERROR("No joint with name '", jointName_, "' in model.");
                returnCode = hresult_t::ERROR_GENERIC;
            }
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            // Get the joint model
            const pinocchio::JointModel & jointModel = model->pinocchioModel_.joints[jointIndex_];

            // Initialize constraint jacobian, drift and multipliers
            jacobian_.setZero(jointModel.nv(), model->pinocchioModel_.nv);
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
             size_t qIndex,
             size_t vIndex)
        {
            typename pinocchio::LieGroupMap::template operation<JointModel>::type lgo;
            lgo.difference(q0.segment(qIndex, jmodel.nq()),
                           q1.segment(qIndex, jmodel.nq()),
                           v.segment(vIndex, jmodel.nv()));
        }

        template<typename JointModel>
        static std::enable_if_t<is_pinocchio_joint_composite_v<JointModel>, void>
        algo(const pinocchio::JointModelBase<JointModel> & jmodel,
             const ConfigVectorIn1 & q0,
             const ConfigVectorIn2 & q1,
             TangentVectorType & v,
             size_t qIndex,
             size_t vIndex)
        {
            for (const auto & joint : jmodel.derived().joints)
            {
                algo(joint.derived(), q0, q1, v, qIndex, vIndex);
                qIndex += joint.nq();
                vIndex += joint.nv();
            }
        }
    };

    template<typename ConfigVectorIn1, typename ConfigVectorIn2>
    Eigen::VectorXd difference(const pinocchio::JointModel & jmodel,
                               const ConfigVectorIn1 & q0,
                               const ConfigVectorIn2 & q1)
    {
        Eigen::VectorXd v(jmodel.nv());
        typedef DifferenceStep<ConfigVectorIn1, ConfigVectorIn2, Eigen::VectorXd> Pass;
        Pass::run(jmodel, typename Pass::ArgsType(q0, q1, v, 0, 0));
        return v;
    }

    hresult_t JointConstraint::computeJacobianAndDrift(const Eigen::VectorXd & q,
                                                       const Eigen::VectorXd & v)
    {
        if (!isAttached_)
        {
            PRINT_ERROR("Constraint not attached to a model.");
            return hresult_t::ERROR_GENERIC;
        }

        // Assuming the model still exists
        auto model = model_.lock();

        // Get the joint model
        const pinocchio::JointModel & jointModel = model->pinocchioModel_.joints[jointIndex_];

        // Add Baumgarte stabilization drift
        const Eigen::VectorXd deltaPosition =
            difference(jointModel, configurationRef_, jointModel.jointConfigSelector(q));
        drift_ = kp_ * deltaPosition + kd_ * jointModel.jointVelocitySelector(v);
        if (isReversed_)
        {
            drift_ *= -1;
        }

        return hresult_t::SUCCESS;
    }
}
