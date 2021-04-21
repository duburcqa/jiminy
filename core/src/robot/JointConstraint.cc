#include "jiminy/core/robot/Robot.h"
#include "jiminy/core/Macros.h"
#include "jiminy/core/Types.h"

#include "jiminy/core/robot/JointConstraint.h"


namespace jiminy
{
    template<>
    std::string const AbstractConstraintTpl<JointConstraint>::type_("JointConstraint");

    JointConstraint::JointConstraint(std::string const & jointName) :
    AbstractConstraintTpl(),
    jointName_(jointName),
    jointIdx_(0),
    configurationRef_()
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

    int32_t const & JointConstraint::getJointIdx(void) const
    {
        return jointIdx_;
    }

    vectorN_t & JointConstraint::getReferenceConfiguration(void)
    {
        return configurationRef_;
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
            if (jointIdx_ == model->pncModel_.njoints)
            {
                PRINT_ERROR("No joint with name '", jointName_, "' in model.");
                returnCode = hresult_t::ERROR_GENERIC;
            }
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            // Get the joint model
            pinocchio::JointModel const & jointModel = model->pncModel_.joints[jointIdx_];

            // Compute the jacobian. It is simply the velocity selector mask.
            jacobian_ = matrixN_t::Zero(jointModel.nv(), model->pncModel_.nv);
            for (int32_t i=0; i < jointModel.nv(); ++i)
            {
                jacobian_(i, jointModel.idx_v() + i) = 1.0;
            }

            // Compute the drift.
            drift_ = vectorN_t::Zero(jointModel.nv());

            // Get the current joint position and use it as reference
            configurationRef_ = jointModel.jointConfigSelector(q);
        }

        return returnCode;
    }

    template<typename JointModel, typename ConfigVectorIn1, typename ConfigVectorIn2>
    auto difference(pinocchio::JointModelBase<JointModel> const & /* jmodel */,
                    Eigen::MatrixBase<ConfigVectorIn1>    const & q0,
                    Eigen::MatrixBase<ConfigVectorIn2>    const & q1)
    {
        typename pinocchio::LieGroupMap::template operation<JointModel>::type lgo;
        return lgo.difference(q0, q1);
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

        return hresult_t::SUCCESS;
    }
}
