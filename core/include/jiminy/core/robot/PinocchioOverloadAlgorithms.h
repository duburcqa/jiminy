/// Re-implementation of dynamics algorithms from pinocchio, adding support
/// of armature (aka rotor inertia) for 1DoF (prismatic, revolute) joints.
///
/// Based on https://github.com/stack-of-tasks/pinocchio/blob/820d0f85fbabddce20924a6e0f781fb2be5029e9/src/algorithm/aba.hxx
///          https://github.com/stack-of-tasks/pinocchio/blob/820d0f85fbabddce20924a6e0f781fb2be5029e9/src/algorithm/rnea.hxx
///          https://github.com/stack-of-tasks/pinocchio/blob/820d0f85fbabddce20924a6e0f781fb2be5029e9/src/algorithm/crba.hxx
///
/// Splitting of algorithms in smaller blocks that can be executed separately.
///
/// Based on https://github.com/stack-of-tasks/pinocchio/blob/820d0f85fbabddce20924a6e0f781fb2be5029e9/src/algorithm/kinematics.hxx
///
/// Copyright (c) 2014-2020, CNRS
/// Copyright (c) 2018-2020, INRIA

#ifndef PINOCCHIO_OVERLOAD_ALGORITHMS_H
#define PINOCCHIO_OVERLOAD_ALGORITHMS_H

#include <functional>

#include "pinocchio/algorithm/aba.hpp"             // `pinocchio::aba`
#include "pinocchio/algorithm/rnea.hpp"            // `pinocchio::rnea`
#include "pinocchio/algorithm/crba.hpp"            // `pinocchio::crba`
#include "pinocchio/algorithm/energy.hpp"          // `pinocchio::kineticEnergy`
#include "pinocchio/multibody/visitor.hpp"         // `pinocchio::fusion::JointUnaryVisitorBase`
#include "pinocchio/multibody/fwd.hpp"             // `pinocchio::ModelTpl`, `pinocchio::DataTpl`
#include "pinocchio/multibody/joint/fwd.hpp"       // `pinocchio::JointModelBase`, `pinocchio::JointDataBase`, ...
#include "pinocchio/spatial/fwd.hpp"               // `Pinocchio::Motion`

#include "jiminy/core/Macros.h"
#include "jiminy/core/engine/EngineMultiRobot.h"


namespace jiminy
{
namespace pinocchio_overload
{
    template<typename Scalar, int Options, template<typename, int> class JointCollectionTpl,
             typename ConfigVectorType, typename TangentVectorType>
    inline Scalar
    kineticEnergy(pinocchio::ModelTpl<Scalar, Options, JointCollectionTpl> const & model,
                  pinocchio::DataTpl<Scalar, Options, JointCollectionTpl>        & data,
                  Eigen::MatrixBase<ConfigVectorType>                      const & q,
                  Eigen::MatrixBase<TangentVectorType>                     const & v,
                  bool_t                                                   const & update_kinematics)
    {
        pinocchio::kineticEnergy(model, data, q, v, update_kinematics);
        data.kinetic_energy += 0.5 * (model.rotorInertia.array() * Eigen::pow(v.array(), 2)).sum();
        return data.kinetic_energy;
    }

    template<typename Scalar, int Options, template<typename, int> class JointCollectionTpl,
             typename ConfigVectorType, typename TangentVectorType1, typename TangentVectorType2,
             typename ForceDerived>
    inline typename pinocchio::DataTpl<Scalar,Options,JointCollectionTpl>::TangentVectorType const &
    rnea(pinocchio::ModelTpl<Scalar,Options,JointCollectionTpl> const & model,
         pinocchio::DataTpl<Scalar,Options,JointCollectionTpl>        & data,
         Eigen::MatrixBase<ConfigVectorType>                    const & q,
         Eigen::MatrixBase<TangentVectorType1>                  const & v,
         Eigen::MatrixBase<TangentVectorType2>                  const & a,
         pinocchio::container::aligned_vector<ForceDerived>     const & fext)
    {
        pinocchio::rnea(model, data, q, v, a, fext);
        data.tau += model.rotorInertia.asDiagonal() * a;
        return data.tau;
    }

    template<typename Scalar, int Options, template<typename, int> class JointCollectionTpl,
             typename ConfigVectorType, typename TangentVectorType1, typename TangentVectorType2>
    inline typename pinocchio::DataTpl<Scalar,Options,JointCollectionTpl>::TangentVectorType const &
    rnea(pinocchio::ModelTpl<Scalar, Options, JointCollectionTpl> const & model,
         pinocchio::DataTpl<Scalar, Options, JointCollectionTpl>        & data,
         Eigen::MatrixBase<ConfigVectorType>                      const & q,
         Eigen::MatrixBase<TangentVectorType1>                    const & v,
         Eigen::MatrixBase<TangentVectorType2>                    const & a)
    {
        pinocchio::rnea(model, data, q, v, a);
        data.tau += model.rotorInertia.asDiagonal() * a;
        return data.tau;
    }

    template<typename Scalar, int Options, template<typename, int> class JointCollectionTpl,
             typename ConfigVectorType>
    inline typename pinocchio::DataTpl<Scalar, Options, JointCollectionTpl>::MatrixXs const &
    crba(pinocchio::ModelTpl<Scalar, Options, JointCollectionTpl> const & model,
         pinocchio::DataTpl<Scalar, Options, JointCollectionTpl>        & data,
         Eigen::MatrixBase<ConfigVectorType>                      const & q)
    {
        pinocchio::crba(model, data, q);
        data.M += model.rotorInertia.asDiagonal();
        return data.M;
    }

    template<typename Scalar, int Options, template<typename, int> class JointCollectionTpl>
    struct AbaBackwardStep :
    public pinocchio::fusion::JointUnaryVisitorBase<AbaBackwardStep<Scalar, Options, JointCollectionTpl> >
    {
        typedef pinocchio::ModelTpl<Scalar, Options, JointCollectionTpl> Model;
        typedef pinocchio::DataTpl<Scalar, Options, JointCollectionTpl> Data;

        typedef boost::fusion::vector<Model const &, Data &> ArgsType;

        template<typename JointModel>
        static void algo(pinocchio::JointModelBase<JointModel> const & jmodel,
                         pinocchio::JointDataBase<typename JointModel::JointDataDerived> & jdata,
                         Model const & model,
                         Data & data)
        {
            /// @brief  See equation 9.28 of Roy Featherstone Rigid Body Dynamics

            typedef typename Model::JointIndex JointIndex;
            typedef typename Data::Inertia Inertia;
            typedef typename Data::Force Force;

            const JointIndex & i = jmodel.id();
            const JointIndex & parent  = model.parents[i];
            typename Inertia::Matrix6 & Ia = data.Yaba[i];

            jmodel.jointVelocitySelector(data.u) -= jdata.S().transpose()*data.f[i];

            // jmodel.calc_aba(jdata.derived(), Ia, parent > 0);
            Scalar const & Im = model.rotorInertia[jmodel.idx_v()];
            calc_aba(jmodel.derived(), jdata.derived(), Ia, Im, parent > 0);

            if (parent > 0)
            {
                Force & pa = data.f[i];
                #if PINOCCHIO_MAJOR_VERSION > 2 || (PINOCCHIO_MAJOR_VERSION == 2 && (PINOCCHIO_MINOR_VERSION > 5 || (PINOCCHIO_MINOR_VERSION == 5 && PINOCCHIO_PATCH_VERSION >= 6)))
                pa.toVector() += Ia * data.a_gf[i].toVector() + jdata.UDinv() * jmodel.jointVelocitySelector(data.u);
                #else
                pa.toVector() += Ia * data.a[i].toVector() + jdata.UDinv() * jmodel.jointVelocitySelector(data.u);
                #endif
                data.Yaba[parent] += pinocchio::internal::SE3actOn<Scalar>::run(data.liMi[i], Ia);
                data.f[parent] += data.liMi[i].act(pa);
            }
        }

        template<typename JointModel, typename Matrix6Like>
        static std::enable_if_t<is_pinocchio_joint_revolute_v<JointModel>
                             || is_pinocchio_joint_revolute_unbounded_v<JointModel>, void>
        calc_aba(JointModel const & model,
                 typename JointModel::JointDataDerived & data,
                 Eigen::MatrixBase<Matrix6Like> & Ia,
                 Scalar const & Im,
                 bool const & update_I)
        {
            using Motion = pinocchio::Motion;

            data.U.noalias() = Ia.col(Motion::ANGULAR + getAxis(model));
            data.Dinv[0] = Scalar(1.0) / (data.U[Motion::ANGULAR + getAxis(model)] + Im);
            data.UDinv.noalias() = data.U * data.Dinv[0];

            if (update_I)
            {
                Ia -= data.UDinv * data.U.transpose();
            }
        }

        template<typename JointModel, typename Matrix6Like>
        static std::enable_if_t<is_pinocchio_joint_revolute_unaligned_v<JointModel>
                             || is_pinocchio_joint_revolute_unbounded_unaligned_v<JointModel>, void>
        calc_aba(JointModel const & model,
                 typename JointModel::JointDataDerived & data,
                 Eigen::MatrixBase<Matrix6Like> & Ia,
                 Scalar const & Im,
                 bool const & update_I)
        {
            using Motion = pinocchio::Motion;

            data.U.noalias() = Ia.template middleCols<3>(Motion::ANGULAR) * model.axis;
            data.Dinv[0] = Scalar(1.0) / (model.axis.dot(data.U.template segment<3>(Motion::ANGULAR)) + Im);
            data.UDinv.noalias() = data.U * data.Dinv[0];

            if (update_I)
            {
                Ia -= data.UDinv * data.U.transpose();
            }
        }

        template<typename JointModel, typename Matrix6Like>
        static std::enable_if_t<is_pinocchio_joint_prismatic_v<JointModel>, void>
        calc_aba(JointModel const & model,
                 typename JointModel::JointDataDerived & data,
                 Eigen::MatrixBase<Matrix6Like> & Ia,
                 Scalar const & Im,
                 bool const & update_I)
        {
            using Motion = pinocchio::Motion;

            data.U.noalias() = Ia.col(Motion::LINEAR + getAxis(model));
            data.Dinv[0] = Scalar(1.0) / (data.U[Motion::LINEAR + getAxis(model)] + Im);
            data.UDinv.noalias() = data.U * data.Dinv[0];

            if (update_I)
            {
                Ia -= data.UDinv * data.U.transpose();
            }
        }

        template<typename JointModel, typename Matrix6Like>
        static std::enable_if_t<is_pinocchio_joint_prismatic_unaligned_v<JointModel>, void>
        calc_aba(JointModel const & model,
                 typename JointModel::JointDataDerived & data,
                 Eigen::MatrixBase<Matrix6Like> & Ia,
                 Scalar const & Im,
                 bool const & update_I)
        {
            using Motion = pinocchio::Motion;

            data.U.noalias() = Ia.template middleCols<3>(Motion::LINEAR) * model.axis;
            data.Dinv[0] = Scalar(1.0) / (model.axis.dot(data.U.template segment<3>(Motion::LINEAR)) + Im);
            data.UDinv.noalias() = data.U * data.Dinv[0];

            if (update_I)
            {
                Ia -= data.UDinv * data.U.transpose();
            }
        }

        template<typename JointModel, typename Matrix6Like>
        static std::enable_if_t<is_pinocchio_joint_planar_v<JointModel>
                             || is_pinocchio_joint_translation_v<JointModel>
                             || is_pinocchio_joint_spherical_v<JointModel>
                             || is_pinocchio_joint_spherical_zyx_v<JointModel>
                             || is_pinocchio_joint_freeflyer_v<JointModel>
                             || is_pinocchio_joint_mimic_v<JointModel>
                             || is_pinocchio_joint_composite_v<JointModel>, void>
        calc_aba(JointModel const & model,
                 typename JointModel::JointDataDerived & data,
                 Eigen::MatrixBase<Matrix6Like> & Ia,
                 Scalar const & Im,
                 bool const & update_I)
        {
            model.calc_aba(data.derived(), Ia, update_I);
        }

        template<int axis>
        static int getAxis(pinocchio::JointModelRevoluteTpl<Scalar, Options, axis> const & model)
        {
            return axis;
        }

        template<int axis>
        static int getAxis(pinocchio::JointModelRevoluteUnboundedTpl<Scalar, Options, axis> const & model)
        {
            return axis;
        }

        template<int axis>
        static int getAxis(pinocchio::JointModelPrismaticTpl<Scalar, Options, axis> const & model)
        {
            return axis;
        }
    };

    template<typename Scalar, int Options, template<typename, int> class JointCollectionTpl,
             typename ConfigVectorType, typename TangentVectorType1,
             typename TangentVectorType2, typename ForceDerived>
    inline const typename pinocchio::DataTpl<Scalar, Options, JointCollectionTpl>::TangentVectorType &
    aba(pinocchio::ModelTpl<Scalar, Options, JointCollectionTpl> const & model,
        pinocchio::DataTpl<Scalar, Options, JointCollectionTpl>        & data,
        Eigen::MatrixBase<ConfigVectorType>                      const & q,
        Eigen::MatrixBase<TangentVectorType1>                    const & v,
        Eigen::MatrixBase<TangentVectorType2>                    const & tau,
        pinocchio::container::aligned_vector<ForceDerived>       const & fext)
    {
        assert(model.check(data) && "data is not consistent with model.");
        assert(q.size() == model.nq && "The joint configuration vector is not of right size");
        assert(v.size() == model.nv && "The joint velocity vector is not of right size");
        assert(tau.size() == model.nv && "The joint acceleration vector is not of right size");

        typedef typename pinocchio::ModelTpl<Scalar, Options, JointCollectionTpl>::JointIndex JointIndex;

        data.v[0].setZero();
        #if PINOCCHIO_MAJOR_VERSION > 2 || (PINOCCHIO_MAJOR_VERSION == 2 && (PINOCCHIO_MINOR_VERSION > 5 || (PINOCCHIO_MINOR_VERSION == 5 && PINOCCHIO_PATCH_VERSION >= 6)))
        data.a_gf[0] = -model.gravity;
        #else
        data.a[0] = -model.gravity;
        #endif
        data.u = tau;

        typedef pinocchio::AbaForwardStep1<Scalar, Options, JointCollectionTpl,
                                           ConfigVectorType, TangentVectorType1> Pass1;
        for (JointIndex i=1; i<(JointIndex)model.njoints; ++i)
        {
            Pass1::run(model.joints[i], data.joints[i],
                       typename Pass1::ArgsType(model, data, q.derived(), v.derived()));
            data.f[i] -= fext[i];
        }

        typedef AbaBackwardStep<Scalar,Options,JointCollectionTpl> Pass2;
        for (JointIndex i=(JointIndex)model.njoints-1; i>0; --i)
        {
            Pass2::run(model.joints[i], data.joints[i],
                       typename Pass2::ArgsType(model, data));
        }

        typedef pinocchio::AbaForwardStep2<Scalar,Options,JointCollectionTpl> Pass3;
        for (JointIndex i=1; i<(JointIndex)model.njoints; ++i)
        {
            Pass3::run(model.joints[i], data.joints[i],
                       typename Pass3::ArgsType(model, data));
        }

        return data.ddq;
    }

    template<typename TangentVectorType>
    struct ForwardKinematicsAccelerationStep :
    public pinocchio::fusion::JointUnaryVisitorBase<ForwardKinematicsAccelerationStep<TangentVectorType> >
    {
        typedef boost::fusion::vector<pinocchio::Model const &,
                                      pinocchio::Data &,
                                      Eigen::MatrixBase<TangentVectorType> const &
                                      > ArgsType;

        template<typename JointModel>
        static void algo(pinocchio::JointModelBase<JointModel> const & jmodel,
                         pinocchio::JointDataBase<typename JointModel::JointDataDerived> & jdata,
                         pinocchio::Model const & model,
                         pinocchio::Data & data,
                         Eigen::MatrixBase<TangentVectorType> const & a)
        {
            uint32_t const & i = jmodel.id();
            uint32_t const & parent = model.parents[i];
            data.a[i]  = jdata.S() * jmodel.jointVelocitySelector(a) + jdata.c() + (data.v[i] ^ jdata.v());
            data.a[i] += data.liMi[i].actInv(data.a[parent]);
        }
    };

    /// \brief Compute only joints spatial accelerations, assuming positions and velocities
    /// are already up-to-date.
    ///
    /// Note that it does not update the internal buffer `data.ddq`. This buffer is updated
    /// by `aba` and `forwardDynamics` algorithms only.
    template<typename TangentVectorType>
    inline void forwardKinematicsAcceleration(pinocchio::Model const & model,
                                              pinocchio::Data & data,
                                              Eigen::MatrixBase<TangentVectorType> const & a)
    {
        typedef ForwardKinematicsAccelerationStep<TangentVectorType> Pass1;
        data.a[0].setZero();
        for (int32_t i = 1; i < model.njoints; ++i)
        {
            Pass1::run(model.joints[i], data.joints[i], typename Pass1::ArgsType(model, data, a));
        }
    }
}
}

#endif //end of JIMINY_ALGORITHMS_MOTOR_INERTIA_H
