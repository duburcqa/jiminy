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

#include "pinocchio/spatial/fwd.hpp"               // `Pinocchio::Inertia`
#include "pinocchio/multibody/visitor.hpp"         // `pinocchio::fusion::JointUnaryVisitorBase`
#include "pinocchio/multibody/fwd.hpp"             // `pinocchio::ModelTpl`, `pinocchio::DataTpl`
#include "pinocchio/multibody/joint/fwd.hpp"       // `pinocchio::JointModelBase`, `pinocchio::JointDataBase`, ...
#include "pinocchio/algorithm/aba.hpp"             // `pinocchio::aba`
#include "pinocchio/algorithm/rnea.hpp"            // `pinocchio::rnea`
#include "pinocchio/algorithm/crba.hpp"            // `pinocchio::crba`
#include "pinocchio/algorithm/energy.hpp"          // `pinocchio::computeKineticEnergy`
#include "pinocchio/algorithm/cholesky.hpp"        // `pinocchio::cholesky::`

#include "jiminy/core/Macros.h"
#include "jiminy/core/engine/EngineMultiRobot.h"


namespace jiminy
{
namespace pinocchio_overload
{
    template<typename Scalar, int Options, template<typename, int> class JointCollectionTpl,
             typename ConfigVectorType, typename TangentVectorType>
    inline Scalar
    computeKineticEnergy(pinocchio::ModelTpl<Scalar, Options, JointCollectionTpl> const & model,
                         pinocchio::DataTpl<Scalar, Options, JointCollectionTpl>        & data,
                         Eigen::MatrixBase<ConfigVectorType>                      const & q,
                         Eigen::MatrixBase<TangentVectorType>                     const & v)
    {
        (void) pinocchio::computeKineticEnergy(model, data, q, v);
        data.kinetic_energy += 0.5 * (model.rotorInertia.array() * v.array().pow(2)).sum();
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
        (void) pinocchio::rnea(model, data, q, v, a, fext);
        data.tau.array() += model.rotorInertia.array() * a.array();
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
        (void) pinocchio::rnea(model, data, q, v, a);
        data.tau.array() += model.rotorInertia.array() * a.array();
        return data.tau;
    }

    template<typename Scalar, int Options, template<typename, int> class JointCollectionTpl,
             typename ConfigVectorType>
    inline typename pinocchio::DataTpl<Scalar, Options, JointCollectionTpl>::MatrixXs const &
    crba(pinocchio::ModelTpl<Scalar, Options, JointCollectionTpl> const & model,
         pinocchio::DataTpl<Scalar, Options, JointCollectionTpl>        & data,
         Eigen::MatrixBase<ConfigVectorType>                      const & q)
    {
        (void) pinocchio::crba(model, data, q);
        data.M.diagonal() += model.rotorInertia;
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
            auto const Im = model.rotorInertia.segment(jmodel.idx_v(), jmodel.nv());
            calc_aba(jmodel.derived(), jdata.derived(), Im, Ia, parent > 0);

            if (parent > 0)
            {
                Force & pa = data.f[i];
                pa.toVector() += Ia * data.a_gf[i].toVector() + jdata.UDinv() * jmodel.jointVelocitySelector(data.u);
                data.Yaba[parent] += pinocchio::internal::SE3actOn<Scalar>::run(data.liMi[i], Ia);
                data.f[parent] += data.liMi[i].act(pa);
            }
        }

        template<typename JointModel, typename VectorLike, typename Matrix6Like>
        static std::enable_if_t<is_pinocchio_joint_revolute_v<JointModel>
                             || is_pinocchio_joint_revolute_unbounded_v<JointModel>, void>
        calc_aba(JointModel const & model,
                 typename JointModel::JointDataDerived & data,
                 Eigen::MatrixBase<VectorLike> const & Im,
                 Eigen::MatrixBase<Matrix6Like> & Ia,
                 bool const & update_I)
        {
            using Inertia = pinocchio::Inertia;

            data.U = Ia.col(Inertia::ANGULAR + getAxis(model));
            data.Dinv[0] = Scalar(1) / (data.U[Inertia::ANGULAR + getAxis(model)] + Im[0]);
            data.UDinv.noalias() = data.U * data.Dinv[0];

            if (update_I)
            {
                Ia.noalias() -= data.UDinv * data.U.transpose();
            }
        }

        template<typename JointModel, typename VectorLike, typename Matrix6Like>
        static std::enable_if_t<is_pinocchio_joint_revolute_unaligned_v<JointModel>
                             || is_pinocchio_joint_revolute_unbounded_unaligned_v<JointModel>, void>
        calc_aba(JointModel const & model,
                 typename JointModel::JointDataDerived & data,
                 Eigen::MatrixBase<VectorLike> const & Im,
                 Eigen::MatrixBase<Matrix6Like> & Ia,
                 bool const & update_I)
        {
            using Inertia = pinocchio::Inertia;

            data.U.noalias() = Ia.template middleCols<3>(Inertia::ANGULAR) * model.axis;
            data.Dinv[0] = Scalar(1) / (model.axis.dot(data.U.template segment<3>(Inertia::ANGULAR)) + Im[0]);
            data.UDinv.noalias() = data.U * data.Dinv[0];

            if (update_I)
            {
                Ia.noalias() -= data.UDinv * data.U.transpose();
            }
        }

        template<typename JointModel, typename VectorLike, typename Matrix6Like>
        static std::enable_if_t<is_pinocchio_joint_prismatic_v<JointModel>, void>
        calc_aba(JointModel const & model,
                 typename JointModel::JointDataDerived & data,
                 Eigen::MatrixBase<VectorLike> const & Im,
                 Eigen::MatrixBase<Matrix6Like> & Ia,
                 bool const & update_I)
        {
            using Inertia = pinocchio::Inertia;

            data.U = Ia.col(Inertia::LINEAR + getAxis(model));
            data.Dinv[0] = Scalar(1) / (data.U[Inertia::LINEAR + getAxis(model)] + Im[0]);
            data.UDinv.noalias() = data.U * data.Dinv[0];

            if (update_I)
            {
                Ia.noalias() -= data.UDinv * data.U.transpose();
            }
        }

        template<typename JointModel, typename VectorLike, typename Matrix6Like>
        static std::enable_if_t<is_pinocchio_joint_prismatic_unaligned_v<JointModel>, void>
        calc_aba(JointModel const & model,
                 typename JointModel::JointDataDerived & data,
                 Eigen::MatrixBase<VectorLike> const & Im,
                 Eigen::MatrixBase<Matrix6Like> & Ia,
                 bool const & update_I)
        {
            using Inertia = pinocchio::Inertia;

            data.U.noalias() = Ia.template middleCols<3>(Inertia::LINEAR) * model.axis;
            data.Dinv[0] = Scalar(1) / (model.axis.dot(data.U.template segment<3>(Inertia::LINEAR)) + Im[0]);
            data.UDinv.noalias() = data.U * data.Dinv[0];

            if (update_I)
            {
                Ia.noalias() -= data.UDinv * data.U.transpose();
            }
        }

        template<typename JointModel, typename VectorLike, typename Matrix6Like>
        static std::enable_if_t<is_pinocchio_joint_planar_v<JointModel>, void>
        calc_aba(JointModel const & /* model */,
                 typename JointModel::JointDataDerived & data,
                 Eigen::MatrixBase<VectorLike> const & Im,
                 Eigen::MatrixBase<Matrix6Like> & Ia,
                 bool const & update_I)
        {
            static Eigen::Matrix<Scalar, 3, 3> StU;

            data.U.template leftCols<2>() = Ia.template leftCols<2>();
            data.U.template rightCols<1>() = Ia.template rightCols<1>();
            StU.template leftCols<2>() = data.U.template topRows<2>().transpose();
            StU.template rightCols<1>() = data.U.template bottomRows<1>();

            StU.diagonal() += Im;
            pinocchio::internal::PerformStYSInversion<Scalar>::run(StU, data.Dinv);
            data.UDinv.noalias() = data.U * data.Dinv;

            if(update_I)
            {
                Ia.noalias() -= data.UDinv * data.U.transpose();
            }
        }

        template<typename JointModel, typename VectorLike, typename Matrix6Like>
        static std::enable_if_t<is_pinocchio_joint_translation_v<JointModel>, void>
        calc_aba(JointModel const & /* model */,
                 typename JointModel::JointDataDerived & data,
                 Eigen::MatrixBase<VectorLike> const & Im,
                 Eigen::MatrixBase<Matrix6Like> & Ia,
                 bool const & update_I)
        {
            static Eigen::Matrix<Scalar, 3, 3> StU;

            using Inertia = pinocchio::Inertia;

            data.U = Ia.template middleCols<3>(Inertia::LINEAR);
            StU = data.U.template middleRows<3>(Inertia::LINEAR);
            StU.diagonal() += Im;

            pinocchio::internal::PerformStYSInversion<Scalar>::run(StU, data.Dinv);
            data.UDinv.noalias() = data.U * data.Dinv;

            if(update_I)
            {
                Ia.noalias() -= data.UDinv * data.U.transpose();
            }
        }

        template<typename JointModel, typename VectorLike, typename Matrix6Like>
        static std::enable_if_t<is_pinocchio_joint_spherical_v<JointModel>, void>
        calc_aba(JointModel const & /* model */,
                 typename JointModel::JointDataDerived & data,
                 Eigen::MatrixBase<VectorLike> const & Im,
                 Eigen::MatrixBase<Matrix6Like> & Ia,
                 bool const & update_I)
        {
            static Eigen::Matrix<Scalar, 3, 3> StU;

            using Inertia = pinocchio::Inertia;

            data.U = Ia.template block<6,3>(0, Inertia::ANGULAR);
            StU = data.U.template middleRows<3>(Inertia::ANGULAR);
            StU.diagonal() += Im;

            pinocchio::internal::PerformStYSInversion<Scalar>::run(StU, data.Dinv);
            data.UDinv.noalias() = data.U * data.Dinv;

            if (update_I)
            {
                Ia.noalias() -= data.UDinv * data.U.transpose();
            }
        }

        template<typename JointModel, typename VectorLike, typename Matrix6Like>
        static std::enable_if_t<is_pinocchio_joint_spherical_zyx_v<JointModel>, void>
        calc_aba(JointModel const & /* model */,
                 typename JointModel::JointDataDerived & data,
                 Eigen::MatrixBase<VectorLike> const & Im,
                 Eigen::MatrixBase<Matrix6Like> & Ia,
                 bool const & update_I)
        {
            static Eigen::Matrix<Scalar, 3, 3> StU;

            using Inertia = pinocchio::Inertia;

            data.U.noalias() = Ia.template middleCols<3>(Inertia::ANGULAR) * data.S.angularSubspace();
            StU.noalias() = data.S.angularSubspace().transpose() * data.U.template middleRows<3>(Inertia::ANGULAR);
            StU.diagonal() += Im;

            pinocchio::internal::PerformStYSInversion<Scalar>::run(StU, data.Dinv);
            data.UDinv.noalias() = data.U * data.Dinv;

            if (update_I)
            {
                Ia.noalias() -= data.UDinv * data.U.transpose();
            }
        }

        template<typename JointModel, typename VectorLike, typename Matrix6Like>
        static std::enable_if_t<is_pinocchio_joint_freeflyer_v<JointModel>, void>
        calc_aba(JointModel const & /* model */,
                 typename JointModel::JointDataDerived & data,
                 Eigen::MatrixBase<VectorLike> const & Im,
                 Eigen::MatrixBase<Matrix6Like> & Ia,
                 bool const & update_I)
        {
            static Eigen::Matrix<Scalar, 6, 6> StU;

            data.U = Ia;
            StU = Ia;
            StU.diagonal() += Im;

            pinocchio::internal::PerformStYSInversion<Scalar>::run(StU, data.Dinv);
            data.UDinv.noalias() = data.U * data.Dinv;

            if(update_I)
            {
                Ia.noalias() -= data.UDinv * data.U.transpose();
            }
        }

        template<typename JointModel, typename VectorLike, typename Matrix6Like>
        static std::enable_if_t<is_pinocchio_joint_composite_v<JointModel>, void>
        calc_aba(JointModel const & /* model */,
                 typename JointModel::JointDataDerived & data,
                 Eigen::MatrixBase<VectorLike> const & Im,
                 Eigen::MatrixBase<Matrix6Like> & Ia,
                 bool const & update_I)
        {
            static Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> StU;

            data.U.noalias() = Ia * data.S.matrix();
            StU.noalias() = data.S.matrix().transpose() * data.U;
            StU.diagonal() += Im;

            pinocchio::internal::PerformStYSInversion<Scalar>::run(StU, data.Dinv);
            data.UDinv.noalias() = data.U * data.Dinv;

            if(update_I)
            {
                Ia.noalias() -= data.UDinv * data.U.transpose();
            }
        }

        template<typename JointModel, typename VectorLike, typename Matrix6Like>
        static std::enable_if_t< is_pinocchio_joint_mimic_v<JointModel>, void>
        calc_aba(JointModel const & model,
                 typename JointModel::JointDataDerived & data,
                 Eigen::MatrixBase<VectorLike> const & /* Im */,
                 Eigen::MatrixBase<Matrix6Like> & Ia,
                 bool const & update_I)
        {
            // TODO: Not implemented
            model.calc_aba(data.derived(), Ia, update_I);
        }

        template<int axis>
        static int getAxis(pinocchio::JointModelRevoluteTpl<Scalar, Options, axis> const & /* model */)
        {
            return axis;
        }

        template<int axis>
        static int getAxis(pinocchio::JointModelRevoluteUnboundedTpl<Scalar, Options, axis> const & /* model */)
        {
            return axis;
        }

        template<int axis>
        static int getAxis(pinocchio::JointModelPrismaticTpl<Scalar, Options, axis> const & /* model */)
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

        data.v[0].setZero();
        data.a_gf[0] = -model.gravity;
        data.u = tau;

        typedef pinocchio::AbaForwardStep1<Scalar, Options, JointCollectionTpl,
                                           ConfigVectorType, TangentVectorType1> Pass1;
        for (int32_t i = 1; i < model.njoints; ++i)
        {
            Pass1::run(model.joints[i], data.joints[i],
                       typename Pass1::ArgsType(model, data, q.derived(), v.derived()));
            data.f[i] -= fext[i];
        }

        typedef AbaBackwardStep<Scalar,Options,JointCollectionTpl> Pass2;
        for (int32_t i = model.njoints - 1; i > 0; --i)
        {
            Pass2::run(model.joints[i], data.joints[i],
                       typename Pass2::ArgsType(model, data));
        }

        typedef pinocchio::AbaForwardStep2<Scalar,Options,JointCollectionTpl> Pass3;
        for (int32_t i = 1; i < model.njoints; ++i)
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
            jointIndex_t const & i = jmodel.id();
            jointIndex_t const & parent = model.parents[i];
            data.a[i]  = jdata.S() * jmodel.jointVelocitySelector(a) + jdata.c() + (data.v[i] ^ jdata.v());
            if (parent > 0)
            {
                data.a[i] += data.liMi[i].actInv(data.a[parent]);
            }
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
        typedef ForwardKinematicsAccelerationStep<TangentVectorType> Pass;
        data.a[0].setZero();
        for (int32_t i = 1; i < model.njoints; ++i)
        {
            Pass::run(model.joints[i], data.joints[i], typename Pass::ArgsType(model, data, a));
        }
    }

    template<typename JacobianType>
    hresult_t computeJMinvJt(pinocchio::Model const & model,
                             pinocchio::Data & data,
                             Eigen::MatrixBase<JacobianType> const & J,
                             bool_t const & updateDecomposition = true)
    {
        // Compute the Cholesky decomposition of mass matrix M if requested
        if (updateDecomposition)
        {
            pinocchio::cholesky::decompose(model, data);
        }

        // Make sure the decomposition of the mass matrix is valid
        if ((data.Dinv.array() < 0.0).any())
        {
            PRINT_ERROR("The inertia matrix is not strictly positive definite.");
            return hresult_t::ERROR_BAD_INPUT;
        }

        /* Compute sDUiJt := sqrt(D)^-1 * U^-1 * J.T
           - Use row-major for sDUiJt and U to enable vectorization
           - Implement custom cholesky::Uiv to compute all columns at once (faster SIMD)
           - TODO: take advantage of the sparsity of J when multiplying by sqrt(D)^-1 */
        Eigen::Matrix<float64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> sDUiJt = J.transpose();
        Eigen::Matrix<float64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> U = data.U;
        std::vector<int> const & nvt = data.nvSubtree_fromRow;
        for(int k = model.nv - 2; k >= 0; --k)
        {
            sDUiJt.row(k).noalias() -= U.row(k).segment(k + 1, nvt[static_cast<std::size_t>(k)] - 1) *
                sDUiJt.middleRows(k + 1, nvt[static_cast<std::size_t>(k)] - 1);
        }
        sDUiJt.array().colwise() *= data.Dinv.array().sqrt();

        /* Compute JMinvJt := sDUiJt.T * sDUiJt
           - TODO: Take advantage of the sparsity pattern of the jacobian which propagates
             through sDUiJt. Reference: Exploiting Sparsity in Operational-Space Dynamics
             (Figure 10 of http://royfeatherstone.org/papers/sparseOSIM.pdf).
             Each constraint must provide a std::vector of std::pair<start,dim> that are
             dependency blocks. */
        data.JMinvJt.resize(J.rows(), J.rows());
        data.JMinvJt.triangularView<Eigen::Lower>().setZero();
        data.JMinvJt.selfadjointView<Eigen::Lower>().rankUpdate(sDUiJt.transpose());

        return hresult_t::SUCCESS;
    }

    template<typename RhsType>
    inline auto solveJMinvJtv(pinocchio::Data & data,
                              Eigen::MatrixBase<RhsType> const & v,
                              bool_t const & updateDecomposition = true)
    {
        // Compute Cholesky decomposition of JMinvJt
        if (updateDecomposition)
        {
            data.llt_JMinvJt.compute(data.JMinvJt);
        }

        // Solve the linear system
        return data.llt_JMinvJt.solve(v);
    }
}
}

#endif //end of JIMINY_ALGORITHMS_MOTOR_INERTIA_H
