#ifndef JIMINY_PINOCCHIO_H
#define JIMINY_PINOCCHIO_H

#include <chrono>
#include <type_traits>

#include "jiminy/core/Macros.h"
#include "jiminy/core/Types.h"


namespace jiminy
{
    hresult_t getJointNameFromPositionIdx(pinocchio::Model const & model,
                                          jointIndex_t     const & idIn,
                                          std::string            & jointNameOut);

    hresult_t getJointNameFromVelocityIdx(pinocchio::Model const & model,
                                          jointIndex_t     const & idIn,
                                          std::string            & jointNameOut);

    hresult_t getJointTypeFromIdx(pinocchio::Model const & model,
                                  jointIndex_t     const & idIn,
                                  joint_t                & jointTypeOut);

    hresult_t getJointTypePositionSuffixes(joint_t                  const & jointTypeIn,
                                           std::vector<std::string>       & jointTypeSuffixesOut);

    hresult_t getJointTypeVelocitySuffixes(joint_t                  const & jointTypeIn,
                                           std::vector<std::string>       & jointTypeSuffixesOut);

    hresult_t getFrameIdx(pinocchio::Model const & model,
                          std::string      const & frameName,
                          frameIndex_t           & frameIdx);
    hresult_t getFramesIdx(pinocchio::Model          const & model,
                           std::vector<std::string>  const & framesNames,
                           std::vector<frameIndex_t>       & framesIdx);

    hresult_t getJointModelIdx(pinocchio::Model const & model,
                               std::string      const & jointName,
                               jointIndex_t           & jointModelIdx);
    hresult_t getJointsModelIdx(pinocchio::Model          const & model,
                                std::vector<std::string>  const & jointsNames,
                                std::vector<jointIndex_t>       & jointsModelIdx);

    hresult_t getJointPositionIdx(pinocchio::Model     const & model,
                                  std::string          const & jointName,
                                  std::vector<int32_t>       & jointPositionIdx);
    hresult_t getJointPositionIdx(pinocchio::Model const & model,
                                  std::string      const & jointName,
                                  int32_t                & jointPositionFirstIdx);
    hresult_t getJointsPositionIdx(pinocchio::Model         const & model,
                                   std::vector<std::string> const & jointsNames,
                                   std::vector<int32_t>           & jointsPositionIdx,
                                   bool_t                   const & firstJointIdxOnly = false);

    hresult_t getJointVelocityIdx(pinocchio::Model     const & model,
                                  std::string          const & jointName,
                                  std::vector<int32_t>       & jointVelocityIdx);
    hresult_t getJointVelocityIdx(pinocchio::Model const & model,
                                  std::string      const & jointName,
                                  int32_t                & jointVelocityFirstIdx);
    hresult_t getJointsVelocityIdx(pinocchio::Model         const & model,
                                   std::vector<std::string> const & jointsNames,
                                   std::vector<int32_t>           & jointsVelocityIdx,
                                   bool_t                   const & firstJointIdxOnly = false);

    hresult_t isPositionValid(pinocchio::Model const & model,
                              vectorN_t        const & position,
                              bool_t                 & isValid,
                              float64_t        const & tol);

    hresult_t insertFlexibilityBeforeJointInModel(pinocchio::Model       & modelInOut,
                                                  std::string      const & childJointNameIn,
                                                  std::string      const & newJointNameIn);

    hresult_t insertFlexibilityAtFixedFrameInModel(pinocchio::Model       & modelInOut,
                                                   std::string      const & frameNameIn);

    hresult_t interpolate(pinocchio::Model const & modelIn,
                          vectorN_t        const & timesIn,
                          matrixN_t        const & positionsIn,
                          vectorN_t        const & timesOut,
                          matrixN_t              & positionsOut);

    template<typename Scalar, typename Vector3Like, typename Matrix3Like>
    void Jlog3(const Scalar & theta,
               const Eigen::MatrixBase<Vector3Like> & log,
               const Eigen::MatrixBase<Matrix3Like> & Jlog)
    {
        PINOCCHIO_ASSERT_MATRIX_SPECIFIC_SIZE(Vector3Like,  log, 3, 1);
        PINOCCHIO_ASSERT_MATRIX_SPECIFIC_SIZE(Matrix3Like, Jlog, 3, 3);

        Matrix3Like & Jlog_ = PINOCCHIO_EIGEN_CONST_CAST(Matrix3Like, Jlog);

        const Scalar theta2 = theta * theta;

        Scalar alpha;
        if(theta < pinocchio::TaylorSeriesExpansion<Scalar>::template precision<3>())
        {
            alpha = Scalar(1) / Scalar(12) + theta2 / Scalar(720);
        }
        else
        {
            Scalar ct,st; pinocchio::SINCOS(theta, &st, &ct);
            const Scalar st_1mct = st / (Scalar(1) - ct);
            alpha = Scalar(1) / theta2 - st_1mct / (Scalar(2) * theta);
        }
        const Scalar diag_value = Scalar(1) - alpha * theta2;

        Jlog_.template triangularView<Eigen::Lower>().setZero();
        Jlog_.template selfadjointView<Eigen::Lower>().rankUpdate(log, alpha);
        Jlog_.template triangularView<Eigen::StrictlyUpper>() = Jlog_.transpose();
        Jlog_.diagonal().array() += diag_value;
        pinocchio::addSkew(log / Scalar(2), Jlog_);
    }

    template<typename Scalar, typename Vector3Like1, typename Matrix3Like1, typename Vector3Like2, typename Matrix3Like2>
    void dJlog3(const Scalar & theta,
                const Eigen::MatrixBase<Vector3Like1> & log,
                const Eigen::MatrixBase<Matrix3Like1> & Jlog,
                const Eigen::MatrixBase<Vector3Like2> & v,
                const Eigen::MatrixBase<Matrix3Like2> & dJlog)
    {
        PINOCCHIO_ASSERT_MATRIX_SPECIFIC_SIZE(Vector3Like1,   log, 3, 1);
        PINOCCHIO_ASSERT_MATRIX_SPECIFIC_SIZE(Matrix3Like1,  Jlog, 3, 3);
        PINOCCHIO_ASSERT_MATRIX_SPECIFIC_SIZE(Vector3Like2,     v, 3, 1);
        PINOCCHIO_ASSERT_MATRIX_SPECIFIC_SIZE(Matrix3Like2, dJlog, 3, 3);

        Matrix3Like2 & dJlog_ = PINOCCHIO_EIGEN_CONST_CAST(Matrix3Like2, dJlog);

        const Vector3Like1 dlog = Jlog * v;
        const Scalar theta2 = theta * theta;
        const Scalar dtheta2 = Scalar(2) * log.dot(v);

        Scalar alpha, dalpha;
        if(theta < pinocchio::TaylorSeriesExpansion<Scalar>::template precision<5>())
        {
            dalpha = dtheta2 * (Scalar(1) - (Scalar(3) / Scalar(42)) * theta2) / (Scalar(720) - Scalar(60) * theta2);
        }
        if(theta < pinocchio::TaylorSeriesExpansion<Scalar>::template precision<3>())
        {
            alpha = Scalar(1) / Scalar(12) + theta2 / Scalar(720);
        }
        else
        {
            Scalar ct,st; pinocchio::SINCOS(theta, &st, &ct);
            const Scalar st_1mct = st / (Scalar(1) - ct);
            alpha = Scalar(1) / theta2 - st_1mct / (Scalar(2) * theta);
            if(theta >= pinocchio::TaylorSeriesExpansion<Scalar>::template precision<5>())
            {
                dalpha = (dtheta2 / theta) * ((st + theta) / (Scalar(4) * (Scalar(1) - ct)) - Scalar(1) / theta) / theta2;
            }
        }
        const Scalar diag_value = - dalpha * theta2 - alpha * dtheta2;

        dJlog_.template triangularView<Eigen::Lower>().setZero();
        dJlog_.template selfadjointView<Eigen::Lower>().rankUpdate(log, dalpha);
        dJlog_.template selfadjointView<Eigen::Lower>().rankUpdate(log, dlog, alpha);
        dJlog_.template triangularView<Eigen::StrictlyUpper>() = dJlog_.transpose();
        dJlog_.diagonal().array() += diag_value;
        pinocchio::addSkew(dlog / Scalar(2), dJlog_);
    }

    /// \brief Convert a force expressed in the global frame of a specific frame to its parent joint frame.
    ///
    /// \param[in] model        Pinocchio model.
    /// \param[in] data         Pinocchio data.
    /// \param[in] frameIdx     Id of the frame.
    /// \param[in] fextInGlobal Force in the global frame to be converted.
    /// \return Force in the parent joint local frame.
    pinocchio::Force convertForceGlobalFrameToJoint(pinocchio::Model const & model,
                                                    pinocchio::Data  const & data,
                                                    frameIndex_t     const & frameIdx,
                                                    pinocchio::Force const & fextInGlobal);

    hresult_t buildGeomFromUrdf(pinocchio::Model         const & model,
                                std::string              const & filename,
                                pinocchio::GeometryType  const & type,
                                pinocchio::GeometryModel       & geomModel,
                                std::vector<std::string> const & packageDirs,
                                bool_t                   const & loadMeshes = true,
                                bool_t                   const & makeConvexMeshes = false);

    hresult_t buildModelsFromUrdf(std::string const & urdfPath,
                                  bool_t const & hasFreeflyer,
                                  std::vector<std::string> const & meshPackageDirs,
                                  pinocchio::Model & pncModel,
                                  pinocchio::GeometryModel & collisionModel,
                                  std::optional<std::reference_wrapper<pinocchio::GeometryModel> > visualModel = std::nullopt,
                                  bool_t const & loadVisualMeshes = false);
}

#endif  // JIMINY_PINOCCHIO_H
