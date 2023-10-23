#ifndef JIMINY_PINOCCHIO_H
#define JIMINY_PINOCCHIO_H

#include <chrono>
#include <type_traits>

#include "jiminy/core/macros.h"
#include "jiminy/core/types.h"


namespace jiminy
{
    hresult_t getJointNameFromPositionIdx(
        const pinocchio::Model & model, const jointIndex_t & idIn, std::string & jointNameOut);

    hresult_t getJointNameFromVelocityIdx(
        const pinocchio::Model & model, const jointIndex_t & idIn, std::string & jointNameOut);

    hresult_t getJointTypeFromIdx(
        const pinocchio::Model & model, const jointIndex_t & idIn, joint_t & jointTypeOut);

    hresult_t getJointTypePositionSuffixes(const joint_t & jointTypeIn,
                                           std::vector<std::string> & jointTypeSuffixesOut);

    hresult_t getJointTypeVelocitySuffixes(const joint_t & jointTypeIn,
                                           std::vector<std::string> & jointTypeSuffixesOut);

    hresult_t getFrameIdx(
        const pinocchio::Model & model, const std::string & frameName, frameIndex_t & frameIdx);
    hresult_t getFramesIdx(const pinocchio::Model & model,
                           const std::vector<std::string> & framesNames,
                           std::vector<frameIndex_t> & framesIdx);

    hresult_t getJointModelIdx(const pinocchio::Model & model,
                               const std::string & jointName,
                               jointIndex_t & jointModelIdx);
    hresult_t getJointsModelIdx(const pinocchio::Model & model,
                                const std::vector<std::string> & jointsNames,
                                std::vector<jointIndex_t> & jointsModelIdx);

    hresult_t getJointPositionIdx(const pinocchio::Model & model,
                                  const std::string & jointName,
                                  std::vector<int32_t> & jointPositionIdx);
    hresult_t getJointPositionIdx(const pinocchio::Model & model,
                                  const std::string & jointName,
                                  int32_t & jointPositionFirstIdx);
    hresult_t getJointsPositionIdx(const pinocchio::Model & model,
                                   const std::vector<std::string> & jointsNames,
                                   std::vector<int32_t> & jointsPositionIdx,
                                   const bool_t & firstJointIdxOnly = false);

    hresult_t getJointVelocityIdx(const pinocchio::Model & model,
                                  const std::string & jointName,
                                  std::vector<int32_t> & jointVelocityIdx);
    hresult_t getJointVelocityIdx(const pinocchio::Model & model,
                                  const std::string & jointName,
                                  int32_t & jointVelocityFirstIdx);
    hresult_t getJointsVelocityIdx(const pinocchio::Model & model,
                                   const std::vector<std::string> & jointsNames,
                                   std::vector<int32_t> & jointsVelocityIdx,
                                   const bool_t & firstJointIdxOnly = false);

    hresult_t isPositionValid(const pinocchio::Model & model,
                              const Eigen::VectorXd & position,
                              bool_t & isValid,
                              const float64_t & tol);

    hresult_t insertFlexibilityBeforeJointInModel(pinocchio::Model & modelInOut,
                                                  const std::string & childJointNameIn,
                                                  const std::string & newJointNameIn);

    hresult_t insertFlexibilityAtFixedFrameInModel(pinocchio::Model & modelInOut,
                                                   const std::string & frameNameIn);

    hresult_t interpolate(const pinocchio::Model & modelIn,
                          const Eigen::VectorXd & timesIn,
                          const Eigen::MatrixXd & positionsIn,
                          const Eigen::VectorXd & timesOut,
                          Eigen::MatrixXd & positionsOut);

    /// \brief Convert a force expressed in the global frame of a specific frame to its parent
    ///        joint frame.
    ///
    /// \param[in] model Pinocchio model.
    /// \param[in] data Pinocchio data.
    /// \param[in] frameIdx Index of the frame.
    /// \param[in] fextInGlobal Force in the global frame to be converted.
    ///
    /// \return Force in the parent joint local frame.
    pinocchio::Force convertForceGlobalFrameToJoint(const pinocchio::Model & model,
                                                    const pinocchio::Data & data,
                                                    const frameIndex_t & frameIdx,
                                                    const pinocchio::Force & fextInGlobal);

    hresult_t buildGeomFromUrdf(const pinocchio::Model & model,
                                const std::string & filename,
                                const pinocchio::GeometryType & type,
                                pinocchio::GeometryModel & geomModel,
                                const std::vector<std::string> & packageDirs,
                                const bool_t & loadMeshes = true,
                                const bool_t & makeConvexMeshes = false);

    hresult_t buildModelsFromUrdf(
        const std::string & urdfPath,
        const bool_t & hasFreeflyer,
        const std::vector<std::string> & meshPackageDirs,
        pinocchio::Model & pncModel,
        pinocchio::GeometryModel & collisionModel,
        std::optional<std::reference_wrapper<pinocchio::GeometryModel>> visualModel = std::nullopt,
        const bool_t & loadVisualMeshes = false);
}

#endif  // JIMINY_PINOCCHIO_H
