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

    hresult_t getBodyIdx(pinocchio::Model const & model,
                         std::string      const & bodyName,
                         frameIndex_t           & bodyIdx);
    hresult_t getBodiesIdx(pinocchio::Model          const & model,
                           std::vector<std::string>  const & bodiesNames,
                           std::vector<frameIndex_t>       & bodiesIdx);

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

    hresult_t insertFlexibilityAtFixedFrameInModel(pinocchio::Model         & modelInOut,
                                                   std::string        const & frameNameIn,
                                                   pinocchio::Inertia const & childBodyInertiaIn,
                                                   std::string        const & newJointNameIn);

    hresult_t interpolate(pinocchio::Model const & modelIn,
                          vectorN_t        const & timesIn,
                          matrixN_t        const & positionsIn,
                          vectorN_t        const & timesOut,
                          matrixN_t              & positionsOut);

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
                                  boost::optional<pinocchio::GeometryModel &> visualModel = boost::none,
                                  bool_t const & loadVisualMeshes = false);
}

#endif  // JIMINY_PINOCCHIO_H
