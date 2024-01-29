#ifndef JIMINY_PINOCCHIO_H
#define JIMINY_PINOCCHIO_H

#include <optional>

#include "pinocchio/multibody/fcl.hpp"  // `pinocchio::GeometryType`

#include "jiminy/core/fwd.h"


namespace jiminy
{
    JointModelType getJointType(const pinocchio::JointModel & jointModel) noexcept;

    hresult_t getJointTypeFromIdx(const pinocchio::Model & model,
                                  pinocchio::JointIndex jointModelIdx,
                                  JointModelType & jointTypeOut);

    hresult_t getJointNameFromPositionIdx(const pinocchio::Model & model,
                                          pinocchio::JointIndex jointModelIdx,
                                          std::string & jointNameOut);

    hresult_t getJointNameFromVelocityIdx(const pinocchio::Model & model,
                                          pinocchio::JointIndex jointModelIdx,
                                          std::string & jointNameOut);

    hresult_t getJointTypePositionSuffixes(JointModelType jointTypeIn,
                                           std::vector<std::string_view> & jointTypeSuffixesOut);

    hresult_t getJointTypeVelocitySuffixes(JointModelType jointTypeIn,
                                           std::vector<std::string_view> & jointTypeSuffixesOut);

    hresult_t getFrameIdx(const pinocchio::Model & model,
                          const std::string & frameName,
                          pinocchio::FrameIndex & frameIdx);
    hresult_t getFramesIdx(const pinocchio::Model & model,
                           const std::vector<std::string> & framesNames,
                           std::vector<pinocchio::FrameIndex> & framesIdx);

    hresult_t getJointModelIdx(const pinocchio::Model & model,
                               const std::string & jointName,
                               pinocchio::JointIndex & jointModelIdx);
    hresult_t getJointsModelIdx(const pinocchio::Model & model,
                                const std::vector<std::string> & jointsNames,
                                std::vector<pinocchio::JointIndex> & jointsModelIdx);

    hresult_t getJointPositionIdx(const pinocchio::Model & model,
                                  const std::string & jointName,
                                  std::vector<Eigen::Index> & jointPositionIdx);
    hresult_t getJointPositionIdx(const pinocchio::Model & model,
                                  const std::string & jointName,
                                  Eigen::Index & jointPositionFirstIdx);
    hresult_t getJointsPositionIdx(const pinocchio::Model & model,
                                   const std::vector<std::string> & jointsNames,
                                   std::vector<Eigen::Index> & jointsPositionIdx,
                                   bool firstJointIdxOnly = false);

    hresult_t getJointVelocityIdx(const pinocchio::Model & model,
                                  const std::string & jointName,
                                  std::vector<Eigen::Index> & jointVelocityIdx);
    hresult_t getJointVelocityIdx(const pinocchio::Model & model,
                                  const std::string & jointName,
                                  Eigen::Index & jointVelocityFirstIdx);
    hresult_t getJointsVelocityIdx(const pinocchio::Model & model,
                                   const std::vector<std::string> & jointsNames,
                                   std::vector<Eigen::Index> & jointsVelocityIdx,
                                   bool firstJointIdxOnly = false);

    bool JIMINY_DLLAPI isPositionValid(
        const pinocchio::Model & model,
        const Eigen::VectorXd & q,
        double tolAbs = Eigen::NumTraits<double>::dummy_precision());

    void swapJoints(pinocchio::Model & model,
                    pinocchio::JointIndex jointIndex1,
                    pinocchio::JointIndex jointIndex2);

    hresult_t insertFlexibilityBeforeJointInModel(pinocchio::Model & modelInOut,
                                                  const std::string & childJointNameIn,
                                                  const std::string & newJointNameIn);

    hresult_t insertFlexibilityAtFixedFrameInModel(pinocchio::Model & modelInOut,
                                                   const std::string & frameNameIn);

    hresult_t JIMINY_DLLAPI interpolate(const pinocchio::Model & modelIn,
                                        const Eigen::VectorXd & timesIn,
                                        const Eigen::MatrixXd & positionsIn,
                                        const Eigen::VectorXd & timesOut,
                                        Eigen::MatrixXd & positionsOut);

    /// \brief Translate a force expressed at the given fixed frame to its parent joint frame.
    ///
    /// \param[in] model Pinocchio model.
    /// \param[in] data Pinocchio data.
    /// \param[in] frameIdx Index of the frame.
    /// \param[in] fextInGlobal Force in the global frame to be converted.
    ///
    /// \return Force in the parent joint local frame.
    pinocchio::Force convertForceGlobalFrameToJoint(const pinocchio::Model & model,
                                                    const pinocchio::Data & data,
                                                    pinocchio::FrameIndex frameIndex,
                                                    const pinocchio::Force & aFf);

    hresult_t buildGeomFromUrdf(const pinocchio::Model & model,
                                const std::string & filename,
                                const pinocchio::GeometryType & type,
                                pinocchio::GeometryModel & geomModel,
                                const std::vector<std::string> & packageDirs,
                                bool loadMeshes = true,
                                bool makeConvexMeshes = false);

    hresult_t JIMINY_DLLAPI buildModelsFromUrdf(
        const std::string & urdfPath,
        bool hasFreeflyer,
        const std::vector<std::string> & meshPackageDirs,
        pinocchio::Model & pncModel,
        pinocchio::GeometryModel & collisionModel,
        std::optional<std::reference_wrapper<pinocchio::GeometryModel>> visualModel = std::nullopt,
        bool loadVisualMeshes = false);
}

#endif  // JIMINY_PINOCCHIO_H
