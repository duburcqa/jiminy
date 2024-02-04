#ifndef JIMINY_PINOCCHIO_H
#define JIMINY_PINOCCHIO_H

#include <optional>

#include "pinocchio/multibody/fcl.hpp"  // `pinocchio::GeometryType`

#include "jiminy/core/fwd.h"


namespace jiminy
{
    JointModelType JIMINY_DLLAPI getJointType(const pinocchio::JointModel & jointModel) noexcept;

    hresult_t JIMINY_DLLAPI getJointTypeFromIndex(const pinocchio::Model & model,
                                                  pinocchio::JointIndex jointIndex,
                                                  JointModelType & jointType);

    hresult_t JIMINY_DLLAPI getJointNameFromPositionIndex(const pinocchio::Model & model,
                                                          pinocchio::JointIndex jointPositionIndex,
                                                          std::string & jointName);

    hresult_t JIMINY_DLLAPI getJointNameFromVelocityIndex(const pinocchio::Model & model,
                                                          pinocchio::JointIndex jointVelocityIndex,
                                                          std::string & jointName);

    hresult_t getJointTypePositionSuffixes(JointModelType jointType,
                                           std::vector<std::string_view> & jointPositionSuffixes);

    hresult_t getJointTypeVelocitySuffixes(JointModelType jointType,
                                           std::vector<std::string_view> & jointVelocitySuffixes);

    hresult_t JIMINY_DLLAPI getFrameIndex(const pinocchio::Model & model,
                                          const std::string & frameName,
                                          pinocchio::FrameIndex & frameIndex);
    hresult_t JIMINY_DLLAPI getFrameIndices(const pinocchio::Model & model,
                                            const std::vector<std::string> & frameNames,
                                            std::vector<pinocchio::FrameIndex> & frameIndices);

    hresult_t JIMINY_DLLAPI getJointIndex(const pinocchio::Model & model,
                                          const std::string & jointName,
                                          pinocchio::JointIndex & jointIndex);
    hresult_t JIMINY_DLLAPI getJointIndices(
        const pinocchio::Model & model,
        const std::vector<std::string> & jointNames,
        std::vector<pinocchio::JointIndex> & jointModelIndices);

    hresult_t JIMINY_DLLAPI getJointPositionFirstIndex(const pinocchio::Model & model,
                                                       const std::string & jointName,
                                                       Eigen::Index & jointPositionFirstIndex);
    hresult_t JIMINY_DLLAPI getJointPositionIndices(
        const pinocchio::Model & model,
        const std::string & jointName,
        std::vector<Eigen::Index> & jointPositionIndices);
    hresult_t JIMINY_DLLAPI getJointsPositionIndices(
        const pinocchio::Model & model,
        const std::vector<std::string> & jointNames,
        std::vector<Eigen::Index> & jointsPositionIndices,
        bool onlyFirstIndex = false);

    hresult_t JIMINY_DLLAPI getJointVelocityFirstIndex(const pinocchio::Model & model,
                                                       const std::string & jointName,
                                                       Eigen::Index & jointVelocityFirstIndex);
    hresult_t JIMINY_DLLAPI getJointVelocityIndices(
        const pinocchio::Model & model,
        const std::string & jointName,
        std::vector<Eigen::Index> & jointVelocityIndices);
    hresult_t JIMINY_DLLAPI getJointsVelocityIndices(
        const pinocchio::Model & model,
        const std::vector<std::string> & jointNames,
        std::vector<Eigen::Index> & jointsVelocityIndices,
        bool onlyFirstIndex = false);

    bool JIMINY_DLLAPI isPositionValid(
        const pinocchio::Model & model,
        const Eigen::VectorXd & q,
        double tolAbs = Eigen::NumTraits<double>::dummy_precision());

    void swapJoints(pinocchio::Model & model,
                    pinocchio::JointIndex jointIndex1,
                    pinocchio::JointIndex jointIndex2);

    hresult_t insertFlexibilityBeforeJointInModel(pinocchio::Model & model,
                                                  const std::string & childJointName,
                                                  const std::string & newJointName);

    hresult_t insertFlexibilityAtFixedFrameInModel(pinocchio::Model & model,
                                                   const std::string & frameName);

    hresult_t JIMINY_DLLAPI interpolatePositions(const pinocchio::Model & model,
                                                 const Eigen::VectorXd & timesIn,
                                                 const Eigen::MatrixXd & positionsIn,
                                                 const Eigen::VectorXd & timesOut,
                                                 Eigen::MatrixXd & positionsOut);

    /// \brief Translate a force expressed at the given fixed frame to its parent joint frame.
    ///
    /// \param[in] model Pinocchio model.
    /// \param[in] data Pinocchio data.
    /// \param[in] frameIndex Index of the frame.
    /// \param[in] fextInGlobal Force in the global frame to be converted.
    ///
    /// \return Force in the parent joint local frame.
    pinocchio::Force convertForceGlobalFrameToJoint(const pinocchio::Model & model,
                                                    const pinocchio::Data & data,
                                                    pinocchio::FrameIndex frameIndex,
                                                    const pinocchio::Force & fextInGlobal);

    hresult_t buildGeometryModelFromUrdf(const pinocchio::Model & model,
                                         const std::string & filename,
                                         const pinocchio::GeometryType & type,
                                         pinocchio::GeometryModel & geomModel,
                                         const std::vector<std::string> & meshPackageDirs,
                                         bool loadMeshes = true,
                                         bool makeConvexMeshes = false);

    hresult_t JIMINY_DLLAPI buildMultipleModelsFromUrdf(
        const std::string & urdfPath,
        bool hasFreeflyer,
        const std::vector<std::string> & meshPackageDirs,
        pinocchio::Model & pinocchioModel,
        pinocchio::GeometryModel & collisionModel,
        std::optional<std::reference_wrapper<pinocchio::GeometryModel>> visualModel = std::nullopt,
        bool loadVisualMeshes = false);
}

#endif  // JIMINY_PINOCCHIO_H
