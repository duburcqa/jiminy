#ifndef JIMINY_PINOCCHIO_H
#define JIMINY_PINOCCHIO_H

#include <optional>

#include "pinocchio/multibody/fcl.hpp"    // `pinocchio::GeometryType`
#include "pinocchio/multibody/frame.hpp"  // `pinocchio::FrameType`

#include "jiminy/core/fwd.h"


namespace jiminy
{
    JointModelType JIMINY_DLLAPI getJointType(const pinocchio::JointModel & jointModel) noexcept;

    JointModelType JIMINY_DLLAPI getJointTypeFromIndex(const pinocchio::Model & model,
                                                       pinocchio::JointIndex jointIndex);

    std::string JIMINY_DLLAPI getJointNameFromPositionIndex(
        const pinocchio::Model & model, pinocchio::JointIndex jointPositionIndex);

    std::string JIMINY_DLLAPI getJointNameFromVelocityIndex(
        const pinocchio::Model & model, pinocchio::JointIndex jointVelocityIndex);

    // FIXME: Return std::span<std::string_view> and add constexpr specifier when moving to C++20
    std::vector<std::string_view> getJointTypePositionSuffixes(JointModelType jointType);

    // FIXME: Return std::span<std::string_view> and add constexpr specifier when moving to C++20
    std::vector<std::string_view> getJointTypeVelocitySuffixes(JointModelType jointType);

    pinocchio::FrameIndex JIMINY_DLLAPI getFrameIndex(
        const pinocchio::Model & model,
        const std::string & frameName,
        pinocchio::FrameType frameType = static_cast<pinocchio::FrameType>(
            pinocchio::JOINT | pinocchio::FIXED_JOINT | pinocchio::BODY | pinocchio::OP_FRAME |
            pinocchio::SENSOR));
    std::vector<pinocchio::FrameIndex> JIMINY_DLLAPI getFrameIndices(
        const pinocchio::Model & model, const std::vector<std::string> & frameNames);

    pinocchio::JointIndex JIMINY_DLLAPI getJointIndex(const pinocchio::Model & model,
                                                      const std::string & jointName);
    std::vector<pinocchio::JointIndex> JIMINY_DLLAPI getJointIndices(
        const pinocchio::Model & model, const std::vector<std::string> & jointNames);

    /// \brief Return only the first position index associated with the joint regardless its number
    ///        of degrees of freedom.
    Eigen::Index JIMINY_DLLAPI getJointPositionFirstIndex(const pinocchio::Model & model,
                                                          const std::string & jointName);
    /// \brief Return all the position indices associated with the joint.
    std::vector<Eigen::Index> JIMINY_DLLAPI getJointPositionIndices(const pinocchio::Model & model,
                                                                    const std::string & jointName);
    std::vector<Eigen::Index> JIMINY_DLLAPI getJointsPositionIndices(
        const pinocchio::Model & model,
        const std::vector<std::string> & jointNames,
        bool onlyFirstIndex = false);

    Eigen::Index JIMINY_DLLAPI getJointVelocityFirstIndex(const pinocchio::Model & model,
                                                          const std::string & jointName);
    std::vector<Eigen::Index> JIMINY_DLLAPI getJointVelocityIndices(const pinocchio::Model & model,
                                                                    const std::string & jointName);
    std::vector<Eigen::Index> JIMINY_DLLAPI getJointsVelocityIndices(
        const pinocchio::Model & model,
        const std::vector<std::string> & jointNames,
        bool onlyFirstIndex = false);

    bool JIMINY_DLLAPI isPositionValid(
        const pinocchio::Model & model,
        const Eigen::VectorXd & q,
        double tolAbs = Eigen::NumTraits<double>::dummy_precision());

    void swapJointIndices(pinocchio::Model & model,
                    pinocchio::JointIndex jointIndex1,
                    pinocchio::JointIndex jointIndex2);

    void addFlexibilityJointBeforeMechanicalJoint(pinocchio::Model & model,
                                                  const std::string & childJointName,
                                                  const std::string & newJointName);

    void addFlexibilityJointAtFixedFrame(pinocchio::Model & model, const std::string & frameName);

    void addBacklashJointAfterMechanicalJoint(pinocchio::Model & model,
                                              const std::string & parentJointName,
                                              const std::string & newJointName);

    Eigen::MatrixXd JIMINY_DLLAPI interpolatePositions(const pinocchio::Model & model,
                                                       const Eigen::VectorXd & timesIn,
                                                       const Eigen::MatrixXd & positionsIn,
                                                       const Eigen::VectorXd & timesOut);

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

    pinocchio::GeometryModel buildGeometryModelFromUrdf(
        const pinocchio::Model & model,
        const std::string & filename,
        const pinocchio::GeometryType & type,
        const std::vector<std::string> & meshPackageDirs,
        bool loadMeshes = true,
        bool makeConvexMeshes = false);

    void JIMINY_DLLAPI buildMultipleModelsFromUrdf(
        const std::string & urdfPath,
        bool hasFreeflyer,
        const std::vector<std::string> & meshPackageDirs,
        pinocchio::Model & pinocchioModel,
        pinocchio::GeometryModel & collisionModel,
        std::optional<std::reference_wrapper<pinocchio::GeometryModel>> visualModel = std::nullopt,
        bool loadVisualMeshes = false);
}

#endif  // JIMINY_PINOCCHIO_H
