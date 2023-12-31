#include <numeric>

#include "pinocchio/parsers/urdf.hpp"  // `pinocchio::urdf::buildGeom`, `pinocchio::urdf::buildModel`
#include "pinocchio/spatial/se3.hpp"         // `pinocchio::SE3`
#include "pinocchio/spatial/force.hpp"       // `pinocchio::Force`
#include "pinocchio/spatial/inertia.hpp"     // `pinocchio::Inertia`
#include "pinocchio/multibody/model.hpp"     // `pinocchio::Model`
#include "pinocchio/multibody/geometry.hpp"  // `pinocchio::GeometryModel`
#include "pinocchio/multibody/data.hpp"      // `pinocchio::Data`
#include "pinocchio/multibody/visitor.hpp"   // `pinocchio::fusion::JointUnaryVisitorBase`
#include "pinocchio/multibody/joint/joint-model-base.hpp"  // `pinocchio::JointModelBase`
#include "pinocchio/algorithm/joint-configuration.hpp"     // `pinocchio::isNormalized`

#include "hpp/fcl/mesh_loader/loader.h"
#include "hpp/fcl/BVH/BVH_model.h"

#include "jiminy/core/telemetry/fwd.h"  // `LogData`
#include "jiminy/core/utilities/helpers.h"
#include "jiminy/core/utilities/pinocchio.h"


namespace jiminy
{
    hresult_t getJointNameFromPositionIdx(
        const pinocchio::Model & model, int32_t idx, std::string & jointNameOut)
    {
        // Iterate over all joints
        for (pinocchio::JointIndex i = 0; i < static_cast<pinocchio::JointIndex>(model.njoints);
             ++i)
        {
            // Get joint starting and ending index in position vector
            const int32_t startIdx = model.joints[i].idx_q();
            const int32_t endIdx = startIdx + model.joints[i].nq();

            // If idx is between start and end, we found the joint we were looking for
            if (startIdx <= idx && idx < endIdx)
            {
                jointNameOut = model.names[i];
                return hresult_t::SUCCESS;
            }
        }

        PRINT_ERROR("Position index out of range.");
        return hresult_t::ERROR_BAD_INPUT;
    }

    hresult_t getJointNameFromVelocityIdx(
        const pinocchio::Model & model, int32_t idx, std::string & jointNameOut)
    {
        // Iterate over all joints
        for (pinocchio::JointIndex i = 0; i < static_cast<pinocchio::JointIndex>(model.njoints);
             ++i)
        {
            // Get joint starting and ending index in velocity vector
            const int32_t startIdx = model.joints[i].idx_v();
            const int32_t endIdx = startIdx + model.joints[i].nv();

            // If idx is between start and end, we found the joint we were looking for
            if (startIdx <= idx && idx < endIdx)
            {
                jointNameOut = model.names[i];
                return hresult_t::SUCCESS;
            }
        }

        PRINT_ERROR("Velocity index out of range.");
        return hresult_t::ERROR_BAD_INPUT;
    }

    struct getJointTypeAlgo : public pinocchio::fusion::JointUnaryVisitorBase<getJointTypeAlgo>
    {
        typedef boost::fusion::vector<JointModelType & /* jointType */> ArgsType;

        template<typename JointModel>
        static void algo(const pinocchio::JointModelBase<JointModel> & model,
                         JointModelType & jointType)
        {
            jointType = getJointType(model.derived());
        }

        template<typename JointModel>
        static std::enable_if_t<is_pinocchio_joint_freeflyer_v<JointModel>, JointModelType>
        getJointType(const JointModel &)
        {
            return JointModelType::FREE;
        }

        template<typename JointModel>
        static std::enable_if_t<is_pinocchio_joint_spherical_v<JointModel> ||
                                    is_pinocchio_joint_spherical_zyx_v<JointModel>,
                                JointModelType>
        getJointType(const JointModel &)
        {
            return JointModelType::SPHERICAL;
        }

        template<typename JointModel>
        static std::enable_if_t<is_pinocchio_joint_translation_v<JointModel>, JointModelType>
        getJointType(const JointModel &)
        {
            return JointModelType::TRANSLATION;
        }

        template<typename JointModel>
        static std::enable_if_t<is_pinocchio_joint_planar_v<JointModel>, JointModelType>
        getJointType(const JointModel &)
        {
            return JointModelType::PLANAR;
        }

        template<typename JointModel>
        static std::enable_if_t<is_pinocchio_joint_prismatic_v<JointModel> ||
                                    is_pinocchio_joint_prismatic_unaligned_v<JointModel>,
                                JointModelType>
        getJointType(const JointModel &)
        {
            return JointModelType::LINEAR;
        }

        template<typename JointModel>
        static std::enable_if_t<is_pinocchio_joint_revolute_v<JointModel> ||
                                    is_pinocchio_joint_revolute_unaligned_v<JointModel>,
                                JointModelType>
        getJointType(const JointModel &)
        {
            return JointModelType::ROTARY;
        }

        template<typename JointModel>
        static std::enable_if_t<is_pinocchio_joint_revolute_unbounded_v<JointModel> ||
                                    is_pinocchio_joint_revolute_unbounded_unaligned_v<JointModel>,
                                JointModelType>
        getJointType(const JointModel &)
        {
            return JointModelType::ROTARY_UNBOUNDED;
        }

        template<typename JointModel>
        static std::enable_if_t<is_pinocchio_joint_mimic_v<JointModel> ||
                                    is_pinocchio_joint_composite_v<JointModel>,
                                JointModelType>
        getJointType(const JointModel &)
        {
            return JointModelType::UNSUPPORTED;
        }
    };

    JointModelType getJointType(const pinocchio::JointModel & jointModel) noexcept
    {
        JointModelType jointTypeOut{JointModelType::UNSUPPORTED};
        getJointTypeAlgo::run(jointModel, typename getJointTypeAlgo::ArgsType(jointTypeOut));
        return jointTypeOut;
    }

    hresult_t getJointTypeFromIdx(
        const pinocchio::Model & model, pinocchio::JointIndex idIn, JointModelType & jointTypeOut)
    {
        if (model.njoints < static_cast<int32_t>(idIn) - 1)
        {
            jointTypeOut = JointModelType::UNSUPPORTED;
            PRINT_ERROR("Joint index '", idIn, "' is out of range.");
            return hresult_t::ERROR_GENERIC;
        }

        jointTypeOut = getJointType(model.joints[idIn]);

        return hresult_t::SUCCESS;
    }

    hresult_t getJointTypePositionSuffixes(JointModelType jointTypeIn,
                                           std::vector<std::string_view> & jointTypeSuffixesOut)
    {
        // If no extra discrimination is needed
        switch (jointTypeIn)
        {
        case JointModelType::LINEAR:
        case JointModelType::ROTARY:
            jointTypeSuffixesOut = {""};
            break;
        case JointModelType::ROTARY_UNBOUNDED:
            jointTypeSuffixesOut = {"Cos", "Sin"};
            break;
        case JointModelType::PLANAR:
            jointTypeSuffixesOut = {"TransX", "TransY"};
            break;
        case JointModelType::TRANSLATION:
            jointTypeSuffixesOut = {"TransX", "TransY", "TransZ"};
            break;
        case JointModelType::SPHERICAL:
            jointTypeSuffixesOut = {"QuatX", "QuatY", "QuatZ", "QuatW"};
            break;
        case JointModelType::FREE:
            jointTypeSuffixesOut = {
                "TransX", "TransY", "TransZ", "QuatX", "QuatY", "QuatZ", "QuatW"};
            break;
        case JointModelType::UNSUPPORTED:
        default:
            jointTypeSuffixesOut = {""};
            PRINT_ERROR("Joints of type 'UNSUPPORTED' do not have fieldnames.");
            return hresult_t::ERROR_GENERIC;
        }

        return hresult_t::SUCCESS;
    }

    hresult_t getJointTypeVelocitySuffixes(JointModelType jointTypeIn,
                                           std::vector<std::string_view> & jointTypeSuffixesOut)
    {
        // If no extra discrimination is needed
        jointTypeSuffixesOut = {""};
        switch (jointTypeIn)
        {
        case JointModelType::LINEAR:
        case JointModelType::ROTARY:
        case JointModelType::ROTARY_UNBOUNDED:
            jointTypeSuffixesOut = {""};
            break;
        case JointModelType::PLANAR:
            jointTypeSuffixesOut = {"LinX", "LinY"};
            break;
        case JointModelType::TRANSLATION:
            jointTypeSuffixesOut = {"LinX", "LinY", "LinZ"};
            break;
        case JointModelType::SPHERICAL:
            jointTypeSuffixesOut = {"AngX", "AngY", "AngZ"};
            break;
        case JointModelType::FREE:
            jointTypeSuffixesOut = {"LinX", "LinY", "LinZ", "AngX", "AngY", "AngZ"};
            break;
        case JointModelType::UNSUPPORTED:
        default:
            jointTypeSuffixesOut = {""};
            PRINT_ERROR("Joints of type 'UNSUPPORTED' do not have fieldnames.");
            return hresult_t::ERROR_GENERIC;
        }

        return hresult_t::SUCCESS;
    }

    hresult_t getFrameIdx(const pinocchio::Model & model,
                          const std::string & frameName,
                          pinocchio::FrameIndex & frameIdx)
    {
        auto frameIt = std::find_if(model.frames.begin(),
                                    model.frames.end(),
                                    [&frameName](const pinocchio::Frame & frame)
                                    { return frame.name == frameName; });

        if (frameIt == model.frames.end())
        {
            PRINT_ERROR("Frame '", frameName, "' not found in robot model.");
            return hresult_t::ERROR_BAD_INPUT;
        }
        frameIdx = std::distance(model.frames.begin(), frameIt);

        return hresult_t::SUCCESS;
    }

    hresult_t getFramesIdx(const pinocchio::Model & model,
                           const std::vector<std::string> & framesNames,
                           std::vector<pinocchio::FrameIndex> & framesIdx)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        framesIdx.resize(0);
        for (const std::string & name : framesNames)
        {
            if (returnCode == hresult_t::SUCCESS)
            {
                pinocchio::FrameIndex frameIdx;
                returnCode = getFrameIdx(model, name, frameIdx);
                framesIdx.push_back(frameIdx);
            }
        }

        return returnCode;
    }

    hresult_t getJointPositionIdx(const pinocchio::Model & model,
                                  const std::string & jointName,
                                  std::vector<Eigen::Index> & jointPositionIdx)
    {
        // It returns all the indices if the joint has multiple degrees of freedom

        if (!model.existJointName(jointName))
        {
            PRINT_ERROR("Joint '", jointName, "' not found in robot model.");
            return hresult_t::ERROR_BAD_INPUT;
        }

        const pinocchio::JointIndex jointModelIdx = model.getJointId(jointName);
        const Eigen::Index jointPositionFirstIdx = model.joints[jointModelIdx].idx_q();
        const Eigen::Index jointNq = model.joints[jointModelIdx].nq();
        jointPositionIdx.resize(static_cast<std::size_t>(jointNq));
        std::iota(jointPositionIdx.begin(), jointPositionIdx.end(), jointPositionFirstIdx);

        return hresult_t::SUCCESS;
    }

    hresult_t getJointPositionIdx(const pinocchio::Model & model,
                                  const std::string & jointName,
                                  Eigen::Index & jointPositionFirstIdx)
    {
        // It returns the first index even if the joint has multiple degrees of freedom

        if (!model.existJointName(jointName))
        {
            PRINT_ERROR("Joint '", jointName, "' not found in robot model.");
            return hresult_t::ERROR_BAD_INPUT;
        }

        pinocchio::JointIndex jointModelIdx = model.getJointId(jointName);
        jointPositionFirstIdx = model.joints[jointModelIdx].idx_q();

        return hresult_t::SUCCESS;
    }

    hresult_t getJointsPositionIdx(const pinocchio::Model & model,
                                   const std::vector<std::string> & jointsNames,
                                   std::vector<Eigen::Index> & jointsPositionIdx,
                                   bool firstJointIdxOnly)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        jointsPositionIdx.clear();
        if (!firstJointIdxOnly)
        {
            std::vector<Eigen::Index> jointPositionIdx;
            for (const std::string & jointName : jointsNames)
            {
                if (returnCode == hresult_t::SUCCESS)
                {
                    returnCode = getJointPositionIdx(model, jointName, jointPositionIdx);
                }
                if (returnCode == hresult_t::SUCCESS)
                {
                    jointsPositionIdx.insert(
                        jointsPositionIdx.end(), jointPositionIdx.begin(), jointPositionIdx.end());
                }
            }
        }
        else
        {
            Eigen::Index jointPositionIdx;
            for (const std::string & jointName : jointsNames)
            {
                if (returnCode == hresult_t::SUCCESS)
                {
                    returnCode = getJointPositionIdx(model, jointName, jointPositionIdx);
                }
                if (returnCode == hresult_t::SUCCESS)
                {
                    jointsPositionIdx.push_back(jointPositionIdx);
                }
            }
        }

        return returnCode;
    }

    hresult_t getJointModelIdx(const pinocchio::Model & model,
                               const std::string & jointName,
                               pinocchio::JointIndex & jointModelIdx)
    {
        // It returns the first index even if the joint has multiple degrees of freedom

        if (!model.existJointName(jointName))
        {
            PRINT_ERROR("Joint '", jointName, "' not found in robot model.");
            return hresult_t::ERROR_BAD_INPUT;
        }

        jointModelIdx = model.getJointId(jointName);

        return hresult_t::SUCCESS;
    }

    hresult_t getJointsModelIdx(const pinocchio::Model & model,
                                const std::vector<std::string> & jointsNames,
                                std::vector<pinocchio::JointIndex> & jointsModelIdx)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        jointsModelIdx.clear();
        pinocchio::JointIndex jointModelIdx;
        for (const std::string & jointName : jointsNames)
        {
            if (returnCode == hresult_t::SUCCESS)
            {
                returnCode = getJointModelIdx(model, jointName, jointModelIdx);
            }
            if (returnCode == hresult_t::SUCCESS)
            {
                jointsModelIdx.push_back(jointModelIdx);
            }
        }

        return returnCode;
    }

    hresult_t getJointVelocityIdx(const pinocchio::Model & model,
                                  const std::string & jointName,
                                  std::vector<Eigen::Index> & jointVelocityIdx)
    {
        // It returns all the indices if the joint has multiple degrees of freedom

        if (!model.existJointName(jointName))
        {
            PRINT_ERROR("Joint '", jointName, "' not found in robot model.");
            return hresult_t::ERROR_BAD_INPUT;
        }

        const pinocchio::JointIndex jointModelIdx = model.getJointId(jointName);
        const Eigen::Index jointVelocityFirstIdx = model.joints[jointModelIdx].idx_v();
        const Eigen::Index jointNv = model.joints[jointModelIdx].nv();
        jointVelocityIdx.resize(static_cast<std::size_t>(jointNv));
        std::iota(jointVelocityIdx.begin(), jointVelocityIdx.end(), jointVelocityFirstIdx);

        return hresult_t::SUCCESS;
    }

    hresult_t getJointVelocityIdx(const pinocchio::Model & model,
                                  const std::string & jointName,
                                  Eigen::Index & jointVelocityFirstIdx)
    {
        // It returns the first index even if the joint has multiple degrees of freedom

        if (!model.existJointName(jointName))
        {
            PRINT_ERROR("Joint '", jointName, "' not found in robot model.");
            return hresult_t::ERROR_BAD_INPUT;
        }

        const pinocchio::JointIndex jointModelIdx = model.getJointId(jointName);
        jointVelocityFirstIdx = model.joints[jointModelIdx].idx_v();

        return hresult_t::SUCCESS;
    }

    hresult_t getJointsVelocityIdx(const pinocchio::Model & model,
                                   const std::vector<std::string> & jointsNames,
                                   std::vector<Eigen::Index> & jointsVelocityIdx,
                                   bool firstJointIdxOnly)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        jointsVelocityIdx.clear();
        if (!firstJointIdxOnly)
        {
            std::vector<Eigen::Index> jointVelocityIdx;
            for (const std::string & jointName : jointsNames)
            {
                if (returnCode == hresult_t::SUCCESS)
                {
                    returnCode = getJointVelocityIdx(model, jointName, jointVelocityIdx);
                }
                if (returnCode == hresult_t::SUCCESS)
                {
                    jointsVelocityIdx.insert(
                        jointsVelocityIdx.end(), jointVelocityIdx.begin(), jointVelocityIdx.end());
                }
            }
        }
        else
        {
            Eigen::Index jointVelocityIdx;
            for (const std::string & jointName : jointsNames)
            {
                if (returnCode == hresult_t::SUCCESS)
                {
                    returnCode = getJointVelocityIdx(model, jointName, jointVelocityIdx);
                }
                if (returnCode == hresult_t::SUCCESS)
                {
                    jointsVelocityIdx.push_back(jointVelocityIdx);
                }
            }
        }

        return returnCode;
    }

    hresult_t isPositionValid(const pinocchio::Model & model,
                              const Eigen::VectorXd & position,
                              bool & isValid,
                              double tol)
    {
        if (model.nq != position.size())
        {
            isValid = false;
            PRINT_ERROR("Size of configuration vector inconsistent with model.");
            return hresult_t::ERROR_BAD_INPUT;
        }

        isValid = pinocchio::isNormalized(model, position, tol);

        return hresult_t::SUCCESS;
    }

    void switchJoints(pinocchio::Model & modelInOut,
                      pinocchio::JointIndex firstJointIdx,
                      pinocchio::JointIndex secondJointIdx)
    {
        assert(firstJointIdx < secondJointIdx &&
               "'firstJointIdx' must be smaller than 'secondJointIdx'.");

        // Only perform swap if firstJointIdx is less that secondJointId
        if (firstJointIdx < secondJointIdx)
        {
            // Update parents for other joints
            for (pinocchio::JointIndex & parent : modelInOut.parents)
            {
                if (firstJointIdx == parent)
                {
                    parent = secondJointIdx;
                }
                else if (secondJointIdx == parent)
                {
                    parent = firstJointIdx;
                }
            }

            // Update frame parents
            for (pinocchio::Frame & frame : modelInOut.frames)
            {
                if (firstJointIdx == frame.parent)
                {
                    frame.parent = secondJointIdx;
                }
                else if (secondJointIdx == frame.parent)
                {
                    frame.parent = firstJointIdx;
                }
            }

            // Update values in subtrees
            for (std::vector<pinocchio::Index> & subtree : modelInOut.subtrees)
            {
                for (pinocchio::Index & index : subtree)
                {
                    if (firstJointIdx == index)
                    {
                        index = secondJointIdx;
                    }
                    else if (secondJointIdx == index)
                    {
                        index = firstJointIdx;
                    }
                }
            }

            // Update values in supports
            for (std::vector<pinocchio::Index> & supports : modelInOut.supports)
            {
                for (pinocchio::Index & index : supports)
                {
                    if (firstJointIdx == index)
                    {
                        index = secondJointIdx;
                    }
                    else if (secondJointIdx == index)
                    {
                        index = firstJointIdx;
                    }
                }
            }

            /* Update vectors based on joint index: effortLimit, velocityLimit, lowerPositionLimit
               and upperPositionLimit. */
            swapMatrixRows(modelInOut.effortLimit,
                           modelInOut.joints[firstJointIdx].idx_v(),
                           modelInOut.joints[firstJointIdx].nv(),
                           modelInOut.joints[secondJointIdx].idx_v(),
                           modelInOut.joints[secondJointIdx].nv());
            swapMatrixRows(modelInOut.velocityLimit,
                           modelInOut.joints[firstJointIdx].idx_v(),
                           modelInOut.joints[firstJointIdx].nv(),
                           modelInOut.joints[secondJointIdx].idx_v(),
                           modelInOut.joints[secondJointIdx].nv());
            swapMatrixRows(modelInOut.lowerPositionLimit,
                           modelInOut.joints[firstJointIdx].idx_q(),
                           modelInOut.joints[firstJointIdx].nq(),
                           modelInOut.joints[secondJointIdx].idx_q(),
                           modelInOut.joints[secondJointIdx].nq());
            swapMatrixRows(modelInOut.upperPositionLimit,
                           modelInOut.joints[firstJointIdx].idx_q(),
                           modelInOut.joints[firstJointIdx].nq(),
                           modelInOut.joints[secondJointIdx].idx_q(),
                           modelInOut.joints[secondJointIdx].nq());
            swapMatrixRows(modelInOut.rotorInertia,
                           modelInOut.joints[firstJointIdx].idx_v(),
                           modelInOut.joints[firstJointIdx].nv(),
                           modelInOut.joints[secondJointIdx].idx_v(),
                           modelInOut.joints[secondJointIdx].nv());
            swapMatrixRows(modelInOut.friction,
                           modelInOut.joints[firstJointIdx].idx_v(),
                           modelInOut.joints[firstJointIdx].nv(),
                           modelInOut.joints[secondJointIdx].idx_v(),
                           modelInOut.joints[secondJointIdx].nv());
            swapMatrixRows(modelInOut.damping,
                           modelInOut.joints[firstJointIdx].idx_v(),
                           modelInOut.joints[firstJointIdx].nv(),
                           modelInOut.joints[secondJointIdx].idx_v(),
                           modelInOut.joints[secondJointIdx].nv());

            /* Switch elements in joint-indexed vectors:
               parents, names, subtrees, joints, jointPlacements, inertias. */
            const pinocchio::JointIndex tempParent = modelInOut.parents[firstJointIdx];
            modelInOut.parents[firstJointIdx] = modelInOut.parents[secondJointIdx];
            modelInOut.parents[secondJointIdx] = tempParent;

            const std::string tempName = modelInOut.names[firstJointIdx];
            modelInOut.names[firstJointIdx] = modelInOut.names[secondJointIdx];
            // std::swap is NOT used to preserve memory alignment
            modelInOut.names[secondJointIdx] = tempName;

            const std::vector<pinocchio::Index> tempSubtree = modelInOut.subtrees[firstJointIdx];
            modelInOut.subtrees[firstJointIdx] = modelInOut.subtrees[secondJointIdx];
            modelInOut.subtrees[secondJointIdx] = tempSubtree;

            const pinocchio::JointModel jointTemp = modelInOut.joints[firstJointIdx];
            modelInOut.joints[firstJointIdx] = modelInOut.joints[secondJointIdx];
            modelInOut.joints[secondJointIdx] = jointTemp;

            const pinocchio::SE3 tempPlacement = modelInOut.jointPlacements[firstJointIdx];
            modelInOut.jointPlacements[firstJointIdx] = modelInOut.jointPlacements[secondJointIdx];
            modelInOut.jointPlacements[secondJointIdx] = tempPlacement;

            const pinocchio::Inertia tempInertia = modelInOut.inertias[firstJointIdx];
            modelInOut.inertias[firstJointIdx] = modelInOut.inertias[secondJointIdx];
            modelInOut.inertias[secondJointIdx] = tempInertia;

            /* Recompute all position and velocity indexes, as we may have switched joints that
               did not have the same size. It skips 'universe' since it is not an actual joint. */
            int32_t incrementalNq = 0;
            int32_t incrementalNv = 0;
            for (std::size_t i = 1; i < modelInOut.joints.size(); ++i)
            {
                pinocchio::JointModel & jmodel = modelInOut.joints[i];
                jmodel.setIndexes(i, incrementalNq, incrementalNv);
                incrementalNq += jmodel.nq();
                incrementalNv += jmodel.nv();
                modelInOut.nqs[i] = jmodel.nq();
                modelInOut.idx_qs[i] = jmodel.idx_q();
                modelInOut.nvs[i] = jmodel.nv();
                modelInOut.idx_vs[i] = jmodel.idx_v();
            }
        }
    }

    hresult_t insertFlexibilityBeforeJointInModel(pinocchio::Model & modelInOut,
                                                  const std::string & childJointNameIn,
                                                  const std::string & newJointNameIn)
    {
        using namespace pinocchio;

        if (!modelInOut.existJointName(childJointNameIn))
        {
            PRINT_ERROR("Child joint does not exist.");
            return hresult_t::ERROR_GENERIC;
        }

        pinocchio::JointIndex childJointIdx = modelInOut.getJointId(childJointNameIn);

        // Flexible joint is placed at the same position as the child joint, in its parent frame
        const SE3 jointPosition = modelInOut.jointPlacements[childJointIdx];

        // Create flexible joint
        const pinocchio::JointIndex newJointIdx =
            modelInOut.addJoint(modelInOut.parents[childJointIdx],
                                JointModelSpherical(),
                                jointPosition,
                                newJointNameIn);

        // Set child joint to be a child of the new joint, at the origin
        modelInOut.parents[childJointIdx] = newJointIdx;
        modelInOut.jointPlacements[childJointIdx] = SE3::Identity();

        // Add new joint to frame list
        pinocchio::FrameIndex childFrameIdx;
        getFrameIdx(modelInOut, childJointNameIn, childFrameIdx);  // Cannot fail at this point
        const pinocchio::FrameIndex newFrameIdx = modelInOut.addJointFrame(
            newJointIdx, static_cast<int32_t>(modelInOut.frames[childFrameIdx].previousFrame));

        // Update child joint previousFrame index
        modelInOut.frames[childFrameIdx].previousFrame = newFrameIdx;
        modelInOut.frames[childFrameIdx].placement = SE3::Identity();

        // Update new joint subtree to include all the joints below it
        for (std::size_t i = 0; i < modelInOut.subtrees[childJointIdx].size(); ++i)
        {
            modelInOut.subtrees[newJointIdx].push_back(modelInOut.subtrees[childJointIdx][i]);
        }

        // Add weightless body
        modelInOut.appendBodyToJoint(newJointIdx, pinocchio::Inertia::Zero(), SE3::Identity());

        /* Pinocchio requires that joints are in increasing order as we move to the leaves of the
           kinematic tree. Here this is no longer the case, as an intermediate joint was appended
           at the end. We put the joint back in order by doing successive permutations. */
        for (pinocchio::JointIndex i = childJointIdx; i < newJointIdx; ++i)
        {
            switchJoints(modelInOut, i, newJointIdx);
        }

        return hresult_t::SUCCESS;
    }

    hresult_t insertFlexibilityAtFixedFrameInModel(pinocchio::Model & modelInOut,
                                                   const std::string & frameNameIn)
    {
        using namespace pinocchio;

        // Make sure the frame exists and is fixed
        if (!modelInOut.existFrame(frameNameIn))
        {
            PRINT_ERROR("Frame does not exist.");
            return hresult_t::ERROR_GENERIC;
        }
        pinocchio::FrameIndex frameIdx;
        getFrameIdx(modelInOut, frameNameIn, frameIdx);  // Cannot fail at this point
        Model::Frame & frame = modelInOut.frames[frameIdx];
        if (frame.type != pinocchio::FrameType::FIXED_JOINT)
        {
            PRINT_ERROR("Frame must be associated with fixed joint.");
            return hresult_t::ERROR_GENERIC;
        }

        /* Get the parent and child actual joints.
           To this end, first get the parent joint, next get the list of frames having it as
           parent, finally goes all the way up into their respective branch to find out whether it
           is part of the correct branch. */
        const pinocchio::JointIndex parentJointIdx = frame.parent;
        std::vector<pinocchio::FrameIndex> childFramesIdx;
        for (pinocchio::FrameIndex i = 1;
             i < static_cast<pinocchio::FrameIndex>(modelInOut.nframes);
             ++i)
        {
            // Skip joints and frames not having the right parent joint
            if (modelInOut.frames[i].type == pinocchio::FrameType::JOINT)
            {
                pinocchio::JointIndex jointIdx = modelInOut.frames[i].parent;
                if (modelInOut.parents[jointIdx] != parentJointIdx)
                {
                    continue;
                }
            }
            else if (modelInOut.frames[i].parent != parentJointIdx)
            {
                continue;
            }

            // Check if the candidate frame is really a child
            pinocchio::FrameIndex childFrameIdx = i;
            do
            {
                childFrameIdx = modelInOut.frames[childFrameIdx].previousFrame;
                if (childFrameIdx == frameIdx)
                {
                    childFramesIdx.push_back(i);
                    break;
                }
            } while (childFrameIdx > 0 &&
                     modelInOut.frames[childFrameIdx].type != pinocchio::FrameType::JOINT);
        }

        // The inertia of the newly created joint is the one of all child frames
        Inertia childBodyInertia = frame.inertia.se3Action(frame.placement);
        for (pinocchio::FrameIndex childFrameIdx : childFramesIdx)
        {
            const pinocchio::Frame & childFrame = modelInOut.frames[childFrameIdx];
            childBodyInertia += childFrame.inertia.se3Action(childFrame.placement);
        }

        // Remove inertia of child body from composite body
        if (childBodyInertia.mass() < 0.0)
        {
            PRINT_ERROR("Child body mass must be positive.");
            return hresult_t::ERROR_GENERIC;
        }
        if (modelInOut.inertias[parentJointIdx].mass() - childBodyInertia.mass() < 0.0)
        {
            PRINT_ERROR("Child body mass too large to be subtracted to joint mass.");
            return hresult_t::ERROR_GENERIC;
        }
        const Inertia childBodyInertiaInv(-childBodyInertia.mass(),
                                          childBodyInertia.lever(),
                                          Symmetric3(-childBodyInertia.inertia().data()));
        modelInOut.inertias[parentJointIdx] += childBodyInertiaInv;

        // Create flexible joint
        const pinocchio::JointIndex newJointIdx = modelInOut.addJoint(
            parentJointIdx, JointModelSpherical(), frame.placement, frame.name);
        modelInOut.inertias[newJointIdx] = childBodyInertia.se3Action(frame.placement.inverse());

        // Get min child joint index for swapping
        pinocchio::JointIndex childMinJointIdx = newJointIdx;
        for (pinocchio::FrameIndex childFrameIdx : childFramesIdx)
        {
            if (modelInOut.frames[childFrameIdx].type == pinocchio::FrameType::JOINT)
            {
                childMinJointIdx =
                    std::min(childMinJointIdx, modelInOut.frames[childFrameIdx].parent);
            }
        }

        // Update information for child joints
        for (pinocchio::FrameIndex childFrameIdx : childFramesIdx)
        {
            // Get joint index for frames that are actual joints
            if (modelInOut.frames[childFrameIdx].type != pinocchio::FrameType::JOINT)
            {
                continue;
            }
            const pinocchio::JointIndex childJointIdx = modelInOut.frames[childFrameIdx].parent;

            // Set child joint to be a child of the new joint
            modelInOut.parents[childJointIdx] = newJointIdx;
            modelInOut.jointPlacements[childJointIdx] =
                frame.placement.actInv(modelInOut.jointPlacements[childJointIdx]);

            // Update new joint subtree to include all the joints below it
            for (std::size_t i = 0; i < modelInOut.subtrees[childJointIdx].size(); ++i)
            {
                modelInOut.subtrees[newJointIdx].push_back(modelInOut.subtrees[childJointIdx][i]);
            }
        }

        // Update information for child frames
        for (pinocchio::FrameIndex childFrameIdx : childFramesIdx)
        {
            // Skip actual joints
            if (modelInOut.frames[childFrameIdx].type == pinocchio::FrameType::JOINT)
            {
                continue;
            }

            // Set child frame to be a child of the new joint
            modelInOut.frames[childFrameIdx].parent = newJointIdx;
            modelInOut.frames[childFrameIdx].placement =
                frame.placement.actInv(modelInOut.frames[childFrameIdx].placement);
        }

        // Replace fixed frame by joint frame
        frame.type = pinocchio::FrameType::JOINT;
        frame.parent = newJointIdx;
        frame.inertia.setZero();
        frame.placement.setIdentity();

        /* Pinocchio requires joints to be stored by increasing index as we go down the kinematic
           tree. Here this is no longer the case, as an intermediate joint was appended at the end.
           We move it back this at the correct place by doing successive permutations. */
        for (pinocchio::JointIndex i = childMinJointIdx; i < newJointIdx; ++i)
        {
            switchJoints(modelInOut, i, newJointIdx);
        }

        return hresult_t::SUCCESS;
    }

    hresult_t interpolate(const pinocchio::Model & modelIn,
                          const Eigen::VectorXd & timesIn,
                          const Eigen::MatrixXd & positionsIn,
                          const Eigen::VectorXd & timesOut,
                          Eigen::MatrixXd & positionsOut)
    {
        // Nothing to do. Return early.
        if (timesIn.size() == 0)
        {
            positionsOut.conservativeResize(0, Eigen::NoChange);
            return hresult_t::SUCCESS;
        }

        if (!std::is_sorted(timesIn.data(), timesIn.data() + timesIn.size()) ||
            !std::is_sorted(timesOut.data(), timesOut.data() + timesOut.size()))
        {
            PRINT_ERROR("Input and output time sequences must be sorted.");
            return hresult_t::ERROR_BAD_INPUT;
        }

        if (timesIn.size() != positionsIn.rows() || modelIn.nq != positionsIn.cols())
        {
            PRINT_ERROR("Input position sequence dimension not consistent with model and time "
                        "sequence. Time expected as first dimension.");
            return hresult_t::ERROR_BAD_INPUT;
        }

        int32_t timesInIdx = -1;
        Eigen::VectorXd qInterp(positionsIn.cols());
        positionsOut.resize(timesOut.size(), positionsIn.cols());
        for (Eigen::Index i = 0; i < timesOut.size(); ++i)
        {
            double t = timesOut[i];
            while (timesInIdx < timesIn.size() - 1 && timesIn[timesInIdx + 1] < t)
            {
                ++timesInIdx;
            }
            if (0 <= timesInIdx && timesInIdx < timesIn.size() - 1)
            {
                // Must use Eigen::Ref/Eigen::VectorXd buffers instead of Transpose Eigen::RowXpr,
                // otherwise `interpolate` result will be wrong for SE3
                const Eigen::Ref<const Eigen::VectorXd> qRight =
                    positionsIn.row(timesInIdx).transpose();
                const Eigen::Ref<const Eigen::VectorXd> qLeft =
                    positionsIn.row(timesInIdx + 1).transpose();
                const double ratio =
                    (t - timesIn[timesInIdx]) / (timesIn[timesInIdx + 1] - timesIn[timesInIdx]);
                pinocchio::interpolate(modelIn, qRight, qLeft, ratio, qInterp);
                positionsOut.row(i) = qInterp;
            }
            else if (timesInIdx < 0)
            {
                positionsOut.row(i) = positionsIn.row(0);
            }
            else
            {
                positionsOut.row(i) = positionsIn.row(timesIn.size() - 1);
            }
        }

        return hresult_t::SUCCESS;
    }

    pinocchio::Force convertForceGlobalFrameToJoint(const pinocchio::Model & model,
                                                    const pinocchio::Data & data,
                                                    pinocchio::FrameIndex frameIdx,
                                                    const pinocchio::Force & fextInGlobal)
    {
        // Compute transform from global frame to local joint frame.
        // Translation: joint_p_frame.
        // Rotation: joint_R_world
        pinocchio::SE3 joint_M_global(
            data.oMi[model.frames[frameIdx].parent].rotation().transpose(),
            model.frames[frameIdx].placement.translation());

        return joint_M_global.act(fextInGlobal);
    }

    class DummyMeshLoader : public hpp::fcl::MeshLoader
    {
    public:
        virtual ~DummyMeshLoader() {}

        DummyMeshLoader() :
        MeshLoader(hpp::fcl::BV_OBBRSS)
        {
        }

        virtual hpp::fcl::BVHModelPtr_t load(const std::string & /* filename */,
                                             const hpp::fcl::Vec3f & /* scale */) override final
        {
            return hpp::fcl::BVHModelPtr_t(new hpp::fcl::BVHModel<hpp::fcl::OBBRSS>);
        }
    };

    hresult_t buildGeomFromUrdf(const pinocchio::Model & model,
                                const std::string & filename,
                                const pinocchio::GeometryType & type,
                                pinocchio::GeometryModel & geomModel,
                                const std::vector<std::string> & packageDirs,
                                bool loadMeshes,
                                bool makeConvexMeshes)
    {
        // Load geometry model
        try
        {
            if (loadMeshes)
            {
                pinocchio::urdf::buildGeom(model, filename, type, geomModel, packageDirs);
            }
            else
            {
                hpp::fcl::MeshLoaderPtr MeshLoaderPtr(new DummyMeshLoader);
                pinocchio::urdf::buildGeom(
                    model, filename, type, geomModel, packageDirs, MeshLoaderPtr);
            }
        }
        catch (const std::exception & e)
        {
            PRINT_ERROR("Something is wrong with the URDF. Impossible to load the collision "
                        "geometries.\nRaised from exception: ",
                        e.what());
            return hresult_t::ERROR_GENERIC;
        }

        // Replace the mesh geometry object by its convex representation if necessary
        if (makeConvexMeshes)
        {
            try
            {
                for (uint32_t i = 0; i < geomModel.geometryObjects.size(); ++i)
                {
                    auto & geometry = geomModel.geometryObjects[i].geometry;
                    if (geometry->getObjectType() == hpp::fcl::OT_BVH)
                    {
                        hpp::fcl::BVHModelPtr_t bvh =
                            std::static_pointer_cast<hpp::fcl::BVHModelBase>(geometry);
                        bvh->buildConvexHull(true);
                        geometry = bvh->convex;
                    }
                }
            }
            catch (const std::logic_error & e)
            {
                PRINT_WARNING(
                    "hpp-fcl not built with qhull. Impossible to convert meshes to convex hulls.");
            }
        }

        return hresult_t::SUCCESS;
    }

    hresult_t buildModelsFromUrdf(
        const std::string & urdfPath,
        bool hasFreeflyer,
        const std::vector<std::string> & meshPackageDirs,
        pinocchio::Model & pncModel,
        pinocchio::GeometryModel & collisionModel,
        std::optional<std::reference_wrapper<pinocchio::GeometryModel>> visualModel,
        bool loadVisualMeshes)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        // Make sure the URDF file exists
        if (!std::ifstream(urdfPath).good())
        {
            PRINT_ERROR("The URDF file '", urdfPath, "' is invalid.");
            return hresult_t::ERROR_BAD_INPUT;
        }

        // Build physics model
        try
        {
            if (hasFreeflyer)
            {
                pinocchio::urdf::buildModel(urdfPath, pinocchio::JointModelFreeFlyer(), pncModel);
            }
            else
            {
                pinocchio::urdf::buildModel(urdfPath, pncModel);
            }
        }
        catch (const std::exception & e)
        {
            PRINT_ERROR("Something is wrong with the URDF. Impossible to build a model from "
                        "it.\nRaised from exception: ",
                        e.what());
            returnCode = hresult_t::ERROR_BAD_INPUT;
        }

        // Build collision model
        if (returnCode == hresult_t::SUCCESS)
        {
            returnCode = buildGeomFromUrdf(pncModel,
                                           urdfPath,
                                           pinocchio::COLLISION,
                                           collisionModel,
                                           meshPackageDirs,
                                           true,
                                           true);
        }

        // Build visual model
        if (returnCode == hresult_t::SUCCESS)
        {
            if (visualModel)
            {
                returnCode = buildGeomFromUrdf(pncModel,
                                               urdfPath,
                                               pinocchio::VISUAL,
                                               *visualModel,
                                               meshPackageDirs,
                                               loadVisualMeshes,
                                               false);
            }
        }

        return returnCode;
    }
}
