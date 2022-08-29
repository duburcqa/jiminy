#include <numeric>

#include "pinocchio/parsers/urdf.hpp"                      // `pinocchio::urdf::buildGeom`, `pinocchio::urdf::buildModel`
#include "pinocchio/spatial/se3.hpp"                       // `pinocchio::SE3`
#include "pinocchio/spatial/force.hpp"                     // `pinocchio::Force`
#include "pinocchio/spatial/inertia.hpp"                   // `pinocchio::Inertia`
#include "pinocchio/multibody/model.hpp"                   // `pinocchio::Model`
#include "pinocchio/multibody/fcl.hpp"                     // `pinocchio::GeometryType`
#include "pinocchio/multibody/geometry.hpp"                // `pinocchio::GeometryModel`
#include "pinocchio/multibody/data.hpp"                    // `pinocchio::Data`
#include "pinocchio/multibody/visitor.hpp"                 // `pinocchio::fusion::JointUnaryVisitorBase`
#include "pinocchio/multibody/joint/joint-model-base.hpp"  // `pinocchio::JointModelBase`
#include "pinocchio/algorithm/joint-configuration.hpp"     // `pinocchio::isNormalized`
#include "pinocchio/algorithm/model.hpp"                   // `pinocchio::buildReducedModel`

#include "hpp/fcl/mesh_loader/loader.h"
#include "hpp/fcl/BVH/BVH_model.h"

#include "jiminy/core/utilities/Helpers.h"
#include "jiminy/core/utilities/Pinocchio.h"


namespace jiminy
{
    hresult_t getJointNameFromPositionIdx(pinocchio::Model const & model,
                                          int32_t          const & idx,
                                          std::string            & jointNameOut)
    {
        // Iterate over all joints.
        for (jointIndex_t i = 0; i < static_cast<jointIndex_t>(model.njoints); ++i)
        {
            // Get joint starting and ending index in position vector.
            int32_t const & startIdx = model.joints[i].idx_q();
            int32_t const endIdx = startIdx + model.joints[i].nq();

            // If inIn is between start and end, we found the joint we were looking for.
            if (startIdx <= idx && idx < endIdx)
            {
                jointNameOut = model.names[i];
                return hresult_t::SUCCESS;
            }
        }

        PRINT_ERROR("Position index out of range.");
        return hresult_t::ERROR_BAD_INPUT;
    }

    hresult_t getJointNameFromVelocityIdx(pinocchio::Model const & model,
                                          int32_t          const & idIn,
                                          std::string            & jointNameOut)
    {
        // Iterate over all joints.
        for (jointIndex_t i = 0; i < static_cast<jointIndex_t>(model.njoints); ++i)
        {
            // Get joint starting and ending index in velocity vector.
            int32_t const & startIdx = model.joints[i].idx_v();
            int32_t const endIdx = startIdx + model.joints[i].nv();

            // If inIn is between start and end, we found the joint we were looking for.
            if (startIdx <= idIn && idIn < endIdx)
            {
                jointNameOut = model.names[i];
                return hresult_t::SUCCESS;
            }
        }

        PRINT_ERROR("Velocity index out of range.");
        return hresult_t::ERROR_BAD_INPUT;
    }

    struct getJointTypeAlgo
    : public pinocchio::fusion::JointUnaryVisitorBase<getJointTypeAlgo>
    {
        typedef boost::fusion::vector<joint_t & /* jointType */> ArgsType;

        template<typename JointModel>
        static void algo(pinocchio::JointModelBase<JointModel> const & model,
                         joint_t & jointType)
        {
            jointType = getJointType(model.derived());
        }

        template<typename JointModel>
        static std::enable_if_t<is_pinocchio_joint_freeflyer_v<JointModel>, joint_t>
        getJointType(JointModel const &)
        {
            return joint_t::FREE;
        }

        template<typename JointModel>
        static std::enable_if_t<is_pinocchio_joint_spherical_v<JointModel>
                             || is_pinocchio_joint_spherical_zyx_v<JointModel>, joint_t>
        getJointType(JointModel const &)
        {
            return joint_t::SPHERICAL;
        }

        template<typename JointModel>
        static std::enable_if_t<is_pinocchio_joint_translation_v<JointModel>, joint_t>
        getJointType(JointModel const &)
        {
            return joint_t::TRANSLATION;
        }

        template<typename JointModel>
        static std::enable_if_t<is_pinocchio_joint_planar_v<JointModel>, joint_t>
        getJointType(JointModel const &)
        {
            return joint_t::PLANAR;
        }

        template<typename JointModel>
        static std::enable_if_t<is_pinocchio_joint_prismatic_v<JointModel>
                             || is_pinocchio_joint_prismatic_unaligned_v<JointModel>, joint_t>
        getJointType(JointModel const &)
        {
            return joint_t::LINEAR;
        }

        template<typename JointModel>
        static std::enable_if_t<is_pinocchio_joint_revolute_v<JointModel>
                             || is_pinocchio_joint_revolute_unaligned_v<JointModel>, joint_t>
        getJointType(JointModel const &)
        {
            return joint_t::ROTARY;
        }

        template<typename JointModel>
        static std::enable_if_t<is_pinocchio_joint_revolute_unbounded_v<JointModel>
                             || is_pinocchio_joint_revolute_unbounded_unaligned_v<JointModel>, joint_t>
        getJointType(JointModel const &)
        {
            return joint_t::ROTARY_UNBOUNDED;
        }

        template<typename JointModel>
        static std::enable_if_t<is_pinocchio_joint_mimic_v<JointModel>
                             || is_pinocchio_joint_composite_v<JointModel>, joint_t>
        getJointType(JointModel const &)
        {
            return joint_t::NONE;
        }
    };

    hresult_t getJointTypeFromIdx(pinocchio::Model const & model,
                                  jointIndex_t     const & idIn,
                                  joint_t                & jointTypeOut)
    {
        if (model.njoints < static_cast<int32_t>(idIn) - 1)
        {
            PRINT_ERROR("Joint index '", idIn, "' is out of range.");
            return hresult_t::ERROR_GENERIC;
        }

        getJointTypeAlgo::run(model.joints[idIn],
            typename getJointTypeAlgo::ArgsType(jointTypeOut));

        return hresult_t::SUCCESS;
    }

    hresult_t getJointTypePositionSuffixes(joint_t                  const & jointTypeIn,
                                           std::vector<std::string>       & jointTypeSuffixesOut)
    {
        jointTypeSuffixesOut = std::vector<std::string>({std::string("")});  // If no extra discrimination is needed
        switch (jointTypeIn)
        {
        case joint_t::LINEAR:
            break;
        case joint_t::ROTARY:
            break;
        case joint_t::ROTARY_UNBOUNDED:
            jointTypeSuffixesOut = std::vector<std::string>({std::string("Cos"),
                                                             std::string("Sin")});
            break;
        case joint_t::PLANAR:
            jointTypeSuffixesOut = std::vector<std::string>({std::string("TransX"),
                                                             std::string("TransY")});
            break;
        case joint_t::TRANSLATION:
            jointTypeSuffixesOut = std::vector<std::string>({std::string("TransX"),
                                                             std::string("TransY"),
                                                             std::string("TransZ")});
            break;
        case joint_t::SPHERICAL:
            jointTypeSuffixesOut = std::vector<std::string>({std::string("QuatX"),
                                                             std::string("QuatY"),
                                                             std::string("QuatZ"),
                                                             std::string("QuatW")});
            break;
        case joint_t::FREE:
            jointTypeSuffixesOut = std::vector<std::string>({std::string("TransX"),
                                                             std::string("TransY"),
                                                             std::string("TransZ"),
                                                             std::string("QuatX"),
                                                             std::string("QuatY"),
                                                             std::string("QuatZ"),
                                                             std::string("QuatW")});
            break;
        case joint_t::NONE:
        default:
            PRINT_ERROR("Joints of type 'NONE' do not have fieldnames.");
            return hresult_t::ERROR_GENERIC;
        }

        return hresult_t::SUCCESS;
    }

    hresult_t getJointTypeVelocitySuffixes(joint_t                  const & jointTypeIn,
                                           std::vector<std::string>       & jointTypeSuffixesOut)
    {
        jointTypeSuffixesOut = std::vector<std::string>({std::string("")});  // If no extra discrimination is needed
        switch (jointTypeIn)
        {
        case joint_t::LINEAR:
            break;
        case joint_t::ROTARY:
            break;
        case joint_t::ROTARY_UNBOUNDED:
            break;
        case joint_t::PLANAR:
            jointTypeSuffixesOut = std::vector<std::string>({std::string("LinX"),
                                                             std::string("LinY")});
            break;
        case joint_t::TRANSLATION:
            jointTypeSuffixesOut = std::vector<std::string>({std::string("LinX"),
                                                             std::string("LinY"),
                                                             std::string("LinZ")});
            break;
        case joint_t::SPHERICAL:
            jointTypeSuffixesOut = std::vector<std::string>({std::string("AngX"),
                                                             std::string("AngY"),
                                                             std::string("AngZ")});
            break;
        case joint_t::FREE:
            jointTypeSuffixesOut = std::vector<std::string>({std::string("LinX"),
                                                             std::string("LinY"),
                                                             std::string("LinZ"),
                                                             std::string("AngX"),
                                                             std::string("AngY"),
                                                             std::string("AngZ")});
            break;
        case joint_t::NONE:
        default:
            PRINT_ERROR("Joints of type 'NONE' do not have fieldnames.");
            return hresult_t::ERROR_GENERIC;
        }

        return hresult_t::SUCCESS;
    }

    hresult_t getFrameIdx(pinocchio::Model const & model,
                          std::string      const & frameName,
                          frameIndex_t           & frameIdx)
    {
        if (!model.existFrame(frameName))
        {
            PRINT_ERROR("Frame '", frameName, "' not found in robot model.");
            return hresult_t::ERROR_BAD_INPUT;
        }

        frameIdx = model.getFrameId(frameName);

        return hresult_t::SUCCESS;
    }

    hresult_t getFramesIdx(pinocchio::Model          const & model,
                           std::vector<std::string>  const & framesNames,
                           std::vector<frameIndex_t>       & framesIdx)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        framesIdx.resize(0);
        for (std::string const & name : framesNames)
        {
            if (returnCode == hresult_t::SUCCESS)
            {
                frameIndex_t frameIdx;
                returnCode = getFrameIdx(model, name, frameIdx);
                framesIdx.push_back(frameIdx);
            }
        }

        return returnCode;
    }

    hresult_t getBodyIdx(pinocchio::Model const & model,
                         std::string      const & bodyName,
                         frameIndex_t           & bodyIdx)
    {
        if (!model.existBodyName(bodyName))
        {
            PRINT_ERROR("Body '", bodyName, "' not found in robot model.");
            return hresult_t::ERROR_BAD_INPUT;
        }

        bodyIdx = model.getBodyId(bodyName);

        return hresult_t::SUCCESS;
    }

    hresult_t getBodiesIdx(pinocchio::Model          const & model,
                           std::vector<std::string>  const & bodiesNames,
                           std::vector<frameIndex_t>       & bodiesIdx)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        bodiesIdx.resize(0);
        for (std::string const & name : bodiesNames)
        {
            if (returnCode == hresult_t::SUCCESS)
            {
                frameIndex_t frameIdx;
                returnCode = getFrameIdx(model, name, frameIdx);
                bodiesIdx.push_back(frameIdx);
            }
        }

        return returnCode;
    }

    hresult_t getJointPositionIdx(pinocchio::Model     const & model,
                                  std::string          const & jointName,
                                  std::vector<int32_t>       & jointPositionIdx)
    {
        // It returns all the indices if the joint has multiple degrees of freedom

        if (!model.existJointName(jointName))
        {
            PRINT_ERROR("Joint '", jointName, "' not found in robot model.");
            return hresult_t::ERROR_BAD_INPUT;
        }

        jointIndex_t const & jointModelIdx = model.getJointId(jointName);
        int32_t const & jointPositionFirstIdx = model.joints[jointModelIdx].idx_q();
        int32_t const & jointNq = model.joints[jointModelIdx].nq();
        jointPositionIdx.resize(static_cast<std::size_t>(jointNq));
        std::iota(jointPositionIdx.begin(), jointPositionIdx.end(), jointPositionFirstIdx);

        return hresult_t::SUCCESS;
    }

    hresult_t getJointPositionIdx(pinocchio::Model const & model,
                                  std::string      const & jointName,
                                  int32_t                & jointPositionFirstIdx)
    {
        // It returns the first index even if the joint has multiple degrees of freedom

        if (!model.existJointName(jointName))
        {
            PRINT_ERROR("Joint '", jointName, "' not found in robot model.");
            return hresult_t::ERROR_BAD_INPUT;
        }

        jointIndex_t const & jointModelIdx = model.getJointId(jointName);
        jointPositionFirstIdx = model.joints[jointModelIdx].idx_q();

        return hresult_t::SUCCESS;
    }

    hresult_t getJointsPositionIdx(pinocchio::Model         const & model,
                                   std::vector<std::string> const & jointsNames,
                                   std::vector<int32_t>           & jointsPositionIdx,
                                   bool_t                   const & firstJointIdxOnly)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        jointsPositionIdx.clear();
        if (!firstJointIdxOnly)
        {
            std::vector<int32_t> jointPositionIdx;
            for (std::string const & jointName : jointsNames)
            {
                if (returnCode == hresult_t::SUCCESS)
                {
                    returnCode = getJointPositionIdx(model, jointName, jointPositionIdx);
                }
                if (returnCode == hresult_t::SUCCESS)
                {
                    jointsPositionIdx.insert(jointsPositionIdx.end(), jointPositionIdx.begin(), jointPositionIdx.end());
                }
            }
        }
        else
        {
            int32_t jointPositionIdx;
            for (std::string const & jointName : jointsNames)
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

    hresult_t getJointModelIdx(pinocchio::Model const & model,
                               std::string      const & jointName,
                               jointIndex_t           & jointModelIdx)
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

    hresult_t getJointsModelIdx(pinocchio::Model          const & model,
                                std::vector<std::string>  const & jointsNames,
                                std::vector<jointIndex_t>       & jointsModelIdx)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        jointsModelIdx.clear();
        jointIndex_t jointModelIdx;
        for (std::string const & jointName : jointsNames)
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

    hresult_t getJointVelocityIdx(pinocchio::Model     const & model,
                                  std::string          const & jointName,
                                  std::vector<int32_t>       & jointVelocityIdx)
    {
        // It returns all the indices if the joint has multiple degrees of freedom

        if (!model.existJointName(jointName))
        {
            PRINT_ERROR("Joint '", jointName, "' not found in robot model.");
            return hresult_t::ERROR_BAD_INPUT;
        }

        jointIndex_t const & jointModelIdx = model.getJointId(jointName);
        int32_t const & jointVelocityFirstIdx = model.joints[jointModelIdx].idx_v();
        int32_t const & jointNv = model.joints[jointModelIdx].nv();
        jointVelocityIdx.resize(static_cast<std::size_t>(jointNv));
        std::iota(jointVelocityIdx.begin(), jointVelocityIdx.end(), jointVelocityFirstIdx);

        return hresult_t::SUCCESS;
    }

    hresult_t getJointVelocityIdx(pinocchio::Model const & model,
                                  std::string      const & jointName,
                                  int32_t                & jointVelocityFirstIdx)
    {
        // It returns the first index even if the joint has multiple degrees of freedom

        if (!model.existJointName(jointName))
        {
            PRINT_ERROR("Joint '", jointName, "' not found in robot model.");
            return hresult_t::ERROR_BAD_INPUT;
        }

        jointIndex_t const & jointModelIdx = model.getJointId(jointName);
        jointVelocityFirstIdx = model.joints[jointModelIdx].idx_v();

        return hresult_t::SUCCESS;
    }

    hresult_t getJointsVelocityIdx(pinocchio::Model         const & model,
                                   std::vector<std::string> const & jointsNames,
                                   std::vector<int32_t>           & jointsVelocityIdx,
                                   bool_t                   const & firstJointIdxOnly)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        jointsVelocityIdx.clear();
        if (!firstJointIdxOnly)
        {
            std::vector<int32_t> jointVelocityIdx;
            for (std::string const & jointName : jointsNames)
            {
                if (returnCode == hresult_t::SUCCESS)
                {
                    returnCode = getJointVelocityIdx(model, jointName, jointVelocityIdx);
                }
                if (returnCode == hresult_t::SUCCESS)
                {
                    jointsVelocityIdx.insert(jointsVelocityIdx.end(), jointVelocityIdx.begin(), jointVelocityIdx.end());
                }
            }
        }
        else
        {
            int32_t jointVelocityIdx;
            for (std::string const & jointName : jointsNames)
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

    hresult_t isPositionValid(pinocchio::Model const & model,
                              vectorN_t        const & position,
                              bool_t                 & isValid,
                              float64_t        const & tol)
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

    void switchJoints(pinocchio::Model        & modelInOut,
                      jointIndex_t      const & firstJointIdx,
                      jointIndex_t      const & secondJointIdx)
    {
        // Only perform swap if firstJointIdx is less that secondJointId
        if (firstJointIdx < secondJointIdx)
        {
            // Update parents for other joints.
            for (std::size_t i = 0; i < modelInOut.parents.size(); ++i)
            {
                if (firstJointIdx == modelInOut.parents[i])
                {
                    modelInOut.parents[i] = secondJointIdx;
                }
                else if (secondJointIdx == modelInOut.parents[i])
                {
                    modelInOut.parents[i] = firstJointIdx;
                }
            }
            // Update frame parents.
            for (std::size_t i = 0; i < modelInOut.frames.size(); ++i)
            {
                if (firstJointIdx == modelInOut.frames[i].parent)
                {
                    modelInOut.frames[i].parent = secondJointIdx;
                }
                else if (secondJointIdx == modelInOut.frames[i].parent)
                {
                    modelInOut.frames[i].parent = firstJointIdx;
                }
            }
            // Update values in subtrees.
            for (std::size_t i = 0; i < modelInOut.subtrees.size(); ++i)
            {
                for (std::size_t j = 0; j < modelInOut.subtrees[i].size(); ++j)
                {
                    if (firstJointIdx == modelInOut.subtrees[i][j])
                    {
                        modelInOut.subtrees[i][j] = secondJointIdx;
                    }
                    else if (secondJointIdx == modelInOut.subtrees[i][j])
                    {
                        modelInOut.subtrees[i][j] = firstJointIdx;
                    }
                }
            }

            // Update vectors based on joint index: effortLimit, velocityLimit,
            // lowerPositionLimit and upperPositionLimit.
            swapVectorBlocks(modelInOut.effortLimit,
                             modelInOut.joints[firstJointIdx].idx_v(),
                             modelInOut.joints[firstJointIdx].nv(),
                             modelInOut.joints[secondJointIdx].idx_v(),
                             modelInOut.joints[secondJointIdx].nv());
            swapVectorBlocks(modelInOut.velocityLimit,
                             modelInOut.joints[firstJointIdx].idx_v(),
                             modelInOut.joints[firstJointIdx].nv(),
                             modelInOut.joints[secondJointIdx].idx_v(),
                             modelInOut.joints[secondJointIdx].nv());

            swapVectorBlocks(modelInOut.lowerPositionLimit,
                             modelInOut.joints[firstJointIdx].idx_q(),
                             modelInOut.joints[firstJointIdx].nq(),
                             modelInOut.joints[secondJointIdx].idx_q(),
                             modelInOut.joints[secondJointIdx].nq());
            swapVectorBlocks(modelInOut.upperPositionLimit,
                             modelInOut.joints[firstJointIdx].idx_q(),
                             modelInOut.joints[firstJointIdx].nq(),
                             modelInOut.joints[secondJointIdx].idx_q(),
                             modelInOut.joints[secondJointIdx].nq());

            // Switch elements in joint-indexed vectors:
            // parents, names, subtrees, joints, jointPlacements, inertias.
            jointIndex_t const tempParent = modelInOut.parents[firstJointIdx];
            modelInOut.parents[firstJointIdx] = modelInOut.parents[secondJointIdx];
            modelInOut.parents[secondJointIdx] = tempParent;

            std::string const tempName = modelInOut.names[firstJointIdx];
            modelInOut.names[firstJointIdx] = modelInOut.names[secondJointIdx];
            modelInOut.names[secondJointIdx] = tempName;  // std::swap is NOT used to preserve memory alignement

            std::vector<pinocchio::Index> const tempSubtree = modelInOut.subtrees[firstJointIdx];
            modelInOut.subtrees[firstJointIdx] = modelInOut.subtrees[secondJointIdx];
            modelInOut.subtrees[secondJointIdx] = tempSubtree;

            pinocchio::JointModel const jointTemp = modelInOut.joints[firstJointIdx];
            modelInOut.joints[firstJointIdx] = modelInOut.joints[secondJointIdx];
            modelInOut.joints[secondJointIdx] = jointTemp;

            pinocchio::SE3 const tempPlacement = modelInOut.jointPlacements[firstJointIdx];
            modelInOut.jointPlacements[firstJointIdx] = modelInOut.jointPlacements[secondJointIdx];
            modelInOut.jointPlacements[secondJointIdx] = tempPlacement;

            pinocchio::Inertia const tempInertia = modelInOut.inertias[firstJointIdx];
            modelInOut.inertias[firstJointIdx] = modelInOut.inertias[secondJointIdx];
            modelInOut.inertias[secondJointIdx] = tempInertia;

            /* Recompute all position and velocity indexes, as we may have
               switched joints that didn't have the same size.
               Skip 'universe' joint since it is not an actual joint. */
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

    hresult_t insertFlexibilityBeforeJointInModel(pinocchio::Model       & modelInOut,
                                                  std::string      const & childJointNameIn,
                                                  std::string      const & newJointNameIn)
    {
        using namespace pinocchio;

        if (!modelInOut.existJointName(childJointNameIn))
        {
            PRINT_ERROR("Child joint does not exist.");
            return hresult_t::ERROR_GENERIC;
        }

        jointIndex_t const & childJointIdx = modelInOut.getJointId(childJointNameIn);

        // Flexible joint is placed at the same position as the child joint, in its parent frame
        SE3 const jointPosition = modelInOut.jointPlacements[childJointIdx];

        // Create flexible joint
        jointIndex_t const newJointIdx = modelInOut.addJoint(modelInOut.parents[childJointIdx],
                                                             JointModelSpherical(),
                                                             jointPosition,
                                                             newJointNameIn);

        // Set child joint to be a child of the new joint, at the origin
        modelInOut.parents[childJointIdx] = newJointIdx;
        modelInOut.jointPlacements[childJointIdx] = SE3::Identity();

        // Add new joint to frame list
        frameIndex_t const & childFrameIdx = modelInOut.getFrameId(childJointNameIn);
        frameIndex_t const & newFrameIdx = modelInOut.addJointFrame(
            newJointIdx, static_cast<int32_t>(modelInOut.frames[childFrameIdx].previousFrame));

        // Update child joint previousFrame index
        modelInOut.frames[childFrameIdx].parent = newJointIdx;
        modelInOut.frames[childFrameIdx].previousFrame = newFrameIdx;
        modelInOut.frames[childFrameIdx].placement = SE3::Identity();

        // Update new joint subtree to include all the joints below it
        for (std::size_t i = 0; i < modelInOut.subtrees[childJointIdx].size(); ++i)
        {
            modelInOut.subtrees[newJointIdx].push_back(modelInOut.subtrees[childJointIdx][i]);
        }

        /* Add weightless body.
           In practice having a zero inertia makes some of pinocchio algorithm
           crash, so we set a very small value instead: 1g. Anything below
           creates numerical instability. */
        float64_t const mass = 1.0e-3;
        float64_t const lengthSemiAxis = 1.0;
        pinocchio::Inertia const inertia = pinocchio::Inertia::FromEllipsoid(
            mass, lengthSemiAxis, lengthSemiAxis, lengthSemiAxis);

        modelInOut.appendBodyToJoint(newJointIdx, inertia, SE3::Identity());

        /* Pinocchio requires that joints are in increasing order as we move to the
           leaves of the kinematic tree. Here this is no longer the case, as an
           intermediate joint was appended at the end. We put back this joint at the
           correct position, by doing successive permutations. */
        for (jointIndex_t i = childJointIdx; i < newJointIdx; ++i)
        {
            switchJoints(modelInOut, i, newJointIdx);
        }

        return hresult_t::SUCCESS;
    }

    hresult_t insertFlexibilityAtFixedFrameInModel(pinocchio::Model         & modelInOut,
                                                   std::string        const & frameNameIn,
                                                   pinocchio::Inertia const & childBodyInertiaIn,
                                                   std::string        const & newJointNameIn)
    {
        using namespace pinocchio;

        // Make sure the frame exists and is fixed
        if (!modelInOut.existFrame(frameNameIn))
        {
            PRINT_ERROR("Frame does not exist.");
            return hresult_t::ERROR_GENERIC;
        }
        frameIndex_t frameIdx;
        ::jiminy::getFrameIdx(modelInOut, frameNameIn, frameIdx);
        Model::Frame & frame = modelInOut.frames[frameIdx];
        if (frame.type != FIXED_JOINT)
        {
            PRINT_ERROR("Frame must be associated with fixed joint.");
            return hresult_t::ERROR_GENERIC;
        }

        /* Get the parent and child actual joints.
           To this end, first get the parent joint, then get the list of
           joints having it as parent, then goes up into the list until
           the coresponding branch is found in order to identify the actual
           child in the tree. */
        jointIndex_t const parentJointIdx = frame.parent;
        std::vector<jointIndex_t> childCandidateJointsIdx;
        for (std::size_t i = 1; i < static_cast<std::size_t>(modelInOut.njoints); ++i)
        {
            if (modelInOut.parents[i] == parentJointIdx)
            {
                childCandidateJointsIdx.push_back(i);
            }
        }

        std::vector<jointIndex_t> childJointsIdx;
        for (jointIndex_t const & childCandidateIdx : childCandidateJointsIdx)
        {
            frameIndex_t childFrameIdx;
            std::string const & childJointName = modelInOut.names[childCandidateIdx];
            ::jiminy::getFrameIdx(modelInOut, childJointName, childFrameIdx);

            do
            {
                childFrameIdx = modelInOut.frames[childFrameIdx].previousFrame;
                if (childFrameIdx == frameIdx)
                {
                    childJointsIdx.push_back(childCandidateIdx);
                    break;
                }
            }
            while (childFrameIdx > 0 && modelInOut.frames[childFrameIdx].type != JOINT);
        }

        // Remove inertia of child body from composite body
        Inertia childBodyInertiaInv;
        childBodyInertiaInv.mass() = - childBodyInertiaIn.mass();
        childBodyInertiaInv.lever() = childBodyInertiaIn.lever();
        childBodyInertiaInv.inertia() = Symmetric3(
            - childBodyInertiaIn.inertia().data());
        modelInOut.appendBodyToJoint(parentJointIdx,
                                     childBodyInertiaInv,
                                     frame.placement);
        modelInOut.nbodies--;  // No need to increment the number of bodies

        // Create flexible joint
        jointIndex_t const newJointIdx = modelInOut.addJoint(parentJointIdx,
                                                             JointModelSpherical(),
                                                             frame.placement,
                                                             newJointNameIn);
        modelInOut.appendBodyToJoint(newJointIdx, childBodyInertiaIn, SE3::Identity());

        // Add new joint to frame list
        frameIndex_t const & newFrameIdx = modelInOut.addJointFrame(
            newJointIdx, static_cast<int32_t>(frameIdx));

        for (jointIndex_t const & childJointIdx : childJointsIdx)
        {
            // Set child joint to be a child of the new joint
            modelInOut.parents[childJointIdx] = newJointIdx;
            modelInOut.jointPlacements[childJointIdx] = frame.placement.actInv(
                modelInOut.jointPlacements[childJointIdx]);

            // Update new joint subtree to include all the joints below it
            for (std::size_t i = 0; i < modelInOut.subtrees[childJointIdx].size(); ++i)
            {
                modelInOut.subtrees[newJointIdx].push_back(
                    modelInOut.subtrees[childJointIdx][i]);
            }
        }

        if (childJointsIdx.size() > 0)
        {
            jointIndex_t const & childJointIdx = *std::min_element(
                childJointsIdx.begin(), childJointsIdx.end());

            // Update child frames parent and previousFrame indices
            frameIndex_t childFrameIdx;
            std::string const & childJointName = modelInOut.names[childJointIdx];
            ::jiminy::getFrameIdx(modelInOut, childJointName, childFrameIdx);
            do
            {
                childFrameIdx = modelInOut.frames[childFrameIdx].previousFrame;

                modelInOut.frames[childFrameIdx].parent = newJointIdx;
                modelInOut.frames[childFrameIdx].placement = frame.placement.actInv(
                   modelInOut.frames[childFrameIdx].placement);

                if (childFrameIdx == frameIdx)
                {
                    modelInOut.frames[childFrameIdx].previousFrame = newFrameIdx;
                    break;
                }
            }
            while (childFrameIdx > 0 && modelInOut.frames[childFrameIdx].type != JOINT);

            /* Pinocchio requires that joints are in increasing order as we move to the
            leaves of the kinematic tree. Here this is no longer the case, as an
            intermediate joint was appended at the end. We put back this joint at the
            correct position, by doing successive permutations. */
            for (jointIndex_t i = childJointIdx; i < newJointIdx; ++i)
            {
                switchJoints(modelInOut, i, newJointIdx);
            }
        }

        return hresult_t::SUCCESS;
    }

    hresult_t interpolate(pinocchio::Model const & modelIn,
                          vectorN_t        const & timesIn,
                          matrixN_t        const & positionsIn,
                          vectorN_t        const & timesOut,
                          matrixN_t              & positionsOut)
    {
        if (!std::is_sorted(timesIn.data(), timesIn.data() + timesIn.size())
         || !std::is_sorted(timesOut.data(), timesOut.data() + timesOut.size()))
        {
            PRINT_ERROR("Input and output time sequences must be sorted.");
            return hresult_t::ERROR_BAD_INPUT;
        }

        if (timesIn.size() != positionsIn.rows() || modelIn.nq != positionsIn.cols())
        {
            PRINT_ERROR("Input position sequence dimension not consistent with model and time sequence. Time expected as first dimension.");
            return hresult_t::ERROR_BAD_INPUT;
        }

        int32_t timesInIdx = -1;
        vectorN_t qInterp(positionsIn.cols());
        positionsOut.resize(timesOut.size(), positionsIn.cols());
        for (Eigen::Index i = 0; i < timesOut.size() ; ++i)
        {
            float64_t t = timesOut[i];
            while (timesInIdx < timesIn.size() - 1 && timesIn[timesInIdx + 1] < t)
            {
                ++timesInIdx;
            }
            if (0 <= timesInIdx && timesInIdx < timesIn.size() - 1)
            {
                // Must use Eigen::Ref/vectorN_t buffers instead of Transpose Eigen::RowXpr, otherwise `interpolate` result will be wrong for SE3
                Eigen::Ref<vectorN_t const> const qRight = positionsIn.row(timesInIdx).transpose();
                Eigen::Ref<vectorN_t const> const qLeft = positionsIn.row(timesInIdx + 1).transpose();
                float64_t const ratio = (t - timesIn[timesInIdx]) / (timesIn[timesInIdx + 1] - timesIn[timesInIdx]);
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

    pinocchio::Force convertForceGlobalFrameToJoint(pinocchio::Model const & model,
                                                    pinocchio::Data  const & data,
                                                    frameIndex_t     const & frameIdx,
                                                    pinocchio::Force const & fextInGlobal)
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

        DummyMeshLoader(void) :
        MeshLoader(hpp::fcl::BV_OBBRSS)
        {
            // Empty on purpose.
        }

        virtual hpp::fcl::BVHModelPtr_t load(std::string     const & /* filename */,
                                             hpp::fcl::Vec3f const & /* scale */) override final
        {
            return boost::shared_ptr<hpp::fcl::BVHModel<hpp::fcl::OBBRSS> >(
                new hpp::fcl::BVHModel<hpp::fcl::OBBRSS>);
        }
    };

    hresult_t buildGeomFromUrdf(pinocchio::Model         const & model,
                                std::string              const & filename,
                                pinocchio::GeometryType  const & type,
                                pinocchio::GeometryModel       & geomModel,
                                std::vector<std::string> const & packageDirs,
                                bool_t                   const & loadMeshes,
                                bool_t                   const & makeConvexMeshes)
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
                pinocchio::urdf::buildGeom(model, filename, type, geomModel, packageDirs, MeshLoaderPtr);
            }
        }
        catch (std::exception const & e)
        {
            PRINT_ERROR("Something is wrong with the URDF. Impossible to load the collision geometries.\n"
                        "Raised from exception: ", e.what());
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
                        hpp::fcl::BVHModelPtr_t bvh = boost::static_pointer_cast<hpp::fcl::BVHModelBase>(geometry);
                        bvh->buildConvexHull(true);
                        geometry = bvh->convex;
                    }
                }
            }
            catch (std::logic_error const & e)
            {
                PRINT_WARNING("hpp-fcl not built with qhull. Impossible to convert meshes to convex hulls.");
            }
        }

        return hresult_t::SUCCESS;
    }

    hresult_t buildModelsFromUrdf(std::string const & urdfPath,
                                  bool_t const & hasFreeflyer,
                                  std::vector<std::string> const & meshPackageDirs,
                                  pinocchio::Model & pncModel,
                                  pinocchio::GeometryModel & collisionModel,
                                  boost::optional<pinocchio::GeometryModel &> visualModel,
                                  bool_t const & loadVisualMeshes)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        // Make sure the URDF file exists
        if (!std::ifstream(urdfPath.c_str()).good())
        {
            PRINT_ERROR("The URDF file does not exist. Impossible to load it.");
            return hresult_t::ERROR_BAD_INPUT;
        }

        // Build physics model
        try
        {
            if (hasFreeflyer)
            {
                pinocchio::urdf::buildModel(
                    urdfPath, pinocchio::JointModelFreeFlyer(), pncModel);
            }
            else
            {
                pinocchio::urdf::buildModel(urdfPath, pncModel);
            }
        }
        catch (std::exception const & e)
        {
            PRINT_ERROR("Something is wrong with the URDF. Impossible to build a model from it.\n"
                        "Raised from exception: ", e.what());
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
            if (visualModel.is_initialized())
            {
                returnCode = buildGeomFromUrdf(pncModel,
                                               urdfPath,
                                               pinocchio::VISUAL,
                                               visualModel.value(),
                                               meshPackageDirs,
                                               loadVisualMeshes,
                                               false);
            }
        }

        return returnCode;
    }

    void buildReducedModel(pinocchio::Model const & inputModel,
                           pinocchio::GeometryModel const & inputGeomModel,
                           std::vector<pinocchio::JointIndex> const & listOfJointsToLock,
                           vectorN_t const & referenceConfiguration,
                           pinocchio::Model & reducedModel,
                           pinocchio::GeometryModel & reducedGeomModel)
    {
        // Fix `parentFrame` not updated for reduced geometry model in Pinocchio < 2.6.0
        pinocchio::buildReducedModel(inputModel,
                                     inputGeomModel,
                                     listOfJointsToLock,
                                     referenceConfiguration,
                                     reducedModel,
                                     reducedGeomModel);
        for (auto const & geom : inputGeomModel.geometryObjects)
        {
            geomIndex_t reducedGeomIdx = reducedGeomModel.getGeometryId(geom.name);
            auto & reducedGeom = reducedGeomModel.geometryObjects[reducedGeomIdx];
            reducedGeom.parentFrame = reducedModel.getBodyId(inputModel.frames[geom.parentFrame].name);
        }
    }
}
