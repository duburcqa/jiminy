#include <numeric>

#include "pinocchio/multibody/model.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"

#include "jiminy/core/utilities/Helpers.h"
#include "jiminy/core/utilities/Pinocchio.h"


namespace jiminy
{
    hresult_t getJointNameFromPositionIdx(pinocchio::Model const & model,
                                          int32_t          const & idIn,
                                          std::string            & jointNameOut)
    {
        // Iterate over all joints.
        for (int32_t i = 0; i < model.njoints; ++i)
        {
            // Get joint starting and ending index in position vector.
            int32_t startIndex = model.joints[i].idx_q();
            int32_t endIndex = startIndex + model.joints[i].nq();

            // If inIn is between start and end, we found the joint we were looking for.
            if (startIndex <= idIn && endIndex > idIn)
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
        for (int32_t i = 0; i < model.njoints; ++i)
        {
            // Get joint starting and ending index in velocity vector.
            int32_t startIndex = model.joints[i].idx_v();
            int32_t endIndex = startIndex + model.joints[i].nv();

            // If inIn is between start and end, we found the joint we were looking for.
            if (startIndex <= idIn && endIndex > idIn)
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
                                  int32_t const & idIn,
                                  joint_t & jointTypeOut)
    {
        if (model.njoints < idIn - 1 || idIn < 0)
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
                          int32_t                & frameIdx)
    {
        if (!model.existFrame(frameName))
        {
            PRINT_ERROR("Frame '", frameName, "' not found in robot model.");
            return hresult_t::ERROR_BAD_INPUT;
        }

        frameIdx = model.getFrameId(frameName);

        return hresult_t::SUCCESS;
    }

    hresult_t getFramesIdx(pinocchio::Model         const & model,
                           std::vector<std::string> const & framesNames,
                           std::vector<int32_t>           & framesIdx)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        framesIdx.resize(0);
        for (std::string const & name : framesNames)
        {
            if (returnCode == hresult_t::SUCCESS)
            {
                int32_t idx;
                returnCode = getFrameIdx(model, name, idx);
                framesIdx.push_back(std::move(idx));
            }
        }

        return returnCode;
    }

    hresult_t getBodyIdx(pinocchio::Model const & model,
                          std::string     const & bodyName,
                          int32_t               & bodyIdx)
    {
        if (!model.existBodyName(bodyName))
        {
            PRINT_ERROR("Body '", bodyName, "' not found in robot model.");
            return hresult_t::ERROR_BAD_INPUT;
        }

        bodyIdx = model.getBodyId(bodyName);

        return hresult_t::SUCCESS;
    }

    hresult_t getBodiesIdx(pinocchio::Model         const & model,
                           std::vector<std::string> const & bodiesNames,
                           std::vector<int32_t>           & bodiesIdx)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        bodiesIdx.resize(0);
        for (std::string const & name : bodiesNames)
        {
            if (returnCode == hresult_t::SUCCESS)
            {
                int32_t idx;
                returnCode = getFrameIdx(model, name, idx);
                bodiesIdx.push_back(std::move(idx));
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

        int32_t const & jointModelIdx = model.getJointId(jointName);
        int32_t const & jointPositionFirstIdx = model.joints[jointModelIdx].idx_q();
        int32_t const & jointNq = model.joints[jointModelIdx].nq();
        jointPositionIdx.resize(jointNq);
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

        int32_t const & jointModelIdx = model.getJointId(jointName);
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
                               int32_t                & jointModelIdx)
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

    hresult_t getJointsModelIdx(pinocchio::Model         const & model,
                                std::vector<std::string> const & jointsNames,
                                std::vector<int32_t>           & jointsModelIdx)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        jointsModelIdx.clear();
        int32_t jointModelIdx;
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

        int32_t const & jointModelIdx = model.getJointId(jointName);
        int32_t const & jointVelocityFirstIdx = model.joints[jointModelIdx].idx_v();
        int32_t const & jointNv = model.joints[jointModelIdx].nv();
        jointVelocityIdx.resize(jointNv);
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

        int32_t const & jointModelIdx = model.getJointId(jointName);
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

    void switchJoints(pinocchio::Model       & modelInOut,
                      uint32_t         const & firstJointIdx,
                      uint32_t         const & secondJointIdx)
    {
        // Only perform swap if firstJointIdx is less that secondJointId
        if (firstJointIdx < secondJointIdx)
        {
            // Update parents for other joints.
            for (uint32_t i = 0; i < modelInOut.parents.size(); ++i)
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
            for (uint32_t i = 0; i < modelInOut.frames.size(); ++i)
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
            for (uint32_t i = 0; i < modelInOut.subtrees.size(); ++i)
            {
                for (uint32_t j = 0; j < modelInOut.subtrees[i].size(); ++j)
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
            uint32_t tempParent = modelInOut.parents[firstJointIdx];
            modelInOut.parents[firstJointIdx] = modelInOut.parents[secondJointIdx];
            modelInOut.parents[secondJointIdx] = tempParent;

            std::string tempName = modelInOut.names[firstJointIdx];
            modelInOut.names[firstJointIdx] = modelInOut.names[secondJointIdx];
            modelInOut.names[secondJointIdx] = tempName;

            std::vector<pinocchio::Index> tempSubtree = modelInOut.subtrees[firstJointIdx];
            modelInOut.subtrees[firstJointIdx] = modelInOut.subtrees[secondJointIdx];
            modelInOut.subtrees[secondJointIdx] = tempSubtree;

            pinocchio::JointModel jointTemp = modelInOut.joints[firstJointIdx];
            modelInOut.joints[firstJointIdx] = modelInOut.joints[secondJointIdx];
            modelInOut.joints[secondJointIdx] = jointTemp;

            pinocchio::SE3 tempPlacement = modelInOut.jointPlacements[firstJointIdx];
            modelInOut.jointPlacements[firstJointIdx] = modelInOut.jointPlacements[secondJointIdx];
            modelInOut.jointPlacements[secondJointIdx] = tempPlacement;

            pinocchio::Inertia tempInertia = modelInOut.inertias[firstJointIdx];
            modelInOut.inertias[firstJointIdx] = modelInOut.inertias[secondJointIdx];
            modelInOut.inertias[secondJointIdx] = tempInertia;

            /* Recompute all position and velocity indexes, as we may have
               switched joints that didn't have the same size.
               Skip 'universe' joint since it is not an actual joint. */
            uint32_t incrementalNq = 0;
            uint32_t incrementalNv = 0;
            for (uint32_t i = 1; i < modelInOut.joints.size(); ++i)
            {
                modelInOut.joints[i].setIndexes(i, incrementalNq, incrementalNv);
                incrementalNq += modelInOut.joints[i].nq();
                incrementalNv += modelInOut.joints[i].nv();
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

        int32_t const & childJointIdx = modelInOut.getJointId(childJointNameIn);

        // Flexible joint is placed at the same position as the child joint, in its parent frame
        SE3 const jointPosition = modelInOut.jointPlacements[childJointIdx];

        // Create flexible joint
        int32_t const newJointIdx = modelInOut.addJoint(modelInOut.parents[childJointIdx],
                                                        JointModelSpherical(),
                                                        jointPosition,
                                                        newJointNameIn);

        // Set child joint to be a child of the new joint, at the origin
        modelInOut.parents[childJointIdx] = newJointIdx;
        modelInOut.jointPlacements[childJointIdx] = SE3::Identity();

        // Add new joint to frame list
        int32_t const & childFrameIdx = modelInOut.getFrameId(childJointNameIn);
        int32_t const & newFrameIdx = modelInOut.addJointFrame(
            newJointIdx, modelInOut.frames[childFrameIdx].previousFrame);

        // Update child joint previousFrame index
        modelInOut.frames[childFrameIdx].parent = newJointIdx;
        modelInOut.frames[childFrameIdx].previousFrame = newFrameIdx;
        modelInOut.frames[childFrameIdx].placement = SE3::Identity();

        // Update new joint subtree to include all the joints below it
        for (uint32_t i = 0; i < modelInOut.subtrees[childJointIdx].size(); ++i)
        {
            modelInOut.subtrees[newJointIdx].push_back(modelInOut.subtrees[childJointIdx][i]);
        }

        /* Add weightless body.
           In practice having a zero inertia makes some of pinocchio algorithm crash,
           so we set a very small value instead: 1.0g. Anything below that creates
           numerical instability. */
        float64_t const mass = 1.0e-3;
        float64_t const length_semiaxis = 1.0;
        pinocchio::Inertia inertia = pinocchio::Inertia::FromEllipsoid(
            mass, length_semiaxis, length_semiaxis, length_semiaxis);

        modelInOut.appendBodyToJoint(newJointIdx, inertia, SE3::Identity());

        /* Pinocchio requires that joints are in increasing order as we move to the
           leaves of the kinematic tree. Here this is no longer the case, as an
           intermediate joint was appended at the end. We put back this joint at the
           correct position, by doing successive permutations. */
        for (int32_t i = childJointIdx; i < newJointIdx; ++i)
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
        int32_t frameIdx;
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
        uint32_t const parentJointIdx = frame.parent;
        std::vector<int32_t> childCandidateJointsIdx;
        for (int32_t i = 1; i < modelInOut.njoints; ++i)
        {
            if (modelInOut.parents[i] == parentJointIdx)
            {
                childCandidateJointsIdx.push_back(i);
            }
        }

        std::vector<int32_t> childJointsIdx;
        for (int32_t const & childCandidateIdx : childCandidateJointsIdx)
        {
            int32_t childFrameIdx;
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
        int32_t const newJointIdx = modelInOut.addJoint(parentJointIdx,
                                                        JointModelSpherical(),
                                                        frame.placement,
                                                        newJointNameIn);
        modelInOut.appendBodyToJoint(newJointIdx, childBodyInertiaIn, SE3::Identity());

        // Add new joint to frame list
        int32_t const & newFrameIdx = modelInOut.addJointFrame(
            newJointIdx, frameIdx);

        for (int32_t const & childJointIdx : childJointsIdx)
        {
            // Set child joint to be a child of the new joint
            modelInOut.parents[childJointIdx] = newJointIdx;
            modelInOut.jointPlacements[childJointIdx] = frame.placement.actInv(
                modelInOut.jointPlacements[childJointIdx]);

            // Update new joint subtree to include all the joints below it
            for (uint32_t i = 0; i < modelInOut.subtrees[childJointIdx].size(); ++i)
            {
                modelInOut.subtrees[newJointIdx].push_back(
                    modelInOut.subtrees[childJointIdx][i]);
            }
        }

        if (childJointsIdx.size() > 0)
        {
            int32_t const & childJointIdx = *std::min_element(
                childJointsIdx.begin(), childJointsIdx.end());

            // Update child frames parent and previousFrame indices
            int32_t childFrameIdx;
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
            for (int32_t i = childJointIdx; i < newJointIdx; ++i)
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
        for (uint32_t i = 0; i < timesOut.size() ; ++i)
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
                                                    int32_t          const & frameIdx,
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
}
