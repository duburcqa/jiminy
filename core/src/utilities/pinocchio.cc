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

    JointModelType getJointTypeFromIndex(const pinocchio::Model & model,
                                         pinocchio::JointIndex jointIndex)
    {
        if (model.njoints < static_cast<int>(jointIndex) - 1)
        {
            THROW_ERROR(lookup_error, "Joint index '", jointIndex, "' is out of range.");
        }

        return getJointType(model.joints[jointIndex]);
    }

    std::string getJointNameFromPositionIndex(const pinocchio::Model & model,
                                              pinocchio::JointIndex jointPositionIndex)
    {
        // Iterate over all joints
        for (int jointIndex = 0; jointIndex < model.njoints; ++jointIndex)
        {
            // Get joint starting and ending index in position vector
            const pinocchio::JointIndex firstPositionIndex = model.idx_qs[jointIndex];
            const pinocchio::JointIndex postPositionIndex =
                firstPositionIndex + model.nqs[jointIndex];

            // If idx is between start and end, we found the joint we were looking for
            if (firstPositionIndex <= jointPositionIndex && jointPositionIndex < postPositionIndex)
            {
                return model.names[jointIndex];
            }
        }

        THROW_ERROR(lookup_error, "Position index out of range.");
    }

    std::string getJointNameFromVelocityIndex(const pinocchio::Model & model,
                                              pinocchio::JointIndex jointVelocityIndex)
    {
        // Iterate over all joints
        for (int jointIndex = 0; jointIndex < model.njoints; ++jointIndex)
        {
            // Get joint starting and ending index in velocity vector
            const pinocchio::JointIndex firstVelocityIndex = model.idx_vs[jointIndex];
            const pinocchio::JointIndex postVelocityIndex =
                firstVelocityIndex + model.nvs[jointIndex];

            // If idx is within range, we found the joint we were looking for
            if (firstVelocityIndex <= jointVelocityIndex && jointVelocityIndex < postVelocityIndex)
            {
                return model.names[jointIndex];
            }
        }

        THROW_ERROR(lookup_error, "Velocity index out of range.");
    }

    std::vector<std::string_view> getJointTypePositionSuffixes(JointModelType jointType)
    {
        switch (jointType)
        {
        case JointModelType::LINEAR:
        case JointModelType::ROTARY:
            return {""};  // No extra discrimination is needed
        case JointModelType::ROTARY_UNBOUNDED:
            return {"Cos", "Sin"};
        case JointModelType::PLANAR:
            return {"TransX", "TransY"};
        case JointModelType::TRANSLATION:
            return {"TransX", "TransY", "TransZ"};
        case JointModelType::SPHERICAL:
            return {"QuatX", "QuatY", "QuatZ", "QuatW"};
        case JointModelType::FREE:
            return {"TransX", "TransY", "TransZ", "QuatX", "QuatY", "QuatZ", "QuatW"};
        case JointModelType::UNSUPPORTED:
        default:
            THROW_ERROR(lookup_error, "Joints of type 'UNSUPPORTED' do not have fieldnames.");
        }
    }

    std::vector<std::string_view> getJointTypeVelocitySuffixes(JointModelType jointType)
    {
        switch (jointType)
        {
        case JointModelType::LINEAR:
        case JointModelType::ROTARY:
        case JointModelType::ROTARY_UNBOUNDED:
            return {""};  // No extra discrimination is needed
        case JointModelType::PLANAR:
            return {"LinX", "LinY"};
        case JointModelType::TRANSLATION:
            return {"LinX", "LinY", "LinZ"};
        case JointModelType::SPHERICAL:
            return {"AngX", "AngY", "AngZ"};
        case JointModelType::FREE:
            return {"LinX", "LinY", "LinZ", "AngX", "AngY", "AngZ"};
        case JointModelType::UNSUPPORTED:
        default:
            THROW_ERROR(lookup_error, "Joints of type 'UNSUPPORTED' do not have fieldnames.");
        }
    }

    pinocchio::FrameIndex getFrameIndex(const pinocchio::Model & model,
                                        const std::string & frameName,
                                        pinocchio::FrameType frameType)
    {
        if (!model.existFrame(frameName, frameType))
        {
            THROW_ERROR(lookup_error, "Frame '", frameName, "' not found in robot model.");
        }
        return model.getFrameId(frameName, frameType);
    }

    std::vector<pinocchio::FrameIndex> getFrameIndices(const pinocchio::Model & model,
                                                       const std::vector<std::string> & frameNames)
    {
        std::vector<pinocchio::FrameIndex> frameIndices{};
        frameIndices.reserve(frameNames.size());
        for (const std::string & name : frameNames)
        {
            frameIndices.push_back(getFrameIndex(model, name));
        }
        return frameIndices;
    }

    pinocchio::JointIndex getJointIndex(const pinocchio::Model & model,
                                        const std::string & jointName)
    {
        if (!model.existJointName(jointName))
        {
            THROW_ERROR(lookup_error, "Joint '", jointName, "' not found in robot model.");
        }

        return model.getJointId(jointName);
    }

    std::vector<pinocchio::JointIndex> getJointIndices(const pinocchio::Model & model,
                                                       const std::vector<std::string> & jointNames)
    {
        std::vector<pinocchio::JointIndex> jointModelIndices{};
        jointModelIndices.reserve(jointNames.size());
        for (const std::string & jointName : jointNames)
        {
            jointModelIndices.push_back(getJointIndex(model, jointName));
        }
        return jointModelIndices;
    }

    Eigen::Index getJointPositionFirstIndex(const pinocchio::Model & model,
                                            const std::string & jointName)
    {
        const pinocchio::JointIndex jointIndex = getJointIndex(model, jointName);
        return model.idx_qs[jointIndex];
    }

    std::vector<Eigen::Index> getJointPositionIndices(const pinocchio::Model & model,
                                                      const std::string & jointName)
    {
        const pinocchio::JointIndex jointIndex = getJointIndex(model, jointName);
        const int jointPositionFirstIndex = model.idx_qs[jointIndex];
        const int jointNq = model.nqs[jointIndex];
        std::vector<Eigen::Index> jointPositionIndices(jointNq);
        std::iota(
            jointPositionIndices.begin(), jointPositionIndices.end(), jointPositionFirstIndex);
        return jointPositionIndices;
    }

    std::vector<Eigen::Index> getJointsPositionIndices(const pinocchio::Model & model,
                                                       const std::vector<std::string> & jointNames,
                                                       bool onlyFirstIndex)
    {
        std::vector<Eigen::Index> jointsPositionIndices{};
        if (!onlyFirstIndex)
        {
            for (const std::string & jointName : jointNames)
            {
                std::vector<Eigen::Index> jointPositionIndices =
                    getJointPositionIndices(model, jointName);
                jointsPositionIndices.insert(jointsPositionIndices.end(),
                                             jointPositionIndices.begin(),
                                             jointPositionIndices.end());
            }
        }
        else
        {
            jointsPositionIndices.reserve(jointNames.size());
            for (const std::string & jointName : jointNames)
            {
                jointsPositionIndices.push_back(getJointPositionFirstIndex(model, jointName));
            }
        }
        return jointsPositionIndices;
    }

    Eigen::Index getJointVelocityFirstIndex(const pinocchio::Model & model,
                                            const std::string & jointName)
    {
        const pinocchio::JointIndex jointIndex = getJointIndex(model, jointName);
        return model.idx_vs[jointIndex];
    }

    std::vector<Eigen::Index> getJointVelocityIndices(const pinocchio::Model & model,
                                                      const std::string & jointName)
    {
        const pinocchio::JointIndex jointIndex = getJointIndex(model, jointName);
        const int jointVelocityFirstIndex = model.idx_vs[jointIndex];
        const int jointNv = model.nvs[jointIndex];
        std::vector<Eigen::Index> jointVelocityIndices(jointNv);
        std::iota(
            jointVelocityIndices.begin(), jointVelocityIndices.end(), jointVelocityFirstIndex);
        return jointVelocityIndices;
    }

    std::vector<Eigen::Index> getJointsVelocityIndices(const pinocchio::Model & model,
                                                       const std::vector<std::string> & jointNames,
                                                       bool onlyFirstIndex)
    {
        std::vector<Eigen::Index> jointsVelocityIndices{};
        if (!onlyFirstIndex)
        {
            for (const std::string & jointName : jointNames)
            {
                std::vector<Eigen::Index> jointVelocityIndices =
                    getJointVelocityIndices(model, jointName);
                jointsVelocityIndices.insert(jointsVelocityIndices.end(),
                                             jointVelocityIndices.begin(),
                                             jointVelocityIndices.end());
            }
        }
        else
        {
            jointsVelocityIndices.reserve(jointNames.size());
            for (const std::string & jointName : jointNames)
            {
                jointsVelocityIndices.push_back(getJointVelocityFirstIndex(model, jointName));
            }
        }
        return jointsVelocityIndices;
    }

    bool isPositionValid(const pinocchio::Model & model, const Eigen::VectorXd & q, double tolAbs)
    {
        if (model.nq != q.size())
        {
            return false;
        }
        return pinocchio::isNormalized(model, q, tolAbs);
    }

    void swapJoints(pinocchio::Model & model,
                    pinocchio::JointIndex jointIndex1,
                    pinocchio::JointIndex jointIndex2)
    {
        // Early return if nothing to do
        if (jointIndex1 == jointIndex2)
        {
            return;
        }

        // Enforce that the second joint index always comes after the first one
        if (jointIndex1 > jointIndex2)
        {
            return swapJoints(model, jointIndex2, jointIndex1);
        }

        // Swap references to the joint indices themself
        do_for(
            [jointIndex1, jointIndex2](auto && args)
            {
                auto && [vec, member] = args;
                // FIXME: Remove explicit `name` capture when moving to C++20
                auto swapIndicesFun = [jointIndex1, jointIndex2, &member_ = member](auto && subvec)
                {
                    for (auto & elem : subvec)
                    {
                        pinocchio::JointIndex * jointIndex;
                        if constexpr (std::is_null_pointer_v<std::decay_t<decltype(member_)>>)
                        {
                            jointIndex = &elem;
                        }
                        else
                        {
                            jointIndex = &(elem.*member_);
                        }

                        if (*jointIndex == jointIndex1)
                        {
                            *jointIndex = jointIndex2;
                        }
                        else if (*jointIndex == jointIndex2)
                        {
                            *jointIndex = jointIndex1;
                        }
                    }
                };

                if constexpr (is_vector_v<typename std::decay_t<decltype(vec)>::value_type>)
                {
                    std::for_each(vec.begin(), vec.end(), swapIndicesFun);
                }
                else
                {
                    swapIndicesFun(vec);
                }
            },
            std::forward_as_tuple(model.parents, nullptr),
            std::forward_as_tuple(model.frames, &pinocchio::Frame::parent),
            std::forward_as_tuple(model.subtrees, nullptr),
            std::forward_as_tuple(model.supports, nullptr));

        // Swap blocks of Eigen::Vector objects storing joint properties
        do_for(
            [jointIndex1, jointIndex2](
                std::tuple<Eigen::VectorXd &, const std::vector<int> &, const std::vector<int> &>
                    args)
            {
                auto & [vec, jointPositionFirstIndices, jointPositionSizes] = args;
                swapMatrixRows(vec,
                               jointPositionFirstIndices[jointIndex1],
                               jointPositionSizes[jointIndex1],
                               jointPositionFirstIndices[jointIndex2],
                               jointPositionSizes[jointIndex2]);
            },
            std::forward_as_tuple(model.lowerPositionLimit, model.idx_qs, model.nqs),
            std::forward_as_tuple(model.upperPositionLimit, model.idx_qs, model.nqs),
            std::forward_as_tuple(model.effortLimit, model.idx_vs, model.nvs),
            std::forward_as_tuple(model.velocityLimit, model.idx_vs, model.nvs),
            std::forward_as_tuple(model.rotorInertia, model.idx_vs, model.nvs),
            std::forward_as_tuple(model.friction, model.idx_vs, model.nvs));

        // Swap elements in joint-indexed vectors
        std::swap(model.parents[jointIndex1], model.parents[jointIndex2]);
        std::swap(model.names[jointIndex1], model.names[jointIndex2]);
        std::swap(model.subtrees[jointIndex1], model.subtrees[jointIndex2]);
        std::swap(model.joints[jointIndex1], model.joints[jointIndex2]);
        std::swap(model.jointPlacements[jointIndex1], model.jointPlacements[jointIndex2]);
        std::swap(model.inertias[jointIndex1], model.inertias[jointIndex2]);

        /* Recompute all position and velocity indexes, as we may have switched joints that
           did not have the same size. It skips 'universe' since it is not an actual joint. */
        int idx_q = 0;
        int idx_v = 0;
        for (std::size_t i = 1; i < model.joints.size(); ++i)
        {
            pinocchio::JointModel & jmodel = model.joints[i];
            jmodel.setIndexes(i, idx_q, idx_v);
            idx_q += jmodel.nq();
            idx_v += jmodel.nv();
            model.nqs[i] = jmodel.nq();
            model.idx_qs[i] = jmodel.idx_q();
            model.nvs[i] = jmodel.nv();
            model.idx_vs[i] = jmodel.idx_v();
        }
    }

    void insertFlexibilityBeforeJointInModel(pinocchio::Model & model,
                                             const std::string & childJointName,
                                             const std::string & newJointName)
    {
        using namespace pinocchio;

        const pinocchio::JointIndex childJointIndex = getJointIndex(model, childJointName);

        // Flexible joint is placed at the same position as the child joint, in its parent frame
        const SE3 & jointPlacement = model.jointPlacements[childJointIndex];

        // Create flexible joint
        const pinocchio::JointIndex newJointIndex = model.addJoint(
            model.parents[childJointIndex], JointModelSpherical(), jointPlacement, newJointName);

        // Set child joint to be a child of the new joint, at the origin
        model.parents[childJointIndex] = newJointIndex;
        model.jointPlacements[childJointIndex].setIdentity();

        // Add new joint to frame list
        const pinocchio::FrameIndex childFrameIndex = getFrameIndex(model, childJointName);
        const pinocchio::FrameIndex newFrameIndex = model.addJointFrame(
            newJointIndex, static_cast<int>(model.frames[childFrameIndex].previousFrame));

        // Update child joint previousFrame index
        model.frames[childFrameIndex].previousFrame = newFrameIndex;
        model.frames[childFrameIndex].placement.setIdentity();

        // Update new joint subtree to include all the joints below it
        for (std::size_t i = 0; i < model.subtrees[childJointIndex].size(); ++i)
        {
            model.subtrees[newJointIndex].push_back(model.subtrees[childJointIndex][i]);
        }

        // Add weightless body
        model.appendBodyToJoint(newJointIndex, pinocchio::Inertia::Zero(), SE3::Identity());

        /* Pinocchio requires that joints are in increasing order as we move to the leaves of the
           kinematic tree. Here this is no longer the case, as an intermediate joint was appended
           at the end. We put the joint back in order by doing successive permutations. */
        for (pinocchio::JointIndex i = childJointIndex; i < newJointIndex; ++i)
        {
            swapJoints(model, i, newJointIndex);
        }
    }

    void insertFlexibilityAtFixedFrameInModel(pinocchio::Model & model,
                                              const std::string & frameName)
    {
        using namespace pinocchio;

        // Make sure the frame exists and is fixed
        if (!model.existFrame(frameName))
        {
            THROW_ERROR(lookup_error, "No frame with name '", frameName, "' found in model.");
        }
        const pinocchio::FrameIndex frameIndex = getFrameIndex(model, frameName);
        Model::Frame & frame = model.frames[frameIndex];
        if (frame.type != pinocchio::FrameType::FIXED_JOINT)
        {
            THROW_ERROR(std::logic_error, "Frame must be associated with fixed joint.");
        }

        /* Get the parent and child actual joints.
           To this end, first get the parent joint, next get the list of frames having it as
           parent, finally goes all the way up into their respective branch to find out whether it
           is part of the correct branch. */
        const pinocchio::JointIndex parentJointIndex = frame.parent;
        std::vector<pinocchio::FrameIndex> childFrameIndices;
        for (int i = 1; i < model.nframes; ++i)
        {
            // Skip joints and frames not having the right parent joint
            if (model.frames[i].type == pinocchio::FrameType::JOINT)
            {
                if (model.parents[model.frames[i].parent] != parentJointIndex)
                {
                    continue;
                }
            }
            else if (model.frames[i].parent != parentJointIndex)
            {
                continue;
            }

            // Check if the candidate frame is really a child
            pinocchio::FrameIndex childFrameIndex = i;
            do
            {
                childFrameIndex = model.frames[childFrameIndex].previousFrame;
                if (childFrameIndex == frameIndex)
                {
                    childFrameIndices.push_back(i);
                    break;
                }
            } while (childFrameIndex > 0 &&
                     model.frames[childFrameIndex].type != pinocchio::FrameType::JOINT);
        }

        // The inertia of the newly created joint is the one of all child frames
        Inertia childBodyInertia = frame.inertia.se3Action(frame.placement);
        for (pinocchio::FrameIndex childFrameIndex : childFrameIndices)
        {
            const pinocchio::Frame & childFrame = model.frames[childFrameIndex];
            childBodyInertia += childFrame.inertia.se3Action(childFrame.placement);
        }

        // Remove inertia of child body from composite body
        if (childBodyInertia.mass() < 0.0)
        {
            THROW_ERROR(std::runtime_error, "Child body mass must be positive.");
        }
        if (model.inertias[parentJointIndex].mass() - childBodyInertia.mass() < 0.0)
        {
            THROW_ERROR(std::runtime_error,
                        "Child body mass too large to be subtracted to joint mass.");
        }
        const Inertia childBodyInertiaInv(-childBodyInertia.mass(),
                                          childBodyInertia.lever(),
                                          Symmetric3(-childBodyInertia.inertia().data()));
        model.inertias[parentJointIndex] += childBodyInertiaInv;

        // Create flexible joint
        const pinocchio::JointIndex newJointIndex =
            model.addJoint(parentJointIndex, JointModelSpherical(), frame.placement, frame.name);
        model.inertias[newJointIndex] = childBodyInertia.se3Action(frame.placement.inverse());

        // Get min child joint index for swapping
        pinocchio::JointIndex childJointIndexMin = newJointIndex;
        for (pinocchio::FrameIndex childFrameIndex : childFrameIndices)
        {
            if (model.frames[childFrameIndex].type == pinocchio::FrameType::JOINT)
            {
                childJointIndexMin =
                    std::min(childJointIndexMin, model.frames[childFrameIndex].parent);
            }
        }

        // Update information for child joints
        for (pinocchio::FrameIndex childFrameIndex : childFrameIndices)
        {
            // Get joint index for frames that are actual joints
            if (model.frames[childFrameIndex].type != pinocchio::FrameType::JOINT)
            {
                continue;
            }
            const pinocchio::JointIndex childJointIndex = model.frames[childFrameIndex].parent;

            // Set child joint to be a child of the new joint
            model.parents[childJointIndex] = newJointIndex;
            model.jointPlacements[childJointIndex] =
                frame.placement.actInv(model.jointPlacements[childJointIndex]);

            // Update new joint subtree to include all the joints below it
            for (std::size_t i = 0; i < model.subtrees[childJointIndex].size(); ++i)
            {
                model.subtrees[newJointIndex].push_back(model.subtrees[childJointIndex][i]);
            }
        }

        // Update information for child frames
        for (pinocchio::FrameIndex childFrameIndex : childFrameIndices)
        {
            // Skip actual joints
            if (model.frames[childFrameIndex].type == pinocchio::FrameType::JOINT)
            {
                continue;
            }

            // Set child frame to be a child of the new joint
            model.frames[childFrameIndex].parent = newJointIndex;
            model.frames[childFrameIndex].placement =
                frame.placement.actInv(model.frames[childFrameIndex].placement);
        }

        // Replace fixed frame by joint frame
        frame.type = pinocchio::FrameType::JOINT;
        frame.parent = newJointIndex;
        frame.inertia.setZero();
        frame.placement.setIdentity();

        /* Pinocchio requires joints to be stored by increasing index as we go down the kinematic
           tree. Here this is no longer the case, as an intermediate joint was appended at the end.
           We move it back this at the correct place by doing successive permutations. */
        for (pinocchio::JointIndex i = childJointIndexMin; i < newJointIndex; ++i)
        {
            swapJoints(model, i, newJointIndex);
        }
    }

    Eigen::MatrixXd interpolatePositions(const pinocchio::Model & model,
                                         const Eigen::VectorXd & timesIn,
                                         const Eigen::MatrixXd & positionsIn,
                                         const Eigen::VectorXd & timesOut)
    {
        Eigen::MatrixXd positionsOut(positionsIn.rows(), timesOut.size());

        if (!std::is_sorted(timesIn.data(), timesIn.data() + timesIn.size()) ||
            !std::is_sorted(timesOut.data(), timesOut.data() + timesOut.size()))
        {
            THROW_ERROR(std::invalid_argument, "Input and output time sequences must be sorted.");
        }

        if (timesIn.size() != positionsIn.cols() || model.nq != positionsIn.rows())
        {
            THROW_ERROR(std::invalid_argument,
                        "Input position matrix not consistent with model and/or "
                        "time sequence. Time expected as second dimension.");
        }

        // Nothing to do. Return early.
        if (timesIn.size() == 0)
        {
            return positionsOut;
        }

        Eigen::Index timeIndexIn = -1;
        Eigen::Index timeIndexOut = 0;
        for (; timeIndexOut < timesOut.size(); ++timeIndexOut)
        {
            const double t = timesOut[timeIndexOut];
            while (timeIndexIn < timesIn.size() - 1 && timesIn[timeIndexIn + 1] < t)
            {
                ++timeIndexIn;
            }
            if (timeIndexIn != -1)
            {
                break;
            }
        }
        positionsOut.leftCols(timeIndexOut).colwise() = positionsIn.col(0);
        for (; timeIndexOut < timesOut.size(); ++timeIndexOut)
        {
            const double t = timesOut[timeIndexOut];
            while (timeIndexIn < timesIn.size() - 1 && timesIn[timeIndexIn + 1] < t)
            {
                ++timeIndexIn;
            }
            if (timeIndexIn == timesIn.size() - 1)
            {
                break;
            }
            auto q = positionsOut.col(timeIndexOut);
            auto qRight = positionsIn.col(timeIndexIn);
            auto qLeft = positionsIn.col(timeIndexIn + 1);
            const double ratio =
                (t - timesIn[timeIndexIn]) / (timesIn[timeIndexIn + 1] - timesIn[timeIndexIn]);
            pinocchio::interpolate(model, qRight, qLeft, ratio, q);
        }
        positionsOut.rightCols(timesOut.size() - timeIndexOut).colwise() =
            positionsIn.col(timesIn.size() - 1);

        return positionsOut;
    }

    pinocchio::Force convertForceGlobalFrameToJoint(const pinocchio::Model & model,
                                                    const pinocchio::Data & data,
                                                    pinocchio::FrameIndex frameIndex,
                                                    const pinocchio::Force & fextInGlobal)
    {
        /* Compute transform from local world aligned to local joint frame.
           Translation: joint_p_frame, Rotation: joint_R_world */
        auto liRw = data.oMi[model.frames[frameIndex].parent].rotation().transpose();
        auto liPf = model.frames[frameIndex].placement.translation();

        pinocchio::Force liFf{};
        liFf.linear().noalias() = liRw * fextInGlobal.linear();
        liFf.angular().noalias() = liRw * fextInGlobal.angular();
        liFf.angular() += liPf.cross(liFf.linear());
        return liFf;
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

    pinocchio::GeometryModel buildGeometryModelFromUrdf(
        const pinocchio::Model & model,
        const std::string & filename,
        const pinocchio::GeometryType & type,
        const std::vector<std::string> & packageDirs,
        bool loadMeshes,
        bool generateConvexMeshes)
    {
        pinocchio::GeometryModel geomModel{};

        // Load geometry model
        try
        {
            if (loadMeshes)
            {
                pinocchio::urdf::buildGeom(model, filename, type, geomModel, packageDirs);
            }
            else
            {
                hpp::fcl::MeshLoaderPtr meshLoaderPtr(new DummyMeshLoader);
                pinocchio::urdf::buildGeom(
                    model, filename, type, geomModel, packageDirs, meshLoaderPtr);
            }
        }
        catch (const std::exception & e)
        {
            THROW_ERROR(std::ios_base::failure,
                        "Something is wrong with the URDF. Impossible to load the collision "
                        "geometries.\nRaised from exception: ",
                        e.what());
        }

        // Replace the mesh geometry object by its convex representation if necessary
        if (generateConvexMeshes)
        {
            try
            {
                for (uint32_t i = 0; i < geomModel.geometryObjects.size(); ++i)
                {
                    auto & geometry = geomModel.geometryObjects[i].geometry;
                    if (geometry->getObjectType() == hpp::fcl::OT_BVH)
                    {
                        hpp::fcl::BVHModelPtr_t bvh_ptr =
                            std::static_pointer_cast<hpp::fcl::BVHModelBase>(geometry);
                        bvh_ptr->buildConvexHull(true);
                        geometry = bvh_ptr->convex;
                    }
                }
            }
            catch (const std::logic_error & e)
            {
                PRINT_WARNING(
                    "hpp-fcl not built with qhull. Impossible to convert meshes to convex hulls.");
            }
        }

        return geomModel;
    }

    void buildMultipleModelsFromUrdf(
        const std::string & urdfPath,
        bool hasFreeflyer,
        const std::vector<std::string> & packageDirs,
        pinocchio::Model & pinocchioModel,
        pinocchio::GeometryModel & collisionModel,
        std::optional<std::reference_wrapper<pinocchio::GeometryModel>> visualModel,
        bool loadVisualMeshes)
    {
        // Make sure the URDF file exists
        if (!std::ifstream(urdfPath).good())
        {
            THROW_ERROR(std::ios_base::failure, "The URDF file '", urdfPath, "' is invalid.");
        }

        // Build physics model
        try
        {
            if (hasFreeflyer)
            {
                pinocchio::urdf::buildModel(
                    urdfPath, pinocchio::JointModelFreeFlyer(), pinocchioModel);
            }
            else
            {
                pinocchio::urdf::buildModel(urdfPath, pinocchioModel);
            }
        }
        catch (const std::exception & e)
        {
            THROW_ERROR(std::ios_base::failure,
                        "Something is wrong with the URDF. Impossible to build a model from "
                        "it.\nRaised from exception: ",
                        e.what());
        }

        // Build collision model
        collisionModel = buildGeometryModelFromUrdf(
            pinocchioModel, urdfPath, pinocchio::COLLISION, packageDirs, true, true);

        // Build visual model
        if (visualModel)
        {
            visualModel->get() = buildGeometryModelFromUrdf(
                pinocchioModel, urdfPath, pinocchio::VISUAL, packageDirs, loadVisualMeshes, false);
        }
    }
}
