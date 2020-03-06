#include <math.h>
#include <climits>
#include <stdlib.h>     /* srand, rand */

#ifndef _WIN32
#include <pwd.h>
#include <unistd.h>
#include <getopt.h>
#else
#include <stdlib.h>
#include <stdio.h>
#endif

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "pinocchio/algorithm/joint-configuration.hpp"

#include "jiminy/core/Utilities.h"
#include "jiminy/core/Engine.h" // Required to get access to MIN_TIME_STEP and MAX_TIME_STEP
#include "jiminy/core/TelemetrySender.h"


namespace jiminy
{
    extern float64_t const MIN_TIME_STEP;
    extern float64_t const MAX_TIME_STEP;

    // ************************* Timer **************************

    Timer::Timer(void) :
    t0(),
    tf(),
    dt(0)
    {
        tic();
    }

    void Timer::tic(void)
    {
        t0 = Time::now();
    }

    void Timer::toc(void)
    {
        tf = Time::now();
        std::chrono::duration<float64_t> timeDiff = tf - t0;
        dt = timeDiff.count();
    }

    // ************ IO file and Directory utilities **************

    #ifndef _WIN32
    std::string getUserDirectory(void)
    {
        struct passwd *pw = getpwuid(getuid());
        return pw->pw_dir;
    }
    #else
    std::string getUserDirectory(void)
    {
        return {getenv("USERPROFILE")};
    }
    #endif

    // ***************** Random number generator *****************
    // Based on Ziggurat generator by Marsaglia and Tsang (JSS, 2000)

    std::mt19937 generator_;
    std::uniform_real_distribution<float32_t> distUniform_(0.0,1.0);

    uint32_t kn[128];
    float32_t fn[128];
    float32_t wn[128];

    void r4_nor_setup(void)
    {
        float64_t const m1 = 2147483648.0;
        float64_t const vn = 9.91256303526217E-03;
        float64_t dn = 3.442619855899;
        float64_t tn = 3.442619855899;

        float64_t q = vn / exp (-0.5 * dn * dn);

        kn[0] = (uint32_t) ((dn / q) * m1);
        kn[1] = 0;

        wn[0] = static_cast<float32_t>(q / m1);
        wn[127] = static_cast<float32_t>(dn / m1);

        fn[0] = 1.0f;
        fn[127] = static_cast<float32_t>(exp(-0.5 * dn * dn));

        for (uint8_t i=126; 1 <= i; i--)
        {
            dn = sqrt (-2.0 * log(vn / dn + exp(-0.5 * dn * dn)));
            kn[i+1] = static_cast<uint32_t>((dn / tn) * m1);
            tn = dn;
            fn[i] = static_cast<float32_t>(exp(-0.5 * dn * dn));
            wn[i] = static_cast<float32_t>(dn / m1);
        }
    }

    float32_t r4_uni(void)
    {
        return distUniform_(generator_);
    }

    float32_t r4_nor(void)
    {
        float32_t const r = 3.442620f;
        int32_t hz;
        uint32_t iz;
        float32_t x;
        float32_t y;

        hz = static_cast<int32_t>(generator_());
        iz = (hz & 127U);

        if (fabs(hz) < kn[iz])
        {
            return static_cast<float32_t>(hz) * wn[iz];
        }
        else
        {
            while(true)
            {
                if (iz == 0)
                {
                    while(true)
                    {
                        x = - 0.2904764f * log(r4_uni());
                        y = - log(r4_uni());
                        if (x * x <= y + y)
                        {
                            break;
                        }
                    }

                    if (hz <= 0)
                    {
                        return - r - x;
                    }
                    else
                    {
                        return + r + x;
                    }
                }

                x = static_cast<float32_t>(hz) * wn[iz];

                if (fn[iz] + r4_uni() * (fn[iz-1] - fn[iz]) < exp (-0.5f * x * x))
                {
                    return x;
                }

                hz = static_cast<int32_t>(generator_());
                iz = (hz & 127);

                if (fabs(hz) < kn[iz])
                {
                    return static_cast<float32_t>(hz) * wn[iz];
                }
            }
        }
    }

    // ************** Random number generator utilities ****************

	void resetRandGenerators(uint32_t seed)
	{
		srand(seed); // Eigen relies on srand for genering random matrix
        generator_.seed(seed);
        r4_nor_setup();
	}

	float64_t randUniform(float64_t const & lo,
	                      float64_t const & hi)
    {
        return lo + r4_uni() * (hi - lo);
    }

	float64_t randNormal(float64_t const & mean,
	                     float64_t const & std)
    {
        return mean + r4_nor() * std;
    }

    vectorN_t randVectorNormal(uint32_t  const & size,
                               float64_t const & mean,
                               float64_t const & std)
    {
        if (std > 0.0)
        {
            return vectorN_t::NullaryExpr(size,
            [&mean, &std] (vectorN_t::Index const &) -> float64_t
            {
                return randNormal(mean, std);
            });
        }
        else
        {
            return vectorN_t::Constant(size, mean);
        }
    }

    vectorN_t randVectorNormal(uint32_t  const & size,
                               float64_t const & std)
    {
        return randVectorNormal(size, 0, std);
    }

    vectorN_t randVectorNormal(vectorN_t const & mean,
                               vectorN_t const & std)
    {
        return vectorN_t::NullaryExpr(std.size(),
        [&mean, &std] (vectorN_t::Index const & i) -> float64_t
        {
            return randNormal(mean[i], std[i]);
        });
    }

    vectorN_t randVectorNormal(vectorN_t const & std)
    {
        return vectorN_t::NullaryExpr(std.size(),
        [&std] (vectorN_t::Index const & i) -> float64_t
        {
            return randNormal(0, std[i]);
        });
    }

    // ******************* Telemetry utilities **********************

    std::vector<std::string> defaultVectorFieldnames(std::string const & baseName,
                                                     uint32_t    const & size)
    {
        std::vector<std::string> fieldnames;
        fieldnames.reserve(size);
        for (uint32_t i=0; i<size; i++)
        {
            fieldnames.emplace_back(baseName + std::to_string(i)); // TODO: MR going to support "." delimiter
        }
        return fieldnames;
    }


    std::string removeFieldnameSuffix(std::string         fieldname,
                                      std::string const & suffix)
    {
        if (!fieldname.compare(fieldname.size() - suffix.size(), suffix.size(), suffix))
        {
            fieldname.erase(fieldname.size() - suffix.size(), fieldname.size());
        }
        return fieldname;
    }

    std::vector<std::string> removeFieldnamesSuffix(std::vector<std::string>         fieldnames,
                                                    std::string              const & suffix)
    {
        std::transform(fieldnames.begin(), fieldnames.end(), fieldnames.begin(),
        [&suffix](std::string const & name) -> std::string
        {
            return removeFieldnameSuffix(name, suffix);
        });
        return fieldnames;
    }

    // ********************** Pinocchio utilities **********************

    void computePositionDerivative(pinocchio::Model            const & model,
                                   Eigen::Ref<vectorN_t const>         q,
                                   Eigen::Ref<vectorN_t const>         v,
                                   Eigen::Ref<vectorN_t>               qDot,
                                   float64_t                           dt)
    {
        /* Hack to compute the configuration vector derivative, including the
           quaternions on SO3 automatically. Note that the time difference must
           not be too small to avoid failure. */

        dt = std::max(MIN_TIME_STEP, dt);
        vectorN_t qNext(q.size());
        pinocchio::integrate(model, q, v*dt, qNext);
        qDot = (qNext - q) / dt;
    }

    result_t getJointNameFromPositionId(pinocchio::Model const & model,
                                        int32_t          const & idIn,
                                        std::string            & jointNameOut)
    {
        result_t returnCode = result_t::ERROR_GENERIC;

        // Iterate over all joints.
        for (int32_t i = 0; i < model.njoints; i++)
        {
            // Get joint starting and ending index in position vector.
            int32_t startIndex = model.joints[i].idx_q();
            int32_t endIndex = startIndex + model.joints[i].nq();

            // If inIn is between start and end, we found the joint we were looking for.
            if(startIndex <= idIn && endIndex > idIn)
            {
                jointNameOut = model.names[i];
                returnCode = result_t::SUCCESS;
                break;
            }
        }

        if (returnCode == result_t::SUCCESS)
        {
            std::cout << "Error - Utilities::getJointNameFromVelocityId - Position index out of range." << std::endl;
        }

        return returnCode;
    }

    result_t getJointNameFromVelocityId(pinocchio::Model const & model,
                                        int32_t          const & idIn,
                                        std::string            & jointNameOut)
    {
        result_t returnCode = result_t::ERROR_GENERIC;

        // Iterate over all joints.
        for(int32_t i = 0; i < model.njoints; i++)
        {
            // Get joint starting and ending index in velocity vector.
            int32_t startIndex = model.joints[i].idx_v();
            int32_t endIndex = startIndex + model.joints[i].nv();

            // If inIn is between start and end, we found the joint we were looking for.
            if(startIndex <= idIn && endIndex > idIn)
            {
                jointNameOut = model.names[i];
                returnCode = result_t::SUCCESS;
                break;
            }
        }

        if (returnCode == result_t::SUCCESS)
        {
            std::cout << "Error - Utilities::getJointNameFromVelocityId - Velocity index out of range." << std::endl;
        }

        return returnCode;
    }

    result_t getJointTypeFromId(pinocchio::Model const & model,
                                int32_t          const & idIn,
                                joint_t                & jointTypeOut)
    {
        result_t returnCode = result_t::SUCCESS;

        if(model.njoints < idIn - 1)
        {
            std::cout << "Error - Utilities::getJointTypeFromId - Joint id out of range." << std::endl;
            returnCode = result_t::ERROR_GENERIC;
        }

        if (returnCode == result_t::SUCCESS)
        {
            auto const & joint = model.joints[idIn];

            if (joint.shortname() == "JointModelFreeFlyer")
            {
                jointTypeOut = joint_t::FREE;
            }
            else if (joint.shortname() == "JointModelSpherical")
            {
                jointTypeOut = joint_t::SPHERICAL;
            }
            else if (joint.shortname() == "JointModelPlanar")
            {
                jointTypeOut = joint_t::PLANAR;
            }
            else if (joint.shortname() == "JointModelPX" ||
                     joint.shortname() == "JointModelPY" ||
                     joint.shortname() == "JointModelPZ")
            {
                jointTypeOut = joint_t::LINEAR;
            }
            else if (joint.shortname() == "JointModelRX" ||
                     joint.shortname() == "JointModelRY" ||
                     joint.shortname() == "JointModelRZ")
            {
                jointTypeOut = joint_t::ROTARY;
            }
            else
            {
                // Unknown joint, throw an error to avoid any wrong manipulation.
                jointTypeOut = joint_t::NONE;
                std::cout << "Error - Utilities::getJointTypeFromId - Unknown joint type." << std::endl;
                returnCode = result_t::ERROR_GENERIC;
            }
        }

        return returnCode;
    }

    result_t getJointTypePositionSuffixes(joint_t                  const & jointTypeIn,
                                          std::vector<std::string>       & jointTypeSuffixesOut)
    {
        result_t returnCode = result_t::SUCCESS;

        jointTypeSuffixesOut = std::vector<std::string>({std::string("")}); // If no extra discrimination is needed
        switch (jointTypeIn)
        {
        case joint_t::LINEAR:
            break;
        case joint_t::ROTARY:
            break;
        case joint_t::PLANAR:
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
            std::cout << "Error - Utilities::getJointFieldnamesFromType - Joints of type 'NONE' do not have fieldnames." << std::endl;
            returnCode = result_t::ERROR_GENERIC;
        }

        return returnCode;
    }

    result_t getJointTypeVelocitySuffixes(joint_t                  const & jointTypeIn,
                                          std::vector<std::string>       & jointTypeSuffixesOut)
    {
        result_t returnCode = result_t::SUCCESS;

        jointTypeSuffixesOut = std::vector<std::string>({std::string("")}); // If no extra discrimination is needed
        switch (jointTypeIn)
        {
        case joint_t::LINEAR:
            break;
        case joint_t::ROTARY:
            break;
        case joint_t::PLANAR:
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
            std::cout << "Error - Utilities::getJointFieldnamesFromType - Joints of type 'NONE' do not have fieldnames." << std::endl;
            returnCode = result_t::ERROR_GENERIC;
        }

        return returnCode;
    }

    result_t getFrameIdx(pinocchio::Model const & model,
                         std::string      const & frameName,
                         int32_t                & frameIdx)
    {
        result_t returnCode = result_t::SUCCESS;

        if (!model.existFrame(frameName))
        {
            std::cout << "Error - Utilities::getFrameIdx - Frame not found in urdf." << std::endl;
            returnCode = result_t::ERROR_BAD_INPUT;
        }

        if (returnCode == result_t::SUCCESS)
        {
            frameIdx = model.getFrameId(frameName);
        }

        return returnCode;
    }

    result_t getFramesIdx(pinocchio::Model         const & model,
                          std::vector<std::string> const & framesNames,
                          std::vector<int32_t>           & framesIdx)
    {
        result_t returnCode = result_t::SUCCESS;

        framesIdx.resize(0);
        for (std::string const & name : framesNames)
        {
            if (returnCode == result_t::SUCCESS)
            {
                int32_t idx;
                returnCode = getFrameIdx(model, name, idx);
                framesIdx.push_back(idx);
            }
        }

        return returnCode;
    }

    result_t getJointPositionIdx(pinocchio::Model     const & model,
                                 std::string          const & jointName,
                                 std::vector<int32_t>       & jointPositionIdx)
    {
        // It returns all the indices if the joint has multiple degrees of freedom

        result_t returnCode = result_t::SUCCESS;

        if (!model.existJointName(jointName))
        {
            std::cout << "Error - Utilities::getJointPositionIdx - Joint not found in urdf." << std::endl;
            returnCode = result_t::ERROR_BAD_INPUT;
        }

        if (returnCode == result_t::SUCCESS)
        {
            int32_t const & jointModelIdx = model.getJointId(jointName);
            int32_t const & jointPositionFirstIdx = model.joints[jointModelIdx].idx_q();
            int32_t const & jointNq = model.joints[jointModelIdx].nq();
            jointPositionIdx.resize(jointNq);
            std::iota(jointPositionIdx.begin(), jointPositionIdx.end(), jointPositionFirstIdx);
        }

        return returnCode;
    }

    result_t getJointPositionIdx(pinocchio::Model const & model,
                                 std::string      const & jointName,
                                 int32_t                & jointPositionFirstIdx)
    {
        // It returns the first index even if the joint has multiple degrees of freedom

        result_t returnCode = result_t::SUCCESS;

        if (!model.existJointName(jointName))
        {
            std::cout << "Error - Utilities::getJointPositionIdx - Joint not found in urdf." << std::endl;
            returnCode = result_t::ERROR_BAD_INPUT;
        }

        if (returnCode == result_t::SUCCESS)
        {
            int32_t const & jointModelIdx = model.getJointId(jointName);
            jointPositionFirstIdx = model.joints[jointModelIdx].idx_q();
        }

        return returnCode;
    }

    result_t getJointsPositionIdx(pinocchio::Model         const & model,
                                  std::vector<std::string> const & jointsNames,
                                  std::vector<int32_t>           & jointsPositionIdx,
                                  bool                     const & firstJointIdxOnly)
    {
        result_t returnCode = result_t::SUCCESS;

        jointsPositionIdx.clear();
        if (!firstJointIdxOnly)
        {
            std::vector<int32_t> jointPositionIdx;
            for (std::string const & jointName : jointsNames)
            {
                if (returnCode == result_t::SUCCESS)
                {
                    returnCode = getJointPositionIdx(model, jointName, jointPositionIdx);
                }
                if (returnCode == result_t::SUCCESS)
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
                if (returnCode == result_t::SUCCESS)
                {
                    returnCode = getJointPositionIdx(model, jointName, jointPositionIdx);
                }
                if (returnCode == result_t::SUCCESS)
                {
                    jointsPositionIdx.push_back(jointPositionIdx);
                }
            }
        }

        return returnCode;
    }

    result_t getJointModelIdx(pinocchio::Model const & model,
                              std::string      const & jointName,
                              int32_t                & jointModelIdx)
    {
        // It returns the first index even if the joint has multiple degrees of freedom

        result_t returnCode = result_t::SUCCESS;

        if (!model.existJointName(jointName))
        {
            std::cout << "Error - Utilities::getJointPositionIdx - Joint not found in urdf." << std::endl;
            returnCode = result_t::ERROR_BAD_INPUT;
        }

        if (returnCode == result_t::SUCCESS)
        {
            jointModelIdx = model.getJointId(jointName);
        }

        return returnCode;
    }

    result_t getJointsModelIdx(pinocchio::Model         const & model,
                               std::vector<std::string> const & jointsNames,
                               std::vector<int32_t>           & jointsModelIdx)
    {
        result_t returnCode = result_t::SUCCESS;

        jointsModelIdx.clear();
        int32_t jointModelIdx;
        for (std::string const & jointName : jointsNames)
        {
            if (returnCode == result_t::SUCCESS)
            {
                returnCode = getJointModelIdx(model, jointName, jointModelIdx);
            }
            if (returnCode == result_t::SUCCESS)
            {
                jointsModelIdx.push_back(jointModelIdx);
            }
        }

        return returnCode;
    }

    result_t getJointVelocityIdx(pinocchio::Model     const & model,
                                 std::string          const & jointName,
                                 std::vector<int32_t>       & jointVelocityIdx)
    {
        // It returns all the indices if the joint has multiple degrees of freedom

        result_t returnCode = result_t::SUCCESS;

        if (!model.existJointName(jointName))
        {
            std::cout << "Error - getJointVelocityIdx - Frame not found in urdf." << std::endl;
            returnCode = result_t::ERROR_BAD_INPUT;
        }

        if (returnCode == result_t::SUCCESS)
        {
            int32_t const & jointModelIdx = model.getJointId(jointName);
            int32_t const & jointVelocityFirstIdx = model.joints[jointModelIdx].idx_v();
            int32_t const & jointNv = model.joints[jointModelIdx].nv();
            jointVelocityIdx.resize(jointNv);
            std::iota(jointVelocityIdx.begin(), jointVelocityIdx.end(), jointVelocityFirstIdx);
        }

        return returnCode;
    }

    result_t getJointVelocityIdx(pinocchio::Model const & model,
                                 std::string      const & jointName,
                                 int32_t                & jointVelocityFirstIdx)
    {
        // It returns the first index even if the joint has multiple degrees of freedom

        result_t returnCode = result_t::SUCCESS;

        if (!model.existJointName(jointName))
        {
            std::cout << "Error - getJointVelocityIdx - Frame not found in urdf." << std::endl;
            returnCode = result_t::ERROR_BAD_INPUT;
        }

        if (returnCode == result_t::SUCCESS)
        {
            int32_t const & jointModelIdx = model.getJointId(jointName);
            jointVelocityFirstIdx = model.joints[jointModelIdx].idx_v();
        }

        return returnCode;
    }

    result_t getJointsVelocityIdx(pinocchio::Model         const & model,
                                  std::vector<std::string> const & jointsNames,
                                  std::vector<int32_t>           & jointsVelocityIdx,
                                  bool                     const & firstJointIdxOnly)
    {
        result_t returnCode = result_t::SUCCESS;

        jointsVelocityIdx.clear();
        if (!firstJointIdxOnly)
        {
            std::vector<int32_t> jointVelocityIdx;
            for (std::string const & jointName : jointsNames)
            {
                if (returnCode == result_t::SUCCESS)
                {
                    returnCode = getJointVelocityIdx(model, jointName, jointVelocityIdx);
                }
                if (returnCode == result_t::SUCCESS)
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
                if (returnCode == result_t::SUCCESS)
                {
                    returnCode = getJointVelocityIdx(model, jointName, jointVelocityIdx);
                }
                if (returnCode == result_t::SUCCESS)
                {
                    jointsVelocityIdx.push_back(jointVelocityIdx);
                }
            }
        }

        return returnCode;
    }

    void switchJoints(pinocchio::Model       & modelInOut,
                      uint32_t         const & firstJointId,
                      uint32_t         const & secondJointId)
    {
        // Only perform swap if firstJointId is less that secondJointId
        if (firstJointId < secondJointId)
        {
            // Update parents for other joints.
            for(uint32_t i = 0; i < modelInOut.parents.size(); i++)
            {
                if(firstJointId == modelInOut.parents[i])
                {
                    modelInOut.parents[i] = secondJointId;
                }
                else if(secondJointId == modelInOut.parents[i])
                {
                    modelInOut.parents[i] = firstJointId;
                }
            }
            // Update frame parents.
            for(uint32_t i = 0; i < modelInOut.frames.size(); i++)
            {
                if(firstJointId == modelInOut.frames[i].parent)
                {
                    modelInOut.frames[i].parent = secondJointId;
                }
                else if(secondJointId == modelInOut.frames[i].parent)
                {
                    modelInOut.frames[i].parent = firstJointId;
                }
            }
            // Update values in subtrees.
            for(uint32_t i = 0; i < modelInOut.subtrees.size(); i++)
            {
                for(uint32_t j = 0; j < modelInOut.subtrees[i].size(); j++)
                {
                    if(firstJointId == modelInOut.subtrees[i][j])
                    {
                        modelInOut.subtrees[i][j] = secondJointId;
                    }
                    else if(secondJointId == modelInOut.subtrees[i][j])
                    {
                        modelInOut.subtrees[i][j] = firstJointId;
                    }
                }
            }

            // Update vectors based on joint index: effortLimit, velocityLimit,
            // lowerPositionLimit and upperPositionLimit.
            swapVectorBlocks(modelInOut.effortLimit,
                             modelInOut.joints[firstJointId].idx_v(),
                             modelInOut.joints[firstJointId].nv(),
                             modelInOut.joints[secondJointId].idx_v(),
                             modelInOut.joints[secondJointId].nv());
            swapVectorBlocks(modelInOut.velocityLimit,
                             modelInOut.joints[firstJointId].idx_v(),
                             modelInOut.joints[firstJointId].nv(),
                             modelInOut.joints[secondJointId].idx_v(),
                             modelInOut.joints[secondJointId].nv());

            swapVectorBlocks(modelInOut.lowerPositionLimit,
                             modelInOut.joints[firstJointId].idx_q(),
                             modelInOut.joints[firstJointId].nq(),
                             modelInOut.joints[secondJointId].idx_q(),
                             modelInOut.joints[secondJointId].nq());
            swapVectorBlocks(modelInOut.upperPositionLimit,
                             modelInOut.joints[firstJointId].idx_q(),
                             modelInOut.joints[firstJointId].nq(),
                             modelInOut.joints[secondJointId].idx_q(),
                             modelInOut.joints[secondJointId].nq());

            // Switch elements in joint-indexed vectors:
            // parents, names, subtrees, joints, jointPlacements, inertias.
            uint32_t tempParent = modelInOut.parents[firstJointId];
            modelInOut.parents[firstJointId] = modelInOut.parents[secondJointId];
            modelInOut.parents[secondJointId] = tempParent;

            std::string tempName = modelInOut.names[firstJointId];
            modelInOut.names[firstJointId] = modelInOut.names[secondJointId];
            modelInOut.names[secondJointId] = tempName;

            std::vector<pinocchio::Index> tempSubtree = modelInOut.subtrees[firstJointId];
            modelInOut.subtrees[firstJointId] = modelInOut.subtrees[secondJointId];
            modelInOut.subtrees[secondJointId] = tempSubtree;

            pinocchio::JointModel jointTemp = modelInOut.joints[firstJointId];
            modelInOut.joints[firstJointId] = modelInOut.joints[secondJointId];
            modelInOut.joints[secondJointId] = jointTemp;

            pinocchio::SE3 tempPlacement = modelInOut.jointPlacements[firstJointId];
            modelInOut.jointPlacements[firstJointId] = modelInOut.jointPlacements[secondJointId];
            modelInOut.jointPlacements[secondJointId] = tempPlacement;

            pinocchio::Inertia tempInertia = modelInOut.inertias[firstJointId];
            modelInOut.inertias[firstJointId] = modelInOut.inertias[secondJointId];
            modelInOut.inertias[secondJointId] = tempInertia;

            /* Recompute all position and velocity indexes, as we may have
               switched joints that didn't have the same size.
               Skip 'universe' joint since it is not an actual joint. */
            uint32_t incrementalNq = 0;
            uint32_t incrementalNv = 0;
            for(uint32_t i = 1; i < modelInOut.joints.size(); i++)
            {
                modelInOut.joints[i].setIndexes(i, incrementalNq, incrementalNv);
                incrementalNq += modelInOut.joints[i].nq();
                incrementalNv += modelInOut.joints[i].nv();
            }
        }
    }

    result_t insertFlexibilityInModel(pinocchio::Model       & modelInOut,
                                      std::string      const & childJointNameIn,
                                      std::string      const & newJointNameIn)
    {
        result_t returnCode = result_t::SUCCESS;

        if(!modelInOut.existJointName(childJointNameIn))
        {
            returnCode = result_t::ERROR_GENERIC;
        }

        if(returnCode == result_t::SUCCESS)
        {
            int32_t childId = modelInOut.getJointId(childJointNameIn);
            // Flexible joint is placed at the same position as the child joint, in its parent frame.
            pinocchio::SE3 jointPosition = modelInOut.jointPlacements[childId];

            // Create joint.
            int32_t newId = modelInOut.addJoint(modelInOut.parents[childId],
                                                pinocchio::JointModelSpherical(),
                                                jointPosition,
                                                newJointNameIn);

            // Set child joint to be a child of the new joint, at the origin.
            modelInOut.parents[childId] = newId;
            modelInOut.jointPlacements[childId] = pinocchio::SE3::Identity();

            // Add new joint to frame list.
            int32_t childFrameId = modelInOut.getFrameId(childJointNameIn);
            int32_t newFrameId = modelInOut.addJointFrame(newId, modelInOut.frames[childFrameId].previousFrame);

            // Update child joint previousFrame id.
            modelInOut.frames[childFrameId].previousFrame = newFrameId;

            // Update new joint subtree to include all the joints below it.
            for(uint32_t i = 0; i < modelInOut.subtrees[childId].size(); i++)
            {
                modelInOut.subtrees[newId].push_back(modelInOut.subtrees[childId][i]);
            }

            /* Add weightless body.
               In practice having a zero inertia makes some of pinocchio algorithm crash,
               so we set a very small value instead: 1g. */
            std::string bodyName = newJointNameIn + "Body";
            pinocchio::Inertia inertia = pinocchio::Inertia::Identity();
            inertia.mass() *= 1.0e-3;
            inertia.FromEllipsoid(inertia.mass(), 1.0, 1.0, 1.0);
            modelInOut.appendBodyToJoint(newId, inertia, pinocchio::SE3::Identity());

            /* Pinocchio requires that joints are in increasing order as we move to the
               leaves of the kinematic tree. Here this is no longer the case, as an
               intermediate joint was appended at the end. We put back this joint at the
               correct position, by doing successive permutations. */
            for(int32_t i = childId; i < newId; i++)
            {
                switchJoints(modelInOut, i, newId);
            }
        }

        return returnCode;
    }

    // ********************** Math utilities *************************

    ///////////////////////////////////////////////////////////////////////////////////////////////
    ///
    /// \brief      Continuously differentiable piecewise-defined saturation function. More
    ///             precisely, it consists in adding fillets at the two discontinuous points:
    ///             - It is perfectly linear for `uc` between `-bevelStart` and `bevelStart`.
    ///             - It is  perfectly constant equal to mi (resp. ma) for `uc` lower than
    ///               `-bevelStop` (resp. higher than `-bevelStop`).
    ///             - Then, two arcs of a circle connect those two modes continuously between
    ///             `bevelStart` and `bevelStop` (resp. `-bevelStop` and `-bevelStart`).
    ///             See the implementation for details about how `uc`, `bevelStart` and `bevelStop`
    ///             are computed.
    ///
    ///////////////////////////////////////////////////////////////////////////////////////////////
    float64_t saturateSoft(float64_t const & in,
                           float64_t const & mi,
                           float64_t const & ma,
                           float64_t const & r)
    {
        float64_t uc, range, middle, bevelL, bevelXc, bevelYc, bevelStart, bevelStop, out;
        float64_t const alpha = M_PI/8;
        float64_t const beta = M_PI/4;

        range = ma - mi;
        middle = (ma + mi)/2;
        uc = 2*(in - middle)/range;

        bevelL = r * tan(alpha);
        bevelStart = 1 - cos(beta)*bevelL;
        bevelStop = 1 + bevelL;
        bevelXc = bevelStop;
        bevelYc = 1 - r;

        if (uc >= bevelStop)
        {
            out = ma;
        }
        else if (uc <= -bevelStop)
        {
            out = mi;
        }
        else if (uc <= bevelStart && uc >= -bevelStart)
        {
            out = in;
        }
        else if (uc > bevelStart)
        {
            out = sqrt(r * r - (uc - bevelXc) * (uc - bevelXc)) + bevelYc;
            out = 0.5 * out * range + middle;
        }
        else if (uc < -bevelStart)
        {
            out = -sqrt(r * r - (uc + bevelXc) * (uc + bevelXc)) - bevelYc;
            out = 0.5 * out * range + middle;
        }
        else
        {
            out = in;
        }
        return out;
    }

    vectorN_t clamp(Eigen::Ref<vectorN_t const>         data,
                    float64_t                   const & minThr,
                    float64_t                   const & maxThr)
    {
        return data.unaryExpr(
        [&minThr, &maxThr](float64_t const & x) -> float64_t
        {
            return clamp(x, minThr, maxThr);
        });
    }

    float64_t clamp(float64_t const & data,
                    float64_t const & minThr,
                    float64_t const & maxThr)
    {
        if (!isnan(data))
        {
            return std::min(std::max(data, minThr), maxThr);
        }
        else
        {
            return 0.0;
        }
    }
}
