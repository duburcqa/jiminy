#ifndef JIMINY_UTILITIES_H
#define JIMINY_UTILITIES_H

#include <chrono>
#include <vector>
#include <random>

#include "pinocchio/multibody/model.hpp"
#include "pinocchio/algorithm/frames.hpp"

#include "jiminy/core/Types.h"

namespace jiminy
{
    class TelemetrySender;

    // *************** Local Mutex /Lock mechanism ******************

    class LockGuardLocal;

    class MutexLocal
    {
        friend LockGuardLocal;

    public:
        // Disable the copy of the class
        MutexLocal(MutexLocal const & mutexLocalIn) = delete;
        MutexLocal & operator = (MutexLocal const & other) = delete;

        MutexLocal(void);
        MutexLocal(MutexLocal && other) = default;

        ~MutexLocal(void);

        bool const & isLocked(void);

    private:
        std::shared_ptr<bool> isLocked_;
    };

    class LockGuardLocal
    {
    public:
        // Disable the copy of the class
        LockGuardLocal(LockGuardLocal const & lockGuardLocalIn) = delete;
        LockGuardLocal & operator = (LockGuardLocal const & other) = delete;

        LockGuardLocal(MutexLocal & mutexLocal);
        LockGuardLocal(LockGuardLocal && other) = default;

        ~LockGuardLocal(void);

    private:
        std::shared_ptr<bool> ownerFlag_;
    };

    // ************************ Timer *******************************

    class Timer
    {
        using Time = std::chrono::high_resolution_clock;

    public:
        Timer(void);
        void tic(void);
        void toc(void);

    public:
        std::chrono::time_point<Time> t0;
        std::chrono::time_point<Time> tf;
        float64_t dt;
    };

    // ************ IO file and Directory utilities *****************

    std::string getUserDirectory(void);

    // ************ Random number generator utilities ***************

    void resetRandGenerators(uint32_t seed);

    float64_t randUniform(float64_t const & lo,
                          float64_t const & hi);

    float64_t randNormal(float64_t const & mean,
                         float64_t const & std);

    vectorN_t randVectorNormal(uint32_t  const & size,
                               float64_t const & mean,
                               float64_t const & std);

    vectorN_t randVectorNormal(uint32_t  const & size,
                               float64_t const & std);

    vectorN_t randVectorNormal(vectorN_t const & std);

    vectorN_t randVectorNormal(vectorN_t const & mean,
                               vectorN_t const & std);

    // ******************* Telemetry utilities **********************

    std::vector<std::string> defaultVectorFieldnames(std::string const & baseName,
                                                     uint32_t    const & size);

    std::string removeFieldnameSuffix(std::string         fieldname,
                                      std::string const & suffix);
    std::vector<std::string> removeFieldnamesSuffix(std::vector<std::string>         fieldnames, // Make a copy
                                                    std::string              const & suffix);

    // ******************** Pinocchio utilities *********************

    void computePositionDerivative(pinocchio::Model            const & model,
                                   Eigen::Ref<vectorN_t const>         q,
                                   Eigen::Ref<vectorN_t const>         v,
                                   Eigen::Ref<vectorN_t>               qDot,
                                   float64_t                           dt = 1e-5); // Make a copy

    // Pinocchio joint types
    enum class joint_t : int32_t
    {
        // CYLINDRICAL are not available so far

        NONE = 0,
        LINEAR = 1,
        ROTARY = 2,
        PLANAR = 3,
        SPHERICAL = 4,
        FREE = 5,
    };

    result_t getJointNameFromPositionId(pinocchio::Model const & model,
                                        int32_t          const & idIn,
                                        std::string            & jointNameOut);

    result_t getJointNameFromVelocityId(pinocchio::Model const & model,
                                        int32_t          const & idIn,
                                        std::string            & jointNameOut);

    result_t getJointTypeFromId(pinocchio::Model const & model,
                                int32_t          const & idIn,
                                joint_t                & jointTypeOut);

    result_t getJointTypePositionSuffixes(joint_t                  const & jointTypeIn,
                                          std::vector<std::string>       & jointTypeSuffixesOut);

    result_t getJointTypeVelocitySuffixes(joint_t                  const & jointTypeIn,
                                          std::vector<std::string>       & jointTypeSuffixesOut);

    result_t getFrameIdx(pinocchio::Model const & model,
                         std::string      const & frameName,
                         int32_t                & frameIdx);
    result_t getFramesIdx(pinocchio::Model         const & model,
                          std::vector<std::string> const & framesNames,
                          std::vector<int32_t>           & framesIdx);

    result_t getJointModelIdx(pinocchio::Model const & model,
                              std::string      const & jointName,
                              int32_t                & jointModelIdx);
    result_t getJointsModelIdx(pinocchio::Model         const & model,
                               std::vector<std::string> const & jointsNames,
                               std::vector<int32_t>           & jointsModelIdx);

    result_t getJointPositionIdx(pinocchio::Model     const & model,
                                 std::string          const & jointName,
                                 std::vector<int32_t>       & jointPositionIdx);
    result_t getJointPositionIdx(pinocchio::Model const & model,
                                 std::string      const & jointName,
                                 int32_t                & jointPositionFirstIdx);
    result_t getJointsPositionIdx(pinocchio::Model         const & model,
                                  std::vector<std::string> const & jointsNames,
                                  std::vector<int32_t>           & jointsPositionIdx,
                                  bool                     const & firstJointIdxOnly = false);

    result_t getJointVelocityIdx(pinocchio::Model     const & model,
                                 std::string          const & jointName,
                                 std::vector<int32_t>       & jointVelocityIdx);
    result_t getJointVelocityIdx(pinocchio::Model const & model,
                                 std::string      const & jointName,
                                 int32_t                & jointVelocityFirstIdx);
    result_t getJointsVelocityIdx(pinocchio::Model         const & model,
                                  std::vector<std::string> const & jointsNames,
                                  std::vector<int32_t>           & jointsVelocityIdx,
                                  bool                     const & firstJointIdxOnly = false);

    result_t insertFlexibilityInModel(pinocchio::Model       & modelInOut,
                                      std::string      const & childJointNameIn,
                                      std::string      const & newJointNameIn);

    // ********************** Math utilities *************************

    template<typename T0, typename T1, typename... Ts>
    typename std::common_type<T0, T1, Ts...>::type min(T0 && val1, T1 && val2, Ts &&... vs);

    float64_t saturateSoft(float64_t const & in,
                           float64_t const & mi,
                           float64_t const & ma,
                           float64_t const & r);

    vectorN_t clamp(Eigen::Ref<vectorN_t const>         data,
                    float64_t                   const & minThr = -INF,
                    float64_t                   const & maxThr = +INF);

    float64_t clamp(float64_t const & data,
                    float64_t const & minThr = -INF,
                    float64_t const & maxThr = +INF);

    // ********************* Std::vector helpers **********************

    template<typename T>
    bool checkDuplicates(std::vector<T> const & vect);

    template<typename T>
    bool checkIntersection(std::vector<T> const & vect1,
                           std::vector<T> const & vect2);

    template<typename T>
    bool checkInclusion(std::vector<T> const & vect1,
                        std::vector<T> const & vect2);

    template<typename T>
    void eraseVector(std::vector<T>       & vect1,
                     std::vector<T> const & vect2);

    // *********************** Miscellaneous **************************

    template<class F, class dF=std::decay_t<F> >
    auto notF(F&& f);

    template<typename type>
    void swapVectorBlocks(Eigen::Matrix<type, Eigen::Dynamic, 1>       & vector,
                          uint32_t                               const & firstBlockStart,
                          uint32_t                               const & firstBlockLength,
                          uint32_t                               const & secondBlockStart,
                          uint32_t                               const & secondBlockLength);
}

#include "jiminy/core/Utilities.tpp"

#endif  // JIMINY_UTILITIES_H
