#ifndef JIMINY_UTILITIES_H
#define JIMINY_UTILITIES_H

#include <chrono>
#include <type_traits>

#include "json/json.h"

#include "jiminy/core/Types.h"


namespace jiminy
{
    // *************** Local Mutex/Lock mechanism ******************

    class MutexLocal
    {
    public:
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
            std::shared_ptr<bool_t> mutexFlag_;
        };

    public:
        // Disable the copy of the class
        MutexLocal(MutexLocal const & mutexLocalIn) = delete;
        MutexLocal & operator = (MutexLocal const & other) = delete;

    public:
        MutexLocal(void);
        MutexLocal(MutexLocal && other) = default;

        ~MutexLocal(void);

        bool_t const & isLocked(void) const;

    private:
        std::shared_ptr<bool_t> isLocked_;
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

    // ************* IO file and Directory utilities ****************

    std::string getUserDirectory(void);

    // **************** Generic template utilities ******************

    template<bool B, class T = void>
    using enable_if_t = typename std::enable_if<B,T>::type;

    template<typename T>
    struct is_vector : std::integral_constant<bool, false> {};

    template<typename T>
    struct is_vector<std::vector<T> > : std::integral_constant<bool, true> {};

    template <typename Base>
    inline std::shared_ptr<Base>
    shared_from_base(std::enable_shared_from_this<Base>* base)
    {
        return base->shared_from_this();
    }
    template <typename Base>
    inline std::shared_ptr<const Base>
    shared_from_base(std::enable_shared_from_this<Base> const* base)
    {
        return base->shared_from_this();
    }
    template <typename That>
    inline std::shared_ptr<That>
    shared_from(That* that)
    {
        return std::static_pointer_cast<That>(shared_from_base(that));
    }

    // *************** Convertion to JSON utilities *****************

    class AbstractIODevice;

    template<typename T>
    enable_if_t<!is_vector<T>::value, Json::Value>
    convertToJson(T const & value);

    template<typename T>
    enable_if_t<is_vector<T>::value, Json::Value>
    convertToJson(T const & value);

    hresult_t jsonDump(configHolder_t                    const & config,
                       std::shared_ptr<AbstractIODevice>       & device);

    // ************* Convertion from JSON utilities *****************

    template<typename T>
    enable_if_t<!is_vector<T>::value, T>
    convertFromJson(Json::Value const & value);

    template<typename T>
    enable_if_t<is_vector<T>::value, T>
    convertFromJson(Json::Value const & value);

    hresult_t jsonLoad(configHolder_t                    & config,
                       std::shared_ptr<AbstractIODevice> & device);

    // ************ Random number generator utilities ***************

    void resetRandGenerators(uint32_t const & seed);

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

    std::string addCircumfix(std::string         fieldname, // Make a copy
                             std::string const & prefix = "",
                             std::string const & suffix = "");
    std::vector<std::string> addCircumfix(std::vector<std::string> const & fieldnamesIn,
                                          std::string              const & prefix = "",
                                          std::string              const & suffix = "");

    std::string removeSuffix(std::string         fieldname, // Make a copy
                             std::string const & suffix);
    std::vector<std::string> removeSuffix(std::vector<std::string> const & fieldnamesIn, // Make a copy
                                          std::string              const & suffix);

    // ******************** Pinocchio utilities *********************

    // Pinocchio joint types
    enum class joint_t : uint8_t
    {
        // CYLINDRICAL are not available so far

        NONE = 0,
        LINEAR = 1,
        ROTARY = 2,
        PLANAR = 3,
        SPHERICAL = 4,
        FREE = 5,
    };

    void computePositionDerivative(pinocchio::Model const & model,
                                   Eigen::Ref<vectorN_t const> q,
                                   Eigen::Ref<vectorN_t const> v,
                                   Eigen::Ref<vectorN_t> qDot,
                                   float64_t dt = 1e-5); // Make a copy

    hresult_t getJointNameFromPositionId(pinocchio::Model const & model,
                                         int32_t          const & idIn,
                                         std::string            & jointNameOut);

    hresult_t getJointNameFromVelocityId(pinocchio::Model const & model,
                                         int32_t          const & idIn,
                                         std::string            & jointNameOut);

    hresult_t getJointTypeFromId(pinocchio::Model const & model,
                                 int32_t          const & idIn,
                                 joint_t                & jointTypeOut);

    hresult_t getJointTypePositionSuffixes(joint_t                  const & jointTypeIn,
                                           std::vector<std::string>       & jointTypeSuffixesOut);

    hresult_t getJointTypeVelocitySuffixes(joint_t                  const & jointTypeIn,
                                           std::vector<std::string>       & jointTypeSuffixesOut);

    hresult_t getFrameIdx(pinocchio::Model const & model,
                          std::string      const & frameName,
                          int32_t                & frameIdx);
    hresult_t getFramesIdx(pinocchio::Model         const & model,
                           std::vector<std::string> const & framesNames,
                           std::vector<int32_t>           & framesIdx);

    hresult_t getJointModelIdx(pinocchio::Model const & model,
                               std::string      const & jointName,
                               int32_t                & jointModelIdx);
    hresult_t getJointsModelIdx(pinocchio::Model         const & model,
                                std::vector<std::string> const & jointsNames,
                                std::vector<int32_t>           & jointsModelIdx);

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

    hresult_t insertFlexibilityInModel(pinocchio::Model       & modelInOut,
                                       std::string      const & childJointNameIn,
                                       std::string      const & newJointNameIn);

    vector6_t computeFrameForceOnParentJoint(pinocchio::Model const & model,
                                             pinocchio::Data  const & data,
                                             int32_t          const & frameId,
                                             vector3_t        const & fextInWorld);

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

    void catInPlace(std::vector<vectorN_t> const & xList,
                    vectorN_t                    & xCat);
    vectorN_t cat(std::vector<vectorN_t> const & xList);

    // ********************* Std::vector helpers **********************

    template<typename T>
    bool_t checkDuplicates(std::vector<T> const & vect);

    template<typename T>
    bool_t checkIntersection(std::vector<T> const & vect1,
                             std::vector<T> const & vect2);

    template<typename T>
    bool_t checkInclusion(std::vector<T> const & vect1,
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
