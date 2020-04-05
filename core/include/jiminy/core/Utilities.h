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

    template<typename T>
    struct type_identity {
        using type = T;
    };

    template<bool B, class T = void>
    using enable_if_t = typename std::enable_if<B,T>::type;

    // ================= enable_shared_from_this ====================

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
    // ======================== is_vector ===========================

    template<typename T>
    struct is_vector : std::false_type {};

    template<typename T>
    struct is_vector<std::vector<T> > : std::true_type {};

    // ========================= is_eigen ===========================

    namespace isEigenObjectDetail {
        template <typename T, int RowsAtCompileTime, int ColsAtCompileTime>
        std::true_type test(Eigen::Matrix<T, RowsAtCompileTime, ColsAtCompileTime> const *);
        template <typename T, int RowsAtCompileTime, int ColsAtCompileTime>
        std::true_type test(Eigen::Ref<Eigen::Matrix<T, RowsAtCompileTime, ColsAtCompileTime> > const *);
        template <typename T, int RowsAtCompileTime, int ColsAtCompileTime>
        std::true_type test(Eigen::Ref<Eigen::Matrix<T, RowsAtCompileTime, ColsAtCompileTime> const> const *);
        std::false_type test(...);
    }

    template <typename T>
    struct isEigenObject :
        public decltype(isEigenObjectDetail::test(std::declval<T*>())) {};

    template<typename T, typename Enable = void>
    struct is_eigen : public std::false_type {};

    template<typename T>
    struct is_eigen<T, typename std::enable_if<isEigenObject<T>::value>::type> : std::true_type {};

    // ====================== is_eigen_vector =======================

    namespace isEigenVectorDetail {
        template <typename T, int RowsAtCompileTime>
        std::true_type test(Eigen::Matrix<T, RowsAtCompileTime, 1> const *);
        template <typename T, int RowsAtCompileTime>
        std::true_type test(Eigen::Ref<Eigen::Matrix<T, RowsAtCompileTime, 1> > const *);
        template <typename T, int RowsAtCompileTime>
        std::true_type test(Eigen::Ref<Eigen::Matrix<T, RowsAtCompileTime, 1> const> const *);
        std::false_type test(...);
    }

    template <typename T>
    struct isEigenVector : public decltype(isEigenVectorDetail::test(std::declval<T*>())) {};

    template<typename T, typename Enable = void>
    struct is_eigen_vector : std::false_type {};

    template<typename T>
    struct is_eigen_vector<T, typename std::enable_if<isEigenVector<T>::value>::type> : std::true_type {};

    // ========================== is_map ============================

    namespace isMapDetail {
        template<typename K, typename T>
        std::true_type test(std::map<K, T> const *);
        template<typename K, typename T>
        std::true_type test(std::unordered_map<K, T> const *);
        std::false_type test(...);
    }

    template <typename T>
    struct isMap : public decltype(isMapDetail::test(std::declval<T*>())) {};

    template<typename T, typename Enable = void>
    struct is_map : std::false_type {};

    template<typename T>
    struct is_map<T, typename std::enable_if<isMap<T>::value>::type> : std::true_type {};

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
                             std::string const & suffix = "",
                             std::string const & delimiter = "");
    std::vector<std::string> addCircumfix(std::vector<std::string> const & fieldnamesIn,
                                          std::string              const & prefix = "",
                                          std::string              const & suffix = "",
                                          std::string              const & delimiter = "");

    std::string removeSuffix(std::string         fieldname, // Make a copy
                             std::string const & suffix);
    std::vector<std::string> removeSuffix(std::vector<std::string> const & fieldnamesIn, // Make a copy
                                          std::string              const & suffix);

    /// \brief Get the value of a single logged variable.
    ///
    /// \param[in] fieldName    Full name of the variable to get
    /// \param[in] header       Header, vector of field names.
    /// \param[in] logData      Corresponding data in the log file.
    ///
    /// \return Vector of values for fieldName. If fieldName is not in the header list, this vector will be empty.
    Eigen::Ref<vectorN_t const> getLogFieldValue(std::string              const & fieldName,
                                                 std::vector<std::string> const & header,
                                                 matrixN_t                const & logData);

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

    hresult_t computePositionDerivative(pinocchio::Model            const & model,
                                        Eigen::Ref<vectorN_t const> const & q,
                                        Eigen::Ref<vectorN_t const> const & v,
                                        Eigen::Ref<vectorN_t>             & qDot,
                                        float64_t                   const & dt);
    hresult_t computePositionDerivative(pinocchio::Model            const & model,
                                        Eigen::Ref<vectorN_t const> const & q,
                                        Eigen::Ref<vectorN_t const> const & v,
                                        vectorN_t                         & qDot,
                                        float64_t                   const & dt);

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

    pinocchio::Force computeFrameForceOnParentJoint(pinocchio::Model const & model,
                                                    pinocchio::Data  const & data,
                                                    int32_t          const & frameId,
                                                    pinocchio::Force const & fextInWorld);

    // ********************** Math utilities *************************

    template<typename T0, typename T1, typename... Ts>
    typename std::common_type<T0, T1, Ts...>::type min(T0 && val1, T1 && val2, Ts &&... vs);

    float64_t saturateSoft(float64_t const & in,
                           float64_t const & mi,
                           float64_t const & ma,
                           float64_t const & r);

    vectorN_t clamp(Eigen::Ref<vectorN_t const> const & data,
                    float64_t                   const & minThr = -INF,
                    float64_t                   const & maxThr = +INF);

    float64_t clamp(float64_t const & data,
                    float64_t const & minThr = -INF,
                    float64_t const & maxThr = +INF);

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

    void catInPlace(std::vector<vectorN_t> const & xList,
                    vectorN_t                    & xCat);
    vectorN_t cat(std::vector<vectorN_t> const & xList);
}

#include "jiminy/core/Utilities.tpp"

#endif  // JIMINY_UTILITIES_H
