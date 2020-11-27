#ifndef JIMINY_UTILITIES_H
#define JIMINY_UTILITIES_H

#include <chrono>
#include <type_traits>

#include "pinocchio/multibody/joint/joints.hpp"

#include "json/json.h"

#include "jiminy/core/Macro.h"
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

    // *************** Conversion to JSON utilities *****************

    class AbstractIODevice;

    template<typename T>
    std::enable_if_t<!is_vector<T>::value, Json::Value>
    convertToJson(T const & value);

    template<typename T>
    std::enable_if_t<is_vector<T>::value, Json::Value>
    convertToJson(T const & value);

    hresult_t jsonDump(configHolder_t                    const & config,
                       std::shared_ptr<AbstractIODevice>       & device);

    // ************* Conversion from JSON utilities *****************

    template<typename T>
    std::enable_if_t<!is_vector<T>::value, T>
    convertFromJson(Json::Value const & value);

    template<typename T>
    std::enable_if_t<is_vector<T>::value, T>
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

    template<typename Scalar, int32_t Options, int32_t axis>
    int32_t getJointAxis(pinocchio::JointModelBase<pinocchio::JointModelRevoluteTpl<Scalar, Options, axis> > const & joint);

    hresult_t getJointNameFromPositionIdx(pinocchio::Model const & model,
                                          int32_t          const & idIn,
                                          std::string            & jointNameOut);

    hresult_t getJointNameFromVelocityIdx(pinocchio::Model const & model,
                                          int32_t          const & idIn,
                                          std::string            & jointNameOut);

    hresult_t getJointTypeFromIdx(pinocchio::Model const & model,
                                  int32_t          const & idIn,
                                  joint_t                & jointTypeOut);

    hresult_t getJointTypePositionSuffixes(joint_t                  const & jointTypeIn,
                                           std::vector<std::string>       & jointTypeSuffixesOut);

    hresult_t getJointTypeVelocitySuffixes(joint_t                  const & jointTypeIn,
                                           std::vector<std::string>       & jointTypeSuffixesOut);

    hresult_t getBodyIdx(pinocchio::Model const & model,
                         std::string      const & bodyName,
                         int32_t                & bodyIdx);
    hresult_t getBodiesIdx(pinocchio::Model         const & model,
                           std::vector<std::string> const & bodiesNames,
                           std::vector<int32_t>           & bodiesIdx);

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

    hresult_t isPositionValid(pinocchio::Model const & model,
                              vectorN_t        const & position,
                              bool_t                 & isValid);

    hresult_t insertFlexibilityInModel(pinocchio::Model       & modelInOut,
                                       std::string      const & childJointNameIn,
                                       std::string      const & newJointNameIn);

    /// \brief Convert a force expressed in the global frame of a specific frame to its parent joint frame.
    ///
    /// \param[in] model        Pinocchio model.
    /// \param[in] data         Pinocchio data.
    /// \param[in] frameIdx     Id of the frame.
    /// \param[in] fextInGlobal Force in the global frame to be converted.
    /// \return Force in the parent joint local frame.
    pinocchio::Force convertForceGlobalFrameToJoint(pinocchio::Model const & model,
                                                    pinocchio::Data  const & data,
                                                    int32_t          const & frameIdx,
                                                    pinocchio::Force const & fextInGlobal);

    // ********************** Math utilities *************************

    template<typename T0, typename T1, typename... Ts>
    typename std::common_type<T0, T1, Ts...>::type min(T0 && val1, T1 && val2, Ts &&... vs);

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

    template<typename type>
    void swapVectorBlocks(Eigen::Matrix<type, Eigen::Dynamic, 1>       & vector,
                          uint32_t                               const & firstBlockStart,
                          uint32_t                               const & firstBlockLength,
                          uint32_t                               const & secondBlockStart,
                          uint32_t                               const & secondBlockLength);
}

#include "jiminy/core/Utilities.tpp"

#endif  // JIMINY_UTILITIES_H
