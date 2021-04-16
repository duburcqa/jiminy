#ifndef JIMINY_UTILITIES_H
#define JIMINY_UTILITIES_H

#include <chrono>
#include <type_traits>

#include "jiminy/core/Macros.h"
#include "jiminy/core/Types.h"


namespace jiminy
{
    // *************** Local Mutex/Lock mechanism ******************

    class LockGuardLocal;

    class MutexLocal
    {
        friend LockGuardLocal;

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

    // ********************** Math utilities *************************

    template<typename T0, typename T1, typename... Ts>
    typename std::common_type<T0, T1, Ts...>::type min(T0 && val1, T1 && val2, Ts &&... vs);

    template<typename DerivedType>
    auto clamp(Eigen::MatrixBase<DerivedType> const & data,
               float64_t const & minThr = -INF,
               float64_t const & maxThr = +INF);

    template<typename DerivedType1, typename DerivedType2, typename DerivedType3>
    Eigen::MatrixBase<DerivedType1> clamp(Eigen::MatrixBase<DerivedType1> const & data,
                                          Eigen::MatrixBase<DerivedType2> const & minThr,
                                          Eigen::MatrixBase<DerivedType2> const & maxThr);

    float64_t clamp(float64_t const & data,
                    float64_t const & minThr = -INF,
                    float64_t const & maxThr = +INF);

    template<typename... Args>
    float64_t minClipped(float64_t val1, float64_t val2, Args ... vs);

    template<typename ...Args>
    std::tuple<bool_t, float64_t> isGcdIncluded(Args... values);

    template<typename InputIt, typename UnaryFunction>
    std::tuple<bool_t, float64_t> isGcdIncluded(InputIt first, InputIt last, UnaryFunction f);

    template<typename InputIt, typename UnaryFunction, typename ...Args>
    std::tuple<bool_t, float64_t> isGcdIncluded(InputIt first, InputIt last, UnaryFunction f, Args... values);

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

#include "jiminy/core/utilities/Helpers.tpp"

#endif  // JIMINY_UTILITIES_H
