#ifndef JIMINY_UTILITIES_H
#define JIMINY_UTILITIES_H

#include <chrono>
#include <type_traits>

#include "jiminy/core/macros.h"
#include "jiminy/core/types.h"


namespace jiminy
{
    // *************** Local Mutex/Lock mechanism ******************

    class LockGuardLocal;

    class MutexLocal
    {
        friend LockGuardLocal;

    public:
        DISABLE_COPY(MutexLocal)

    public:
        MutexLocal();
        MutexLocal(MutexLocal && other) = default;

        ~MutexLocal();

        const bool_t & isLocked() const;

    private:
        std::shared_ptr<bool_t> isLocked_;
    };

    class LockGuardLocal
    {
    public:
        DISABLE_COPY(LockGuardLocal)

    public:
        LockGuardLocal(MutexLocal & mutexLocal);
        LockGuardLocal(LockGuardLocal && other) = default;

        ~LockGuardLocal();

    private:
        std::shared_ptr<bool_t> mutexFlag_;
    };

    // ************************ Timer *******************************

    class Timer
    {
        using Time = std::chrono::high_resolution_clock;

    public:
        Timer();
        void tic();
        void toc();

    public:
        std::chrono::time_point<Time> t0;
        std::chrono::time_point<Time> tf;
        float64_t dt;
    };

    // ************* IO file and Directory utilities ****************

    std::string getUserDirectory();

    // ******************* Telemetry utilities **********************

    struct logData_t;

    bool_t endsWith(const std::string & fullString, const std::string & ending);

    std::vector<std::string> defaultVectorFieldnames(const std::string & baseName,
                                                     const uint32_t & size);


    std::string addCircumfix(std::string fieldname,  // Copy on purpose
                             const std::string & prefix = "",
                             const std::string & suffix = "",
                             const std::string & delimiter = "");
    std::vector<std::string> addCircumfix(const std::vector<std::string> & fieldnamesIn,
                                          const std::string & prefix = "",
                                          const std::string & suffix = "",
                                          const std::string & delimiter = "");

    std::string removeSuffix(std::string fieldname,  // Copy on purpose
                             const std::string & suffix);
    std::vector<std::string> removeSuffix(const std::vector<std::string> & fieldnamesIn,
                                          const std::string & suffix);

    /// \brief Value of a single logged variable (by copy).
    ///
    /// \param[in] logData Corresponding data in the log file.
    /// \param[in] fieldName Full name of the variable to get.
    ///
    /// \return Vector of values for a given variable as a contiguous array.
    Eigen::VectorXd getLogVariable(const logData_t & logData, const std::string & fieldname);

    // ********************** Math utilities *************************

    template<typename T0, typename T1, typename... Ts>
    typename std::common_type_t<T0, T1, Ts...> min(T0 && val1, T1 && val2, Ts &&... vs);

    template<typename DerivedType1, typename DerivedType2, typename DerivedType3>
    Eigen::MatrixBase<DerivedType1> clamp(const Eigen::MatrixBase<DerivedType1> & data,
                                          const Eigen::MatrixBase<DerivedType2> & minThr,
                                          const Eigen::MatrixBase<DerivedType2> & maxThr);

    template<typename... Args>
    float64_t minClipped(float64_t val1, float64_t val2, Args... vs);

    template<typename... Args>
    std::tuple<bool_t, float64_t> isGcdIncluded(Args... values);

    template<typename InputIt, typename UnaryFunction>
    std::tuple<bool_t, float64_t> isGcdIncluded(InputIt first, InputIt last, UnaryFunction f);

    template<typename InputIt, typename UnaryFunction, typename... Args>
    std::tuple<bool_t, float64_t>
    isGcdIncluded(InputIt first, InputIt last, UnaryFunction f, Args... values);

    // ********************* Std::vector helpers **********************

    template<typename T, typename A>
    bool_t checkDuplicates(const std::vector<T, A> & vect);

    template<typename T, typename A>
    bool_t checkIntersection(const std::vector<T, A> & vect1, const std::vector<T, A> & vect2);

    template<typename T, typename A>
    bool_t checkInclusion(const std::vector<T, A> & vect1, const std::vector<T, A> & vect2);

    template<typename T, typename A>
    void eraseVector(std::vector<T, A> & vect1, const std::vector<T, A> & vect2);

    // *********************** Miscellaneous **************************

    template<typename Derived>
    void swapMatrixBlocks(const Eigen::MatrixBase<Derived> & matrixIn,
                          const Eigen::Index & firstBlockStart,
                          const Eigen::Index & firstBlockLength,
                          const Eigen::Index & secondBlockStart,
                          const Eigen::Index & secondBlockLength);
}

#include "jiminy/core/utilities/helpers.hxx"

#endif  // JIMINY_UTILITIES_H
