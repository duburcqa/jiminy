#ifndef JIMINY_HELPERS_H
#define JIMINY_HELPERS_H

#include <chrono>
#include <memory>
#include <type_traits>

#include "jiminy/core/fwd.h"


namespace jiminy
{
    struct LogData;

    // ******************************* Local Mutex/Lock mechanism ****************************** //

    class LockGuardLocal;

    class JIMINY_DLLAPI MutexLocal
    {
        friend LockGuardLocal;

    public:
        DISABLE_COPY(MutexLocal)

    public:
        explicit MutexLocal() = default;
        MutexLocal(MutexLocal && other) = default;

        ~MutexLocal();

        bool isLocked() const noexcept;

    private:
        std::shared_ptr<bool> isLocked_{std::make_shared<bool>(false)};
    };

    class JIMINY_DLLAPI LockGuardLocal
    {
    public:
        DISABLE_COPY(LockGuardLocal)

    public:
        explicit LockGuardLocal(MutexLocal & mutexLocal) noexcept;
        LockGuardLocal(LockGuardLocal && other) = default;

        ~LockGuardLocal();

    private:
        std::shared_ptr<bool> mutexFlag_;
    };

    // ***************************************** Timer ***************************************** //

    class JIMINY_DLLAPI Timer
    {
        using clock = std::chrono::high_resolution_clock;

    public:
        explicit Timer() noexcept;

        void tic() noexcept;

        template<typename Period = std::ratio<1>>
        double toc() const noexcept;

    private:
        std::chrono::time_point<clock> t0_{};
    };

    // ****************************** Generic template utilities ******************************* //

    template<class F, class... Args>
    std::enable_if_t<!(... && !std::is_same_v<std::invoke_result_t<F, Args>, void>)>
    do_for(F func, Args &&... args);

    template<class F, class... Args>
    std::enable_if_t<(... && !std::is_same_v<std::invoke_result_t<F, Args>, void>),
                     std::tuple<std::invoke_result_t<F, Args>...>>
    do_for(F func, Args &&... args);

    // ******************************** enable_shared_from_this ******************************** //

    template<typename Base>
    inline std::shared_ptr<Base> shared_from_base(std::enable_shared_from_this<Base> * base);

    template<typename Base>
    inline std::shared_ptr<const Base>
    shared_from_base(const std::enable_shared_from_this<Base> * base);

    template<typename T>
    inline std::shared_ptr<T> shared_from(T * derived);

    // **************************** IO file and Directory utilities **************************** //

    std::string JIMINY_DLLAPI getUserDirectory();

    // ********************************** Telemetry utilities ********************************** //

    bool endsWith(const std::string & str, const std::string & substr);

    std::string addCircumfix(std::string fieldname,  // Make a copy
                             const std::string_view & prefix = {},
                             const std::string_view & suffix = {},
                             const std::string_view & delimiter = {});
    std::vector<std::string> addCircumfix(const std::vector<std::string> & fieldnames,
                                          const std::string_view & prefix = {},
                                          const std::string_view & suffix = {},
                                          const std::string_view & delimiter = {});

    std::string removeSuffix(std::string fieldname,  // Make a copy
                             const std::string & suffix);
    std::vector<std::string> removeSuffix(const std::vector<std::string> & fieldnames,
                                          const std::string & suffix);

    /// \brief Value of a single logged variable (by copy).
    ///
    /// \param[in] logData Corresponding data in the log file.
    /// \param[in] fieldName Full name of the variable to get.
    ///
    /// \return Vector of values for a given variable as a contiguous array.
    Eigen::VectorXd JIMINY_DLLAPI getLogVariable(const LogData & logData,
                                                 const std::string & fieldname);

    // ************************************* Math utilities ************************************ //

    template<typename... Args>
    std::enable_if_t<std::conjunction_v<std::is_same<Args, double>...>, const double &>
    minClipped(const double & value1, const double & value2, const Args &... values);

    template<typename... Args>
    std::enable_if_t<std::conjunction_v<std::is_same<Args, double>...>,
                     std::tuple<bool, const double &>>
    isGcdIncluded(const Args &... values);

    template<typename InputIt, typename UnaryFunction>
    std::enable_if_t<std::is_invocable_r_v<const double &,
                                           UnaryFunction,
                                           typename std::iterator_traits<InputIt>::reference>,
                     std::tuple<bool, const double &>>
    isGcdIncluded(InputIt first, InputIt last, const UnaryFunction & func);

    template<typename InputIt, typename UnaryFunction, typename... Args>
    std::enable_if_t<std::is_invocable_r_v<const double &,
                                           UnaryFunction,
                                           typename std::iterator_traits<InputIt>::reference> &&
                         std::conjunction_v<std::is_same<Args, double>...>,
                     std::tuple<bool, const double &>>
    isGcdIncluded(InputIt first, InputIt last, const UnaryFunction & func, const Args &... values);

    // ********************************** Std::vector helpers ********************************** //

    template<typename T>
    std::enable_if_t<is_vector_v<T>, bool> checkDuplicates(const T & vect);

    template<typename T1, typename T2>
    std::enable_if_t<is_vector_v<T1> && is_vector_v<T2>, bool> checkIntersection(const T1 & vec1,
                                                                                 const T2 & vec2);

    template<typename T1, typename T2>
    std::enable_if_t<is_vector_v<T1> && is_vector_v<T2>, bool> checkInclusion(const T1 & vec1,
                                                                              const T2 & vec2);

    template<typename T1, typename T2>
    std::enable_if_t<is_vector_v<T1> && is_vector_v<T2>, void> eraseVector(const T1 & vec1,
                                                                           const T2 & vec2);

    // ************************************* Miscellaneous ************************************* //

    /// \brief Swap two disjoint row-blocks of data in a matrix.
    ///
    /// \details Let b1, b2 be two row-blocks of arbitrary sizes of a matrix B s.t.
    ///          B = (... b1 ... b2 ...).T. This function re-assigns B to (... b2 ... b1 ...).T.
    ///
    /// \pre firstBlockStart + firstBlockSize <= secondBlockStart
    ///
    /// \param[in, out] matrix Matrix to modify.
    /// \param[in] firstBlockStart Start index of the first block.
    /// \param[in] firstBlockSize Length of the first block.
    /// \param[in] secondBlockStart Start index of the second block.
    /// \param[in] secondBlockSize Length of the second block.
    template<typename Derived>
    void swapMatrixRows(const Eigen::MatrixBase<Derived> & matrixIn,
                        Eigen::Index firstBlockStart,
                        Eigen::Index firstBlockSize,
                        Eigen::Index secondBlockStart,
                        Eigen::Index secondBlockSize);
}

#include "jiminy/core/utilities/helpers.hxx"

#endif  // JIMINY_HELPERS_H
