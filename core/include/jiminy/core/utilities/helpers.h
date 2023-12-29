#ifndef JIMINY_UTILITIES_H
#define JIMINY_UTILITIES_H

#include <chrono>
#include <memory>
#include <type_traits>

#include "jiminy/core/fwd.h"


namespace jiminy
{
    struct LogData;

    // *************** Local Mutex/Lock mechanism ******************

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

    // ************************ Timer *******************************

    class JIMINY_DLLAPI Timer
    {
        using Time = std::chrono::high_resolution_clock;

    public:
        explicit Timer() noexcept;

        void tic() noexcept;
        void toc() noexcept;

    public:
        std::chrono::time_point<Time> t0{};
        std::chrono::time_point<Time> tf{};
        double dt{0.0};
    };

    // ****************************** Generic template utilities ******************************* //

    template<class F, class... Args>
    std::enable_if_t<!(... && !std::is_same_v<std::invoke_result_t<F, Args>, void>)>
    do_for(F f, Args &&... args);

    template<class F, class... Args>
    std::enable_if_t<(... && !std::is_same_v<std::invoke_result_t<F, Args>, void>),
                     std::tuple<std::invoke_result_t<F, Args>...>>
    do_for(F f, Args &&... args);

    // ******************************** enable_shared_from_this ******************************** //

    template<typename Base>
    inline std::shared_ptr<Base> shared_from_base(std::enable_shared_from_this<Base> * base);

    template<typename Base>
    inline std::shared_ptr<const Base>
    shared_from_base(const std::enable_shared_from_this<Base> * base);

    template<typename T>
    inline std::shared_ptr<T> shared_from(T * derived);

    // ************* IO file and Directory utilities ****************

    std::string JIMINY_DLLAPI getUserDirectory();

    // ******************* Telemetry utilities **********************

    bool endsWith(const std::string & text, const std::string & ending);

    std::string addCircumfix(std::string fieldname,  // Make a copy
                             const std::string_view & prefix = {},
                             const std::string_view & suffix = {},
                             const std::string_view & delimiter = {});
    std::vector<std::string> addCircumfix(const std::vector<std::string> & fieldnamesIn,
                                          const std::string_view & prefix = {},
                                          const std::string_view & suffix = {},
                                          const std::string_view & delimiter = {});

    std::string removeSuffix(std::string fieldname,  // Make a copy
                             const std::string & suffix);
    std::vector<std::string> removeSuffix(const std::vector<std::string> & fieldnamesIn,
                                          const std::string & suffix);

    /// \brief Value of a single logged variable (by copy).
    ///
    /// \param[in] logData Corresponding data in the log file.
    /// \param[in] fieldName Full name of the variable to get.
    ///
    /// \return Vector of values for a given variable as a contiguous array.
    Eigen::VectorXd JIMINY_DLLAPI getLogVariable(const LogData & logData,
                                                 const std::string & fieldname);

    // ********************** Math utilities *************************

    template<typename DerivedType1, typename DerivedType2, typename DerivedType3>
    Eigen::MatrixBase<DerivedType1> clamp(const Eigen::MatrixBase<DerivedType1> & data,
                                          const Eigen::MatrixBase<DerivedType2> & minThr,
                                          const Eigen::MatrixBase<DerivedType2> & maxThr);

    template<typename... Args>
    double minClipped(double val1, double val2, Args... vs);

    template<typename... Args>
    std::tuple<bool, double> isGcdIncluded(Args... values);

    template<typename InputIt, typename UnaryFunction>
    std::tuple<bool, double> isGcdIncluded(InputIt first, InputIt last, UnaryFunction f);

    template<typename InputIt, typename UnaryFunction, typename... Args>
    std::tuple<bool, double>
    isGcdIncluded(InputIt first, InputIt last, UnaryFunction f, Args... values);

    // ********************* Std::vector helpers **********************

    template<typename T, typename A>
    bool checkDuplicates(const std::vector<T, A> & vect);

    template<typename T1, typename A1, typename T2, typename A2>
    bool checkIntersection(const std::vector<T1, A1> & vect1, const std::vector<T2, A2> & vect2);

    template<typename T1, typename A1, typename T2, typename A2>
    bool checkInclusion(const std::vector<T1, A1> & vect1, const std::vector<T2, A2> & vect2);

    template<typename T1, typename A1, typename T2, typename A2>
    void eraseVector(std::vector<T1, A1> & vect1, const std::vector<T2, A2> & vect2);

    // *********************** Miscellaneous **************************

    template<typename Derived>
    void swapMatrixRows(const Eigen::MatrixBase<Derived> & matrixIn,
                        Eigen::Index firstBlockStart,
                        Eigen::Index firstBlockLength,
                        Eigen::Index secondBlockStart,
                        Eigen::Index secondBlockLength);
}

#include "jiminy/core/utilities/helpers.hxx"

#endif  // JIMINY_UTILITIES_H
