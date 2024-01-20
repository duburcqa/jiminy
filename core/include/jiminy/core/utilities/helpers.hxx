#ifndef JIMINY_HELPERS_HXX
#define JIMINY_HELPERS_HXX

#include <algorithm>
#include <numeric>


namespace jiminy
{
    // ***************************************** Timer ***************************************** //

    template<typename Period>
    double Timer::toc() const noexcept
    {
        const typename clock::duration dt_{clock::now() - t0_};
        return std::chrono::duration_cast<std::chrono::duration<double, Period>>(dt_).count();
    }

    // ****************************** Generic template utilities ******************************* //

    template<class F, class... Args>
    std::enable_if_t<!(... && !std::is_same_v<std::invoke_result_t<F, Args>, void>)>
    do_for(F func, Args &&... args)
    {
        (func(std::forward<Args>(args)), ...);
    }

    template<class F, class... Args>
    std::enable_if_t<(... && !std::is_same_v<std::invoke_result_t<F, Args>, void>),
                     std::tuple<std::invoke_result_t<F, Args>...>>
    do_for(F func, Args &&... args)
    {
        return std::tuple{func(std::forward<Args>(args))...};
    }

    // ******************************** enable_shared_from_this ******************************** //

    template<typename Base>
    inline std::shared_ptr<Base> shared_from_base(std::enable_shared_from_this<Base> * base)
    {
        return base->shared_from_this();
    }

    template<typename Base>
    inline std::shared_ptr<const Base>
    shared_from_base(const std::enable_shared_from_this<Base> * base)
    {
        return base->shared_from_this();
    }

    template<typename T>
    inline std::shared_ptr<T> shared_from(T * derived)
    {
        return std::static_pointer_cast<T>(shared_from_base(derived));
    }

    // ************************************* Math utilities ************************************ //

    inline const double & minClipped()
    {
        return INF;
    }

    inline const double & minClipped(const double & val)
    {
        if (val > EPS)
        {
            return val;
        }
        return INF;
    }

    template<typename... Args>
    const double & minClipped(const double & value1, const double & value2, const Args &... values)
    {
        const bool isValid1 = value1 > EPS;
        const bool isValid2 = value2 > EPS;
        if (isValid1 && isValid2)
        {
            return minClipped(std::min(value1, value2), values...);
        }
        else if (isValid2)
        {
            return minClipped(value2, values...);
        }
        else if (isValid1)
        {
            return minClipped(value1, values...);
        }
        return minClipped(values...);
    }

    template<typename... Args>
    std::tuple<bool, const double &> isGcdIncluded(const Args &... values)
    {
        const double & minValue = minClipped(values...);
        if (!std::isfinite(minValue))
        {
            return {true, INF};
        }
        auto lambda = [&minValue](double value)
        {
            if (value < EPS)
            {
                return true;
            }
            return std::fmod(value, minValue) < EPS;
        };
        // Taking advantage of C++17 "fold expression"
        return {(... && lambda(values)), minValue};
    }

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
    template<typename InputIt, typename UnaryFunction>
    std::enable_if_t<std::is_invocable_r_v<const double &,
                                           UnaryFunction,
                                           typename std::iterator_traits<InputIt>::reference>,
                     std::tuple<bool, const double &>>
    isGcdIncluded(InputIt first, InputIt last, const UnaryFunction & func)
    {
        const double & minValue = std::transform_reduce(first, last, INF, minClipped<>, func);
        if (!std::isfinite(minValue))
        {
            return {true, INF};
        }
        auto lambda = [minValue, &func](const auto & elem)
        {
            const double value = func(elem);
            if (value < EPS)
            {
                return true;
            }
            return std::fmod(value, minValue) < EPS;
        };
        return {std::all_of(first, last, lambda), minValue};
    }

    template<typename InputIt, typename UnaryFunction, typename... Args>
    std::enable_if_t<std::is_invocable_r_v<const double &,
                                           UnaryFunction,
                                           typename std::iterator_traits<InputIt>::reference>,
                     std::tuple<bool, const double &>>
    isGcdIncluded(InputIt first, InputIt last, const UnaryFunction & func, const Args &... values)
    {
        const auto [isIncluded1, value1] = isGcdIncluded(values...);
        const auto [isIncluded2, value2] = isGcdIncluded(first, last, func);
        if (!isIncluded1 || !isIncluded2)
        {
            return {false, INF};
        }
        if (value1 < value2)  // inf < inf := false
        {
            if (!std::isfinite(value2))
            {
                return {true, value1};
            }
            return {std::fmod(value2, value1) < EPS, value1};
        }
        else
        {
            if (!std::isfinite(value2))
            {
                if (std::isfinite(value1))
                {
                    return {true, value1};
                }
                return {true, INF};
            }
            return {std::fmod(value1, value2) < EPS, value2};
        }
    }
#pragma GCC diagnostic pop

    // ********************************** Std::vector helpers ********************************** //

    template<typename T>
    std::enable_if_t<is_vector_v<T>, bool> checkDuplicates(const T & vec)
    {
        const auto vecEnd = vec.cend();
        for (auto vecIt = vec.cbegin(); vecIt != vecEnd; ++vecIt)
        {
            if (std::find(std::next(vecIt), vecEnd, *vecIt) != vecEnd)
            {
                return true;
            }
        }
        return false;
    }

    template<typename T1, typename T2>
    std::enable_if_t<is_vector_v<T1> && is_vector_v<T2>, bool> checkIntersection(const T1 & vec1,
                                                                                 const T2 & vec2)
    {
        const auto vec2It =
            std::find_if(vec2.cbegin(),
                         vec2.cend(),
                         [vec1Begin = vec1.cbegin(), vec1End = vec1.cend()](const auto & elem2)
                         {
                             auto vec1It = std::find(vec1Begin, vec1End, elem2);
                             return (vec1It != vec1End);
                         });
        return (vec2It != vec2.cend());
    }

    template<typename T1, typename T2>
    std::enable_if_t<is_vector_v<T1> && is_vector_v<T2>, bool> checkInclusion(const T1 & vec1,
                                                                              const T2 & vec2)
    {
        const auto vec1End = vec1.cend();
        for (const auto & elem2 : vec2)
        {
            const auto vec1It = std::find(vec1.cbegin(), vec1End, elem2);
            if (vec1It == vec1End)
            {
                return false;
            }
        }
        return true;
    }

    template<typename T1, typename T2>
    std::enable_if_t<is_vector_v<T1> && is_vector_v<T2>, void> eraseVector(T1 & vec1,
                                                                           const T2 & vec2)
    {
        vec1.erase(
            std::remove_if(vec1.begin(),
                           vec1.end(),
                           [vec2Begin = vec2.cbegin(), vec2End = vec2.cend()](const auto & elem1)
                           {
                               auto vec2It = std::find(vec2Begin, vec2End, elem1);
                               return (vec2It != vec2End);
                           }),
            vec1.end());
    }

    // ************************************* Miscellaneous ************************************* //

    /// \brief Swap two disjoint row-blocks of data in a matrix.
    ///
    /// \details Let b1, b2 be two row-blocks of arbitrary sizes of a matrix B s.t.
    ///          B = (... b1 ... b2 ...).T. This function re-assigns B to (... b2 ... b1 ...).T.
    ///
    /// \pre firstBlockStart + firstBlockLength <= secondBlockStart
    ///
    /// \param[in, out] matrix Matrix to modify.
    /// \param[in] firstBlockStart Start index of the first block.
    /// \param[in] firstBlockLength Length of the first block.
    /// \param[in] secondBlockStart Start index of the second block.
    /// \param[in] secondBlockLength Length of the second block.
    template<typename Derived>
    void swapMatrixRows(const Eigen::MatrixBase<Derived> & matrixIn,
                        Eigen::Index firstBlockStart,
                        Eigen::Index firstBlockLength,
                        Eigen::Index secondBlockStart,
                        Eigen::Index secondBlockLength)
    {
        // Get plain matrix type and cast away constness
        using Matrix = typename Eigen::MatrixBase<Derived>::PlainObject;
        Eigen::MatrixBase<Derived> & matrix = const_cast<Eigen::MatrixBase<Derived> &>(matrixIn);

        // Extract first plus middle block by copy
        const Eigen::Index middleBlockStart = firstBlockStart + firstBlockLength;
        const Eigen::Index middleBlockLength = secondBlockStart - middleBlockStart;
        assert(middleBlockLength >= 0 && "First and second blocks must not overlap");
        const Eigen::Index firstMiddleBlockLength = firstBlockLength + middleBlockLength;
        const Matrix firstMiddleBlock = matrix.middleRows(firstBlockStart, firstMiddleBlockLength);

        // Re-assign first block to second block
        auto secondBlock = matrix.middleRows(secondBlockStart, secondBlockLength);
        matrix.middleRows(firstBlockStart, secondBlockLength) = secondBlock;

        // Shift middle block
        auto middleBlock = firstMiddleBlock.bottomRows(middleBlockLength);
        matrix.middleRows(firstBlockStart + secondBlockLength, middleBlockLength) = middleBlock;

        // Re-assign second block to first block
        auto firstBlock = firstMiddleBlock.topRows(firstBlockLength);
        const Eigen::Index secondBlockEnd = secondBlockStart + secondBlockLength;  // Excluded
        matrix.middleRows(secondBlockEnd - firstBlockLength, firstBlockLength) = firstBlock;
    }
}

#endif  // JIMINY_HELPERS_HXX
