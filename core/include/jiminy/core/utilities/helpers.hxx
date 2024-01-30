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
    std::enable_if_t<std::conjunction_v<std::is_same<Args, double>...>, const double &>
    minClipped(const double & value1, const double & value2, const Args &... values)
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
    std::enable_if_t<std::conjunction_v<std::is_same<Args, double>...>,
                     std::tuple<bool, const double &>>
    isGcdIncluded(const Args &... values)
    {
        const double & minValue = minClipped(values...);
        if (!std::isfinite(minValue))
        {
            return {true, INF};
        }
        /* FIXME: In some cases, order of evaluation is not always respected with MSVC, although
           they pretend it has been fixed but it. As a result, 'isIncluded' must be explicitly
           computed first. For reference, see:
           https://devblogs.microsoft.com/cppblog/compiler-improvements-in-vs-2015-update-2/#order-of-initializer-list
        */
        bool isIncluded = (
            [&minValue](double value)
            {
                if (value < EPS)
                {
                    return true;
                }
                return std::fmod(value, minValue) < EPS;
            }(values) && ...);
        return {isIncluded, minValue};
    }

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
        // FIXME: Order of evaluation is not always respected with MSVC.
        bool isIncluded = std::all_of(first, last, lambda);
        return {isIncluded, minValue};
    }

    template<typename InputIt, typename UnaryFunction, typename... Args>
    std::enable_if_t<std::is_invocable_r_v<const double &,
                                           UnaryFunction,
                                           typename std::iterator_traits<InputIt>::reference> &&
                         std::conjunction_v<std::is_same<Args, double>...>,
                     std::tuple<bool, const double &>>
    isGcdIncluded(InputIt first, InputIt last, const UnaryFunction & func, const Args &... values)
    {
        auto && [isIncluded1, value1] = isGcdIncluded(values...);
        auto && [isIncluded2, value2] = isGcdIncluded(first, last, func);
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

    template<typename Derived>
    void swapMatrixRows(const Eigen::MatrixBase<Derived> & mat,
                        Eigen::Index firstBlockStart,
                        Eigen::Index firstBlockSize,
                        Eigen::Index secondBlockStart,
                        Eigen::Index secondBlockSize)
    {
        /* Aliasing is NOT an issue when shifting up the rows of a matrix, regardless its storage
           order. As a result, it is only necessary to backup the first block plus the overlapping
           part of the middle block if any. Then, assign the first block to the second, reconstruct
           the middle block while accounting for a potential overlap, and finally assign the second
           block to the backup of the first one. */

        // Make sure that the first block is actually before the second one
        if (firstBlockStart > secondBlockStart)
        {
            return swapMatrixRows(
                mat, secondBlockStart, secondBlockSize, firstBlockStart, firstBlockSize);
        }

        // Get plain matrix type and cast away constness
        using Matrix = typename Eigen::MatrixBase<Derived>::PlainObject;
        Derived & derived = const_cast<Derived &>(mat.derived());

        // Backup the first block plus the overlapping part of the middle block if any
        const Eigen::Index middleBlockStart = firstBlockStart + firstBlockSize;
        const Eigen::Index middleBlockSize = secondBlockStart - middleBlockStart;
        assert(middleBlockSize >= 0 && "The blocks must be disjoint");
        const Eigen::Index overlapBlockSize =
            std::max(std::min(firstBlockSize + middleBlockSize, secondBlockSize), firstBlockSize);
        const Matrix overlapBlock = derived.middleRows(firstBlockStart, overlapBlockSize);

        // Re-assign the first block to the second one without copy in all cases
        auto secondBlock = derived.middleRows(secondBlockStart, secondBlockSize);
        derived.middleRows(firstBlockStart, secondBlockSize) = secondBlock;

        // Shift the disjoint part of the middle block if any
        const Eigen::Index newMiddleBlockStart = firstBlockStart + secondBlockSize;
        const Eigen::Index middleOverlapBlockSize =
            std::max(overlapBlockSize - firstBlockSize, Eigen::Index{0});
        const Eigen::Index middleDisjointBlockSize = middleBlockSize - middleOverlapBlockSize;
        if (middleDisjointBlockSize > 0)
        {
            const Eigen::Index middleBlockEnd = middleBlockStart + middleBlockSize;
            const Eigen::Index newMiddleBlockEnd = newMiddleBlockStart + middleBlockSize;
            auto middleDisjointBlock = derived.middleRows(middleBlockEnd - middleDisjointBlockSize,
                                                          middleDisjointBlockSize);
            derived.middleRows(newMiddleBlockEnd - middleDisjointBlockSize,
                               middleDisjointBlockSize) = middleDisjointBlock;
        }

        // Shift the overlapping part of the middle block if any
        if (middleOverlapBlockSize > 0)
        {
            auto middleOverlapBlock = overlapBlock.bottomRows(middleOverlapBlockSize);
            derived.middleRows(newMiddleBlockStart, middleOverlapBlockSize) = middleOverlapBlock;
        }

        // Re-assign the second block to the first one
        const Eigen::Index secondBlockEnd = secondBlockStart + secondBlockSize;
        auto firstBlock = overlapBlock.topRows(firstBlockSize);
        derived.middleRows(secondBlockEnd - firstBlockSize, firstBlockSize) = firstBlock;
    }
}

#endif  // JIMINY_HELPERS_HXX
