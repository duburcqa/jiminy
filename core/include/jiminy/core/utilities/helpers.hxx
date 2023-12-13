#include <algorithm>
#include <numeric>


namespace jiminy
{
    // ****************************** Generic template utilities ******************************* //

    template<class F, class... Args>
    std::enable_if_t<!(... && !std::is_same_v<std::invoke_result_t<F, Args>, void>)>
    do_for(F f, Args &&... args)
    {
        (f(std::forward<Args>(args)), ...);
    }

    template<class F, class... Args>
    std::enable_if_t<(... && !std::is_same_v<std::invoke_result_t<F, Args>, void>),
                     std::tuple<std::invoke_result_t<F, Args>...>>
    do_for(F f, Args &&... args)
    {
        return std::tuple{f(std::forward<Args>(args))...};
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

    // *************************** Math *****************************

    template<typename T>
    T min(T && t)
    {
        return std::forward<T>(t);
    }

    template<typename T0, typename T1, typename... Ts>
    typename std::common_type_t<T0, T1, Ts...> min(T0 && val1, T1 && val2, Ts &&... vs)
    {
        return min(std::min(val1, val2), std::forward<Ts>(vs)...);
    }

    template<typename DerivedType>
    auto clamp(const Eigen::MatrixBase<DerivedType> & data, float64_t minThr, float64_t maxThr)
    {
        return data.unaryExpr([&minThr, &maxThr](float64_t x) -> float64_t
                              { return std::clamp(x, minThr, maxThr); });
    }

    template<typename DerivedType1, typename DerivedType2, typename DerivedType3>
    Eigen::MatrixBase<DerivedType1> clamp(const Eigen::MatrixBase<DerivedType1> & data,
                                          const Eigen::MatrixBase<DerivedType2> & minThr,
                                          const Eigen::MatrixBase<DerivedType2> & maxThr)
    {
        return data.array().max(minThr).min(maxThr);
    }

    inline float64_t minClipped()
    {
        return INF;
    }

    inline float64_t minClipped(float64_t val)
    {
        if (val > EPS)
        {
            return std::forward<float64_t>(val);
        }
        return INF;
    }

    template<typename... Args>
    float64_t minClipped(float64_t val1, float64_t val2, Args... vs)
    {
        const bool_t isValid1 = val1 > EPS;
        const bool_t isValid2 = val2 > EPS;
        if (isValid1 && isValid2)
        {
            return minClipped(std::min(val1, val2), std::forward<Args>(vs)...);
        }
        else if (isValid2)
        {
            return minClipped(val2, std::forward<Args>(vs)...);
        }
        else if (isValid1)
        {
            return minClipped(val1, std::forward<Args>(vs)...);
        }
        return minClipped(std::forward<Args>(vs)...);
    }

    template<typename... Args>
    std::tuple<bool_t, float64_t> isGcdIncluded(Args... values)
    {
        const float64_t minValue = minClipped(std::forward<Args>(values)...);
        if (!std::isfinite(minValue))
        {
            return {true, INF};
        }
        auto lambda = [&minValue](float64_t value)
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

    template<typename InputIt, typename UnaryFunction>
    std::tuple<bool_t, float64_t> isGcdIncluded(InputIt first, InputIt last, UnaryFunction f)
    {
        const float64_t minValue = std::transform_reduce(first, last, INF, minClipped<>, f);
        if (!std::isfinite(minValue))
        {
            return {true, INF};
        }
        auto lambda = [&minValue, &f](const auto & elem)
        {
            const float64_t value = f(elem);
            if (value < EPS)
            {
                return true;
            }
            return std::fmod(value, minValue) < EPS;
        };
        return {std::all_of(first, last, lambda), minValue};
    }

    template<typename InputIt, typename UnaryFunction, typename... Args>
    std::tuple<bool_t, float64_t>
    isGcdIncluded(InputIt first, InputIt last, UnaryFunction f, Args... values)
    {
        const auto [isIncluded1, value1] = isGcdIncluded(std::forward<Args>(values)...);
        const auto [isIncluded2, value2] = isGcdIncluded(first, last, f);
        if (!isIncluded1 || !isIncluded2)
        {
            return {false, INF};
        }
        if (value1 < value2)  // inf < inf : false
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

    // ******************** Std::vector helpers *********************

    template<typename T, typename A>
    bool_t checkDuplicates(const std::vector<T, A> & vect)
    {
        for (auto it = vect.begin(); it != vect.end(); ++it)
        {
            if (std::find(it + 1, vect.end(), *it) != vect.end())
            {
                return true;
            }
        }
        return false;
    }

    template<typename T, typename A>
    bool_t checkIntersection(const std::vector<T, A> & vect1, const std::vector<T, A> & vect2)
    {
        auto vect2It = std::find_if(vect2.begin(),
                                    vect2.end(),
                                    [&vect1](const auto & elem2)
                                    {
                                        auto vect1It =
                                            std::find(vect1.begin(), vect1.end(), elem2);
                                        return (vect1It != vect1.end());
                                    });
        return (vect2It != vect2.end());
    }

    template<typename T, typename A>
    bool_t checkInclusion(const std::vector<T, A> & vect1, const std::vector<T, A> & vect2)
    {
        for (const auto & elem2 : vect2)
        {
            auto vect1It = std::find(vect1.begin(), vect1.end(), elem2);
            if (vect1It == vect1.end())
            {
                return false;
            }
        }
        return true;
    }

    template<typename T, typename A>
    void eraseVector(std::vector<T, A> & vect1, const std::vector<T, A> & vect2)
    {
        vect1.erase(std::remove_if(vect1.begin(),
                                   vect1.end(),
                                   [&vect2](const auto & elem1)
                                   {
                                       auto vect2It = std::find(vect2.begin(), vect2.end(), elem1);
                                       return (vect2It != vect2.end());
                                   }),
                    vect1.end());
    }

    // *********************** Miscellaneous **************************

    /// \brief Swap two non-overlapping row-blocks of data in a matrix.
    ///
    /// \details Given two blocks of arbitrary sizes b1, b2 in a matrix B = (... b1 ... b2 ...),
    ///          this function re-assigns B to (... b2 ... b1 ...).
    ///
    /// \pre firstBlockStart + firstBlockLength <= secondBlockStart
    ///
    /// \param[in, out] matrix Matrix to modify.
    /// \param[in] firstBlockStart Start index of the first block.
    /// \param[in] firstBlockLength Length of the first block.
    /// \param[in] secondBlockStart Start index of the second block.
    /// \param[in] secondBlockLength Length of the second block.
    template<typename Derived>
    void swapMatrixBlocks(const Eigen::MatrixBase<Derived> & matrixIn,
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
        auto middleBlock = firstMiddleBlock.topRows(middleBlockLength);
        matrix.middleRows(firstBlockStart + secondBlockLength, middleBlockLength) = middleBlock;

        // Re-assign second block to first block
        auto firstBlock = firstMiddleBlock.bottomRows(firstBlockLength);
        const Eigen::Index secondBlockEnd = secondBlockStart + secondBlockLength;  // Excluded
        matrix.middleRows(secondBlockEnd - firstBlockLength, firstBlockLength) = firstBlock;
    }
}