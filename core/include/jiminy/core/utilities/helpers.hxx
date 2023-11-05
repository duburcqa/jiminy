#include <algorithm>
#include <numeric>


namespace jiminy
{
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
    auto clamp(const Eigen::MatrixBase<DerivedType> & data,
               const float64_t & minThr,
               const float64_t & maxThr)
    {
        return data.unaryExpr([&minThr, &maxThr](const float64_t & x) -> float64_t
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
        auto lambda = [&minValue](const float64_t & value)
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

    /// \brief Swap two blocks of data in a vector.
    ///
    /// \details Given two uneven blocks in a vector v = (... v1 ... v2 ...), this function
    /// modifies
    ///          v to v = (... v2 ... v1 ...). v1 and v2 can be of arbitrary size.
    ///
    /// \pre firstBlockStart + firstBlockLength <= secondBlockStart
    /// \pre secondBlockStart + secondBlockLength <= vector.size()
    ///
    /// \param[in, out] vector Vector to modify.
    /// \param[in] firstBlockStart Start index of the first block.
    /// \param[in] firstBlockLength Length of the first block.
    /// \param[in] secondBlockStart Start index of the second block.
    /// \param[in] secondBlockLength Length of the second block.
    template<typename type>
    void swapVectorBlocks(Eigen::Matrix<type, Eigen::Dynamic, 1> & vector,
                          const Eigen::Index & firstBlockStart,
                          const Eigen::Index & firstBlockLength,
                          const Eigen::Index & secondBlockStart,
                          const Eigen::Index & secondBlockLength)
    {
        // Extract both blocks by copy
        Eigen::Matrix<type, Eigen::Dynamic, 1> firstBlock =
            vector.segment(firstBlockStart, firstBlockLength);
        Eigen::Matrix<type, Eigen::Dynamic, 1> secondBlock =
            vector.segment(secondBlockStart, secondBlockLength);

        // Extract content between the blocks
        const Eigen::Index middleLength = secondBlockStart - (firstBlockStart + firstBlockLength);
        Eigen::Matrix<type, Eigen::Dynamic, 1> middleBlock =
            vector.segment(firstBlockStart + firstBlockLength, middleLength);

        // Reorder vector
        vector.segment(firstBlockStart, secondBlockLength) = secondBlock;
        vector.segment(firstBlockStart + secondBlockLength, middleLength) = middleBlock;
        vector.segment(firstBlockStart + secondBlockLength + middleLength, firstBlockLength) =
            firstBlock;
    }
}