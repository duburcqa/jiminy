#include <iostream>
#include <memory>
#include <algorithm>

#include "jiminy/core/io/JsonWriter.h"

namespace jiminy
{
    // *************************** Math *****************************

    template<typename T>
    T min(T && t)
    {
        return std::forward<T>(t);
    }

    template<typename T0, typename T1, typename... Ts>
    typename std::common_type<T0, T1, Ts...>::type min(T0 && val1, T1 && val2, Ts &&... vs)
    {
        if (val2 < val1)
        {
            return min(val2, std::forward<Ts>(vs)...);
        }
        else
        {
            return min(val1, std::forward<Ts>(vs)...);
        }
    }

    // *************** Convertion to JSON utilities *****************

    template<typename T>
    std::enable_if_t<!is_vector<T>::value, Json::Value>
    convertToJson(T const & value)
    {
        return {value};
    }

    template<>
    Json::Value convertToJson<vectorN_t>(vectorN_t const & value);

    template<>
    Json::Value convertToJson<matrixN_t>(matrixN_t const & value);

    template<>
    Json::Value convertToJson<flexibleJointData_t>(flexibleJointData_t const & value);

    template<>
    Json::Value convertToJson<heatMapFunctor_t>(heatMapFunctor_t const & value);

    template<>
    Json::Value convertToJson<configHolder_t>(configHolder_t const & value);

    template<typename T>
    std::enable_if_t<is_vector<T>::value, Json::Value>
    convertToJson(T const & value)
    {
        Json::Value root;

        using TVal = typename T::value_type;
        if (std::is_same<TVal, std::string>::value)
        {
            root["type"] = "list(string)";
        }
        else if (std::is_same<TVal, vectorN_t>::value
              || std::is_same<TVal, matrixN_t>::value)
        {
            root["type"] = "list(array)";
        }
        else if (std::is_same<TVal, flexibleJointData_t>::value)
        {
            root["type"] = "list(flexibility)";
        }
        else
        {
            PRINT_ERROR("Unknown data type: ", root.type())
            root["type"] = "unknown";
        }

        Json::Value vec(Json::arrayValue);
        for (auto const & elem : value)
        {
            vec.append(convertToJson(elem));
        }
        root["value"] = vec;

        return root;
    }

    // ************* Convertion from JSON utilities *****************

    template<typename T>
    std::enable_if_t<!is_vector<T>::value, T>
    convertFromJson(Json::Value const & value)
    {
        T::undefined_template_specialization_for_this_type;
    }

    template<>
    std::string convertFromJson<std::string>(Json::Value const & value);

    template<>
    bool_t convertFromJson<bool_t>(Json::Value const & value);

    template<>
    int32_t convertFromJson<int32_t>(Json::Value const & value);

    template<>
    uint32_t convertFromJson<uint32_t>(Json::Value const & value);

    template<>
    float64_t convertFromJson<float64_t>(Json::Value const & value);

    template<>
    vectorN_t convertFromJson<vectorN_t>(Json::Value const & value);

    template<>
    matrixN_t convertFromJson<matrixN_t>(Json::Value const & value);

    template<>
    flexibleJointData_t convertFromJson<flexibleJointData_t>(Json::Value const & value);

    template<>
    heatMapFunctor_t convertFromJson<heatMapFunctor_t>(Json::Value const & value);

    template<>
    configHolder_t convertFromJson<configHolder_t>(Json::Value const & value);

    template<typename T>
    std::enable_if_t<is_vector<T>::value, T>
    convertFromJson(Json::Value const & value)
    {
        T vec;
        if (value.size() > 0)
        {
            vec.resize(value.size());
            for (auto itr = value.begin() ; itr != value.end() ; ++itr)
            {
                vec[itr.index()] = convertFromJson<typename T::value_type>((*itr));
            }
        }
        return vec;
    }

    // ******************** Std::vector helpers *********************

    template<typename T>
    bool_t checkDuplicates(std::vector<T> const & vect)
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

    template<typename T>
    bool_t checkIntersection(std::vector<T> const & vect1,
                             std::vector<T> const & vect2)
    {
        auto vect2It = std::find_if(vect2.begin(), vect2.end(),
        [&vect1](auto const & elem2)
        {
            auto vect1It = std::find(vect1.begin(), vect1.end(), elem2);
            return (vect1It != vect1.end());
        });
        return (vect2It != vect2.end());
    }

    template<typename T>
    bool_t checkInclusion(std::vector<T> const & vect1,
                          std::vector<T> const & vect2)
    {
        for (auto const & elem2 : vect2)
        {
            auto vect1It = std::find(vect1.begin(), vect1.end(), elem2);
            if (vect1It == vect1.end())
            {
                return false;
            }
        }
        return true;
    }

    template<typename T>
    void eraseVector(std::vector<T>       & vect1,
                     std::vector<T> const & vect2)
    {
        vect1.erase(std::remove_if(vect1.begin(), vect1.end(),
        [&vect2](auto const & elem1)
        {
            auto vect2It = std::find(vect2.begin(), vect2.end(), elem1);
            return (vect2It != vect2.end());
        }), vect1.end());
    }

    // *********************** Miscellaneous **************************

    ///////////////////////////////////////////////////////////////////////////////////////////////
    ///
    /// \brief Swap two blocks of data in a vector.
    ///
    /// \details Given two uneven blocks in a vector v = (... v1 ... v2 ...), this function modifies
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
    ///////////////////////////////////////////////////////////////////////////////////////////////
    template<typename type>
    void swapVectorBlocks(Eigen::Matrix<type, Eigen::Dynamic, 1>       & vector,
                          uint32_t                               const & firstBlockStart,
                          uint32_t                               const & firstBlockLength,
                          uint32_t                               const & secondBlockStart,
                          uint32_t                               const & secondBlockLength)
    {
        // Extract both blocks.
        Eigen::Matrix< type, Eigen::Dynamic, 1 > firstBlock = vector.segment(firstBlockStart, firstBlockLength);
        Eigen::Matrix< type, Eigen::Dynamic, 1 > secondBlock = vector.segment(secondBlockStart, secondBlockLength);

        // Extract content between the blocks.
        uint32_t middleLength = secondBlockStart - (firstBlockStart + firstBlockLength);
        Eigen::Matrix< type, Eigen::Dynamic, 1 > middleBlock = vector.segment(firstBlockStart + firstBlockLength, middleLength);

        // Reorder vector.
        vector.segment(firstBlockStart, secondBlockLength) = secondBlock;
        vector.segment(firstBlockStart + secondBlockLength, middleLength) = middleBlock;
        vector.segment(firstBlockStart + secondBlockLength + middleLength, firstBlockLength) = firstBlock;
    }
}