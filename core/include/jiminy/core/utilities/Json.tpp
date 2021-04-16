#include "jiminy/core/io/JsonWriter.h"


namespace jiminy
{
    // *************** Convertion to JSON utilities *****************

    template<typename T>
    std::enable_if_t<!is_vector_v<T>, Json::Value>
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
    std::enable_if_t<is_vector_v<T>, Json::Value>
    convertToJson(T const & value)
    {
        Json::Value root;

        using TVal = typename T::value_type;
        if (std::is_same<TVal, std::string>::value)  // C++17 conditional constexpr is not supported by gcc<7.3
        {
            root["type"] = "list(string)";
        }
        else if (std::is_same<TVal, vectorN_t>::value
              || std::is_same<TVal, matrixN_t>::value)  // constexpr
        {
            root["type"] = "list(array)";
        }
        else if (std::is_same<TVal, flexibleJointData_t>::value)  // constexpr
        {
            root["type"] = "list(flexibility)";
        }
        else
        {
            PRINT_ERROR("Unknown data type: ", root.type());
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
    std::enable_if_t<!is_vector_v<T>, T>
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
    std::enable_if_t<is_vector_v<T>, T>
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
}