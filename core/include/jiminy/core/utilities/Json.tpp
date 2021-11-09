#include "json/json.h"

#include "jiminy/core/io/JsonWriter.h"
#include "jiminy/core/Macros.h"


namespace jiminy
{
    // *************** Convertion to JSON utilities *****************

    template<typename T>
    std::enable_if_t<!is_vector_v<T>, Json::Value>
    convertToJson(T const & value)
    {
        return {value};
    }

    template<typename T, int RowsAtCompileTime>
    Json::Value convertToJson(Eigen::Matrix<T, RowsAtCompileTime, 1> const & value)
    {
        Json::Value row(Json::arrayValue);
        for (Eigen::Index i = 0; i < value.size(); ++i)
        {
            row.append(value[i]);
        }
        return row;
    }

    template<typename T>
    Json::Value convertToJson(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> const & value)
    {
        Json::Value mat(Json::arrayValue);
        if (value.rows() > 0)
        {
            for (Eigen::Index i = 0; i<value.rows(); ++i)
            {
                Json::Value row(Json::arrayValue);
                for (Eigen::Index j = 0; j<value.cols(); ++j)
                {
                    row.append(value(i, j));
                }
                mat.append(row);
            }
        }
        else
        {
            mat.append(Json::Value(Json::arrayValue));
        }
        return mat;
    }

    template<>
    Json::Value convertToJson<flexibleJointData_t>(flexibleJointData_t const & value);

    template<>
    Json::Value convertToJson<heightmapFunctor_t>(heightmapFunctor_t const & value);

    template<typename T, typename A>
    constexpr std::enable_if_t<!std::is_same<T, vectorN_t>::value
                            && !std::is_same<T, matrixN_t>::value, const char *>
    getJsonVectorType(std::vector<T, A> const & /* value */)
    {
        return "unknown";
    }

    template<typename T, typename A>
    constexpr std::enable_if_t<std::is_same<T, vectorN_t>::value
                            || std::is_same<T, matrixN_t>::value, const char *>
    getJsonVectorType(std::vector<T, A> const & /* value */)
    {
        return "list(array)";
    }

    template<>
    constexpr const char * getJsonVectorType<std::string>(std::vector<std::string> const & /* value */)
    {
        return "list(string)";
    }

    template<>
    constexpr const char * getJsonVectorType<flexibleJointData_t>(std::vector<flexibleJointData_t> const & /* value */)
    {
        return "list(flexibility)";
    }

    template<typename T>
    std::enable_if_t<is_vector_v<T>, Json::Value>
    convertToJson(T const & value)
    {
        Json::Value root;
        root["type"] = getJsonVectorType(value);
        Json::Value vec(Json::arrayValue);
        for (auto const & elem : value)
        {
            vec.append(convertToJson(elem));
        }
        root["value"] = vec;
        return root;
    }

    template<>
    Json::Value convertToJson<configHolder_t>(configHolder_t const & value);

    // ************* Convertion from JSON utilities *****************

    template<typename T>
    std::enable_if_t<!is_vector_v<T>, T>
    convertFromJson(Json::Value const & /* value */)
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
    heightmapFunctor_t convertFromJson<heightmapFunctor_t>(Json::Value const & value);

    template<typename T>
    std::enable_if_t<is_vector_v<T>, T>
    convertFromJson(Json::Value const & value)
    {
        T vec;
        if (!value.empty())
        {
            vec.resize(value.size());
            for (auto itr = value.begin() ; itr != value.end() ; ++itr)
            {
                vec[itr.index()] = convertFromJson<typename T::value_type>((*itr));
            }
        }
        return vec;
    }

    template<>
    configHolder_t convertFromJson<configHolder_t>(Json::Value const & value);
}