#include "json/json.h"

#include "jiminy/core/io/json_writer.h"
#include "jiminy/core/macros.h"


namespace jiminy
{
    // *************** Convertion to JSON utilities *****************

    template<typename T>
    std::enable_if_t<!is_vector_v<T>, Json::Value> convertToJson(const T & value)
    {
        return {value};
    }

    template<typename T, int RowsAtCompileTime>
    Json::Value convertToJson(const Eigen::Matrix<T, RowsAtCompileTime, 1> & value)
    {
        Json::Value row(Json::arrayValue);
        for (Eigen::Index i = 0; i < value.size(); ++i)
        {
            row.append(value[i]);
        }
        return row;
    }

    template<typename T>
    Json::Value convertToJson(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> & value)
    {
        Json::Value mat(Json::arrayValue);
        if (value.rows() > 0)
        {
            for (Eigen::Index i = 0; i < value.rows(); ++i)
            {
                Json::Value row(Json::arrayValue);
                for (Eigen::Index j = 0; j < value.cols(); ++j)
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
    Json::Value convertToJson<flexibleJointData_t>(const flexibleJointData_t & value);

    template<>
    Json::Value convertToJson<heightmapFunctor_t>(const heightmapFunctor_t & value);

    template<typename T, typename A>
    constexpr std::enable_if_t<!is_eigen_v<T>, const char *>
    getJsonVectorType(const std::vector<T, A> & /* value */)
    {
        return "unknown";
    }

    template<typename T, typename A>
    constexpr std::enable_if_t<is_eigen_v<T>, const char *>
    getJsonVectorType(const std::vector<T, A> & /* value */)
    {
        return "list(array)";
    }

    template<>
    constexpr const char *
    getJsonVectorType<std::string>(const std::vector<std::string> & /* value */)
    {
        return "list(string)";
    }

    template<>
    constexpr const char *
    getJsonVectorType<flexibleJointData_t>(const std::vector<flexibleJointData_t> & /* value */)
    {
        return "list(flexibility)";
    }

    template<typename T>
    std::enable_if_t<is_vector_v<T>, Json::Value> convertToJson(const T & value)
    {
        Json::Value root;
        root["type"] = getJsonVectorType(value);
        Json::Value vec(Json::arrayValue);
        for (const auto & elem : value)
        {
            vec.append(convertToJson(elem));
        }
        root["value"] = vec;
        return root;
    }

    template<>
    Json::Value convertToJson<configHolder_t>(const configHolder_t & value);

    // ************* Convertion from JSON utilities *****************

    template<typename T>
    std::enable_if_t<!is_vector_v<T>, T> convertFromJson(const Json::Value & /* value */)
    {
        T::undefined_template_specialization_for_this_type;
    }

    template<>
    std::string convertFromJson<std::string>(const Json::Value & value);

    template<>
    bool_t convertFromJson<bool_t>(const Json::Value & value);

    template<>
    int32_t convertFromJson<int32_t>(const Json::Value & value);

    template<>
    uint32_t convertFromJson<uint32_t>(const Json::Value & value);

    template<>
    float64_t convertFromJson<float64_t>(const Json::Value & value);

    template<>
    Eigen::VectorXd convertFromJson<Eigen::VectorXd>(const Json::Value & value);

    template<>
    Eigen::MatrixXd convertFromJson<Eigen::MatrixXd>(const Json::Value & value);

    template<>
    flexibleJointData_t convertFromJson<flexibleJointData_t>(const Json::Value & value);

    template<>
    heightmapFunctor_t convertFromJson<heightmapFunctor_t>(const Json::Value & value);

    template<typename T>
    std::enable_if_t<is_vector_v<T>, T> convertFromJson(const Json::Value & value)
    {
        T vec;
        if (!value.empty())
        {
            vec.resize(value.size());
            for (auto itr = value.begin(); itr != value.end(); ++itr)
            {
                vec[itr.index()] = convertFromJson<typename T::value_type>((*itr));
            }
        }
        return vec;
    }

    template<>
    configHolder_t convertFromJson<configHolder_t>(const Json::Value & value);
}