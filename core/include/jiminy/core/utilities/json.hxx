#include "json/json.h"

#include "jiminy/core/fwd.h"
#include "jiminy/core/io/json_writer.h"


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
    Json::Value convertToJson(const MatrixX<T> & value)
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
    Json::Value convertToJson<FlexibleJointData>(const FlexibleJointData & value);

    template<>
    Json::Value convertToJson<HeightmapFunctor>(const HeightmapFunctor & value);

    template<typename T, typename A>
    constexpr std::enable_if_t<!is_eigen_any_v<T>, const char *>
    getJsonVectorType(const std::vector<T, A> & /* value */)
    {
        return "unknown";
    }

    template<typename T, typename A>
    constexpr std::enable_if_t<is_eigen_any_v<T>, const char *>
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
    getJsonVectorType<FlexibleJointData>(const std::vector<FlexibleJointData> & /* value */)
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
    Json::Value convertToJson<GenericConfig>(const GenericConfig & value);

    // ************* Conversion from JSON utilities *****************

    template<typename T>
    std::enable_if_t<!is_vector_v<T> && !is_eigen_vector_v<T>, T>
    convertFromJson(const Json::Value & /* value */) = delete;

    template<>
    std::string convertFromJson<std::string>(const Json::Value & value);

    template<>
    bool convertFromJson<bool>(const Json::Value & value);

    template<>
    int32_t convertFromJson<int32_t>(const Json::Value & value);

    template<>
    uint32_t convertFromJson<uint32_t>(const Json::Value & value);

    template<>
    double convertFromJson<double>(const Json::Value & value);

    template<typename T>
    std::enable_if_t<is_eigen_vector_v<T>, T> convertFromJson(const Json::Value & value)
    {
        T vec{};
        if (value.size() > 0)
        {
            vec.resize(value.size());
            for (auto it = value.begin(); it != value.end(); ++it)
            {
                vec[it.index()] = convertFromJson<typename T::Scalar>(*it);
            }
        }
        return vec;
    }

    template<>
    Eigen::MatrixXd convertFromJson<Eigen::MatrixXd>(const Json::Value & value);

    template<>
    FlexibleJointData convertFromJson<FlexibleJointData>(const Json::Value & value);

    template<>
    HeightmapFunctor convertFromJson<HeightmapFunctor>(const Json::Value & value);

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
    GenericConfig convertFromJson<GenericConfig>(const Json::Value & value);
}