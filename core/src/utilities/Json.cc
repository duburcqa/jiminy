

#include "jiminy/core/io/AbstractIODevice.h"
#include "jiminy/core/io/MemoryDevice.h"
#include "jiminy/core/io/JsonWriter.h"
#include "jiminy/core/io/JsonLoader.h"

#include "jiminy/core/utilities/Json.h"
#include "jiminy/core/utilities/Json.tpp"


namespace jiminy
{
    // *************** Convertion to JSON utilities *****************

    template<>
    Json::Value convertToJson<flexibleJointData_t>(flexibleJointData_t const & value)
    {
        Json::Value flex;
        flex["frameName"] = convertToJson(value.frameName);
        flex["stiffness"] = convertToJson(value.stiffness);
        flex["damping"] = convertToJson(value.damping);
        flex["inertia"] = convertToJson(value.inertia);
        return flex;
    }

    template<>
    Json::Value convertToJson<heightmapFunctor_t>(heightmapFunctor_t const & /* value */)
    {
        return {"not supported"};
    }

    class AppendBoostVariantToJson : public boost::static_visitor<>
    {
    public:
        explicit AppendBoostVariantToJson(Json::Value & root) :
        root_(root),
        field_()
        {
            // Empty on purpose
        }

        ~AppendBoostVariantToJson(void) = default;

        template<typename T>
        void operator()(T const & value)
        {
            root_[field_] = convertToJson(value);
        }

    public:
        Json::Value & root_;
        std::string field_;
    };

    template<>
    Json::Value convertToJson<configHolder_t>(configHolder_t const & value)
    {
        Json::Value root{Json::objectValue};
        AppendBoostVariantToJson visitor(root);
        for (auto const & option : value)
        {
            visitor.field_ = option.first;
            boost::apply_visitor(visitor, option.second);
        }
        return root;
    }

    Json::Value convertToJson(configHolder_t const & value)
    {
        return convertToJson<configHolder_t>(value);
    }

    hresult_t jsonDump(configHolder_t                    const & config,
                       std::shared_ptr<AbstractIODevice>       & device)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        // Create the memory device if necessary (the device is nullptr)
        if (!device)
        {
            device = std::make_shared<MemoryDevice>(0U);
        }

        // Wrapper the memory device in a JsonWriter
        JsonWriter ioWrite(device);

        // Convert the configuration in Json and write it in the device
        returnCode = ioWrite.dump(convertToJson(config));

        return returnCode;
    }

    // ************* Convertion from JSON utilities *****************

    template<>
    std::string convertFromJson<std::string>(Json::Value const & value)
    {
        return value.asString();
    }

    template<>
    bool_t convertFromJson<bool_t>(Json::Value const & value)
    {
        return value.asBool();
    }

    template<>
    int32_t convertFromJson<int32_t>(Json::Value const & value)
    {
        return value.asInt();
    }

    template<>
    uint32_t convertFromJson<uint32_t>(Json::Value const & value)
    {
        return value.asUInt();
    }

    template<>
    float64_t convertFromJson<float64_t>(Json::Value const & value)
    {
        return value.asDouble();
    }

    template<>
    vectorN_t convertFromJson<vectorN_t>(Json::Value const & value)
    {
        vectorN_t vec;
        if (value.size() > 0)
        {
            vec.resize(value.size());
            for (auto it = value.begin() ; it != value.end() ; ++it)
            {
                vec[it.index()] = convertFromJson<float64_t>(*it);
            }
        }
        return vec;
    }

    template<>
    matrixN_t convertFromJson<matrixN_t>(Json::Value const & value)
    {
        matrixN_t mat;
        if (value.size() > 0)
        {
            auto it = value.begin() ;
            if (it->size() > 0)
            {
                mat.resize(value.size(), it->size());
                for (; it != value.end() ; ++it)
                {
                    mat.row(it.index()) = convertFromJson<vectorN_t>(*it);
                }
            }
        }
        return mat;
    }

    template<>
    flexibleJointData_t convertFromJson<flexibleJointData_t>(Json::Value const & value)
    {
        return {
            convertFromJson<std::string>(value["frameName"]),
            convertFromJson<vectorN_t>(value["stiffness"]),
            convertFromJson<vectorN_t>(value["damping"]),
            convertFromJson<vectorN_t>(value["inertia"])
        };
    }

    template<>
    heightmapFunctor_t convertFromJson<heightmapFunctor_t>(Json::Value const & /* value */)
    {
        return {
            heightmapFunctor_t(
                [](vector3_t const & /* pos */) -> std::pair<float64_t, vector3_t>
                {
                    return {0.0, vector3_t::UnitZ()};
                })
        };
    }

    template<>
    configHolder_t convertFromJson<configHolder_t>(Json::Value const & value)
    {
        configHolder_t config;
        for (auto root = value.begin() ; root != value.end() ; ++root)
        {
            configField_t field;

            if (root->type() == Json::objectValue)
            {
                std::vector<std::string> keys = root->getMemberNames();
                std::vector<std::string> const stdVectorAttrib{
                    "type",
                    "value"
                };
                if (keys == stdVectorAttrib)
                {
                    std::string type = (*root)["type"].asString();
                    Json::Value data = (*root)["value"];
                    if (type == "list(string)")
                    {
                        field = convertFromJson<std::vector<std::string> >(data);
                    }
                    else if (type == "list(array)")
                    {
                        if (data.begin()->size() == 0
                         || data.begin()->begin()->type() == Json::realValue)
                        {
                            field = convertFromJson<std::vector<vectorN_t> >(data);
                        }
                        else
                        {
                            field = convertFromJson<std::vector<matrixN_t> >(data);
                        }
                    }
                    else if (type == "list(flexibility)")
                    {
                        field = convertFromJson<flexibilityConfig_t>(data);
                    }
                    else
                    {
                        PRINT_ERROR("Unknown data type: std::vector<", type, ">");
                        field = std::string{"ValueError"};
                    }
                }
                else
                {
                    field = convertFromJson<configHolder_t>(*root);
                }
            }
            else if (root->type() == Json::stringValue)
            {
                field = convertFromJson<std::string>(*root);
            }
            else if (root->type() == Json::booleanValue)
            {
                field = convertFromJson<bool_t>(*root);
            }
            else if (root->type() == Json::realValue)
            {
                field = convertFromJson<float64_t>(*root);
            }
            else if (root->isConvertibleTo(Json::uintValue))
            {
                /* One must use `Json::isConvertibleTo` instead of checking type since JSON
                   format as no way to distinguish between both, so that (u)int32_t are
                   always parsed as int64_t. */
                field = convertFromJson<uint32_t>(*root);
            }
            else if (root->isConvertibleTo(Json::intValue))
            {
                field = convertFromJson<int32_t>(*root);
            }
            else if (root->type() == Json::arrayValue)
            {
                if (root->size() > 0)
                {
                    auto it = root->begin();
                    if (it->type() == Json::realValue)
                    {
                        field = convertFromJson<vectorN_t>(*root);
                    }
                    else if (it->type() == Json::arrayValue)
                    {
                        field = convertFromJson<matrixN_t>(*root);
                    }
                    else
                    {
                        PRINT_ERROR("Unknown data type: std::vector<", it->type(), ">");
                        field = std::string{"ValueError"};
                    }
                }
                else
                {
                    field = vectorN_t();
                }
            }
            else
            {
                PRINT_ERROR("Unknown data type: ", root->type());
                field = std::string{"ValueError"};
            }

            config[root.key().asString()] = field;
        }
        return config;
    }

    configHolder_t convertFromJson(Json::Value const & value)
    {
        return convertFromJson<configHolder_t>(value);
    }

    hresult_t jsonLoad(configHolder_t                    & config,
                       std::shared_ptr<AbstractIODevice> & device)
    {

        hresult_t returnCode = hresult_t::SUCCESS;

        JsonLoader ioRead(device);
        returnCode = ioRead.load();

        if (returnCode == hresult_t::SUCCESS)
        {
            config = convertFromJson<configHolder_t>(*ioRead.getRoot());
        }

        return returnCode;
    }
}
