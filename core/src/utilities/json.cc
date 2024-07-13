#include "jiminy/core/io/abstract_io_device.h"
#include "jiminy/core/io/memory_device.h"
#include "jiminy/core/io/json_writer.h"
#include "jiminy/core/io/json_loader.h"

#include "jiminy/core/utilities/json.h"
#include "jiminy/core/utilities/json.hxx"


namespace jiminy
{
    // *************** Convertion to JSON utilities *****************

    template<>
    Json::Value convertToJson<FlexibilityJointConfig>(const FlexibilityJointConfig & value)
    {
        Json::Value flex;
        flex["frameName"] = convertToJson(value.frameName);
        flex["stiffness"] = convertToJson(value.stiffness);
        flex["damping"] = convertToJson(value.damping);
        flex["inertia"] = convertToJson(value.inertia);
        return flex;
    }

    template<>
    Json::Value convertToJson<HeightmapFunction>(const HeightmapFunction & /* value */)
    {
        return {"unsupported"};
    }

    class AppendBoostVariantToJson : public boost::static_visitor<>
    {
    public:
        explicit AppendBoostVariantToJson(Json::Value & root) noexcept :
        root_{root}
        {
        }

        ~AppendBoostVariantToJson() = default;

        template<typename T>
        void operator()(const T & value)
        {
            root_[field_] = convertToJson(value);
        }

    public:
        Json::Value & root_;
        std::string field_{};
    };

    template<>
    Json::Value convertToJson<GenericConfig>(const GenericConfig & value)
    {
        Json::Value root{Json::objectValue};
        AppendBoostVariantToJson visitor(root);
        for (const auto & option : value)
        {
            visitor.field_ = option.first;
            boost::apply_visitor(visitor, option.second);
        }
        return root;
    }

    Json::Value convertToJson(const GenericConfig & value)
    {
        return convertToJson<GenericConfig>(value);
    }

    void jsonDump(const GenericConfig & config, std::shared_ptr<AbstractIODevice> & device)
    {
        // Create the memory device if necessary (the device is nullptr)
        if (!device)
        {
            device = std::make_shared<MemoryDevice>(0U);
        }

        // Wrapper the memory device in a JsonWriter
        JsonWriter ioWrite(device);

        // Convert the configuration in Json and write it in the device
        ioWrite.dump(convertToJson(config));
    }

    // ************* Convertion from JSON utilities *****************

    template<>
    std::string convertFromJson<std::string>(const Json::Value & value)
    {
        return value.asString();
    }

    template<>
    bool convertFromJson<bool>(const Json::Value & value)
    {
        return value.asBool();
    }

    template<>
    int32_t convertFromJson<int32_t>(const Json::Value & value)
    {
        return value.asInt();
    }

    template<>
    uint32_t convertFromJson<uint32_t>(const Json::Value & value)
    {
        return value.asUInt();
    }

    template<>
    double convertFromJson<double>(const Json::Value & value)
    {
        return value.asDouble();
    }

    template<>
    Eigen::MatrixXd convertFromJson<Eigen::MatrixXd>(const Json::Value & value)
    {
        Eigen::MatrixXd mat{};
        if (value.size() > 0)
        {
            auto it = value.begin();
            if (it->size() > 0)
            {
                mat.resize(value.size(), it->size());
                for (; it != value.end(); ++it)
                {
                    mat.row(it.index()) = convertFromJson<Eigen::VectorXd>(*it);
                }
            }
        }
        return mat;
    }

    template<>
    FlexibilityJointConfig convertFromJson<FlexibilityJointConfig>(const Json::Value & value)
    {
        return {convertFromJson<std::string>(value["frameName"]),
                convertFromJson<Eigen::VectorXd>(value["stiffness"]),
                convertFromJson<Eigen::VectorXd>(value["damping"]),
                convertFromJson<Eigen::VectorXd>(value["inertia"])};
    }

    template<>
    HeightmapFunction convertFromJson<HeightmapFunction>(const Json::Value & /* value */)
    {
        return {[](const Eigen::Vector2d & /* xy */,
                   double & height,
                   std::optional<Eigen::Ref<Eigen::Vector3d>> normal) -> void
                {
                    height = 0.0;
                    if (normal.has_value())
                    {
                        normal.value() = Eigen::Vector3d::UnitZ();
                    }
                }};
    }

    template<>
    GenericConfig convertFromJson<GenericConfig>(const Json::Value & value)
    {
        GenericConfig config{};
        for (auto root = value.begin(); root != value.end(); ++root)
        {
            GenericConfig::mapped_type field;

            if (root->type() == Json::objectValue)
            {
                std::vector<std::string> keys = root->getMemberNames();
                const std::vector<std::string> stdVectorAttrib{"type", "value"};
                if (keys == stdVectorAttrib)
                {
                    std::string type = (*root)["type"].asString();
                    Json::Value data = (*root)["value"];
                    if (type == "list(string)")
                    {
                        field = convertFromJson<std::vector<std::string>>(data);
                    }
                    else if (type == "list(array)")
                    {
                        if (data.begin()->size() == 0 ||
                            data.begin()->begin()->type() == Json::realValue)
                        {
                            field = convertFromJson<std::vector<Eigen::VectorXd>>(data);
                        }
                        else
                        {
                            field = convertFromJson<std::vector<Eigen::MatrixXd>>(data);
                        }
                    }
                    else if (type == "list(flexibility)")
                    {
                        field = convertFromJson<FlexibilityConfig>(data);
                    }
                    else
                    {
                        JIMINY_THROW(
                            std::invalid_argument, "Unknown data type: std::vector<", type, ">");
                    }
                }
                else
                {
                    field = convertFromJson<GenericConfig>(*root);
                }
            }
            else if (root->type() == Json::stringValue)
            {
                field = convertFromJson<std::string>(*root);
            }
            else if (root->type() == Json::booleanValue)
            {
                field = convertFromJson<bool>(*root);
            }
            else if (root->type() == Json::realValue)
            {
                field = convertFromJson<double>(*root);
            }
            else if (root->isConvertibleTo(Json::uintValue))
            {
                /* One must use `Json::isConvertibleTo` instead of checking type since JSON format
                   as no way to distinguish between both, so that (u)int32_t are always parsed as
                   int64_t. */
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
                        field = convertFromJson<Eigen::VectorXd>(*root);
                    }
                    else if (it->isConvertibleTo(Json::uintValue))
                    {
                        field = convertFromJson<VectorX<uint32_t>>(*root);
                    }
                    else if (it->type() == Json::arrayValue)
                    {
                        field = convertFromJson<Eigen::MatrixXd>(*root);
                    }
                    else
                    {
                        JIMINY_THROW(std::invalid_argument,
                                     "Unknown data type: std::vector<",
                                     it->type(),
                                     ">");
                    }
                }
                else
                {
                    field = Eigen::VectorXd{};
                }
            }
            else
            {
                JIMINY_THROW(std::invalid_argument, "Unknown data type: ", root->type());
            }

            config[root.key().asString()] = field;
        }
        return config;
    }

    GenericConfig convertFromJson(const Json::Value & value)
    {
        return convertFromJson<GenericConfig>(value);
    }

    void jsonLoad(GenericConfig & config, std::shared_ptr<AbstractIODevice> & device)
    {
        JsonLoader ioRead(device);
        ioRead.load();

        config = convertFromJson<GenericConfig>(*ioRead.getRoot());
    }
}
