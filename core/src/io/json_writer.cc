#include <ostream>

#include "json/json.h"

#include "jiminy/core/io/abstract_io_device.h"
#include "jiminy/core/io/json_writer.h"


namespace jiminy
{
    JsonWriter::JsonWriter(const std::shared_ptr<AbstractIODevice> & device) noexcept :
    device_{device}
    {
    }

    void JsonWriter::dump(const Json::Value & input)
    {
        device_->open(OpenMode::WRITE_ONLY);

        std::stringbuf buffer;
        Json::StreamWriterBuilder builder;
        builder["commentStyle"] = "None";
        builder["indentation"] = "  ";
        builder["enableYAMLCompatibility"] = true;
        builder["dropNullPlaceholders"] = false;
        builder["useSpecialFloats"] = true;
        builder["precision"] = 9;
        std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());
        std::ostream output(&buffer);
        writer->write(input, &output);

        // FIXME: Use `view` to get a string_view rather than a string copy when moving to C++20
        const std::string data = buffer.str();
        device_->resize(static_cast<int64_t>(data.size()));
        device_->write(data);

        device_->close();
    }
}
