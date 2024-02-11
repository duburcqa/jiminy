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
        device_->resize(static_cast<int64_t>(buffer.str().size()));

        device_->write(buffer.str());

        device_->close();
    }
}
