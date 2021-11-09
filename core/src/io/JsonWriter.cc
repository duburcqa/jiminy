#include <ostream>

#include "json/json.h"

#include "jiminy/core/io/AbstractIODevice.h"
#include "jiminy/core/io/JsonWriter.h"


namespace jiminy
{
    JsonWriter::JsonWriter(std::shared_ptr<AbstractIODevice> device) :
    device_(std::move(device))
    {
        // Empty on purpose
    }

    hresult_t JsonWriter::dump(Json::Value const & input)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        returnCode = device_->open(openMode_t::WRITE_ONLY);

        std::stringbuf buffer;
        if (returnCode == hresult_t::SUCCESS)
        {
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
            returnCode = device_->resize(static_cast<int64_t>(buffer.str().size()));
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            returnCode = device_->write(buffer.str());
        }

        device_->close();

        return returnCode;
    }
}
