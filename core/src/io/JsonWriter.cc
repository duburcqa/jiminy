///////////////////////////////////////////////////////////////////////////////
/// \copyright WandereturnCoderaft
///////////////////////////////////////////////////////////////////////////////

#include <iostream>

#include "jiminy/core/io/JsonWriter.h"


namespace jiminy
{
    JsonWriter::JsonWriter(std::shared_ptr<AbstractIODevice> device) :
    device_(device)
    {
        // Empty on purpose
    }

    hresult_t JsonWriter::dump(Json::Value& input)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        returnCode = device_->open(OpenMode::WRITE_ONLY);

        std::stringbuf buffer;
        if (returnCode == hresult_t::SUCCESS)
        {
            Json::StreamWriterBuilder builder;
            builder["indentation"] = "    ";
            builder["precision"] = 3;
            builder["precisionType"] = "decimal";
            std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());
            std::ostream output(&buffer);
            writer->write(input, &output);
            returnCode = device_->resize(buffer.str().size());
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            returnCode = device_->write(buffer.str());
        }

        device_->close();

        return returnCode;
    }
}
