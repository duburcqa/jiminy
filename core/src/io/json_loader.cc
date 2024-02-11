#include "json/json.h"

#include "jiminy/core/io/abstract_io_device.h"
#include "jiminy/core/io/json_loader.h"


namespace jiminy
{
    JsonLoader::JsonLoader(const std::shared_ptr<AbstractIODevice> & device) noexcept :
    device_{device}
    {
    }

    void JsonLoader::load()
    {
        device_->open(OpenMode::READ_ONLY);

        auto size = device_->bytesAvailable();
        payload_.resize(size);
        device_->read(payload_);

        std::string errs;
        Json::CharReaderBuilder rbuilder;
        std::unique_ptr<Json::CharReader> reader(rbuilder.newCharReader());
        const bool isParsingOk = reader->parse(
            (payload_.data()), payload_.data() + payload_.size(), rootJson_.get(), &errs);
        if (!isParsingOk)
        {
            THROW_ERROR(std::ios_base::failure, "Impossible to parse JSON content.");
        }

        device_->close();
    }

    const Json::Value * JsonLoader::getRoot()
    {
        return rootJson_.get();
    }
}
