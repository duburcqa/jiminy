#include "json/json.h"

#include "jiminy/core/io/abstract_io_device.h"
#include "jiminy/core/io/json_loader.h"


namespace jiminy
{
    JsonLoader::JsonLoader(const std::shared_ptr<AbstractIODevice> & device) noexcept :
    device_{device}
    {
    }

    hresult_t JsonLoader::load()
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        returnCode = device_->open(OpenMode::READ_ONLY);

        if (returnCode == hresult_t::SUCCESS)
        {
            auto size = device_->bytesAvailable();
            payload_.resize(size);
            returnCode = device_->read(payload_);
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            std::string errs;
            Json::CharReaderBuilder rbuilder;
            std::unique_ptr<Json::CharReader> reader(rbuilder.newCharReader());
            const bool isParsingOk = reader->parse(
                (payload_.data()), payload_.data() + payload_.size(), rootJson_.get(), &errs);
            if (!isParsingOk)
            {
                returnCode = hresult_t::ERROR_GENERIC;
            }
        }

        device_->close();

        return returnCode;
    }

    const Json::Value * JsonLoader::getRoot()
    {
        return rootJson_.get();
    }
}
