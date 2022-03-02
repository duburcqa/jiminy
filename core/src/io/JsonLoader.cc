#include "json/json.h"

#include "jiminy/core/io/AbstractIODevice.h"
#include "jiminy/core/io/JsonLoader.h"


namespace jiminy
{
    JsonLoader::JsonLoader(std::shared_ptr<AbstractIODevice> device) :
    rootJson_(std::make_unique<Json::Value>()),
    payload_(),
    device_(device)
    {
        // Empty on purpose.
    }

    hresult_t JsonLoader::load()
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        returnCode = device_->open(openMode_t::READ_ONLY);

        if (returnCode == hresult_t::SUCCESS)
        {
            auto size = device_->bytesAvailable();
            payload_.resize(static_cast<std::size_t>(size));
            returnCode = device_->read(payload_.data(), size);
        }

        if (returnCode == hresult_t::SUCCESS)
        {
           std::string errs;
           Json::CharReaderBuilder rbuilder;
           std::unique_ptr<Json::CharReader> reader(rbuilder.newCharReader());
           bool_t const isParsingOk = reader->parse((payload_.data()),
                                                     payload_.data() + payload_.size(),
                                                     rootJson_.get(),
                                                     &errs);
           if (!isParsingOk)
           {
               returnCode = hresult_t::ERROR_GENERIC;
           }
        }

        device_->close();

        return returnCode;
    }

    Json::Value const * JsonLoader::getRoot()
    {
        return rootJson_.get();
    }
}  // End of namespace jiminy.
