#ifndef JIMINY_JSON_LOADER_H
#define JIMINY_JSON_LOADER_H

#include "jiminy/core/fwd.h"


namespace Json
{
    class Value;
}

namespace jiminy
{
    class AbstractIODevice;

    /// \brief   Helper class to load and parse JSON files.
    ///
    /// \details This module depends on jsoncpp library to load all relevant data. It provides
    ///          methods to extract data from a json document.
    class JIMINY_DLLAPI JsonLoader
    {
    public:
        explicit JsonLoader(const std::shared_ptr<AbstractIODevice> & device) noexcept;

        /// \brief Load json data from device and parse it to root json.
        hresult_t load();

        /// \brief Reference to the parsed root json document.
        ///
        /// \details Use to allow child classes to implement custom parsers.
        const Json::Value * getRoot();

    protected:
        /// \brief Hold the parsed document.
        std::unique_ptr<Json::Value> rootJson_{std::make_unique<Json::Value>()};
        std::vector<char> payload_{};
        std::shared_ptr<AbstractIODevice> device_;
    };
}

#endif  // JIMINY_JSON_LOADER_H
