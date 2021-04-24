///////////////////////////////////////////////////////////////////////////////
///
/// \brief   Helper class to load and parse JSON files.
///
/// \details This module depends on jsoncpp library to load all relevant data.
///          It provides methods to extract data from a json document.
///
///////////////////////////////////////////////////////////////////////////////

#ifndef JIMINY_JSON_LOADER_H
#define JIMINY_JSON_LOADER_H

#include "jiminy/core/Types.h"


namespace Json
{
    class Value;
}

namespace jiminy
{
    class AbstractIODevice;

    class JsonLoader
    {
    public:
        explicit JsonLoader(std::shared_ptr<AbstractIODevice> device);
        ~JsonLoader() = default;

        ///////////////////////////////////////////////////////////////////////
        /// \brief Load json data from device and parse it to root json.
        ///////////////////////////////////////////////////////////////////////
        hresult_t load();

        ///////////////////////////////////////////////////////////////////////
        /// \brief Get a reference to the parsed root json document.
        ///
        /// \details Use to allow child classes to implement custom parsers.
        ///
        /// \retval The parsed JSON document.
        ///////////////////////////////////////////////////////////////////////
        Json::Value const * getRoot();

    protected:
        std::unique_ptr<Json::Value> rootJson_;  ///< To hold the parsed document.
        std::vector<char_t> payload_;
        std::shared_ptr<AbstractIODevice> device_;
    };
}  // End of namespace jiminy.
#endif  // JIMINY_JSON_LOADER_H
