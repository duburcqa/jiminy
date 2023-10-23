#ifndef JIMINY_JSON_H
#define JIMINY_JSON_H

#include "jiminy/core/types.h"


namespace Json
{
    class Value;
}

namespace jiminy
{
    class AbstractIODevice;

    // *************** Conversion to JSON utilities *****************

    Json::Value convertToJson(const configHolder_t & value);

    hresult_t jsonDump(const configHolder_t & config, std::shared_ptr<AbstractIODevice> & device);

    // ************* Conversion from JSON utilities *****************

    configHolder_t convertFromJson(const Json::Value & value);

    hresult_t jsonLoad(configHolder_t & config, std::shared_ptr<AbstractIODevice> & device);
}

#endif  // JIMINY_JSON_H