#ifndef JIMINY_JSON_H
#define JIMINY_JSON_H

#include "jiminy/core/macros.h"
#include "jiminy/core/types.h"


namespace Json
{
    class Value;
}

namespace jiminy
{
    class AbstractIODevice;

    // *************** Conversion to JSON utilities *****************

    Json::Value JIMINY_DLLAPI convertToJson(const configHolder_t & value);

    hresult_t JIMINY_DLLAPI jsonDump(const configHolder_t & config,
                                     std::shared_ptr<AbstractIODevice> & device);

    // ************* Conversion from JSON utilities *****************

    configHolder_t JIMINY_DLLAPI convertFromJson(const Json::Value & value);

    hresult_t JIMINY_DLLAPI jsonLoad(configHolder_t & config,
                                     std::shared_ptr<AbstractIODevice> & device);
}

#endif  // JIMINY_JSON_H