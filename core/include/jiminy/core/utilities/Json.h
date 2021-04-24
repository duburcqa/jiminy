#ifndef JIMINY_JSON_H
#define JIMINY_JSON_H

#include "jiminy/core/Types.h"


namespace Json
{
    class Value;
}

namespace jiminy
{
    class AbstractIODevice;

    // *************** Conversion to JSON utilities *****************

    Json::Value convertToJson(configHolder_t const & value);

    hresult_t jsonDump(configHolder_t                    const & config,
                       std::shared_ptr<AbstractIODevice>       & device);

    // ************* Conversion from JSON utilities *****************

    configHolder_t convertFromJson(Json::Value const & value);

    hresult_t jsonLoad(configHolder_t                    & config,
                       std::shared_ptr<AbstractIODevice> & device);
}

#endif  // JIMINY_JSON_H