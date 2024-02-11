#ifndef JIMINY_JSON_H
#define JIMINY_JSON_H

#include "jiminy/core/fwd.h"


namespace Json
{
    class Value;
}

namespace jiminy
{
    class AbstractIODevice;

    // *************** Conversion to JSON utilities *****************

    Json::Value JIMINY_DLLAPI convertToJson(const GenericConfig & value);

    void JIMINY_DLLAPI jsonDump(const GenericConfig & config,
                                std::shared_ptr<AbstractIODevice> & device);

    // ************* Conversion from JSON utilities *****************

    GenericConfig JIMINY_DLLAPI convertFromJson(const Json::Value & value);

    void JIMINY_DLLAPI jsonLoad(GenericConfig & config,
                                std::shared_ptr<AbstractIODevice> & device);
}

#endif  // JIMINY_JSON_H