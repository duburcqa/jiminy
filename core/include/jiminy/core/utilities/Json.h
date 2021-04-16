#ifndef JIMINY_JSON_H
#define JIMINY_JSON_H

#include "json/json.h"

#include "jiminy/core/Macros.h"
#include "jiminy/core/Types.h"


namespace jiminy
{
    // *************** Conversion to JSON utilities *****************

    class AbstractIODevice;

    template<typename T>
    std::enable_if_t<!is_vector_v<T>, Json::Value>
    convertToJson(T const & value);

    template<typename T>
    std::enable_if_t<is_vector_v<T>, Json::Value>
    convertToJson(T const & value);

    hresult_t jsonDump(configHolder_t                    const & config,
                       std::shared_ptr<AbstractIODevice>       & device);

    // ************* Conversion from JSON utilities *****************

    template<typename T>
    std::enable_if_t<is_vector_v<T>, T>
    convertFromJson(Json::Value const & value);

    template<typename T>
    std::enable_if_t<!is_vector_v<T>, T>
    convertFromJson(Json::Value const & value);

    hresult_t jsonLoad(configHolder_t                    & config,
                       std::shared_ptr<AbstractIODevice> & device);
}

#include "jiminy/core/utilities/Json.tpp"

#endif  // JIMINY_JSON_H