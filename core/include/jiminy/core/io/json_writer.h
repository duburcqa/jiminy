#ifndef JIMINY_JSON_WRITER_H
#define JIMINY_JSON_WRITER_H

#include "jiminy/core/fwd.h"


namespace Json
{
    class Value;
}

namespace jiminy
{
    class AbstractIODevice;

    class JIMINY_DLLAPI JsonWriter
    {
    public:
        explicit JsonWriter(const std::shared_ptr<AbstractIODevice> & device) noexcept;

        /// \brief Dump current content to device.
        hresult_t dump(const Json::Value & input);

    private:
        std::shared_ptr<AbstractIODevice> device_;
    };
}

#endif
