#ifndef JIMINY_JSON_WRITER_H
#define JIMINY_JSON_WRITER_H

#include "jiminy/core/macros.h"
#include "jiminy/core/types.h"


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
        JsonWriter(std::shared_ptr<AbstractIODevice> device);
        ~JsonWriter() = default;

        /// \brief Dump current content to device.
        hresult_t dump(const Json::Value & input);

        /// \brief In case the constructor can't init the backend use set to init.
        void setBackend(std::shared_ptr<AbstractIODevice> device);

    private:
        std::shared_ptr<AbstractIODevice> device_;
    };
}

#endif
