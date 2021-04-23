#ifndef JIMINY_JSON_WRITER_H
#define JIMINY_JSON_WRITER_H

#include "jiminy/core/Types.h"


namespace Json
{
    class Value;
}

namespace jiminy
{
    class AbstractIODevice;

    class JsonWriter
    {
    public:
        JsonWriter(std::shared_ptr<AbstractIODevice> device);

        ~JsonWriter() = default;

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief Dump current content to device.
        /////////////////////////////////////////////////////////////////////////////////////////////////////////////
        hresult_t dump(Json::Value const & input);

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief In case the constructor can't init the backend use set to init.
        /////////////////////////////////////////////////////////////////////////////////////////////////////////////
        void setBackend(std::shared_ptr<AbstractIODevice> device);
    private:
        std::shared_ptr<AbstractIODevice> device_;
    };
}

#endif
