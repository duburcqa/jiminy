
#include "jiminy/core/io/abstract_io_device.h"


namespace jiminy
{
    OpenMode operator|(OpenMode modeA, OpenMode modeB)
    {
        return static_cast<OpenMode>(static_cast<int32_t>(modeA) | static_cast<int32_t>(modeB));
    }

    OpenMode operator&(OpenMode modeA, OpenMode modeB)
    {
        return static_cast<OpenMode>(static_cast<int32_t>(modeA) & static_cast<int32_t>(modeB));
    }

    OpenMode operator|=(OpenMode & modeA, OpenMode modeB)
    {
        return modeA = modeA | modeB;
    }

    OpenMode operator&=(OpenMode & modeA, OpenMode modeB)
    {
        return modeA = modeA & modeB;
    }

    OpenMode operator~(OpenMode mode)
    {
        return static_cast<OpenMode>(~static_cast<int32_t>(mode));
    }

    AbstractIODevice::AbstractIODevice(OpenMode supportedModes) noexcept :
    supportedModes_(supportedModes)
    {
    }

    void AbstractIODevice::open(OpenMode modes)
    {
        if (isOpen())
        {
            JIMINY_THROW(bad_control_flow, "Device already open.");
        }

        if ((modes & supportedModes_) != modes)
        {
            JIMINY_THROW(std::invalid_argument,
                         "At least one of the selected modes is not supported by the device.");
        }

        doOpen(modes);
        modes_ = modes;
    }

    void AbstractIODevice::close()
    {
        if (!isOpen())
        {
            JIMINY_THROW(bad_control_flow, "Device not open.");
        }

        doClose();
        modes_ = NOT_OPEN;
    }

    OpenMode AbstractIODevice::openModes() const
    {
        return modes_;
    }

    OpenMode AbstractIODevice::supportedModes() const
    {
        return supportedModes_;
    }

    bool AbstractIODevice::isWritable() const
    {
        return (modes_ & OpenMode::WRITE_ONLY) || (modes_ & OpenMode::READ_WRITE);
    }

    bool AbstractIODevice::isReadable() const
    {
        return (modes_ & OpenMode::READ_ONLY) || (modes_ & OpenMode::READ_WRITE);
    }

    bool AbstractIODevice::isOpen() const
    {
        return (modes_ != OpenMode::NOT_OPEN);
    }

    bool AbstractIODevice::isSequential() const
    {
        return false;
    }

    std::size_t AbstractIODevice::size()
    {
        return bytesAvailable();
    }

    void AbstractIODevice::resize(std::size_t /* size */)
    {
        JIMINY_THROW(not_implemented_error, "Method not available.");
    }

    void AbstractIODevice::seek(std::ptrdiff_t /* pos */)
    {
        JIMINY_THROW(not_implemented_error, "Method not available.");
    }

    std::ptrdiff_t AbstractIODevice::pos()
    {
        return -1;
    }

    std::size_t AbstractIODevice::bytesAvailable()
    {
        return 0;
    }

    void AbstractIODevice::write(const void * data, std::size_t dataSize)
    {
        std::size_t toWrite = dataSize;
        const uint8_t * bufferPos = static_cast<const uint8_t *>(data);

        while (toWrite > 0)
        {
            std::ptrdiff_t writtenBytes = writeData(bufferPos + (dataSize - toWrite), toWrite);
            if (writtenBytes <= 0)
            {
                JIMINY_THROW(std::ios_base::failure, "No data was written. Device probably full.");
            }
            toWrite -= writtenBytes;
        }
    }

    void AbstractIODevice::read(void * data, std::size_t dataSize)
    {
        std::size_t toRead = dataSize;
        uint8_t * bufferPos = static_cast<uint8_t *>(data);

        while (toRead > 0)
        {
            std::ptrdiff_t readBytes = readData(bufferPos + (dataSize - toRead), toRead);
            if (readBytes <= 0)
            {
                JIMINY_THROW(std::ios_base::failure, "No data was read. Device probably empty.");
            }
            toRead -= readBytes;
        }
    }
}
