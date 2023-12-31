
#include "jiminy/core/io/abstract_io_device.h"


namespace jiminy
{
    openMode_t operator|(openMode_t modeA, openMode_t modeB)
    {
        return static_cast<openMode_t>(static_cast<int32_t>(modeA) | static_cast<int32_t>(modeB));
    }

    openMode_t operator&(openMode_t modeA, openMode_t modeB)
    {
        return static_cast<openMode_t>(static_cast<int32_t>(modeA) & static_cast<int32_t>(modeB));
    }

    openMode_t operator|=(openMode_t & modeA, openMode_t modeB)
    {
        return modeA = modeA | modeB;
    }

    openMode_t operator&=(openMode_t & modeA, openMode_t modeB)
    {
        return modeA = modeA & modeB;
    }

    openMode_t operator~(openMode_t mode)
    {
        return static_cast<openMode_t>(~static_cast<int32_t>(mode));
    }

    AbstractIODevice::AbstractIODevice(openMode_t supportedModes) noexcept :
    supportedModes_(supportedModes)
    {
    }

    hresult_t AbstractIODevice::open(openMode_t modes)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (isOpen())
        {
            PRINT_ERROR("Already open.");
            returnCode = lastError_ = hresult_t::ERROR_GENERIC;
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            if ((modes & supportedModes_) != modes)
            {
                std::cout << "supportedModes_: " << supportedModes_ << " | modes: " << modes
                          << std::endl;
                throw std::runtime_error("ERROR");
                PRINT_ERROR("At least one of the selected modes is not supported.");
                returnCode = lastError_ = hresult_t::ERROR_GENERIC;
            }
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            returnCode = doOpen(modes);
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            modes_ = modes;
        }

        return returnCode;
    }

    hresult_t AbstractIODevice::close()
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (!isOpen())
        {
            returnCode = hresult_t::ERROR_GENERIC;
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            returnCode = doClose();
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            modes_ = NOT_OPEN;
        }

        return returnCode;
    }

    openMode_t AbstractIODevice::openModes() const
    {
        return modes_;
    }

    openMode_t AbstractIODevice::supportedModes() const
    {
        return supportedModes_;
    }

    bool AbstractIODevice::isWritable() const
    {
        return (modes_ & openMode_t::WRITE_ONLY) || (modes_ & openMode_t::READ_WRITE);
    }

    bool AbstractIODevice::isReadable() const
    {
        return (modes_ & openMode_t::READ_ONLY) || (modes_ & openMode_t::READ_WRITE);
    }

    bool AbstractIODevice::isOpen() const
    {
        return (modes_ != openMode_t::NOT_OPEN);
    }

    bool AbstractIODevice::isSequential() const
    {
        return false;
    }

    std::size_t AbstractIODevice::size()
    {
        return bytesAvailable();
    }

    hresult_t AbstractIODevice::resize(std::size_t /* size */)
    {
        lastError_ = hresult_t::ERROR_GENERIC;
        PRINT_ERROR("This method is not available.");
        return lastError_;
    }

    hresult_t AbstractIODevice::seek(std::ptrdiff_t /* pos */)
    {
        lastError_ = hresult_t::ERROR_GENERIC;
        PRINT_ERROR("This method is not available.");
        return lastError_;
    }

    std::ptrdiff_t AbstractIODevice::pos()
    {
        return -1;
    }

    std::size_t AbstractIODevice::bytesAvailable()
    {
        return 0;
    }

    hresult_t AbstractIODevice::getLastError() const
    {
        return lastError_;
    }

    hresult_t AbstractIODevice::write(const void * data, std::size_t dataSize)
    {
        std::size_t toWrite = dataSize;
        const uint8_t * bufferPos = static_cast<const uint8_t *>(data);

        while (toWrite > 0)
        {
            std::ptrdiff_t writtenBytes = writeData(bufferPos + (dataSize - toWrite), toWrite);
            if (writtenBytes <= 0)
            {
                lastError_ = hresult_t::ERROR_GENERIC;
                PRINT_ERROR("No data was written. The device is full is probably full.");
                return lastError_;
            }
            toWrite -= writtenBytes;
        }

        return hresult_t::SUCCESS;
    }

    hresult_t AbstractIODevice::read(void * data, std::size_t dataSize)
    {
        std::size_t toRead = dataSize;
        uint8_t * bufferPos = static_cast<uint8_t *>(data);

        while (toRead > 0)
        {
            std::ptrdiff_t readBytes = readData(bufferPos + (dataSize - toRead), toRead);
            if (readBytes <= 0)
            {
                lastError_ = hresult_t::ERROR_GENERIC;
                PRINT_ERROR("No data was read. The device is full is probably empty.");
                return lastError_;
            }
            toRead -= readBytes;
        }

        return hresult_t::SUCCESS;
    }
}
