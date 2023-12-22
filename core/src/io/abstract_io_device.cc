#include "jiminy/core/exceptions.h"

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

    AbstractIODevice::AbstractIODevice() :
    modes_(openMode_t::NOT_OPEN),
    supportedModes_(openMode_t::NOT_OPEN),
    lastError_(hresult_t::SUCCESS),
    io_(nullptr)
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
                PRINT_ERROR("At least of the modes ", modes, " is not supported.");
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

    bool_t AbstractIODevice::isWritable() const
    {
        return (modes_ & openMode_t::WRITE_ONLY) || (modes_ & openMode_t::READ_WRITE);
    }

    bool_t AbstractIODevice::isReadable() const
    {
        return (modes_ & openMode_t::READ_ONLY) || (modes_ & openMode_t::READ_WRITE);
    }

    bool_t AbstractIODevice::isOpen() const
    {
        return (modes_ != openMode_t::NOT_OPEN);
    }

    bool_t AbstractIODevice::isSequential() const
    {
        return false;
    }

    int64_t AbstractIODevice::size()
    {
        return bytesAvailable();
    }

    hresult_t AbstractIODevice::resize(int64_t /* size */)
    {
        lastError_ = hresult_t::ERROR_GENERIC;
        PRINT_ERROR("This method is not available.");
        return lastError_;
    }

    hresult_t AbstractIODevice::seek(int64_t /* pos */)
    {
        lastError_ = hresult_t::ERROR_GENERIC;
        PRINT_ERROR("This method is not available.");
        return lastError_;
    }

    int64_t AbstractIODevice::pos()
    {
        return 0;
    }

    int64_t AbstractIODevice::bytesAvailable()
    {
        return 0;
    }

    hresult_t AbstractIODevice::getLastError() const
    {
        return lastError_;
    }

    hresult_t AbstractIODevice::write(const void * data, int64_t dataSize)
    {
        int64_t toWrite = dataSize;
        const uint8_t * bufferPos = static_cast<const uint8_t *>(data);

        while (toWrite > 0)
        {
            int64_t writtenBytes = writeData(bufferPos + (dataSize - toWrite), toWrite);
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

    hresult_t AbstractIODevice::read(void * data, int64_t dataSize)
    {
        int64_t toRead = dataSize;
        uint8_t * bufferPos = static_cast<uint8_t *>(data);

        while (toRead > 0)
        {
            int64_t readBytes = readData(bufferPos + (dataSize - toRead), toRead);
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

    hresult_t AbstractIODevice::setBlockingMode(bool_t /* shouldBlock */)
    {
        lastError_ = hresult_t::ERROR_GENERIC;
        PRINT_ERROR("This methid is not available.");
        return lastError_;
    }

    bool_t AbstractIODevice::isBackendValid()
    {
        return (io_.get() != nullptr);
    }

    void AbstractIODevice::setBackend(std::unique_ptr<AbstractIODevice> io)
    {
        io_ = std::move(io);
        supportedModes_ = io_->supportedModes();
    }

    void AbstractIODevice::removeBackend()
    {
        io_.reset();
        supportedModes_ = openMode_t::NOT_OPEN;
    }

    // Specific implementation - std::vector<uint8_t>
    template<>
    hresult_t AbstractIODevice::read<std::vector<uint8_t>>(std::vector<uint8_t> & v)
    {
        int64_t toRead = static_cast<int64_t>(v.size() * sizeof(uint8_t));
        uint8_t * bufferPos = reinterpret_cast<uint8_t *>(v.data());
        return read(bufferPos, toRead);
    }

    // Specific implementation - std::vector<char_t>
    template<>
    hresult_t AbstractIODevice::read<std::vector<char_t>>(std::vector<char_t> & v)
    {
        int64_t toRead = static_cast<int64_t>(v.size() * sizeof(char_t));
        uint8_t * bufferPos = reinterpret_cast<uint8_t *>(v.data());
        return read(bufferPos, toRead);
    }

    // Specific implementation - std::string
    template<>
    hresult_t AbstractIODevice::write<std::string_view>(const std::string_view & str)
    {
        int64_t toWrite = static_cast<int64_t>(str.size());
        const uint8_t * bufferPos = reinterpret_cast<const uint8_t *>(str.data());
        return write(bufferPos, toWrite);
    }

    // Specific implementation - std::vector<uint8_t>
    template<>
    hresult_t AbstractIODevice::write<std::vector<uint8_t>>(const std::vector<uint8_t> & v)
    {
        int64_t toWrite = static_cast<int64_t>(v.size() * sizeof(uint8_t));
        const uint8_t * bufferPos = reinterpret_cast<const uint8_t *>(v.data());
        return write(bufferPos, toWrite);
    }

    // Specific implementation - std::vector<char_t>
    template<>
    hresult_t AbstractIODevice::write<std::vector<char_t>>(const std::vector<char_t> & v)
    {
        int64_t toWrite = static_cast<int64_t>(v.size() * sizeof(char_t));
        const uint8_t * bufferPos = reinterpret_cast<const uint8_t *>(v.data());
        return write(bufferPos, toWrite);
    }

    // Specific implementation - std::vector<uint64_t>
    template<>
    hresult_t AbstractIODevice::write<std::vector<uint64_t>>(const std::vector<uint64_t> & v)
    {
        int64_t toWrite = static_cast<int64_t>(v.size() * sizeof(uint64_t));
        const uint8_t * bufferPos = reinterpret_cast<const uint8_t *>(&v[0]);
        return write(bufferPos, toWrite);
    }
}
