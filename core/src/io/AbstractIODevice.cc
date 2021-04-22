///////////////////////////////////////////////////////////////////////////////
///
/// \brief Contains the AbstractIODevice class methods implementations.
///
///////////////////////////////////////////////////////////////////////////////

#include <iostream>

#include "jiminy/core/Macros.h"
#include "jiminy/core/io/AbstractIODevice.h"


namespace jiminy
{
    openMode_t operator|(openMode_t const & modeA,
                         openMode_t const & modeB)
    {
        return static_cast<openMode_t>(static_cast<int32_t>(modeA) | static_cast<int32_t>(modeB));
    }

    openMode_t operator&(openMode_t const & modeA,
                         openMode_t const & modeB)
    {
        return static_cast<openMode_t>(static_cast<int32_t>(modeA) & static_cast<int32_t>(modeB));
    }

    openMode_t operator|=(openMode_t       & modeA,
                          openMode_t const & modeB)
    {
        return modeA = modeA | modeB;
    }

    openMode_t operator&=(openMode_t       & modeA,
                          openMode_t const & modeB)
    {
        return modeA = modeA & modeB;
    }

    openMode_t operator~(openMode_t const & mode)
    {
        return static_cast<openMode_t>(~static_cast<int32_t>(mode));
    }

    AbstractIODevice::AbstractIODevice() :
    modes_(openMode_t::NOT_OPEN),
    supportedModes_(openMode_t::NOT_OPEN),
    lastError_(hresult_t::SUCCESS),
    io_(nullptr)
    {
        // Empty on purpose.
    }

    hresult_t AbstractIODevice::open(openMode_t const & modes)
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

    hresult_t AbstractIODevice::close(void)
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

    openMode_t const & AbstractIODevice::openModes(void) const
    {
        return modes_;
    }

    openMode_t const & AbstractIODevice::supportedModes(void) const
    {
        return supportedModes_;
    }

    bool_t AbstractIODevice::isWritable(void) const
    {
        return (modes_ & openMode_t::WRITE_ONLY) || (modes_ & openMode_t::READ_WRITE);
    }

    bool_t AbstractIODevice::isReadable(void) const
    {
        return (modes_ & openMode_t::READ_ONLY) || (modes_ & openMode_t::READ_WRITE);
    }

    bool_t AbstractIODevice::isOpen(void) const
    {
        return (modes_ != openMode_t::NOT_OPEN);
    }

    bool_t AbstractIODevice::isSequential(void) const
    {
        return false;
    }

    int64_t AbstractIODevice::size(void)
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

    int64_t AbstractIODevice::pos(void)
    {
        return 0;
    }

    int64_t AbstractIODevice::bytesAvailable(void)
    {
        return 0;
    }

    hresult_t AbstractIODevice::getLastError(void) const
    {
        return lastError_;
    }

    hresult_t AbstractIODevice::write(void    const * data,
                                      int64_t         dataSize)
    {
        int64_t toWrite = dataSize;
        uint8_t const * bufferPos = static_cast<uint8_t const *>(data);

        while (toWrite > 0)
        {
            int64_t writtenBytes = writeData(bufferPos + (dataSize - toWrite), toWrite);
            if (writtenBytes <= 0)
            {
                lastError_ = hresult_t::ERROR_GENERIC;
                PRINT_ERROR("Something went wrong. No data was written.");
                return lastError_;
            }
            toWrite -= writtenBytes;
        }

        return hresult_t::SUCCESS;
    }

    hresult_t AbstractIODevice::read(void    * data,
                                     int64_t   dataSize)
    {
        int64_t toRead = dataSize;
        uint8_t * bufferPos = static_cast<uint8_t *>(data);

        while (toRead > 0)
        {
            int64_t readBytes = readData(bufferPos + (dataSize - toRead), toRead);
            if (readBytes <= 0)
            {
                lastError_ = hresult_t::ERROR_GENERIC;
                PRINT_ERROR("Something went wrong. No data was read.");
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

    bool_t AbstractIODevice::isBackendValid(void)
    {
        return (io_.get() != nullptr);
    }

    void AbstractIODevice::setBackend(std::unique_ptr<AbstractIODevice> io)
    {
        io_ = std::move(io);
        supportedModes_ = io_->supportedModes();
    }

    void AbstractIODevice::removeBackend(void)
    {
        io_.reset();
        supportedModes_ = openMode_t::NOT_OPEN;
    }

    // Specific implementation - std::vector<uint8_t>
    template<>
    hresult_t AbstractIODevice::read<std::vector<uint8_t> >(std::vector<uint8_t> & v)
    {
        int64_t toRead = static_cast<int64_t>(v.size() * sizeof(uint8_t));
        uint8_t * bufferPos = reinterpret_cast<uint8_t *>(v.data());
        return read(bufferPos, toRead);
    }

    // Specific implementation - std::vector<char_t>
    template<>
    hresult_t AbstractIODevice::read<std::vector<char_t> >(std::vector<char_t> & v)
    {
        int64_t toRead = static_cast<int64_t>(v.size() * sizeof(char_t));
        uint8_t * bufferPos = reinterpret_cast<uint8_t *>(v.data());
        return read(bufferPos, toRead);
    }

    // Specific implementation - std::string
    template<>
    hresult_t AbstractIODevice::write<std::string>(std::string const & str)
    {
        int64_t toWrite = static_cast<int64_t>(str.size());
        uint8_t const * bufferPos = reinterpret_cast<uint8_t const *>(str.c_str());
        return write(bufferPos, toWrite);
    }

    // Specific implementation - std::vector<uint8_t>
    template<>
    hresult_t AbstractIODevice::write<std::vector<uint8_t> >(std::vector<uint8_t> const & v)
    {
        int64_t toWrite = static_cast<int64_t>(v.size() * sizeof(uint8_t));
        uint8_t const * bufferPos = reinterpret_cast<uint8_t const *>(v.data());
        return write(bufferPos, toWrite);
    }

    // Specific implementation - std::vector<char_t>
    template<>
    hresult_t AbstractIODevice::write<std::vector<char_t> >(std::vector<char_t> const & v)
    {
        int64_t toWrite = static_cast<int64_t>(v.size() * sizeof(char_t));
        uint8_t const * bufferPos = reinterpret_cast<uint8_t const *>(v.data());
        return write(bufferPos, toWrite);
    }

    // Specific implementation - std::vector<uint64_t>
    template<>
    hresult_t AbstractIODevice::write<std::vector<uint64_t> >(std::vector<uint64_t> const & v)
    {
        int64_t toWrite = static_cast<int64_t>(v.size() * sizeof(uint64_t));
        uint8_t const * bufferPos = reinterpret_cast<uint8_t const *>(&v[0]);
        return write(bufferPos, toWrite);
    }
}
