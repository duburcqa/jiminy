///////////////////////////////////////////////////////////////////////////////
///
/// \brief Contains the AbstractIODevice class methods implementations.
///
///////////////////////////////////////////////////////////////////////////////

#include <iostream>

#include "jiminy/core/AbstractIODevice.h"


namespace jiminy
{
    enum OpenMode operator | (enum OpenMode const & modeA,
                              enum OpenMode const & modeB)
    {
        return static_cast<enum OpenMode>(static_cast<int32_t>(modeA) | static_cast<int32_t>(modeB));
    }

    enum OpenMode operator & (enum OpenMode const & modeA,
                              enum OpenMode const & modeB)
    {
        return static_cast<enum OpenMode>(static_cast<int32_t>(modeA) & static_cast<int32_t>(modeB));
    }

    enum OpenMode operator |= (enum OpenMode       & modeA,
                               enum OpenMode const & modeB)
    {
        return modeA = modeA | modeB;
    }

    enum OpenMode operator &= (enum OpenMode       & modeA,
                               enum OpenMode const & modeB)
    {
        return modeA = modeA & modeB;
    }

    enum OpenMode operator ~(enum OpenMode mode)
    {
        int32_t tmpMode = ~static_cast<int32_t>(mode);
        return static_cast<enum OpenMode>(tmpMode);
    }

    result_t AbstractIODevice::open(OpenMode modes)
    {
        if (isOpen())
        {
            std::cout << "Error - AbstractIODevice::open - Already open." << std::endl;
            lastError_ = result_t::ERROR_GENERIC;
            return lastError_;
        }

        if ((modes & supportedModes_) != modes)
        {
            std::cout << "Error - AbstractIODevice::open - At least of the modes " << modes << " is not supported." << std::endl;
            lastError_ = result_t::ERROR_BAD_INPUT;
            return lastError_;
        }

        result_t returnCode = doOpen(modes);
        if (returnCode == result_t::SUCCESS)
        {
            modes_ = modes;
        }
        else
        {
            lastError_ = returnCode;
        }

        return returnCode;
    }

    void AbstractIODevice::close(void)
    {
        if (!isOpen())
        {
            lastError_ = result_t::ERROR_GENERIC;
            return;
        }

        doClose();
        modes_ = NOT_OPEN;
    }

    enum OpenMode AbstractIODevice::openModes(void) const
    {
        return modes_;
    }

    enum OpenMode AbstractIODevice::supportedModes(void) const
    {
        return supportedModes_;
    }

    bool_t AbstractIODevice::isWritable(void) const
    {
        return (modes_ & OpenMode::WRITE_ONLY) || (modes_ & OpenMode::READ_WRITE);
    }

    bool_t AbstractIODevice::isReadable(void) const
    {
        return (modes_ & OpenMode::READ_ONLY) || (modes_ & OpenMode::READ_WRITE);
    }

    bool_t AbstractIODevice::isOpen(void) const
    {
        return (modes_ != OpenMode::NOT_OPEN);
    }

    bool_t AbstractIODevice::isSequential(void) const
    {
        return false;
    }

    int64_t AbstractIODevice::size(void)
    {
        return bytesAvailable();
    }

    result_t AbstractIODevice::seek(int64_t pos)
    {
        (void) pos;
        std::cout << "Error - AbstractIODevice::seek - This methid is not available." << std::endl;
        return result_t::ERROR_GENERIC;
    }

    int64_t AbstractIODevice::pos(void)
    {
        return 0;
    }

    int64_t AbstractIODevice::bytesAvailable(void)
    {
        return 0;
    }

    result_t AbstractIODevice::getLastError(void) const
    {
        return lastError_;
    }

    result_t AbstractIODevice::write(void    const * data,
                                     int64_t         dataSize)
    {
        int64_t toWrite = dataSize;
        uint8_t const* bufferPos = static_cast<uint8_t const*>(data);

        while (toWrite > 0)
        {
            int64_t writtenBytes = writeData(bufferPos + (dataSize - toWrite), toWrite);
            if (writtenBytes < 0)
            {
                return lastError_;
            }

            if (writtenBytes == 0)
            {
                std::cout << "Error - AbstractIODevice::write - Something went wrong. No data was written." << std::endl;
                lastError_ = result_t::ERROR_GENERIC;
                return lastError_;
            }

            toWrite -= writtenBytes;
        }

        return result_t::SUCCESS;
    }

    result_t AbstractIODevice::read(void    * data,
                                    int64_t   dataSize)
    {
        int64_t toRead = dataSize;
        uint8_t* bufferPos = static_cast<uint8_t*>(data);

        while (toRead > 0)
        {
            int64_t readBytes = readData(bufferPos + (dataSize - toRead), toRead);
            if (readBytes < 0)
            {
                return lastError_;
            }

            if (readBytes == 0)
            {
                std::cout << "Error - AbstractIODevice::write - Something went wrong. No data was read." << std::endl;
                lastError_ = result_t::ERROR_GENERIC;
                return lastError_;
            }
            toRead -= readBytes;
        }

        return result_t::SUCCESS;
    }

    result_t AbstractIODevice::setBlockingMode(bool_t shouldBlock)
    {
        (void) shouldBlock;
        std::cout << "Error - AbstractIODevice::setBlockingMode - This methid is not available." << std::endl;
        lastError_ = result_t::ERROR_GENERIC;
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
        supportedModes_ = OpenMode::NOT_OPEN;
    }

    // Specific implementation - std::vector<uint8_t>
    template<>
    result_t AbstractIODevice::read<std::vector<uint8_t> >(std::vector<uint8_t>& v)
    {
        int64_t toRead = static_cast<int64_t>(v.size() * sizeof(uint8_t));
        uint8_t* bufferPos = reinterpret_cast<uint8_t*>(v.data());

        return read(bufferPos, toRead);
    }

    // Specific implementation - std::vector<char_t>
    template<>
    result_t AbstractIODevice::read<std::vector<char_t> >(std::vector<char_t>& v)
    {
        int64_t toRead = static_cast<int64_t>(v.size() * sizeof(char_t));
        uint8_t* bufferPos = reinterpret_cast<uint8_t*>(v.data());

        return read(bufferPos, toRead);
    }

    // Specific implementation - std::string
    template<>
    result_t AbstractIODevice::write<std::string>(std::string const& str)
    {
        int64_t toWrite = static_cast<int64_t>(str.size());
        uint8_t const* bufferPos = reinterpret_cast<uint8_t const*>(str.c_str());
        return write(bufferPos, toWrite);
    }

    // Specific implementation - std::vector<uint8_t>
    template<>
    result_t AbstractIODevice::write<std::vector<uint8_t> >(std::vector<uint8_t> const& v)
    {
        int64_t toWrite = static_cast<int64_t>(v.size() * sizeof(uint8_t));
        uint8_t const* bufferPos = reinterpret_cast<uint8_t const*>(v.data());

        return write(bufferPos, toWrite);
    }

    // Specific implementation - std::vector<char_t>
    template<>
    result_t AbstractIODevice::write<std::vector<char_t> >(std::vector<char_t> const& v)
    {
        int64_t toWrite = static_cast<int64_t>(v.size() * sizeof(char_t));
        uint8_t const* bufferPos = reinterpret_cast<uint8_t const*>(v.data());

        return write(bufferPos, toWrite);
    }

    // Specific implementation - std::vector<uint64_t>
    template<>
    result_t AbstractIODevice::write<std::vector<uint64_t> >(std::vector<uint64_t> const& v)
    {
        int64_t toWrite = static_cast<int64_t>(v.size() * sizeof(uint64_t));
        uint8_t const* bufferPos = reinterpret_cast<uint8_t const*>(&v[0]);

        return write(bufferPos, toWrite);
    }
}
