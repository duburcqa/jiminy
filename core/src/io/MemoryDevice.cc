#include <iostream>

#include "jiminy/core/Macros.h"
#include "jiminy/core/io/MemoryDevice.h"


namespace jiminy
{
    MemoryDevice::MemoryDevice(uint64_t const & size) :
    buffer_(static_cast<std::size_t>(size)),
    currentPos_(0)
    {
        supportedModes_ = openMode_t::READ_ONLY | openMode_t::WRITE_ONLY | openMode_t::READ_WRITE | openMode_t::NON_BLOCKING | openMode_t::APPEND;
    }


    MemoryDevice::MemoryDevice(MemoryDevice const & other) :
    buffer_(other.buffer_),
    currentPos_(other.currentPos_)
    {
        supportedModes_ = other.supportedModes_;
        modes_ = other.modes_;
    }


    MemoryDevice::MemoryDevice(MemoryDevice && other) :
    buffer_(std::move(other.buffer_)),
    currentPos_(other.currentPos_)
    {
        supportedModes_ = other.supportedModes_;
        modes_ = other.modes_;
        other.close();
    }

    MemoryDevice::MemoryDevice(std::vector<uint8_t> && initBuffer) :
    buffer_(std::move(initBuffer)),
    currentPos_(0)
    {
        supportedModes_ = openMode_t::READ_ONLY | openMode_t::WRITE_ONLY | openMode_t::READ_WRITE | openMode_t::NON_BLOCKING | openMode_t::APPEND;
    }

    MemoryDevice::~MemoryDevice(void)
    {
        close();
    }

    MemoryDevice & MemoryDevice::operator=(MemoryDevice const & other)
    {
        buffer_ = other.buffer_;
        currentPos_ = other.currentPos_;
        modes_ = other.modes_;

        return *this;
    }

    MemoryDevice & MemoryDevice::operator=(MemoryDevice && other)
    {
        buffer_ = std::move(other.buffer_);
        currentPos_ = other.currentPos_;
        modes_ = other.modes_;
        other.close();

        return *this;
    }

    hresult_t MemoryDevice::seek(int64_t pos)
    {
        if ((pos < 0) || pos > size())
        {
            lastError_ = hresult_t::ERROR_GENERIC;
            PRINT_ERROR("The requested position '", pos, "' is out of scope.");
            return lastError_;
        }

        currentPos_ = pos;
        return hresult_t::SUCCESS;
    }

    int64_t MemoryDevice::readData(void    * data,
                                   int64_t   dataSize)
    {
        // Read no more than availables bytes.
        int64_t toRead = bytesAvailable();
        if (dataSize < toRead)
        {
            toRead = dataSize;
        }

        std::memcpy(data, buffer_.data() + currentPos_, static_cast<std::size_t>(toRead));
        currentPos_ += toRead;
        return toRead;
    }

    int64_t MemoryDevice::writeData(void    const * data,
                                    int64_t         dataSize)
    {
        // Write no more than availables bytes.
        int64_t toWrite = bytesAvailable();
        if (dataSize < toWrite)
        {
            toWrite = dataSize;
        }

        std::memcpy(buffer_.data() + currentPos_, data, static_cast<std::size_t>(toWrite));
        currentPos_ += toWrite;

        return toWrite;
    }

    hresult_t MemoryDevice::setBlockingMode(bool_t)
    {
        // Since this is a memory device, it can't block when doing its job,
        // thus we don't care about blocking mode and answer 'OK no problem'.
        return hresult_t::SUCCESS;
    }

    hresult_t MemoryDevice::doOpen(openMode_t const & modes)
    {
        if (!(modes & openMode_t::APPEND))
        {
            currentPos_ = 0;
        }

        return hresult_t::SUCCESS;
    }

    hresult_t MemoryDevice::doClose(void)
    {
        return hresult_t::SUCCESS;
    }

    hresult_t MemoryDevice::resize(int64_t size)
    {
        buffer_.resize(static_cast<std::size_t>(size));
        return hresult_t::SUCCESS;
    }
}
