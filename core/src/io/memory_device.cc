
#include "jiminy/core/io/memory_device.h"


namespace jiminy
{
    MemoryDevice::MemoryDevice(std::size_t sizeIn) noexcept :
    AbstractIODevice(
#ifndef _WIN32
        OpenMode::NON_BLOCKING |
#endif
        OpenMode::READ_ONLY | OpenMode::WRITE_ONLY | OpenMode::READ_WRITE | OpenMode::APPEND),
    buffer_(sizeIn),
    currentPos_{0}
    {
    }

    MemoryDevice::MemoryDevice(MemoryDevice && other) :
    AbstractIODevice(std::move(static_cast<AbstractIODevice &&>(other))),
    buffer_(std::move(other.buffer_)),
    currentPos_{other.currentPos_}
    {
        other.close();
    }

    MemoryDevice::MemoryDevice(std::vector<uint8_t> && initBuffer) noexcept :
    AbstractIODevice(
#ifndef _WIN32
        OpenMode::NON_BLOCKING |
#endif
        OpenMode::READ_ONLY | OpenMode::WRITE_ONLY | OpenMode::READ_WRITE | OpenMode::APPEND),
    buffer_(std::move(initBuffer)),
    currentPos_{0}
    {
    }

    MemoryDevice::~MemoryDevice()
    {
        close();
    }

    hresult_t MemoryDevice::seek(std::ptrdiff_t pos)
    {
        if ((pos < 0) || static_cast<std::size_t>(pos) > size())
        {
            lastError_ = hresult_t::ERROR_GENERIC;
            PRINT_ERROR("The requested position '", pos, "' is out of scope.");
            return lastError_;
        }
        currentPos_ = pos;
        return hresult_t::SUCCESS;
    }

    std::ptrdiff_t MemoryDevice::readData(void * data, std::size_t dataSize)
    {
        // Read no more than available bytes
        std::size_t toRead = std::min(dataSize, bytesAvailable());
        std::memcpy(data, buffer_.data() + currentPos_, toRead);
        currentPos_ += toRead;
        return toRead;
    }

    std::ptrdiff_t MemoryDevice::writeData(const void * data, std::size_t dataSize)
    {
        // Write no more than available bytes
        std::size_t toWrite = std::min(dataSize, bytesAvailable());
        std::memcpy(buffer_.data() + currentPos_, data, toWrite);
        currentPos_ += toWrite;
        return toWrite;
    }

    hresult_t MemoryDevice::doOpen(OpenMode modes)
    {
        if (!(modes & OpenMode::APPEND))
        {
            currentPos_ = 0;
        }

        return hresult_t::SUCCESS;
    }

    hresult_t MemoryDevice::doClose()
    {
        return hresult_t::SUCCESS;
    }

    hresult_t MemoryDevice::resize(std::size_t size)
    {
        buffer_.resize(size);
        return hresult_t::SUCCESS;
    }
}
