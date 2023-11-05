#ifndef JIMINY_CORE_MEMORY_DEVICE_H
#define JIMINY_CORE_MEMORY_DEVICE_H

#include "jiminy/core/io/abstract_io_device.h"


namespace jiminy
{
    /// \brief Manage a memory buffer with IODevice interface.
    class MemoryDevice : public AbstractIODevice
    {
    public:
        MemoryDevice(const uint64_t & size);
        MemoryDevice(const MemoryDevice & other);
        MemoryDevice(MemoryDevice && other);

        MemoryDevice(std::vector<uint8_t> && initBuffer);

        virtual ~MemoryDevice();

        MemoryDevice & operator=(const MemoryDevice & other);
        MemoryDevice & operator=(MemoryDevice && other);

        int64_t size() override { return static_cast<int64_t>(buffer_.size()); }

        bool_t isSequential() const override { return false; }

        int64_t pos() override { return currentPos_; }

        int64_t bytesAvailable() override { return size() - currentPos_; }

        hresult_t seek(int64_t pos) override;

        int64_t readData(void * data, int64_t dataSize) override;
        int64_t writeData(const void * data, int64_t dataSize) override;

        hresult_t setBlockingMode(bool_t) override;

        hresult_t resize(int64_t size) override;

    protected:
        hresult_t doOpen(const openMode_t & modes) override;
        hresult_t doClose() override;

    private:
        std::vector<uint8_t> buffer_;
        int64_t currentPos_;
    };
}

#endif  // JIMINY_CORE_MEMORY_DEVICE_H
