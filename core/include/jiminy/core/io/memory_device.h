#ifndef JIMINY_CORE_MEMORY_DEVICE_H
#define JIMINY_CORE_MEMORY_DEVICE_H

#include "jiminy/core/fwd.h"
#include "jiminy/core/io/abstract_io_device.h"


namespace jiminy
{
    /// \brief Manage a memory buffer with IODevice interface.
    class JIMINY_DLLAPI MemoryDevice : public AbstractIODevice
    {
    public:
        explicit MemoryDevice(std::size_t size) noexcept;
        MemoryDevice(MemoryDevice && other);
        explicit MemoryDevice(std::vector<uint8_t> && initBuffer) noexcept;
        virtual ~MemoryDevice();

        std::size_t size() override { return buffer_.size(); }
        hresult_t seek(std::ptrdiff_t pos) override;
        std::ptrdiff_t pos() override { return currentPos_; }
        hresult_t resize(std::size_t size) override;
        std::size_t bytesAvailable() override { return size() - currentPos_; }

        bool isSequential() const override { return false; }

    protected:
        hresult_t doOpen(openMode_t modes) override;
        hresult_t doClose() override;

        std::ptrdiff_t readData(void * data, std::size_t dataSize) override;
        std::ptrdiff_t writeData(const void * data, std::size_t dataSize) override;

    private:
        std::vector<uint8_t> buffer_;
        std::ptrdiff_t currentPos_{0};
    };
}

#endif  // JIMINY_CORE_MEMORY_DEVICE_H
