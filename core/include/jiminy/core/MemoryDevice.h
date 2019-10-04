#ifndef EXO_SIMU_CORE_MEMORY_DEVICE_H
#define EXO_SIMU_CORE_MEMORY_DEVICE_H

#include "jiminy/core/AbstractIODevice.h"


namespace jiminy
{
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /// \brief Manage a memory buffer with IODevice interface.
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    class MemoryDevice : public AbstractIODevice
    {
    public:
        MemoryDevice(int64_t size);
        MemoryDevice(MemoryDevice const& other);
        MemoryDevice(MemoryDevice&& other);

        MemoryDevice(std::vector<uint8_t>&& initBuffer);

        virtual ~MemoryDevice(void);

        MemoryDevice& operator=(MemoryDevice const& other);
        MemoryDevice& operator=(MemoryDevice&& other);

        int64_t size(void) const override
        {
            return buffer_.size();
        }

        bool_t isSequential(void) const override
        {
            return false;
        }

        int64_t pos(void) const override
        {
            return currentPos_;
        }

        int64_t bytesAvailable(void) const override
        {
            return (buffer_.size() - currentPos_);
        }

        result_t seek(int64_t pos) override;

        int64_t readData(void    * data,
                         int64_t   dataSize) override;
        int64_t writeData(void    const * data,
                          int64_t         dataSize) override;

        result_t setBlockingMode(bool_t) override;

        void resize(int64_t size);

    protected:
        result_t doOpen(enum OpenMode modes) override;
        void doClose() override;

    private:
        std::vector<uint8_t> buffer_;
        int64_t currentPos_;
    };
}

#endif // EXO_SIMU_CORE_MEMORY_DEVICE_H
