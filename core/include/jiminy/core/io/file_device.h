#ifndef SYSTEM_FILE_DEVICE_H
#define SYSTEM_FILE_DEVICE_H

#include <string>

#include "jiminy/core/io/abstract_io_device.h"


namespace jiminy
{
    /// \brief Class to manipulate a file.
    class FileDevice : public AbstractIODevice
    {
    public:
        FileDevice(const std::string & filename);
        virtual ~FileDevice();

        int64_t size() override;
        hresult_t resize(int64_t sizeIn) override;
        hresult_t seek(int64_t pos) override;
        int64_t pos() override;
        int64_t bytesAvailable() override;

        int64_t readData(void * data, int64_t dataSize) override;
        int64_t writeData(const void * data, int64_t dataSize) override;

        const std::string & name() const;

    protected:
        hresult_t doOpen(const openMode_t & mode) override;
        hresult_t doClose() override;

        std::string filename_;
        /// \brief File descriptor.
        int32_t fileDescriptor_;
    };
}

#endif  // FILESYSTEM_FILE_H