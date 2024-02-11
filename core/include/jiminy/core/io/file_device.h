#ifndef SYSTEM_FILE_DEVICE_H
#define SYSTEM_FILE_DEVICE_H

#include <string>

#include "jiminy/core/fwd.h"
#include "jiminy/core/io/abstract_io_device.h"


namespace jiminy
{
    /// \brief Class to manipulate a file.
    class JIMINY_DLLAPI FileDevice : public AbstractIODevice
    {
    public:
        explicit FileDevice(const std::string & filename) noexcept;
        virtual ~FileDevice();

        std::size_t size() override;
        void seek(std::ptrdiff_t pos) override;
        std::ptrdiff_t pos() override;
        void resize(std::size_t size) override;
        std::size_t bytesAvailable() override;

        const std::string & name() const;

    protected:
        void doOpen(OpenMode mode) override;
        void doClose() override;

        std::ptrdiff_t readData(void * data, std::size_t dataSize) override;
        std::ptrdiff_t writeData(const void * data, std::size_t dataSize) override;

    protected:
        std::string filename_;
        /// \brief File descriptor.
        int32_t fileDescriptor_{-1};
    };
}

#endif  // FILESYSTEM_FILE_H