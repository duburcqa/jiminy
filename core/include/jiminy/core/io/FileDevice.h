///////////////////////////////////////////////////////////////////////////////
///
/// \brief Contains the File class to manipulate a file.
///
///////////////////////////////////////////////////////////////////////////////

#ifndef SYSTEM_FILE_DEVICE_H
#define SYSTEM_FILE_DEVICE_H

#include <string>

#include "jiminy/core/io/AbstractIODevice.h"


namespace jiminy
{
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /// \brief Class to manipulate a file.
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    class FileDevice : public AbstractIODevice
    {
    public:
        FileDevice(std::string const & filename);
        virtual ~FileDevice(void);

        int64_t size(void) override;
        hresult_t resize(int64_t sizeIn) override;
        hresult_t seek(int64_t pos) override;
        int64_t pos(void) override;
        int64_t bytesAvailable(void) override;

        int64_t readData(void    * data,
                         int64_t   dataSize) override;
        int64_t writeData(void    const * data,
                          int64_t         dataSize) override;

        std::string const & name(void) const;

    protected:
        hresult_t doOpen(enum OpenMode mode) override;
        hresult_t doClose(void) override;

        std::string filename_;
        int32_t fileDescriptor_;  ///< File descriptor.
    };
}

#endif // FILESYSTEM_FILE_H