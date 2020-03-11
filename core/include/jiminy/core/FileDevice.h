///////////////////////////////////////////////////////////////////////////////
///
/// \brief Contains the File class to manipulate a file.
///
///////////////////////////////////////////////////////////////////////////////

#ifndef WDC_SYSTEM_FILE_DEVICE_H
#define WDC_SYSTEM_FILE_DEVICE_H

// C++ STD
#include <string>

#include "jiminy/core/AbstractIODevice.h"


namespace jiminy
{
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /// \brief Class to manipulate a file.
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    class FileDevice : public AbstractIODevice
    {
    public:
        FileDevice(std::string const& filename);
        virtual ~FileDevice(void);

        int64_t size(void) override;
        result_t seek(int64_t pos) override;
        int64_t pos(void) override;
        int64_t bytesAvailable(void) override;

        int64_t readData(void    * data,
                         int64_t   dataSize) override;
        int64_t writeData(void    const * data,
                          int64_t         dataSize) override;

        std::string const& name(void) const;

    protected:
        result_t doOpen(enum OpenMode mode) override;
        void doClose(void) override;

        std::string filename_;
        int32_t fileDescriptor_;  ///< File descriptor.
    };
}

#endif // WDC_FILESYSTEM_FILE_H