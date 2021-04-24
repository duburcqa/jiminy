///////////////////////////////////////////////////////////////////////////////
///
/// \brief Contains the FileDevice class methods implementations.
///
///////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "jiminy/core/Macros.h"
#include "jiminy/core/io/FileDevice.h"

#ifndef _WIN32
#include <unistd.h>
#else
/* This is intended as a drop-in replacement for unistd.h on Windows.
 * https://stackoverflow.com/a/826027/1202830
 */
#include <stdlib.h>
#include <io.h>

/* The specific versions for Windows read, write, open and close being #defined here only handle files. */
#define lseek _lseek
#define open _open
#define close _close
#define write _write
#define read _read
#define ftruncate _chsize

#define S_IRUSR _S_IREAD
#define S_IWUSR _S_IWRITE

#ifdef _WIN64
#define ssize_t __int64
#else
#define ssize_t long
#endif
#endif


namespace jiminy
{
    FileDevice::FileDevice(std::string const & filename) :
    filename_(filename),
    fileDescriptor_(-1)
    {
        supportedModes_ = openMode_t::READ_ONLY     | openMode_t::WRITE_ONLY | openMode_t::READ_WRITE |
                          openMode_t::NON_BLOCKING  | openMode_t::TRUNCATE   | openMode_t::NEW_ONLY   |
                          openMode_t::EXISTING_ONLY | openMode_t::APPEND     | openMode_t::SYNC;
        #ifndef _WIN32
        supportedModes_ |= openMode_t::NON_BLOCKING | openMode_t::SYNC;
        #endif
    }

    FileDevice::~FileDevice(void)
    {
        #if defined(close)
        #   pragma push_macro("close")
        #   undef close
        #endif
        close();
        #if defined(close)
        #   pragma pop_macro("close")
        #endif
    }

    hresult_t FileDevice::doOpen(openMode_t const & mode)
    {
        int32_t posixFLags = 0;
        if (mode & openMode_t::READ_ONLY)
        {
            posixFLags |= O_RDONLY;
        }
        if (mode & openMode_t::WRITE_ONLY)
        {
            posixFLags |= O_WRONLY;
            posixFLags |= O_CREAT;
        }
        if (mode & openMode_t::READ_WRITE)
        {
            posixFLags |= O_RDWR;
        }
        if (mode & openMode_t::TRUNCATE)
        {
            posixFLags |= O_TRUNC;
        }
        if (mode & openMode_t::NEW_ONLY)
        {
            posixFLags |= O_EXCL;
        }
        if (mode & openMode_t::EXISTING_ONLY)
        {
            posixFLags &= ~O_CREAT;
        }
        if (mode & openMode_t::APPEND)
        {
            posixFLags |= O_APPEND;
        }
        #ifndef _WIN32
        if (mode & openMode_t::NON_BLOCKING)
        {
            posixFLags |= O_NONBLOCK;
        }
        if (mode & openMode_t::SYNC)
        {
            posixFLags |= O_SYNC;
        }
        #endif
        #ifdef _WIN32
        posixFLags |= _O_BINARY;
        #endif

        int32_t const rc = ::open(filename_.c_str(), posixFLags, S_IRUSR | S_IWUSR);
        if (rc < 0)
        {
            lastError_ = hresult_t::ERROR_GENERIC;
            PRINT_ERROR("Impossible to open the file using the desired mode.");
            return lastError_;
        }

        fileDescriptor_ = rc;

        return hresult_t::SUCCESS;
    }

    hresult_t FileDevice::doClose(void)
    {
        int32_t const rc = ::close(fileDescriptor_);
        if (rc < 0)
        {
            lastError_ = hresult_t::ERROR_GENERIC;
            PRINT_ERROR("Impossible to close the file.");
            return lastError_;
        }
        else
        {
            fileDescriptor_ = -1;
        }
        return hresult_t::SUCCESS;
    }

    hresult_t FileDevice::seek(int64_t pos)
    {
        ssize_t const rc = ::lseek(fileDescriptor_, pos, SEEK_SET);
        if (rc < 0)
        {
            lastError_ = hresult_t::ERROR_GENERIC;
            PRINT_ERROR("The file is not open, or the requested position '", pos, "' is out of scope.");
            return lastError_;
        }
        return hresult_t::SUCCESS;
    }

    int64_t FileDevice::pos(void)
    {
        ssize_t const pos_cur = ::lseek(fileDescriptor_, 0, SEEK_CUR);
        if (pos_cur < 0)
        {
            lastError_ = hresult_t::ERROR_GENERIC;
            PRINT_ERROR("The file is not open, or the position would be negative or beyond the end.");
        }
        return pos_cur;
    }

    int64_t FileDevice::size(void)
    {
        struct stat st;
        int32_t rc = ::fstat(fileDescriptor_, &st);
        if (rc < 0)
        {
            lastError_ = hresult_t::ERROR_GENERIC;
            PRINT_ERROR("Impossible to access the file.");
        }
        return st.st_size;
    }

    int64_t FileDevice::bytesAvailable(void)
    {
        if (!isReadable())
        {
            return 0;
        }
        return size() - pos();
    }

    int64_t FileDevice::readData(void * data, int64_t dataSize)
    {
        ssize_t const readBytes = ::read(fileDescriptor_, data, static_cast<size_t>(dataSize));
        if (readBytes < 0)
        {
            lastError_ = hresult_t::ERROR_GENERIC;
            PRINT_ERROR("The file is not open, or data buffer is outside accessible address space.");
        }
        return readBytes;
    }

    int64_t FileDevice::writeData(void const * data, int64_t dataSize)
    {
        ssize_t const writtenBytes = ::write(fileDescriptor_, data, static_cast<size_t>(dataSize));
        if (writtenBytes < 0)
        {
            lastError_ = hresult_t::ERROR_GENERIC;
            PRINT_ERROR("The file is not open, or data buffer is outside accessible address space.");
        }
        return writtenBytes;
    }

    std::string const & FileDevice::name(void) const
    {
        return filename_;
    }

    hresult_t FileDevice::resize(int64_t sizeIn)
    {
        int32_t const rc = ::ftruncate(fileDescriptor_, sizeIn);
        if (rc < 0)
        {
            lastError_ = hresult_t::ERROR_GENERIC;
            PRINT_ERROR("The file is not open.");
            return lastError_;
        }
        return hresult_t::SUCCESS;
    }
}