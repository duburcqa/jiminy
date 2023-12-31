#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "jiminy/core/io/file_device.h"

#ifndef _WIN32
#    include <unistd.h>
#else
/* This is intended as a drop-in replacement for unistd.h on Windows.
 * https://stackoverflow.com/a/826027/1202830
 */
#    include <stdlib.h>
#    include <io.h>

/* The specific versions for Windows read, write, open and close being #defined here only handle
 * files. */
#    define lseek _lseek
#    define open _open
#    define close _close
#    define write _write
#    define read _read
#    define ftruncate _chsize

#    define S_IRUSR _S_IREAD
#    define S_IWUSR _S_IWRITE

#    ifdef _WIN64
#        define ssize_t __int64
#    else
#        define ssize_t long
#    endif
#endif


namespace jiminy
{
    FileDevice::FileDevice(const std::string & filename) noexcept :
    AbstractIODevice(
#ifndef _WIN32
        openMode_t::NON_BLOCKING | openMode_t::SYNC |
#endif
        openMode_t::READ_ONLY | openMode_t::WRITE_ONLY | openMode_t::READ_WRITE |
        openMode_t::TRUNCATE | openMode_t::NEW_ONLY | openMode_t::EXISTING_ONLY |
        openMode_t::APPEND),
    filename_{filename}
    {
    }

    FileDevice::~FileDevice()
    {
#if defined(close)
#    pragma push_macro("close")
#    undef close
#endif
        close();
#if defined(close)
#    pragma pop_macro("close")
#endif
    }

    hresult_t FileDevice::doOpen(openMode_t mode)
    {
        int32_t openFlags = 0;
        if (mode & openMode_t::READ_ONLY)
        {
            openFlags |= O_RDONLY;
        }
        if (mode & openMode_t::WRITE_ONLY)
        {
            openFlags |= O_WRONLY;
            openFlags |= O_CREAT;
        }
        if (mode & openMode_t::READ_WRITE)
        {
            openFlags |= O_RDWR;
        }
        if (mode & openMode_t::TRUNCATE)
        {
            openFlags |= O_TRUNC;
        }
        if (mode & openMode_t::NEW_ONLY)
        {
            openFlags |= O_EXCL;
        }
        if (mode & openMode_t::EXISTING_ONLY)
        {
            openFlags &= ~O_CREAT;
        }
        if (mode & openMode_t::APPEND)
        {
            openFlags |= O_APPEND;
        }
#ifndef _WIN32
        if (mode & openMode_t::NON_BLOCKING)
        {
            openFlags |= O_NONBLOCK;
        }
        if (mode & openMode_t::SYNC)
        {
            openFlags |= O_SYNC;
        }
#endif
#ifdef _WIN32
        openFlags |= _O_BINARY;
#endif

        const int32_t rc = ::open(filename_.c_str(), openFlags, S_IRUSR | S_IWUSR);
        if (rc < 0)
        {
            lastError_ = hresult_t::ERROR_GENERIC;
            PRINT_ERROR("Impossible to open the file using the desired mode.");
            return lastError_;
        }

        fileDescriptor_ = rc;

        return hresult_t::SUCCESS;
    }

    hresult_t FileDevice::doClose()
    {
        const int32_t rc = ::close(fileDescriptor_);
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

    hresult_t FileDevice::seek(std::ptrdiff_t pos)
    {
        const ssize_t rc = ::lseek(fileDescriptor_, pos, SEEK_SET);
        if (rc < 0)
        {
            lastError_ = hresult_t::ERROR_GENERIC;
            PRINT_ERROR(
                "The file is not open, or the requested position '", pos, "' is out of scope.");
            return lastError_;
        }
        return hresult_t::SUCCESS;
    }

    std::ptrdiff_t FileDevice::pos()
    {
        const ssize_t pos_cur = ::lseek(fileDescriptor_, 0, SEEK_CUR);
        if (pos_cur < 0)
        {
            lastError_ = hresult_t::ERROR_GENERIC;
            PRINT_ERROR(
                "The file is not open, or the position would be negative or beyond the end.");
        }
        return pos_cur;
    }

    std::size_t FileDevice::size()
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

    std::size_t FileDevice::bytesAvailable()
    {
        if (!isReadable())
        {
            return 0;
        }
        return size() - static_cast<std::size_t>(pos());
    }

    std::ptrdiff_t FileDevice::readData(void * data, std::size_t dataSize)
    {
        const ssize_t readBytes = ::read(fileDescriptor_, data, dataSize);
        if (readBytes < 0)
        {
            lastError_ = hresult_t::ERROR_GENERIC;
            PRINT_ERROR(
                "The file is not open, or data buffer is outside accessible address space.");
        }
        return static_cast<std::ptrdiff_t>(readBytes);
    }

    std::ptrdiff_t FileDevice::writeData(const void * data, std::size_t dataSize)
    {
        const ssize_t writtenBytes = ::write(fileDescriptor_, data, dataSize);
        if (writtenBytes < 0)
        {
            lastError_ = hresult_t::ERROR_GENERIC;
            PRINT_ERROR(
                "The file is not open, or data buffer is outside accessible address space.");
        }
        return writtenBytes;
    }

    const std::string & FileDevice::name() const
    {
        return filename_;
    }

    hresult_t FileDevice::resize(std::size_t size)
    {
        const int rc = ::ftruncate(fileDescriptor_, size);
        if (rc < 0)
        {
            lastError_ = hresult_t::ERROR_GENERIC;
            PRINT_ERROR("The file is not open.");
            return lastError_;
        }
        return hresult_t::SUCCESS;
    }
}