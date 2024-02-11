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
        OpenMode::NON_BLOCKING | OpenMode::SYNC |
#endif
        OpenMode::READ_ONLY | OpenMode::WRITE_ONLY | OpenMode::READ_WRITE | OpenMode::TRUNCATE |
        OpenMode::NEW_ONLY | OpenMode::EXISTING_ONLY | OpenMode::APPEND),
    filename_{filename}
    {
    }

    FileDevice::~FileDevice()
    {
        if (isOpen())
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
    }

    void FileDevice::doOpen(OpenMode mode)
    {
        int32_t openFlags = 0;
        if (mode & OpenMode::READ_ONLY)
        {
            openFlags |= O_RDONLY;
        }
        if (mode & OpenMode::WRITE_ONLY)
        {
            openFlags |= O_WRONLY;
            openFlags |= O_CREAT;
        }
        if (mode & OpenMode::READ_WRITE)
        {
            openFlags |= O_RDWR;
        }
        if (mode & OpenMode::TRUNCATE)
        {
            openFlags |= O_TRUNC;
        }
        if (mode & OpenMode::NEW_ONLY)
        {
            openFlags |= O_EXCL;
        }
        if (mode & OpenMode::EXISTING_ONLY)
        {
            openFlags &= ~O_CREAT;
        }
        if (mode & OpenMode::APPEND)
        {
            openFlags |= O_APPEND;
        }
#ifndef _WIN32
        if (mode & OpenMode::NON_BLOCKING)
        {
            openFlags |= O_NONBLOCK;
        }
        if (mode & OpenMode::SYNC)
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
            THROW_ERROR(std::ios_base::failure,
                        "Impossible to open the file using the desired mode.");
        }
        fileDescriptor_ = rc;
    }

    void FileDevice::doClose()
    {
        const int32_t rc = ::close(fileDescriptor_);
        if (rc < 0)
        {
            THROW_ERROR(std::ios_base::failure, "Impossible to close the file.");
        }
        fileDescriptor_ = -1;
    }

    void FileDevice::seek(std::ptrdiff_t pos)
    {
        const ssize_t rc = ::lseek(fileDescriptor_, pos, SEEK_SET);
        if (rc < 0)
        {
            THROW_ERROR(std::ios_base::failure,
                        "File not open, or requested position '",
                        pos,
                        "' is out of scope.");
        }
    }

    std::ptrdiff_t FileDevice::pos()
    {
        const ssize_t pos_cur = ::lseek(fileDescriptor_, 0, SEEK_CUR);
        if (pos_cur < 0)
        {
            THROW_ERROR(std::ios_base::failure,
                        "File not open, or position would be negative or beyond the end.");
        }
        return pos_cur;
    }

    std::size_t FileDevice::size()
    {
        struct stat st;
        int32_t rc = ::fstat(fileDescriptor_, &st);
        if (rc < 0)
        {
            THROW_ERROR(std::ios_base::failure, "Impossible to access the file.");
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
            THROW_ERROR(std::ios_base::failure,
                        "File not open, or data buffer is outside accessible address space.");
        }
        return static_cast<std::ptrdiff_t>(readBytes);
    }

    std::ptrdiff_t FileDevice::writeData(const void * data, std::size_t dataSize)
    {
        const ssize_t writtenBytes = ::write(fileDescriptor_, data, dataSize);
        if (writtenBytes < 0)
        {
            THROW_ERROR(std::ios_base::failure,
                        "File not open, or data buffer is outside accessible address space.");
        }
        return writtenBytes;
    }

    const std::string & FileDevice::name() const
    {
        return filename_;
    }

    void FileDevice::resize(std::size_t size)
    {
        const int rc = ::ftruncate(fileDescriptor_, size);
        if (rc < 0)
        {
            THROW_ERROR(std::ios_base::failure, "File not open.");
        }
    }
}