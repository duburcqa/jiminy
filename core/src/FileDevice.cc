///////////////////////////////////////////////////////////////////////////////
///
/// \brief Contains the FileDevice class methods implementations.
///
///////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include "jiminy/core/FileDevice.h"


namespace jiminy
{
    FileDevice::FileDevice(std::string const& filename) :
    filename_(filename),
    fileDescriptor_(-1)
    {
        supportedModes_ = OpenMode::READ_ONLY     | OpenMode::WRITE_ONLY | OpenMode::READ_WRITE |
                          OpenMode::NON_BLOCKING  | OpenMode::TRUNCATE   | OpenMode::NEW_ONLY   |
                          OpenMode::EXISTING_ONLY | OpenMode::APPEND     | OpenMode::SYNC;
    }

    FileDevice::~FileDevice(void)
    {
        close();
    }

    result_t FileDevice::doOpen(enum OpenMode mode)
    {
        int32_t posixFLags = 0;
        if (mode & OpenMode::READ_ONLY)
        {
            posixFLags |= O_RDONLY;
        }
        if (mode & OpenMode::WRITE_ONLY)
        {
            posixFLags |= O_WRONLY;
            posixFLags |= O_CREAT;
        }
        if (mode & OpenMode::READ_WRITE)
        {
            posixFLags |= O_RDWR;
        }
        if (mode & OpenMode::NON_BLOCKING)
        {
            posixFLags |= O_NONBLOCK;
        }
        if (mode & OpenMode::TRUNCATE)
        {
            posixFLags |= O_TRUNC;
        }
        if (mode & OpenMode::NEW_ONLY)
        {
            posixFLags |= O_EXCL;
        }
        if (mode & OpenMode::EXISTING_ONLY)
        {
            posixFLags &= ~O_CREAT;
        }
        if (mode & OpenMode::APPEND)
        {
            posixFLags |= O_APPEND;
        }
        if (mode & OpenMode::SYNC)
        {
            posixFLags |= O_SYNC;
        }

        int32_t const rc = ::open(filename_.c_str(), posixFLags, S_IRUSR | S_IWUSR);
        if (rc < 0)
        {
            lastError_ = result_t::ERROR_GENERIC;
            std::cout << "Error - MemoryDevice::doOpen - Impossible to open the file using the desired mode." << std::endl;
            return lastError_;
        }

        fileDescriptor_ = rc;

        return result_t::SUCCESS;
    }

    void FileDevice::doClose(void)
    {
        int32_t const rc = ::close(fileDescriptor_);
        if (rc < 0)
        {
            lastError_ = result_t::ERROR_GENERIC;
            std::cout << "Error - MemoryDevice::doClose - Impossible to close the file." << std::endl;
        }
        else
        {
            fileDescriptor_ = -1;
        }
    }

    result_t FileDevice::seek(int64_t pos)
    {
        ssize_t const rc = ::lseek(fileDescriptor_, pos, SEEK_SET);
        if (rc < 0)
        {
            lastError_ = result_t::ERROR_GENERIC;
            std::cout << "Error - MemoryDevice::seek - The file is not open, or the requested position '" << pos << "' is out of scope." << std::endl;
            return lastError_;
        }

        return result_t::SUCCESS;
    }

    int64_t FileDevice::pos(void)
    {
        ssize_t const rc = ::lseek(fileDescriptor_, 0, SEEK_CUR);
        if (rc < 0)
        {
            lastError_ = result_t::ERROR_GENERIC;
            std::cout << "Error - MemoryDevice::pos - The file is not open, or the position would be negative or beyond the end." << std::endl;
        }

        return rc;
    }

    int64_t FileDevice::size(void)
    {
        struct stat st;
        int32_t rc = ::fstat(fileDescriptor_, &st);
        if (rc < 0)
        {
            lastError_ = result_t::ERROR_GENERIC;
            std::cout << "Error - MemoryDevice::size - Impossible to access the file." << std::endl;
            return rc;
        }

        return st.st_size;
    }

    int64_t FileDevice::bytesAvailable(void)
    {
        if (not isReadable())
        {
            return 0;
        }

        return size() - pos();
    }

    int64_t FileDevice::readData(void* data, int64_t dataSize)
    {
        ssize_t const rc = ::read(fileDescriptor_, data, static_cast<size_t>(dataSize));
        if (rc < 0)
        {
            lastError_ = result_t::ERROR_GENERIC;
            std::cout << "Error - MemoryDevice::readData - The file is not open, or data buffer is outside accessible address space." << std::endl;
        }

        return rc;
    }

    int64_t FileDevice::writeData(void const* data, int64_t dataSize)
    {
        ssize_t const rc = ::write(fileDescriptor_, data, static_cast<size_t>(dataSize));
        if (rc < 0)
        {
            lastError_ = result_t::ERROR_GENERIC;
            std::cout << "Error - MemoryDevice::writeData - The file is not open, or data buffer is outside accessible address space." << std::endl;
        }

        return rc;
    }

    std::string const& FileDevice::name(void) const
    {
        return filename_;
    }
}