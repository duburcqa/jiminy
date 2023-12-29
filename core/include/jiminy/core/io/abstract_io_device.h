#ifndef JIMINY_ABSTRACT_IO_DEVICE_H
#define JIMINY_ABSTRACT_IO_DEVICE_H

#include <memory>

#include "jiminy/core/fwd.h"


namespace jiminy
{
    /// \brief Possible modes for a device (their availability depend of the concrete device).
    ///
    /// \remarks Plain enum is used instead of enum class because otherwise conversion to bool is
    ///          undefined.
    enum openMode_t
    {
        /// \brief Device is not opened.
        NOT_OPEN = 0x000,
        /// \brief Read only mode.
        READ_ONLY = 0x001,
        /// \brief Write only mode.
        WRITE_ONLY = 0x002,
        /// \brief Read/Write mode.
        ///
        /// \warning (READ_ONLY | WRITE_ONLY) != READ_WRITE
        READ_WRITE = 0x004,
        /// \brief Open the device in append mode.
        APPEND = 0x008,
        /// \brief Truncate the device at opening.
        TRUNCATE = 0x010,
        /// \brief Do not use intermediate buffer if possible.
        UNBUFFERED = 0x020,
        /// \brief Create the device at opening, fail if the device already exists.
        NEW_ONLY = 0x040,
        /// \brief Do not create the device if it does not exists.
        EXISTING_ONLY = 0x080,
#ifndef _WIN32
        /// \brief Open the device in non blocking mode.
        NON_BLOCKING = 0x100,
        /// \brief Open the device in sync mode (ensure that write are finished at return).
        SYNC = 0x200,
#endif
    };

    // Facility operators to avoid cast.
    openMode_t operator|(openMode_t modeA, openMode_t modeB);
    openMode_t operator&(openMode_t modeA, openMode_t modeB);
    openMode_t operator|=(openMode_t & modeA, openMode_t modeB);
    openMode_t operator&=(openMode_t & modeA, openMode_t modeB);
    openMode_t operator~(openMode_t mode);

    /// \brief Base interface class to handle all possibles I/O that can act as a stream (file /
    ///        TCP socket / pipe and so on).
    class JIMINY_DLLAPI AbstractIODevice
    {
    public:
        explicit AbstractIODevice(openMode_t supportedModes) noexcept;
        explicit AbstractIODevice(AbstractIODevice &&) = default;
        virtual ~AbstractIODevice() = default;

        /// \brief Open the device.
        ///
        /// \param mode Mode to apply for opening the device.
        ///
        /// \return hresult_t::SUCCESS if successful, another hresult_t value otherwise.
        hresult_t open(openMode_t mode);

        /// \brief Write data in the device.
        ///
        /// \details The default implementation manage only POD type. For specific type, the
        ///          template shall be extended with specific implementation.
        ///
        /// \param Value to write into the device.
        ///
        /// \return hresult_t::SUCCESS if successful, another hresult_t value otherwise.
        hresult_t close();

        /// \brief Current opening modes.
        openMode_t openModes() const;

        /// \brief Supported opening modes.
        openMode_t supportedModes() const;

        /// \brief Whether the device is writable.
        bool isWritable() const;

        /// \brief Whether the device is readable.
        bool isReadable() const;

        /// \brief Whether the device is opened.
        bool isOpen() const;

        /// \brief Whether the device is sequential (i.e socket), false if the device support
        ///        random-access (i.e regular file).
        virtual bool isSequential() const;

        /// \brief The size of the device.
        ///
        /// \details For random-access devices, this function returns the size of the device.
        ///          For sequential devices, bytesAvailable() is returned.
        virtual std::size_t size();

        /// \brief Move the current position cursor to pos if possible.
        ///
        /// \param pos Desired new position of the cursor.
        ///
        /// \return hresult_t::SUCCESS if successful, another hresult_t value otherwise.
        virtual hresult_t seek(std::ptrdiff_t pos);

        /// \brief The current cursor position (0 if there is not concept of position cursor).
        virtual std::ptrdiff_t pos();

        /// \brief Resize the device to provided size.
        virtual hresult_t resize(std::size_t size);

        /// \brief Returns the number of bytes that are available for reading. Commonly used with
        ///        sequential device.
        virtual std::size_t bytesAvailable();

        /// \brief Write data in the device.
        ///
        /// \details The default implementation manage only POD type. For specific type, the
        ///          template shall be extended with specific implementation.
        ///
        /// \param Value to write into the device.
        ///
        /// \return hresult_t::SUCCESS if successful, another hresult_t value otherwise.
        template<typename T>
        std::enable_if_t<!is_contiguous_container_v<T> && std::is_trivially_copyable_v<T>,
                         hresult_t>
        write(const T & value);

        template<typename T>
        std::enable_if_t<is_contiguous_container_v<T>, hresult_t> write(const T & value);

        template<typename T>
        std::enable_if_t<!is_contiguous_container_v<T> && !std::is_trivially_copyable_v<T>,
                         hresult_t>
        write(const T & value) = delete;

        /// \brief Read data in the device.
        ///
        /// \details The default implementation manage only POD type. For specific type, the
        ///          template shall be extended with specific implementation.
        ///
        /// \param Value to store read data.
        ///
        /// \return hresult_t::SUCCESS if successful, another hresult_t value otherwise.
        template<typename T>
        std::enable_if_t<!is_contiguous_container_v<remove_cvref_t<T>> &&
                             std::is_trivially_copyable_v<remove_cvref_t<T>>,
                         hresult_t>
        read(T && value);

        template<typename T>
        std::enable_if_t<is_contiguous_container_v<remove_cvref_t<T>>, hresult_t> read(T && value);

        template<typename T>
        std::enable_if_t<!is_contiguous_container_v<remove_cvref_t<T>> &&
                             !std::is_trivially_copyable_v<remove_cvref_t<T>>,
                         hresult_t>
        read(T && value) = delete;

        /// \brief Retrieve the latest error. Useful for calls that do not return an error code
        ///        directly.
        hresult_t getLastError() const;

    protected:
        virtual hresult_t doOpen(openMode_t mode) = 0;
        virtual hresult_t doClose() = 0;

        /// \brief Write data in the device.
        ///
        /// \param data Buffer of data to write.
        /// \param dataSize Number of bytes to write.
        ///
        /// \return the number of bytes written, -1 in case of error.
        virtual std::ptrdiff_t writeData(const void * data, std::size_t dataSize) = 0;

        /// \brief Write data in the device.
        ///
        /// \param data Buffer of data to write.
        /// \param dataSize Number of bytes to write.
        ///
        /// \return hresult_t::SUCCESS if successful, another hresult_t value otherwise.
        virtual hresult_t write(const void * data, std::size_t dataSize);

        /// \brief Read data from the device.
        ///
        /// \param data Buffer to store read data.
        /// \param dataSize Number of bytes to read.
        ///
        /// \return hresult_t::SUCCESS if successful, another hresult_t value otherwise.
        virtual hresult_t read(void * data, std::size_t dataSize);

        /// \brief Read data in the device.
        ///
        /// \param data Buffer of data to read.
        /// \param dataSize Number of bytes to read.
        ///
        /// \return the number of bytes read, -1 in case of error.
        virtual std::ptrdiff_t readData(void * data, std::size_t dataSize) = 0;

    protected:
        /// \brief Supported modes of the device.
        const openMode_t supportedModes_;
        /// \brief Current opening mode.
        openMode_t modes_{openMode_t::NOT_OPEN};
        /// \brief Latest generated error.
        hresult_t lastError_{hresult_t::SUCCESS};
    };
}

#include "jiminy/core/io/abstract_io_device.hxx"

#endif  // JIMINY_ABSTRACT_IO_DEVICE_H
