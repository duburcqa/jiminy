#ifndef JIMINY_ABSTRACT_IO_DEVICE_HXX
#define JIMINY_ABSTRACT_IO_DEVICE_HXX

#include <string>
#include <vector>


namespace jiminy
{
    // POD types
    template<typename T>
    std::enable_if_t<!is_contiguous_container_v<T> && std::is_trivially_copyable_v<T>, hresult_t>
    AbstractIODevice::write(const T & value)
    {
        std::size_t toWrite = sizeof(T);
        const void * const bufferPos = static_cast<const void *>(&value);
        return write(bufferPos, toWrite);
    }

    template<typename T>
    std::enable_if_t<!is_contiguous_container_v<remove_cvref_t<T>> &&
                         std::is_trivially_copyable_v<remove_cvref_t<T>>,
                     hresult_t>
    AbstractIODevice::read(T && value)
    {
        std::size_t toRead = sizeof(T);
        void * const bufferPos = static_cast<void *>(&value);
        return read(bufferPos, toRead);
    }

    // Contiguous container
    template<typename T>
    std::enable_if_t<is_contiguous_container_v<T>, hresult_t>
    AbstractIODevice::write(const T & value)
    {
        const std::size_t toWrite = value.size() * sizeof(typename T::value_type);
        const void * const bufferPos = static_cast<const void *>(value.data());
        return write(bufferPos, toWrite);
    }

    template<typename T>
    std::enable_if_t<is_contiguous_container_v<remove_cvref_t<T>>, hresult_t>
    AbstractIODevice::read(T && value)
    {
        const std::size_t toRead = value.size() * sizeof(typename remove_cvref_t<T>::value_type);
        void * const bufferPos = static_cast<void *>(value.data());
        return read(bufferPos, toRead);
    }
}

#endif  // JIMINY_ABSTRACT_IO_DEVICE_HXX
