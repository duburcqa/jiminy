#ifndef JIMINY_ABSTRACT_IO_DEVICE_HXX
#define JIMINY_ABSTRACT_IO_DEVICE_HXX

#include <string>
#include <vector>


namespace jiminy
{
    // Generic implementation - POD types
    template<typename T>
    hresult_t AbstractIODevice::write(const T & valueIn)
    {
        return write(reinterpret_cast<const uint8_t * const>(&valueIn), sizeof(T));
    }

    template<typename T>
    hresult_t AbstractIODevice::read(T & valueIn)
    {
        int64_t toRead = sizeof(T);
        uint8_t * bufferPos = reinterpret_cast<uint8_t *>(&valueIn);
        return read(bufferPos, toRead);
    }

    //--------------------------------- Specific implementations --------------------------------//

    // read
    template<>
    hresult_t AbstractIODevice::read<std::vector<uint8_t>>(std::vector<uint8_t> & v);
    template<>
    hresult_t AbstractIODevice::read<std::vector<char>>(std::vector<char> & v);
    template<>
    hresult_t AbstractIODevice::read<std::string>(std::string & str) = delete;

    // write
    template<>
    hresult_t AbstractIODevice::write<std::string_view>(const std::string_view & str);
    template<>
    hresult_t AbstractIODevice::write<std::vector<uint8_t>>(const std::vector<uint8_t> & v);
    template<>
    hresult_t AbstractIODevice::write<std::vector<char>>(const std::vector<char> & v);
    template<>
    hresult_t AbstractIODevice::write<std::vector<uint64_t>>(const std::vector<uint64_t> & v);
}

#endif  // JIMINY_ABSTRACT_IO_DEVICE_HXX
