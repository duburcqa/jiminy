///////////////////////////////////////////////////////////////////////////////
///
/// \brief Contains templated function implementation of the AbstractIODevice class.
///
///////////////////////////////////////////////////////////////////////////////

#ifndef JIMINY_ABSTRACT_IO_DEVICE_TPP
#define JIMINY_ABSTRACT_IO_DEVICE_TPP

#include <string>
#include <vector>


namespace jiminy
{
    // Generic implementation - POD types
    template<typename T>
    hresult_t AbstractIODevice::write(T const & valueIn)
    {
        return write(reinterpret_cast<uint8_t const * const>(&valueIn), sizeof(T));
    }

    template<typename T>
    hresult_t AbstractIODevice::read(T & valueIn)
    {
        int64_t toRead = sizeof(T);
        uint8_t * bufferPos = reinterpret_cast<uint8_t *>(&valueIn);
        return read(bufferPos, toRead);
    }

    //---------------------------------------- Specific implementations ----------------------------------------//
    // read
    template<> hresult_t AbstractIODevice::read<std::vector<uint8_t> >(std::vector<uint8_t> & v);
    template<> hresult_t AbstractIODevice::read<std::vector<char_t> >(std::vector<char_t> & v);
    template<> hresult_t AbstractIODevice::read<std::string>(std::string & str) = delete;

    // write
    template<> hresult_t AbstractIODevice::write<std::string>(std::string const & str);
    template<> hresult_t AbstractIODevice::write<std::vector<uint8_t> >(std::vector<uint8_t> const & v);
    template<> hresult_t AbstractIODevice::write<std::vector<char_t> >(std::vector<char_t> const & v);
    template<> hresult_t AbstractIODevice::write<std::vector<uint64_t> >(std::vector<uint64_t> const & v);
}

#endif // JIMINY_ABSTRACT_IO_DEVICE_TPP
