///////////////////////////////////////////////////////////////////////////////
///
/// \brief   Manage the data structures of the telemetry.
///
///////////////////////////////////////////////////////////////////////////////

#ifndef JIMINY_TELEMETRY_DATA_TPP
#define JIMINY_TELEMETRY_DATA_TPP

#include <iostream>
#include <string>


namespace jiminy
{
    template <typename T>
    hresult_t TelemetryData::internalRegisterVariable(struct memHeader       *   header,
                                                      std::string      const   & variableName,
                                                      T                      * & positionInBufferOut)
    {
        char_t * const memAddress = reinterpret_cast<char_t *>(header);

        // Check in local cache
        auto entry = entriesMap_.find(variableName);
        if (entry != entriesMap_.end())
        {
            positionInBufferOut = static_cast<T *>(entry->second);
            return hresult_t::SUCCESS;
        }

        // Check in memory
        int32_t positionInBuffer = findEntry(header, variableName);
        if (positionInBuffer != -1)
        {
            char_t * address = memAddress + header->startDataSection + sizeof(T) * static_cast<uint32_t>(positionInBuffer);
            entriesMap_[variableName] = static_cast<void *>(address);
            positionInBufferOut = static_cast<T *>(entriesMap_[variableName]);
            return hresult_t::SUCCESS;
        }

        if (!header->isRegisteringAvailable)
        {
            PRINT_ERROR("Entry not found: register it if possible.");
            return hresult_t::ERROR_GENERIC;
        }

        if ((header->nextFreeNameOffset + static_cast<int64_t>(variableName.size()) + 1) >= header->startDataSection)
        {
            PRINT_ERROR("Trying to allocate too much memory to hold telemetry constants and variables. "
                        "Try using shorter names or register less variables.");
            return hresult_t::ERROR_GENERIC;
        }

        char_t * const namePos = memAddress + header->nextFreeNameOffset;  // Compute record address
        memcpy(namePos, variableName.data(), variableName.size());
        header->nextFreeNameOffset += variableName.size();
        header->nextFreeNameOffset += 1U;  // Null-terminated.

        char_t * const dataPos = memAddress + header->nextFreeDataOffset;
        entriesMap_[variableName] = static_cast<void *>(dataPos);
        positionInBufferOut = static_cast<T *>(entriesMap_[variableName]);
        header->nextFreeDataOffset += sizeof(T);

        return hresult_t::SUCCESS;
    }
} // namespace jiminy

#endif // JIMINY_TELEMETRY_DATA_TPP