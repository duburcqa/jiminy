///////////////////////////////////////////////////////////////////////////////
///
/// \brief TelemetryData Implementation.
///
//////////////////////////////////////////////////////////////////////////////

#include <iostream>

#include "jiminy/core/TelemetryData.h"


namespace jiminy
{
    TelemetryData::TelemetryData() :
    constantsMem_("telemetryConstants", CONSTANTS_MEM_SIZE),
    constantsHeader_(),
    integersMem_("telemetryIntegers", INTEGERS_MEM_SIZE),
    integersHeader_(),
    floatsMem_("telemetryFloats", FLOATS_MEM_SIZE),
    floatsHeader_(),
    entriesMap_()
    {
        constantsMem_.create();
        constantsHeader_ = static_cast<struct memHeader *>(constantsMem_.address());

        integersMem_.create();
        integersHeader_ = static_cast<struct memHeader *>(integersMem_.address());

        floatsMem_.create();
        floatsHeader_ = static_cast<struct memHeader *>(floatsMem_.address());
    }

    TelemetryData::~TelemetryData()
    {
        // empty on purpose
    }

    void TelemetryData::reset()
    {
        entriesMap_.clear();

        std::memset(constantsMem_.address(), 0, CONSTANTS_MEM_SIZE);
        std::memset(integersMem_.address(),  0, INTEGERS_MEM_SIZE);
        std::memset(floatsMem_.address(),    0, FLOATS_MEM_SIZE);

        constantsHeader_->startNameSection = sizeof(struct memHeader);
        constantsHeader_->nextFreeNameOffset = sizeof(struct memHeader);
        constantsHeader_->startDataSection = CONSTANTS_MEM_SIZE; // Set to the end, because it make no sense for constants to have a data section.
        constantsHeader_->nextFreeDataOffset = CONSTANTS_MEM_SIZE;
        constantsHeader_->isRegisteringAvailable = true;

        integersHeader_->startNameSection = sizeof(struct memHeader);
        integersHeader_->nextFreeNameOffset = sizeof(struct memHeader);
        integersHeader_->startDataSection = INTEGERS_MEM_SIZE / 2U;
        integersHeader_->nextFreeDataOffset = INTEGERS_MEM_SIZE / 2U;
        integersHeader_->isRegisteringAvailable = true;

        floatsHeader_->startNameSection = sizeof(struct memHeader);
        floatsHeader_->nextFreeNameOffset = sizeof(struct memHeader);
        floatsHeader_->startDataSection = FLOATS_MEM_SIZE / 2U;
        floatsHeader_->nextFreeDataOffset = FLOATS_MEM_SIZE / 2U;
        floatsHeader_->isRegisteringAvailable = true;
    }

    template<>
    result_t TelemetryData::registerVariable<int32_t>(std::string const   & variableName,
                                                      int32_t           * & positionInBufferOut)
    {
        return internalRegisterVariable(integersHeader_, variableName, positionInBufferOut);
    }

    template<>
    result_t TelemetryData::registerVariable<float32_t>(std::string const   & variableName,
                                                        float32_t         * & positionInBufferOut)
    {
        return internalRegisterVariable(floatsHeader_, variableName, positionInBufferOut);
    }

    result_t TelemetryData::registerConstant(std::string const & variableNameIn,
                                             std::string const & constantValueIn)
    {
        // Targeted shared memory.
        struct memHeader * const header = constantsHeader_;
        char_t * const memAddress = reinterpret_cast<char_t *>(header);

        if (!header->isRegisteringAvailable)
        {
            std::cout << "result_t - TelemetryData::registerConstant - Registration is locked." << std::endl;
            return result_t::ERROR_GENERIC;
        }

        std::string const fullConstant = variableNameIn + "=" + constantValueIn;
        if ((header->nextFreeNameOffset + static_cast<int64_t>(fullConstant.size()) + 1) >= header->startDataSection)
        {
            std::cout << "result_t - TelemetryData::registerConstant - Maximum number of registration exceeded." << std::endl;
            return result_t::ERROR_GENERIC;
        }

        if (findEntry(header, fullConstant) != -1)
        {
            std::cout << "result_t - TelemetryData::registerConstant - A constant with this name was already registered." << std::endl;
            return result_t::ERROR_GENERIC;
        }

        char_t * const namePos = memAddress + header->nextFreeNameOffset; // Compute record address
        memcpy(namePos, fullConstant.data(), fullConstant.size());
        header->nextFreeNameOffset += fullConstant.size();
        header->nextFreeNameOffset += 1U; // Null-terminated.

        return result_t::SUCCESS;;
    }

    int32_t TelemetryData::findEntry(struct memHeader        * header,
                                     std::string       const & name)
    {
        char_t * const memAddress = reinterpret_cast<char_t *>(header);

        int32_t position = 0;
        int64_t currentPos = header->startNameSection;
        while (currentPos < header->nextFreeNameOffset)
        {
            std::string const entry(memAddress + currentPos);
            if (entry == name)
            {
                return position;
            }

            currentPos += entry.size();
            currentPos += 1U; // Null-terminated

            position++;
        }

        return -1;
    }

    void TelemetryData::formatHeader(std::vector<char_t> & header) const
    {
        // Lock registering.
        constantsHeader_->isRegisteringAvailable = false;
        integersHeader_->isRegisteringAvailable = false;
        floatsHeader_->isRegisteringAvailable = false;

        header.clear();
        header.reserve(64 * 1024);

        // Record format version.
        header.resize(sizeof(int32_t)); // version.
        header[0] = ((TELEMETRY_VERSION & 0x000000ff) >> 0);
        header[1] = ((TELEMETRY_VERSION & 0x0000ff00) >> 8);
        header[2] = ((TELEMETRY_VERSION & 0x00ff0000) >> 16);
        header[3] = ((TELEMETRY_VERSION & 0xff000000) >> 24);

        // Record constants.
        header.insert(header.end(), START_CONSTANTS.data(), START_CONSTANTS.data() + START_CONSTANTS.size());
        header.push_back('\0');
        char_t const* startConstants = reinterpret_cast<char_t*>(constantsHeader_) + constantsHeader_->startNameSection;
        char_t const* stopConstants = reinterpret_cast<char_t*>(constantsHeader_) + constantsHeader_->nextFreeNameOffset;
        header.insert(header.end(), startConstants, stopConstants);

        // Record entries numbers.
        std::string entriesNumbers;
        entriesNumbers += NUM_INTS;
        entriesNumbers += std::to_string((integersHeader_->nextFreeDataOffset - integersHeader_->startDataSection) /
                                         static_cast<int64_t>(sizeof(int32_t)) + 1); // +1 because we add Global.Time
        entriesNumbers += '\0';
        entriesNumbers += NUM_FLOATS;
        entriesNumbers += std::to_string((floatsHeader_->nextFreeDataOffset - floatsHeader_->startDataSection) /
                                         static_cast<int64_t>(sizeof(float32_t)));
        entriesNumbers += '\0';
        header.insert(header.end(), entriesNumbers.data(), entriesNumbers.data() + entriesNumbers.size());

        // Record header - GLobal.Time - integers, floats.
        header.insert(header.end(), START_COLUMNS.data(), START_COLUMNS.data() + START_COLUMNS.size());
        header.push_back('\0');

        header.insert(header.end(), GLOBAL_TIME.data(), GLOBAL_TIME.data() + GLOBAL_TIME.size());
        header.push_back('\0');

        char_t const* startIntegersHeader = reinterpret_cast<char_t*>(integersHeader_) + integersHeader_->startNameSection;
        char_t const* stopIntegersHeader  = reinterpret_cast<char_t*>(integersHeader_) + integersHeader_->nextFreeNameOffset;
        header.insert(header.end(), startIntegersHeader, stopIntegersHeader);

        char_t const* startFloatsHeader = reinterpret_cast<char_t*>(floatsHeader_) + floatsHeader_->startNameSection;
        char_t const* stopFloatsHeader  = reinterpret_cast<char_t*>(floatsHeader_) + floatsHeader_->nextFreeNameOffset;
        header.insert(header.end(), startFloatsHeader, stopFloatsHeader);

        // Start data section.
        header.insert(header.end(), START_DATA.data(), START_DATA.data() + START_DATA.size());
    }

    void TelemetryData::getData(char_t  const * & intAddrOut,
                                int64_t         & intSizeOut,
                                char_t  const * & floatAddrOut,
                                int64_t         & floatSizeOut) const
    {
        char_t * const integermemAddress = reinterpret_cast<char_t *>(integersHeader_);
        intAddrOut = integermemAddress + integersHeader_->startDataSection;
        intSizeOut = integersHeader_->nextFreeDataOffset - integersHeader_->startDataSection;

        char_t * const floatmemAddress = reinterpret_cast<char_t *>(floatsHeader_);
        floatAddrOut = floatmemAddress + floatsHeader_->startDataSection;
        floatSizeOut = floatsHeader_->nextFreeDataOffset - floatsHeader_->startDataSection;
    }
}// end of namespace jiminy