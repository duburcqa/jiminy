///////////////////////////////////////////////////////////////////////////////
///
/// \brief   Manage the data structures of the telemetry.
///
///////////////////////////////////////////////////////////////////////////////

#ifndef EXO_SIMU_TELEMETRY_DATA_H
#define EXO_SIMU_TELEMETRY_DATA_H

#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "jiminy/core/TelemetrySender.h"
#include "jiminy/core/Types.h"


namespace jiminy
{
    int32_t     const TELEMETRY_VERSION = 1;             ///< Version of the telemetry format.
    std::string const NUM_INTS("NumIntEntries=");        ///< Number of integers in the data section.
    std::string const NUM_FLOATS("NumFloatEntries=");    ///< Number of floats in the data section.
    std::string const GLOBAL_TIME("Global.Time");        ///< Special column
    std::string const START_CONSTANTS("StartConstants"); ///< Marker of the beginning the constants section.
    std::string const START_COLUMNS("StartColumns");     ///< Marker of the beginning the columns section.
    std::string const START_LINE_TOKEN("StartLine");     ///< Marker of the beginning of a line of data.
    std::string const START_DATA("StartData");           ///< Marker of the beginning of the data section.
    std::string const START_INPUT_FRAME("INPUT_START");  ///< Marker of the beginning of a input ethercat frame.
    std::string const START_OUTPUT_FRAME("OUTPUT_START");///< Marker of the beginning of a output ethercat frame.

    std::size_t const CONSTANTS_MEM_SIZE = 16U * 1024U;
    std::size_t const INTEGERS_MEM_SIZE  = 32U * 1024U;
    std::size_t const FLOATS_MEM_SIZE    = 42U * 1024U;

    struct memHeader
    {
        int64_t startNameSection;       ///< Starting position of the naming section (in bytes).
        int64_t nextFreeNameOffset;     ///< Position of the next available position to record a name (in bytes).

        int64_t startDataSection;       ///< Starting position of the data section (in bytes).
        int64_t nextFreeDataOffset;     ///< Position of the next available position to record a data (in bytes).

        bool_t isRegisteringAvailable;  ///< True if registering is available, false otherwise.
    };

    class MemoryBuffer
    {
    public:
        // Disable the copy of the class
        MemoryBuffer(MemoryBuffer const &) = delete;
        MemoryBuffer & operator=(MemoryBuffer const &) = delete;

    public:
        ///////////////////////////////////////////////////////////////////////
        /// \brief       Constructor.
        ///
        /// \param  name  Name of the shared memory.
        /// \param  size  Size of the shared memory in bytes.
        ///////////////////////////////////////////////////////////////////////
        MemoryBuffer(std::string const & name,
                     std::size_t         size) :
        name_(name),
        size_(size),
        memAddress_(nullptr)
        {
            // Empty on purpose.
        };

        ///////////////////////////////////////////////////////////////////////
        /// \brief       Destructor
        ///////////////////////////////////////////////////////////////////////
        ~MemoryBuffer(void)
        {
            free(memAddress_);
        };

        ///////////////////////////////////////////////////////////////////////
        /// \brief       Create the shm context.
        /// \details     Create the shm if required, open it, map it and set
        /// the rights to use it.
        ///
        /// \retval definition::S_OK if successful.
        /// \retval the corresponding errno otherwise.
        ///////////////////////////////////////////////////////////////////////
        result_t create(void)
        {
            if (memAddress_ != nullptr)
            {
                // The shared memory has already been created
                return result_t::SUCCESS;
            }

            // Create the shared memory, set its mode to R/W and get the R/W access rights.
            memAddress_ = malloc(size_);

            if (memAddress_ == nullptr)
            {
                std::cout << "Error - MemoryBuffer::create - Memory allocation for the shared memory '" << name_ << "' failed." << std::endl;
                return result_t::ERROR_GENERIC;
            }

            return result_t::SUCCESS;
        };

        ///////////////////////////////////////////////////////////////////////
        /// \brief       Getter on the mapped shm.
        ///
        /// \return The address of the shm in this process.
        ///////////////////////////////////////////////////////////////////////
        void* address(void)
        {
            return memAddress_;
        };

        std::string name_;  ///< Name of the shared memory.
        std::size_t size_;  ///< Size in bytes of the shared memory.
        void* memAddress_;  ///< Address of the shared memory in this processus.
    };

    ////////////////////////////////////////////////////////////////////////
    /// \class TelemetryData
    /// \brief Manage the telemetry buffers.
    ////////////////////////////////////////////////////////////////////////
    class TelemetryData
    {
    public:
        // Disable the copy of the class
        TelemetryData(TelemetryData const &) = delete;
        TelemetryData & operator=(TelemetryData const &) = delete;

    public:
        ////////////////////////////////////////////////////////////////////////
        /// \brief Constructor.
        ////////////////////////////////////////////////////////////////////////
        TelemetryData(void);

        ////////////////////////////////////////////////////////////////////////
        /// \brief Destructor.
        ////////////////////////////////////////////////////////////////////////
        ~TelemetryData(void);

        ////////////////////////////////////////////////////////////////////////
        /// \brief Reset the telemetry before starting to use the telemetry.
        ////////////////////////////////////////////////////////////////////////
        void reset(void);

        ////////////////////////////////////////////////////////////////////////
        /// \brief Register a new variable in for telemetry.
        /// \warning The only supported types are int32_t and float32_t.
        ///
        /// \param[in]  variableNameIn       Name of the variable to register.
        /// \param[out] positionInBufferOut  Pointer on the allocated buffer that will hold the variable.
        ///
        /// \return S_OK if successful, the corresponding telemetry error otherwise.
        ////////////////////////////////////////////////////////////////////////
        template <typename T>
        result_t registerVariable(std::string const   & variableNameIn,
                                  T                 * & positionInBufferOut);

        ////////////////////////////////////////////////////////////////////////
        /// \brief Register a constant for the telemetry.
        ///
        /// \param[in] invariantNameIn  Name of the invariant.
        /// \param[in] valueIn          Value of the invariant.
        ///
        /// \return S_OK if successful, the corresponding telemetry error otherwise.
        ////////////////////////////////////////////////////////////////////////
        result_t registerConstant(std::string const & invariantNameIn,
                                  std::string const & valueIn);

        ////////////////////////////////////////////////////////////////////////
        /// \brief Format the telemetry header with the current recorded informations.
        /// \warning Calling this method will disable further registrations.
        ///
        /// \param[out] header  header to populate.
        ////////////////////////////////////////////////////////////////////////
        void formatHeader(std::vector<char_t> & header) const;

        ////////////////////////////////////////////////////////////////////////
        /// \brief Get data information to use them.
        ///
        /// \param[out] intAddrOut    Pointer on the int data array.
        /// \param[out] intSize       Size of the int data array.
        /// \param[out] floatAddrOut  Pointer on the float data array.
        /// \param[out] floatSize     Size of the float data array.
        ////////////////////////////////////////////////////////////////////////
        void getData(char_t  const * & intAddrOut,
                     int64_t         & intSize,
                     char_t  const * & floatAddrOut,
                     int64_t         & floatSize) const;

    private:
        ////////////////////////////////////////////////////////////////////////
        /// \brief Register a new variable in for telemetry.
        ///
        /// \param[in]  header               Shared memory header where the variable shall be recorded to.
        /// \param[in]  variableNameIn       Name of the variable to register.
        /// \param[out] positionInBufferOut  Pointer on the allocated buffer that will hold the variable.
        ///
        /// \return S_OK if successful, the corresponding telemetry error otherwise.
        ////////////////////////////////////////////////////////////////////////
        template <typename T>
        result_t internalRegisterVariable(struct memHeader       *   header,
                                          std::string      const   & variableNameIn,
                                          T                      * & positionInBufferOut);

        ////////////////////////////////////////////////////////////////////////
        /// \brief Search for an already registered entry into the shared memory.
        ///
        /// \param header   Pointer to the shared memory header where to search for the entry.
        /// \param name     Name for the entry to search for.
        ///
        /// \return -1 is the entry was not found, the position of the entry otherwise
        ////////////////////////////////////////////////////////////////////////
        int32_t findEntry(struct memHeader       * header,
                          std::string      const & name);

        MemoryBuffer constantsMem_;                ///< Shared memory to handle constants.
        struct memHeader * constantsHeader_;   ///< Header of the constants shm.

        MemoryBuffer integersMem_;                 ///< Shared memory to handle integers variables.
        struct memHeader * integersHeader_;    ///< Header of the integers shm.

        MemoryBuffer floatsMem_;                   ///< Shared memory to handle floats variables.
        struct memHeader * floatsHeader_;      ///< Header of the floats shm.

        /// Local cache to avoid looking into the shms (avoid search).
        std::unordered_map<std::string, void*> entriesMap_;
    };


    template <typename T>
    result_t TelemetryData::internalRegisterVariable(struct memHeader       *   header,
                                                     std::string      const   & variableName,
                                                     T                      * & positionInBufferOut)
    {
        char_t * const memAddress = reinterpret_cast<char_t*>(header);

        // Check in local cache before.
        auto entry = entriesMap_.find(variableName);
        if (entry != entriesMap_.end())
        {
            positionInBufferOut = static_cast<T*>(entry->second);
            return result_t::SUCCESS;
        }

        // Check in shared memory.
        int32_t positionInBuffer = findEntry(header, variableName);
        if (positionInBuffer != -1)
        {
            char_t * address = memAddress + header->startDataSection + sizeof(T) * static_cast<uint32_t>(positionInBuffer);
            entriesMap_[variableName] = static_cast<void*>(address);
            positionInBufferOut = static_cast<T*>(entriesMap_[variableName]);
            return result_t::SUCCESS;
        }

        if (not header->isRegisteringAvailable)
        {
            std::cout << "Error - TelemetryData::updateValue - Entry not found: register it if possible." << std::endl;
            return result_t::ERROR_GENERIC;
        }

        if ((header->nextFreeNameOffset + static_cast<int64_t>(variableName.size()) + 1) >= header->startDataSection)
        {
            std::cout << "Error - TelemetryData::updateValue - TODO" << std::endl; //TODO: write appropriate error message
            return result_t::ERROR_GENERIC;
        }

        char_t * const namePos = memAddress + header->nextFreeNameOffset; // Compute record address
        memcpy(namePos, variableName.data(), variableName.size());
        header->nextFreeNameOffset += variableName.size();
        header->nextFreeNameOffset += 1U; // Null-terminated.


        char_t * const dataPos = memAddress + header->nextFreeDataOffset;
        entriesMap_[variableName] = static_cast<void*>(dataPos);
        positionInBufferOut = static_cast<T*>(entriesMap_[variableName]);
        header->nextFreeDataOffset += sizeof(T);

        return result_t::SUCCESS;
    }
} // namespace jiminy

#endif // EXO_SIMU_TELEMETRY_DATA_H