///////////////////////////////////////////////////////////////////////////////
///
/// \brief       Declaration of the TelemetryRecorder class, responsible of recording data
///              to files.
///
///////////////////////////////////////////////////////////////////////////////

#ifndef EXO_SIMU_TELEMETRY_RECORDER_H
#define EXO_SIMU_TELEMETRY_RECORDER_H

#include "jiminy/core/MemoryDevice.h"
#include "jiminy/core/TelemetryData.h"


namespace jiminy
{
    uint32_t const MAX_BUFFER_SIZE = (256U * 1024U); // 256Ko

    class TelemetryData;

    ////////////////////////////////////////////////////////////////////////
    /// \class TelemetryRecorder
    ////////////////////////////////////////////////////////////////////////
    class TelemetryRecorder
    {
        // Disable the copy of the class
        TelemetryRecorder(TelemetryRecorder const &) = delete;
        TelemetryRecorder & operator=(TelemetryRecorder const &) = delete;
    public:
        ////////////////////////////////////////////////////////////////////////
        /// \brief Constructor.
        ////////////////////////////////////////////////////////////////////////
        TelemetryRecorder(std::shared_ptr<TelemetryData const> const & telemetryDataInstance);

        ////////////////////////////////////////////////////////////////////////
        /// \brief Destructor.
        ////////////////////////////////////////////////////////////////////////
        ~TelemetryRecorder(void);

        ////////////////////////////////////////////////////////////////////////
        /// \brief Initialize the recorder.
        ////////////////////////////////////////////////////////////////////////
        result_t initialize(void);

        bool const & getIsInitialized(void);

        ////////////////////////////////////////////////////////////////////////
        /// \brief Reset the recorder.
        ////////////////////////////////////////////////////////////////////////
        void reset(void);

        ////////////////////////////////////////////////////////////////////////
        /// \brief Create a new line in the record with the current telemetry data.
        ////////////////////////////////////////////////////////////////////////
        result_t flushDataSnapshot(float64_t const & timestamp);

        ////////////////////////////////////////////////////////////////////////
        /// \brief Get access to the memory device holding the data
        ////////////////////////////////////////////////////////////////////////
        void writeDataBinary(std::string const & filename);
        void getData(std::vector<std::string>             & header,
                     std::vector<float32_t>               & timestamps,
                     std::vector<std::vector<int32_t> >   & intData,
                     std::vector<std::vector<float32_t> > & floatData);
    private:
        ////////////////////////////////////////////////////////////////////////
        /// \brief   Create a new file to continue the recording.
        /// \details Each chunk shall have a size defined by LARGE_LOG_SIZE_GB and shall
        ///          be suffixed by an increasing natural number.
        ///
        /// \return  SUCCESS if successful, the corresponding telemetry error otherwise.
        ////////////////////////////////////////////////////////////////////////
        result_t createNewChunk();

    private:
        ///////////////////////////////////////////////////////////////////////
        /// Private attributes
        ///////////////////////////////////////////////////////////////////////
        std::shared_ptr<TelemetryData const> telemetryData_;
        std::vector<MemoryDevice> flows_;

        bool isInitialized_;

        int64_t recordedBytesLimits_;
        int64_t recordedBytesDataLine_;
        int64_t recordedBytes_;             ///< Bytes recorded in the file.
        int64_t headerSize_;                ///< Size in byte of the header.

        char_t const * integersAddress_;    ///< Address of the integer data section.
        int64_t integerSectionSize_;        ///< Size in bytes of the integer data section.

        char_t const * floatsAddress_;      ///< Address of the float data section.
        int64_t floatSectionSize_;          ///< Size in byte of the float data section.
    };
}

#endif // EXO_SIMU_TELEMETRY_RECORDER_H