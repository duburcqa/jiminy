///////////////////////////////////////////////////////////////////////////////
///
/// \brief       Declaration of the TelemetryRecorder class, responsible of recording data
///              to files.
///
///////////////////////////////////////////////////////////////////////////////

#ifndef JIMINY_TELEMETRY_RECORDER_H
#define JIMINY_TELEMETRY_RECORDER_H

#include "jiminy/core/io/MemoryDevice.h"


namespace jiminy
{
    class TelemetryData;

    struct logData_t
    {
    public:
        logData_t(void) :
        header(),
        version(0),
        timeUnit(0.0),
        numInt(0),
        numFloat(0),
        timestamps(),
        intData(),
        floatData()
        {
            // Empty on purpose.
        }

    public:
        std::vector<std::string> header;
        int32_t version;
        float64_t timeUnit;
        uint32_t numInt;
        uint32_t numFloat;
        std::vector<int64_t> timestamps;
        std::vector<std::vector<int64_t> > intData;
        std::vector<std::vector<float64_t> > floatData;
    };

    ////////////////////////////////////////////////////////////////////////
    /// \class TelemetryRecorder
    ////////////////////////////////////////////////////////////////////////
    class TelemetryRecorder
    {
        // Disable the copy of the class
        TelemetryRecorder(TelemetryRecorder const &) = delete;
        TelemetryRecorder & operator=(TelemetryRecorder const &) = delete;
    public:
        TelemetryRecorder(void);
        ~TelemetryRecorder(void);

        ////////////////////////////////////////////////////////////////////////
        /// \brief Initialize the recorder.
        /// \param[in] telemetryData Data to log.
        /// \param[in] timeUnit Unit with which the time will be logged.
        ///                     Note that time is logged as an int.
        ////////////////////////////////////////////////////////////////////////
        hresult_t initialize(TelemetryData       * telemetryData,
                             float64_t     const & timeUnit);

        bool_t const & getIsInitialized(void);

        /// \brief Get the maximum time that can be logged with the current precision.
        /// \return Max time, in second.
        float64_t getMaximumLogTime(void) const;

        /// \brief Get the maximum time that can be logged with the given precision.
        /// \return Max time, in second.
        static float64_t getMaximumLogTime(float64_t const & timeUnit);

        ////////////////////////////////////////////////////////////////////////
        /// \brief Reset the recorder.
        ////////////////////////////////////////////////////////////////////////
        void reset(void);

        ////////////////////////////////////////////////////////////////////////
        /// \brief Create a new line in the record with the current telemetry data.
        ////////////////////////////////////////////////////////////////////////
        hresult_t flushDataSnapshot(float64_t const & timestamp);

        ////////////////////////////////////////////////////////////////////////
        /// \brief Get access to the memory device holding the data
        ////////////////////////////////////////////////////////////////////////
        hresult_t writeDataBinary(std::string const & filename);
        static hresult_t getData(logData_t                                  & logData,
                                 std::vector<AbstractIODevice *>            & flows,
                                 int64_t                              const & integerSectionSize,
                                 int64_t                              const & floatSectionSize,
                                 int64_t                              const & headerSize,
                                 int64_t                                      recordedBytesDataLine = -1);
        hresult_t getData(logData_t & logData);
    private:
        ////////////////////////////////////////////////////////////////////////
        /// \brief   Create a new file to continue the recording.
        /// \details Each chunk shall have a size defined by LARGE_LOG_SIZE_GB and shall
        ///          be suffixed by an increasing natural number.
        ///
        /// \return  SUCCESS if successful, the corresponding telemetry error otherwise.
        ////////////////////////////////////////////////////////////////////////
        hresult_t createNewChunk();

    private:
        ///////////////////////////////////////////////////////////////////////
        /// Private attributes
        ///////////////////////////////////////////////////////////////////////
        std::vector<MemoryDevice> flows_;

        bool_t isInitialized_;

        int64_t recordedBytesLimits_;
        int64_t recordedBytesDataLine_;
        int64_t recordedBytes_;             ///< Bytes recorded in the file.
        int64_t headerSize_;                ///< Size in byte of the header.

        char_t const * integersAddress_;    ///< Address of the integer data section.
        int64_t integerSectionSize_;        ///< Size in bytes of the integer data section.

        char_t const * floatsAddress_;      ///< Address of the float data section.
        int64_t floatSectionSize_;          ///< Size in byte of the float data section.
        float64_t timeUnit_;                ///< Precision to use when logging the time.
    };
}

#endif // JIMINY_TELEMETRY_RECORDER_H