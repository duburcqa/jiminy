///////////////////////////////////////////////////////////////////////////////
///
/// \brief       Declaration of the TelemetryRecorder class, responsible of recording data
///              to files.
///
///////////////////////////////////////////////////////////////////////////////

#ifndef JIMINY_TELEMETRY_RECORDER_H
#define JIMINY_TELEMETRY_RECORDER_H

#include <deque>

#include "jiminy/core/io/MemoryDevice.h"


namespace jiminy
{
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
        TelemetryRecorder(void) = default;
        ~TelemetryRecorder(void);

        ////////////////////////////////////////////////////////////////////////
        /// \brief Initialize the recorder.
        /// \param[in] telemetryData Data to log.
        /// \param[in] timeUnit Unit with which the time will be logged.
        ///                     Note that time is logged.
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

        hresult_t getLog(logData_t & logData);
        static hresult_t readLog(std::string const & filename,
                                 logData_t         & logData);

        hresult_t writeLog(std::string const & filename);

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
        std::deque<MemoryDevice> flows_;

        bool_t isInitialized_;

        int64_t recordedBytesLimits_;
        int64_t recordedBytesDataLine_;
        int64_t recordedBytes_;             ///< Bytes recorded in the file.
        int64_t headerSize_;                ///< Size in byte of the header.

        std::deque<std::pair<std::string, int64_t> > const * integersRegistry_;  ///< Pointer to the integer registry
        int64_t integerSectionSize_;                                             ///< Size in bytes of the integer data section
        std::deque<std::pair<std::string, float64_t> > const * floatsRegistry_;  ///< Pointer to the float registry
        int64_t floatSectionSize_;                                               ///< Size in bytes of the float data section

        float64_t timeUnitInv_;             ///< Precision to use when logging the time.
    };
}

#endif // JIMINY_TELEMETRY_RECORDER_H