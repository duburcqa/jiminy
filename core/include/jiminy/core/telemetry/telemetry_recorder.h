#ifndef JIMINY_TELEMETRY_RECORDER_H
#define JIMINY_TELEMETRY_RECORDER_H

#include <deque>

#include "jiminy/core/fwd.h"
#include "jiminy/core/io/memory_device.h"


namespace jiminy
{
    struct JIMINY_DLLAPI LogData
    {
        int32_t version;
        static_map_t<std::string, std::string> constants;
        double timeUnit;
        VectorX<int64_t> times;
        std::vector<std::string> variableNames;
        MatrixX<int64_t> integerValues;
        MatrixX<double> floatValues;
    };

    class TelemetryData;

    /// \class This class is responsible of writing recorded data to devices.
    class JIMINY_DLLAPI TelemetryRecorder
    {
    public:
        DISABLE_COPY(TelemetryRecorder)

    public:
        TelemetryRecorder() = default;
        ~TelemetryRecorder();

        /// \brief Initialize the recorder.
        ///
        /// \param[in] telemetryData Data to log.
        /// \param[in] timeUnit Unit with which the time will be logged.
        hresult_t initialize(TelemetryData * telemetryData, double timeUnit);

        bool getIsInitialized();

        /// \brief Maximum time that can be logged with the current precision.
        double getMaximumLogTime() const;

        /// \brief Maximum time that can be logged with the given precision.
        static double getMaximumLogTime(double timeUnit);

        /// \brief Reset the recorder.
        void reset();

        /// \brief Create a new line in the record with the current telemetry data.
        hresult_t flushDataSnapshot(double timestamp);

        hresult_t getLog(LogData & logData);
        static hresult_t readLog(const std::string & filename, LogData & logData);

        hresult_t writeLog(const std::string & filename);

    private:
        /// \brief Create a new file to continue the recording.
        ///
        /// \return SUCCESS if successful, the corresponding telemetry error otherwise.
        hresult_t createNewChunk();

    private:
        std::deque<MemoryDevice> flows_;

        bool isInitialized_;

        int64_t recordedBytesLimits_;
        int64_t recordedBytesDataLine_;
        /// \brief Bytes recorded in the file.
        int64_t recordedBytes_;
        /// \brief Size in byte of the header.
        int64_t headerSize_;

        /// \brief Pointer to the integer registry.
        const std::deque<std::pair<std::string, int64_t>> * integersRegistry_;
        /// \brief Size in bytes of the integer data section.
        int64_t integerSectionSize_;
        /// \brief Pointer to the float registry.
        const std::deque<std::pair<std::string, double>> * floatsRegistry_;
        /// \brief Size in bytes of the float data section.
        int64_t floatSectionSize_;

        /// \brief Precision to use when logging the time.
        double timeUnitInv_;
    };
}

#endif  // JIMINY_TELEMETRY_RECORDER_H