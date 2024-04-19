#ifndef JIMINY_TELEMETRY_RECORDER_H
#define JIMINY_TELEMETRY_RECORDER_H

#include "jiminy/core/fwd.h"
#include "jiminy/core/io/memory_device.h"


namespace jiminy
{
    struct LogData;
    class TelemetryData;

    /// \class This class is responsible of writing recorded data to devices.
    class JIMINY_DLLAPI TelemetryRecorder
    {
    public:
        JIMINY_DISABLE_COPY(TelemetryRecorder)

    public:
        explicit TelemetryRecorder() = default;
        ~TelemetryRecorder();

        /// \brief Initialize the recorder.
        ///
        /// \param[in] telemetryData Data to log.
        /// \param[in] timeUnit Unit with which the time will be logged.
        void initialize(TelemetryData * telemetryData, double timeUnit);

        /// \brief Reset the recorder.
        void reset();

        /// \brief Create a new line in the record with the current telemetry data.
        void flushSnapshot(double time);

        LogData getLog();
        static LogData readLog(const std::string & filename);

        void writeLog(const std::string & filename);

        bool getIsInitialized();

        /// \brief Maximum time that can be logged with the current precision.
        double getLogDurationMax() const;

        /// \brief Maximum time that can be logged with the given precision.
        static double getLogDurationMax(double timeUnit);

    private:
        /// \brief Create a new file to continue the recording.
        void createNewChunk();

    private:
        std::deque<MemoryDevice> flows_{};

        bool isInitialized_{false};

        std::size_t recordedBytesLimit_{0};
        std::size_t recordedBytesPerLine_{0};
        /// \brief Bytes recorded in the file.
        std::size_t recordedBytes_{0};
        /// \brief Size in byte of the header.
        std::size_t headerSize_{0};

        /// \brief Pointer to the integer registry.
        const static_map_t<std::string, int64_t, false> * integerRegistry_{nullptr};
        /// \brief Size in bytes of the integer data section.
        int64_t integerSectionSize_{-1};
        /// \brief Pointer to the float registry.
        const static_map_t<std::string, double, false> * floatRegistry_{nullptr};
        /// \brief Size in bytes of the float data section.
        int64_t floatSectionSize_{-1};

        /// \brief Precision to use when logging the time.
        double timeUnitInv_{qNAN};
    };
}

#endif  // JIMINY_TELEMETRY_RECORDER_H
