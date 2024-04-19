#include <math.h>
#include <cmath>
#include <iomanip>
#include <fstream>

#include "jiminy/core/telemetry/fwd.h"
#include "jiminy/core/io/file_device.h"
#include "jiminy/core/telemetry/telemetry_data.h"

#include "jiminy/core/telemetry/telemetry_recorder.h"


namespace jiminy
{
    inline constexpr std::size_t TELEMETRY_MIN_BUFFER_SIZE{256UL * 1024UL};  // 256Ko

    TelemetryRecorder::~TelemetryRecorder()
    {
        reset();
    }

    void TelemetryRecorder::initialize(TelemetryData * telemetryData, double timeUnit)
    {
        if (isInitialized_)
        {
            JIMINY_THROW(bad_control_flow, "TelemetryRecorder already initialized.");
        }

        // Log the time unit as constant
        timeUnitInv_ = 1.0 / timeUnit;
        std::ostringstream timeUnitStr;
        int precision = -static_cast<int>(std::ceil(std::log10(STEPPER_MIN_TIMESTEP)));
        timeUnitStr << std::scientific << std::setprecision(precision) << timeUnit;
        // FIXME: remove explicit conversion to `std::string` when moving to C++20
        telemetryData->registerConstant(std::string{TIME_UNIT}, timeUnitStr.str());

        // Clear the MemoryDevice buffer
        flows_.clear();

        // Get telemetry data infos
        integerRegistry_ = telemetryData->getRegistry<int64_t>();
        integerSectionSize_ = sizeof(int64_t) * integerRegistry_->size();
        floatRegistry_ = telemetryData->getRegistry<double>();
        floatSectionSize_ = sizeof(double) * floatRegistry_->size();
        recordedBytesPerLine_ = integerSectionSize_ + floatSectionSize_ + START_LINE_TOKEN.size() +
                                sizeof(int64_t);  // int64_t for Global.Time

        // Get the header
        std::vector<char> header = telemetryData->formatHeader();
        headerSize_ = header.size();

        // Create a new MemoryDevice and open it
        createNewChunk();

        // Write the Header
        flows_[0].write(header);

        recordedBytes_ = headerSize_;
        isInitialized_ = true;
    }

    double TelemetryRecorder::getLogDurationMax(double timeUnit)
    {
        return static_cast<double>(std::numeric_limits<int64_t>::max()) * timeUnit;
    }

    double TelemetryRecorder::getLogDurationMax() const
    {
        return getLogDurationMax(1.0 / timeUnitInv_);
    }

    bool TelemetryRecorder::getIsInitialized()
    {
        return isInitialized_;
    }

    void TelemetryRecorder::reset()
    {
        // Return early if not initialized
        if (!isInitialized_)
        {
            return;
        }

        // Close the current MemoryDevice, if any and if it was opened
        if (!flows_.empty())
        {
            flows_.back().close();
        }

        isInitialized_ = false;
    }

    void TelemetryRecorder::createNewChunk()
    {
        // Close the current MemoryDevice, if any and if it was opened
        const bool hasHeader = !flows_.empty();
        if (hasHeader)
        {
            flows_.back().close();
        }

        /* Create a new chunk.
           The size of the first chunk is chosen to be large enough to contain the whole header,
           including constants. This does not really affect the performances since it is written
           only once per simulation. The optimized buffer size is used for the log data. */
        std::size_t headerSize = 0;
        if (!hasHeader)
        {
            headerSize = headerSize_;
        }
        std::size_t bufferSizeMax = std::max(TELEMETRY_MIN_BUFFER_SIZE, headerSize);
        std::size_t recordedDataLinesMax = (bufferSizeMax - headerSize) / recordedBytesPerLine_;
        recordedBytesLimit_ = headerSize + recordedDataLinesMax * recordedBytesPerLine_;
        flows_.emplace_back(recordedBytesLimit_);
        flows_.back().open(OpenMode::READ_WRITE);

        recordedBytes_ = 0;
    }

    void TelemetryRecorder::flushSnapshot(double time)
    {
        if (recordedBytes_ == recordedBytesLimit_)
        {
            createNewChunk();
        }

        // Write new line token
        flows_.back().write(START_LINE_TOKEN);

        // Write time
        flows_.back().write(static_cast<int64_t>(std::round(time * timeUnitInv_)));

        // Write data, integers first
        for (const std::pair<std::string, int64_t> & keyValue : *integerRegistry_)
        {
            flows_.back().write(keyValue.second);
        }

        // Write data, floats last
        for (const std::pair<std::string, double> & keyValue : *floatRegistry_)
        {
            flows_.back().write(keyValue.second);
        }

        // Update internal counter
        recordedBytes_ += recordedBytesPerLine_;
    }

    void TelemetryRecorder::writeLog(const std::string & filename)
    {
        FileDevice myFile(filename);
        myFile.open(OpenMode::WRITE_ONLY | OpenMode::TRUNCATE);
        if (!myFile.isOpen())
        {
            JIMINY_THROW(std::ios_base::failure,
                         "Impossible to create the log file. Check if root folder "
                         "exists and if you have writing permissions.");
        }

        for (auto & flow : flows_)
        {
            const std::ptrdiff_t posOld = flow.pos();
            flow.seek(0);

            std::vector<uint8_t> bufferChunk;
            bufferChunk.resize(static_cast<std::size_t>(posOld));
            flow.read(bufferChunk);
            myFile.write(bufferChunk);

            flow.seek(posOld);
        }

        myFile.close();
    }

    LogData parseLogDataRaw(std::vector<AbstractIODevice *> & flows,
                            int64_t integerSectionSize,
                            int64_t floatSectionSize,
                            int64_t headerSize)
    {
        LogData logData{};

        // Set data structure
        const Eigen::Index numInt =
            static_cast<Eigen::Index>(integerSectionSize / sizeof(int64_t));
        const Eigen::Index numFloat = static_cast<Eigen::Index>(floatSectionSize / sizeof(double));
        logData.integerValues.resize(numInt, 0);
        logData.floatValues.resize(numFloat, 0);

        // Process the provided data
        Eigen::Index timeIndex = 0;
        if (!flows.empty())
        {
            bool isReadingHeaderDone = false;
            for (auto & flow : flows)
            {
                // Save the cursor position and move it to the beginning
                const int64_t posOld = flow->pos();
                flow->seek(0);

                // Dealing with version flag, constants, and variable names
                if (!isReadingHeaderDone)
                {
                    // Read version flag and check if valid
                    int32_t version;
                    flow->read(version);
                    if (version != TELEMETRY_VERSION)
                    {
                        JIMINY_THROW(
                            std::runtime_error,
                            "Log telemetry version not supported. Impossible to read log.");
                    }
                    logData.version = version;

                    // Read the rest of the header
                    std::vector<char> headerCharBuffer;
                    headerCharBuffer.resize(static_cast<std::size_t>(headerSize - flow->pos()));
                    flow->read(headerCharBuffer);

                    // Parse constants
                    bool isLastConstant = false;
                    auto posHeaderIt = headerCharBuffer.begin();
                    posHeaderIt +=
                        START_CONSTANTS.size() + 1 + START_LINE_TOKEN.size();  // Skip tokens
                    while (true)
                    {
                        // Find position of the next constant
                        auto posHeaderNextIt = std::search(posHeaderIt,
                                                           headerCharBuffer.end(),
                                                           START_LINE_TOKEN.begin(),
                                                           START_LINE_TOKEN.end());
                        isLastConstant = posHeaderNextIt == headerCharBuffer.end();
                        if (isLastConstant)
                        {
                            posHeaderNextIt = std::search(posHeaderIt,
                                                          headerCharBuffer.end(),
                                                          START_COLUMNS.begin(),
                                                          START_COLUMNS.end());
                        }

                        // Split key and value
                        auto posDelimiterIt = std::search(posHeaderIt,
                                                          posHeaderNextIt,
                                                          TELEMETRY_CONSTANT_DELIMITER.begin(),
                                                          TELEMETRY_CONSTANT_DELIMITER.end());
                        const std::string key(posHeaderIt, posDelimiterIt);
                        const std::string value(posDelimiterIt +
                                                    TELEMETRY_CONSTANT_DELIMITER.size(),
                                                posHeaderNextIt - 1);  // Last char is '\0'
                        logData.constants.emplace_back(key, value);

                        // Stop if it was the last one
                        if (isLastConstant)
                        {
                            posHeaderIt =
                                posHeaderNextIt + START_COLUMNS.size() + 1;  // Skip last '\0'
                            break;
                        }
                        posHeaderIt = posHeaderNextIt + START_LINE_TOKEN.size();
                    }

                    // Parse variable names
                    const char * pHeader = &(*posHeaderIt);
                    std::size_t posHeader = 0;
                    while (true)
                    {
                        // std::string constructor automatically reads till next '\0'
                        const std::string fieldname = std::string(pHeader + posHeader);
                        if (fieldname == START_DATA)
                        {
                            break;
                        }
                        posHeader += fieldname.size() + 1;  // Skip last '\0'
                        logData.variableNames.push_back(fieldname);
                    }

                    isReadingHeaderDone = true;
                }

                // Look for timeUnit constant - if not found, use default time unit
                double timeUnit = STEPPER_MIN_TIMESTEP;
                for (const auto & [key, value] : logData.constants)
                {
                    if (key == TIME_UNIT)
                    {
                        std::istringstream totalSString(value);
                        totalSString >> timeUnit;
                        break;
                    }
                }
                logData.timeUnit = timeUnit;

                // Allocate memory
                const int64_t startLineTokenSize = static_cast<int64_t>(START_LINE_TOKEN.size());
                const int64_t recordedBytesDataLine =
                    startLineTokenSize + sizeof(uint64_t) + integerSectionSize + floatSectionSize;
                const Eigen::Index numData =
                    timeIndex + flow->bytesAvailable() / recordedBytesDataLine;
                logData.times.conservativeResize(numData);
                logData.integerValues.conservativeResize(Eigen::NoChange, numData);
                logData.floatValues.conservativeResize(Eigen::NoChange, numData);

                // Read all available data lines: [token, time, integers, floats]
                char startLineTokenBuffer;
                while (flow->bytesAvailable() > 0)
                {
                    /* Check if actual data are still available.
                       It is necessary because a pre-allocated memory may not be full. */
                    flow->read(startLineTokenBuffer);
                    if (startLineTokenBuffer != START_LINE_TOKEN[0])
                    {
                        break;
                    }
                    flow->seek(flow->pos() + startLineTokenSize - 1);

                    // Read data line
                    flow->read(logData.times[timeIndex]);
                    flow->read(logData.integerValues.col(timeIndex));
                    flow->read(logData.floatValues.col(timeIndex));

                    // Increment time counter
                    ++timeIndex;
                }

                // Restore the cursor position
                flow->seek(posOld);
            }
        }

        // Remove uninitialized data if any. It occurs whenever the last memory buffer is not full.
        logData.times.conservativeResize(timeIndex);
        logData.integerValues.conservativeResize(Eigen::NoChange, timeIndex);
        logData.floatValues.conservativeResize(Eigen::NoChange, timeIndex);

        return logData;
    }

    LogData TelemetryRecorder::getLog()
    {
        std::vector<AbstractIODevice *> abstractFlows_;
        for (MemoryDevice & device : flows_)
        {
            abstractFlows_.push_back(&device);
        }

        return parseLogDataRaw(
            abstractFlows_, integerSectionSize_, floatSectionSize_, headerSize_);
    }

    LogData TelemetryRecorder::readLog(const std::string & filename)
    {
        std::ifstream file = std::ifstream(filename, std::ios::in | std::ifstream::binary);
        if (!file.is_open())
        {
            JIMINY_THROW(std::ios_base::failure,
                         "Impossible to open the log file. Check that the file "
                         "exists and that you have reading permissions.");
        }

        // Make sure the log file is not corrupted
        if (!file.good())
        {
            JIMINY_THROW(std::ios_base::failure, "Corrupted log file.");
        }

        // Skip the version flag
        int32_t header_version_length = sizeof(int32_t);
        file.seekg(header_version_length);

        std::vector<std::string> headerBuffer;
        std::string subHeaderBuffer;

        // Reach the beginning of the constants
        while (std::getline(file, subHeaderBuffer, '\0').good() &&
               subHeaderBuffer != START_CONSTANTS)
        {
        }

        // Get all the logged constants
        while (std::getline(file, subHeaderBuffer, '\0').good() &&
               subHeaderBuffer != START_COLUMNS)
        {
            headerBuffer.push_back(subHeaderBuffer);
        }

        // Get the names of the logged variables
        while (std::getline(file, subHeaderBuffer, '\0').good() && subHeaderBuffer != START_DATA)
        {
        }

        // Extract the number of integers and floats from the list of logged constants
        const std::string & headerNumIntEntries = headerBuffer[headerBuffer.size() - 2];
        int64_t delimiter = headerNumIntEntries.find(TELEMETRY_CONSTANT_DELIMITER);
        const int32_t NumIntEntries = std::stoi(headerNumIntEntries.substr(delimiter + 1));
        const std::string & headerNumFloatEntries = headerBuffer[headerBuffer.size() - 1];
        delimiter = headerNumFloatEntries.find(TELEMETRY_CONSTANT_DELIMITER);
        const int32_t NumFloatEntries = std::stoi(headerNumFloatEntries.substr(delimiter + 1));

        // Deduce the parameters required to parse the whole binary log file
        int64_t integerSectionSize = (NumIntEntries - 1) * sizeof(int64_t);  // Remove Global.Time
        int64_t floatSectionSize = NumFloatEntries * sizeof(double);
        int64_t headerSize = static_cast<int64_t>(file.tellg());  // Last '\0' is included

        // Close the file
        file.close();

        FileDevice device(filename);
        device.open(OpenMode::READ_ONLY);
        std::vector<AbstractIODevice *> flows;
        flows.push_back(&device);
        return parseLogDataRaw(flows, integerSectionSize, floatSectionSize, headerSize);
    }
}
