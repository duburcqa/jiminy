///////////////////////////////////////////////////////////////////////////////
///
/// \brief TelemetryRecorder Implementation.
///
//////////////////////////////////////////////////////////////////////////////

#include <math.h>
#include <cmath>
#include <iomanip>
#include <fstream>


#include "jiminy/core/io/FileDevice.h"
#include "jiminy/core/telemetry/TelemetryData.h"
#include "jiminy/core/Constants.h"

#include "jiminy/core/telemetry/TelemetryRecorder.h"


namespace jiminy
{
    TelemetryRecorder::TelemetryRecorder(void) :
    flows_(),
    isInitialized_(false),
    recordedBytesLimits_(0),
    recordedBytesDataLine_(0),
    recordedBytes_(0),
    headerSize_(0),
    integersRegistry_(nullptr),
    integerSectionSize_(0),
    floatsRegistry_(nullptr),
    floatSectionSize_(0),
    timeUnitInv_(1.0)
    {
        // Empty on purpose
    }

    TelemetryRecorder::~TelemetryRecorder(void)
    {
        if (!flows_.empty())
        {
            flows_.back().close();
        }
    }

    hresult_t TelemetryRecorder::initialize(TelemetryData       * telemetryData,
                                            float64_t     const & timeUnit)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (isInitialized_)
        {
            PRINT_ERROR("TelemetryRecorder already initialized.");
            returnCode = hresult_t::ERROR_INIT_FAILED;
        }
        // Log the time unit as constant
        timeUnitInv_ = 1.0 / timeUnit;
        std::ostringstream timeUnitStr;
        int precision = - static_cast<int>(std::ceil(std::log10(STEPPER_MIN_TIMESTEP)));
        timeUnitStr << std::scientific << std::setprecision(precision) << timeUnit;
        telemetryData->registerConstant(TIME_UNIT, timeUnitStr.str());

        std::vector<char_t> header;
        if (returnCode == hresult_t::SUCCESS)
        {
            // Clear the MemoryDevice buffer
            flows_.clear();

            // Get telemetry data infos
            integersRegistry_ = telemetryData->getRegistry<int64_t>();
            integerSectionSize_ = sizeof(int64_t) * integersRegistry_->size();
            floatsRegistry_ = telemetryData->getRegistry<float64_t>();
            floatSectionSize_ = sizeof(float64_t) * floatsRegistry_->size();
            recordedBytesDataLine_ = integerSectionSize_ + floatSectionSize_
                                   + static_cast<int64_t>(START_LINE_TOKEN.size() + sizeof(uint64_t));  // uint64_t for Global.Time

            // Get the header
            telemetryData->formatHeader(header);
            headerSize_ = static_cast<int64_t>(header.size());

            // Create a new MemoryDevice and open it
            returnCode = createNewChunk();
        }

        // Write the Header
        if (returnCode == hresult_t::SUCCESS)
        {
            returnCode = flows_[0].write(header);
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            recordedBytes_ = headerSize_;
            isInitialized_ = true;
        }

        return returnCode;
    }

    float64_t TelemetryRecorder::getMaximumLogTime(float64_t const & timeUnit)
    {
        return static_cast<float64_t>(std::numeric_limits<int64_t>::max()) * timeUnit;
    }

    float64_t TelemetryRecorder::getMaximumLogTime(void) const
    {
        return getMaximumLogTime(1.0 / timeUnitInv_);
    }

    bool_t const & TelemetryRecorder::getIsInitialized(void)
    {
        return isInitialized_;
    }

    void TelemetryRecorder::reset(void)
    {
        // Close the current MemoryDevice, if any and if it was opened
        if (!flows_.empty())
        {
            flows_.back().close();
        }

        isInitialized_ = false;
    }

    hresult_t TelemetryRecorder::createNewChunk()
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        // Close the current MemoryDevice, if any and if it was opened
        if (!flows_.empty())
        {
            flows_.back().close();
        }

        /* Create a new chunk.
           The size of the first chunk is chosen to be large enough
           to contain the whole header (with constants). Doing this
           does not really affect the performances since it is written
           only once, at init of the simulation. The optimized buffer
           size is used for the log data. */
        int64_t isHeaderThere = flows_.empty();
        int64_t maxBufferSize = std::max(TELEMETRY_MIN_BUFFER_SIZE, isHeaderThere * headerSize_);
        int64_t maxRecordedDataLines = (maxBufferSize - isHeaderThere * headerSize_) / recordedBytesDataLine_;
        recordedBytesLimits_ = isHeaderThere * headerSize_ + maxRecordedDataLines * recordedBytesDataLine_;
        flows_.emplace_back(recordedBytesLimits_);
        returnCode = flows_.back().open(openMode_t::READ_WRITE);

        if (returnCode == hresult_t::SUCCESS)
        {
            recordedBytes_ = 0;
        }

        return returnCode;
    }

    hresult_t TelemetryRecorder::flushDataSnapshot(float64_t const & timestamp)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (recordedBytes_ == recordedBytesLimits_)
        {
            returnCode = createNewChunk();
        }

        if (returnCode == hresult_t::SUCCESS)
        {
            // Write new line token
            flows_.back().write(START_LINE_TOKEN);

            // Write time
            flows_.back().write(static_cast<int64_t>(std::round(timestamp * timeUnitInv_)));

            // Write data, integers first
            for (std::pair<std::string, int64_t> const & keyValue : *integersRegistry_)
            {
                flows_.back().write(keyValue.second);
            }

            // Write data, floats last
            for (std::pair<std::string, float64_t> const & keyValue : *floatsRegistry_)
            {
                flows_.back().write(keyValue.second);
            }

            // Update internal counter
            recordedBytes_ += recordedBytesDataLine_;
        }

        return returnCode;
    }

    hresult_t TelemetryRecorder::writeDataBinary(std::string const & filename)
    {
        FileDevice myFile(filename);
        myFile.open(openMode_t::WRITE_ONLY | openMode_t::TRUNCATE);
        if (myFile.isOpen())
        {
            for (auto & flow : flows_)
            {
                int64_t const pos_old = flow.pos();
                flow.seek(0);

                std::vector<uint8_t> bufferChunk;
                bufferChunk.resize(static_cast<std::size_t>(pos_old));
                flow.read(bufferChunk);
                myFile.write(bufferChunk);

                flow.seek(pos_old);
            }

            myFile.close();
        }
        else
        {
            PRINT_ERROR("Impossible to create the log file. Check if root folder exists and if you have writing permissions.");
            return hresult_t::ERROR_BAD_INPUT;
        }
        return hresult_t::SUCCESS;
    }

    hresult_t TelemetryRecorder::getData(logData_t & logData,
                                         std::vector<AbstractIODevice *> & flows,
                                         int64_t const & integerSectionSize,
                                         int64_t const & floatSectionSize,
                                         int64_t const & headerSize)
    {
        logData.constants.clear();
        logData.fieldnames.clear();
        logData.timestamps.clear();
        logData.intData.clear();
        logData.floatData.clear();

        if (!flows.empty())
        {
            int64_t timestamp;
            std::vector<int64_t> intDataLine;
            logData.numInt = static_cast<std::size_t>(integerSectionSize) / sizeof(int64_t);
            intDataLine.resize(logData.numInt);
            std::vector<float64_t> floatDataLine;
            logData.numFloat = static_cast<std::size_t>(floatSectionSize) / sizeof(float64_t);
            floatDataLine.resize(logData.numFloat);

            bool_t isReadingHeaderDone = false;
            for (auto & flow : flows)
            {
                // Save the cursor position and move it to the beginning
                int64_t const pos_old = flow->pos();
                flow->seek(0);

                // Dealing with version flag, constants, and variable names
                if (!isReadingHeaderDone)
                {
                    // Read version flag and check if valid
                    int32_t version;
                    flow->readData(&version, sizeof(int32_t));
                    if (version != TELEMETRY_VERSION)
                    {
                        PRINT_ERROR("Log telemetry version not supported. Impossible to read log.");
                        return hresult_t::ERROR_BAD_INPUT;
                    }
                    logData.version = version;

                    // Read the rest of the header
                    std::vector<char_t> headerCharBuffer;
                    headerCharBuffer.resize(static_cast<std::size_t>(headerSize - flow->pos()));
                    flow->read(headerCharBuffer);

                    // Parse constants
                    bool_t isLastConstant = false;
                    auto posHeaderIt = headerCharBuffer.begin();
                    posHeaderIt += START_CONSTANTS.size() + 1 + START_LINE_TOKEN.size();  // Skip tokens
                    while (true)
                    {
                        // Find position of the next constant
                        auto posHeaderNextIt = std::search(
                            posHeaderIt,
                            headerCharBuffer.end(),
                            START_LINE_TOKEN.begin(),
                            START_LINE_TOKEN.end());
                        isLastConstant = posHeaderNextIt == headerCharBuffer.end();
                        if (isLastConstant)
                        {
                            posHeaderNextIt = std::search(
                                posHeaderIt,
                                headerCharBuffer.end(),
                                START_COLUMNS.begin(),
                                START_COLUMNS.end());
                        }

                        // Split key and value
                        auto posDelimiterIt = std::search(
                            posHeaderIt,
                            posHeaderNextIt,
                            TELEMETRY_CONSTANT_DELIMITER.begin(),
                            TELEMETRY_CONSTANT_DELIMITER.end());
                        std::string const key(posHeaderIt, posDelimiterIt);
                        std::string const value(
                            posDelimiterIt + TELEMETRY_CONSTANT_DELIMITER.size(),
                            posHeaderNextIt - 1);  // Last char is '\0'
                        logData.constants.emplace_back(key, value);

                        // Stop if it was the last one
                        if (isLastConstant)
                        {
                            posHeaderIt = posHeaderNextIt + START_COLUMNS.size() + 1;  // Skip last '\0'
                            break;
                        }
                        posHeaderIt = posHeaderNextIt + START_LINE_TOKEN.size();
                    }

                    // Parse variable names
                    char_t const * pHeader = &(*posHeaderIt);
                    std::size_t posHeader = 0;
                    while (true)
                    {
                        // std::string constructor automatically reads till next '\0'
                        std::string const fieldname = std::string(pHeader + posHeader);
                        if (fieldname == START_DATA)
                        {
                            break;
                        }
                        posHeader += fieldname.size() + 1;  // Skip last '\0'
                        logData.fieldnames.push_back(fieldname);
                    }

                    isReadingHeaderDone = true;
                }

                // Look for timeUnit constant - if not found, use default time unit
                float64_t timeUnit = STEPPER_MIN_TIMESTEP;
                for (auto const & [key, value] : logData.constants)
                {
                    if (key == TIME_UNIT)
                    {
                        std::istringstream totalSString(value);
                        totalSString >> timeUnit;
                        break;
                    }
                }
                logData.timeUnit = timeUnit;

                // Read all available data lines: [token, time, integers, floats]
                char_t startLineTokenBuffer;
                int64_t const startLineTokenSize = static_cast<int64_t>(START_LINE_TOKEN.size());
                int64_t const recordedBytesDataLine = integerSectionSize + floatSectionSize
                    + startLineTokenSize + static_cast<int64_t>(sizeof(uint64_t));
                std::size_t const numberLines = static_cast<std::size_t>(
                    flow->bytesAvailable() / recordedBytesDataLine);
                logData.timestamps.reserve(logData.timestamps.size() + numberLines);
                while (flow->bytesAvailable() > 0)
                {
                    // Check if actual data are still available.
                    // It is necessary because a pre-allocated memory may not be full.
                    flow->readData(&startLineTokenBuffer, 1);
                    if (startLineTokenBuffer != START_LINE_TOKEN[0])
                    {
                        break;
                    }
                    flow->seek(flow->pos() + startLineTokenSize - 1);

                    // Read data line
                    flow->readData(&timestamp, sizeof(int64_t));
                    flow->readData(intDataLine.data(), integerSectionSize);
                    flow->readData(floatDataLine.data(), floatSectionSize);
                    logData.timestamps.emplace_back(timestamp);
                    logData.intData.emplace_back(intDataLine);
                    logData.floatData.emplace_back(floatDataLine);
                }

                // Restore the cursor position
                flow->seek(pos_old);
            }
        }

        return hresult_t::SUCCESS;
    }

    hresult_t TelemetryRecorder::getData(logData_t & logData)
    {
        std::vector<AbstractIODevice *> abstractFlows_;
        for (MemoryDevice & device: flows_)
        {
            abstractFlows_.push_back(&device);
        }

        return getData(logData,
                       abstractFlows_,
                       integerSectionSize_,
                       floatSectionSize_,
                       headerSize_);
    }
}