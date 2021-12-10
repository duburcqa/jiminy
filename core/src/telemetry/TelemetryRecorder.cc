///////////////////////////////////////////////////////////////////////////////
///
/// \brief TelemetryRecorder Implementation.
///
//////////////////////////////////////////////////////////////////////////////

#include <math.h>
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
    integersAddress_(),
    integerSectionSize_(0),
    floatsAddress_(),
    floatSectionSize_(0),
    timeUnit_(0.0)
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
        // Log the time unit as constant.
        timeUnit_ = timeUnit;
        telemetryData->registerConstant(TIME_UNIT, std::to_string(timeUnit_));

        std::vector<char_t> header;
        if (returnCode == hresult_t::SUCCESS)
        {
            // Clear the MemoryDevice buffer
            flows_.clear();

            // Get telemetry data infos
            telemetryData->getData(integersAddress_,
                                   integerSectionSize_,
                                   floatsAddress_,
                                   floatSectionSize_);
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
        return static_cast<float64_t>(std::numeric_limits<int64_t>::max()) / timeUnit;
    }

    float64_t TelemetryRecorder::getMaximumLogTime(void) const
    {
        return getMaximumLogTime(timeUnit_);
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
        int64_t maxRecordedDataLines = ((maxBufferSize - isHeaderThere * headerSize_) / recordedBytesDataLine_);
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
            flows_.back().write(static_cast<int64_t>(std::round(timestamp * timeUnit_)));

            // Write data, integers first
            flows_.back().write(reinterpret_cast<uint8_t const *>(integersAddress_), integerSectionSize_);

            // Write data, floats last
            flows_.back().write(reinterpret_cast<uint8_t const *>(floatsAddress_), floatSectionSize_);

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

    hresult_t TelemetryRecorder::getData(logData_t                                  & logData,
                                         std::vector<AbstractIODevice *>            & flows,
                                         int64_t                              const & integerSectionSize,
                                         int64_t                              const & floatSectionSize,
                                         int64_t                              const & headerSize,
                                         int64_t                                      recordedBytesDataLine)
    {
        logData.header.clear();
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

                // Dealing with version flag, constants, header, and descriptor
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
                    headerCharBuffer.resize(static_cast<std::size_t>(headerSize) - sizeof(int32_t));
                    flow->read(headerCharBuffer);

                    // Parse header
                    char_t const * pHeader = &headerCharBuffer[0];
                    std::size_t posHeader = 0;
                    std::string fieldHeader(pHeader);
                    while (true)
                    {
                        logData.header.push_back(std::move(fieldHeader));
                        posHeader += logData.header.back().size() + 1;
                        fieldHeader = std::string(pHeader + posHeader);
                        if (fieldHeader.size() == 0 || posHeader >= headerCharBuffer.size())
                        {
                            break;
                        }
                    }
                    isReadingHeaderDone = true;
                }

                // In header, look for timeUnit constant - if not found, use default time unit.
                float64_t timeUnit = TELEMETRY_DEFAULT_TIME_UNIT;
                auto const lastConstantIt = std::find(logData.header.begin(), logData.header.end(), START_COLUMNS);
                for (auto constantIt = logData.header.begin() ; constantIt != lastConstantIt ; ++constantIt)
                {
                    std::size_t const delimiter = constantIt->find(TELEMETRY_CONSTANT_DELIMITER);
                    if (constantIt->substr(0, delimiter) == TIME_UNIT)
                    {
                        timeUnit = std::stof(constantIt->substr(delimiter + 1));
                        break;
                    }
                }
                logData.timeUnit = timeUnit;

                // Dealing with data lines, starting with new line flag, time, integers, and ultimately floats
                if (recordedBytesDataLine > 0)
                {
                    uint32_t numberLines = static_cast<uint32_t>(flow->bytesAvailable() / recordedBytesDataLine);
                    logData.timestamps.reserve(logData.timestamps.size() + numberLines);
                    logData.intData.reserve(logData.intData.size() + numberLines);
                    logData.floatData.reserve(logData.floatData.size() + numberLines);
                }

                std::vector<char_t> startLineTokenBuffer;
                startLineTokenBuffer.resize(START_LINE_TOKEN.size());
                while (flow->bytesAvailable() > 0)
                {
                    flow->read(startLineTokenBuffer);
                    flow->readData(&timestamp, sizeof(int64_t));
                    flow->readData(intDataLine.data(), integerSectionSize);
                    flow->readData(floatDataLine.data(), floatSectionSize);

                    if (startLineTokenBuffer[0] != START_LINE_TOKEN[0])
                    {
                        // The buffer is not full, must stop reading !
                        break;
                    }

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
                       headerSize_,
                       recordedBytesDataLine_);
    }
}