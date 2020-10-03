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
    timeLoggingPrecision_(0.0)
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
                                            float64_t     const & timeLoggingPrecision)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (isInitialized_)
        {
            std::cout << "Error - TelemetryRecorder::initialize - TelemetryRecorder already initialized." << std::endl;
            returnCode = hresult_t::ERROR_INIT_FAILED;
        }
        // Log the time unit as constant.
        timeLoggingPrecision_ = timeLoggingPrecision;
        telemetryData->registerConstant(TIME_UNIT, std::to_string(timeLoggingPrecision_));

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
                                   + static_cast<int64_t>(START_LINE_TOKEN.size() + sizeof(uint32_t));

            // Get the header
            telemetryData->formatHeader(header);
            headerSize_ = header.size();

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

    float64_t TelemetryRecorder::getMaximumLogTime(void) const
    {
        return std::numeric_limits<int32_t>::max() / timeLoggingPrecision_;
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
        uint32_t isHeaderThere = flows_.empty();
        uint32_t maxBufferSize = std::max(TELEMETRY_MAX_BUFFER_SIZE, isHeaderThere * headerSize_);
        uint32_t maxRecordedDataLines = ((maxBufferSize - isHeaderThere * headerSize_) / recordedBytesDataLine_);
        recordedBytesLimits_ = isHeaderThere * headerSize_ + maxRecordedDataLines * recordedBytesDataLine_;
        flows_.emplace_back(recordedBytesLimits_);
        returnCode = flows_.back().open(OpenMode::READ_WRITE);

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
            flows_.back().write(static_cast<int32_t>(std::round(timestamp * timeLoggingPrecision_)));

            // Write data, integers first
            flows_.back().write(reinterpret_cast<uint8_t const*>(integersAddress_), integerSectionSize_);

            // Write data, floats last
            flows_.back().write(reinterpret_cast<uint8_t const*>(floatsAddress_), floatSectionSize_);

            // Update internal counter
            recordedBytes_ += recordedBytesDataLine_;
        }

        return returnCode;
    }

    hresult_t TelemetryRecorder::writeDataBinary(std::string const & filename)
    {
        FileDevice myFile(filename);
        myFile.open(OpenMode::WRITE_ONLY | OpenMode::TRUNCATE);
        if (myFile.isOpen())
        {
            for (auto & flow : flows_)
            {
                int64_t const pos_old = flow.pos();
                flow.seek(0);

                std::vector<uint8_t> bufferChunk;
                bufferChunk.resize(pos_old);
                flow.read(bufferChunk);
                myFile.write(bufferChunk);

                flow.seek(pos_old);
            }

            myFile.close();
        }
        else
        {
            std::cout << "Error - Engine::writeLogTxt - Impossible to create the log file. Check if root folder exists and if you have writing permissions." << std::endl;
            return hresult_t::ERROR_BAD_INPUT;
        }
        return hresult_t::SUCCESS;
    }

    void TelemetryRecorder::getData(std::vector<std::string>                   & header,
                                    std::vector<float64_t>                     & timestamps,
                                    std::vector<std::vector<int32_t> >         & intData,
                                    std::vector<std::vector<float32_t> >       & floatData,
                                    std::vector<AbstractIODevice *>            & flows,
                                    int64_t                              const & integerSectionSize,
                                    int64_t                              const & floatSectionSize,
                                    int64_t                              const & headerSize,
                                    int64_t                                      recordedBytesDataLine)
    {
        header.clear();
        timestamps.clear();
        intData.clear();
        floatData.clear();

        if (!flows.empty())
        {
            int32_t timestamp;
            std::vector<int32_t> intDataLine;
            intDataLine.resize(integerSectionSize / sizeof(int32_t));
            std::vector<float32_t> floatDataLine;
            floatDataLine.resize(floatSectionSize / sizeof(float32_t));

            bool_t isReadingHeaderDone = false;
            for (auto & flow : flows)
            {
                // Save the cursor position and move it to the beginning
                int64_t const pos_old = flow->pos();
                flow->seek(0);

                // Dealing with version flag, constants, header, and descriptor
                if (!isReadingHeaderDone)
                {
                    int64_t header_version_length = sizeof(int32_t);
                    flow->seek(header_version_length); // Skip the version flag
                    std::vector<char_t> headerCharBuffer;
                    headerCharBuffer.resize(headerSize - header_version_length);
                    flow->read(headerCharBuffer);
                    char_t const * pHeader = &headerCharBuffer[0];
                    uint32_t posHeader = 0;
                    std::string fieldHeader(pHeader);
                    while (true)
                    {
                        header.emplace_back(std::move(fieldHeader));
                        posHeader += header.back().size() + 1;
                        fieldHeader = std::string(pHeader + posHeader);
                        if (fieldHeader.size() == 0 || posHeader >= headerCharBuffer.size())
                        {
                            break;
                        }
                        if (posHeader + fieldHeader.size() > headerCharBuffer.size())
                        {
                            fieldHeader = std::string(pHeader + posHeader, headerCharBuffer.size() - posHeader);
                            header.emplace_back(std::move(fieldHeader));
                            break;
                        }
                    }
                    isReadingHeaderDone = true;
                }

                // In header, look for timeUnit constant - if not found, use default time unit.
                float64_t timeUnit = TELEMETRY_DEFAULT_TIME_UNIT;
                auto const lastConstantIt = std::find(header.begin(), header.end(), START_COLUMNS);
                for (auto constantIt = header.begin() ; constantIt != lastConstantIt ; ++constantIt)
                {
                    int32_t const delimiter = constantIt->find("=");
                    if (constantIt->substr(0, delimiter) == TIME_UNIT)
                    {
                        timeUnit = std::stof(constantIt->substr(delimiter + 1));
                        break;
                    }
                }

                // Dealing with data lines, starting with new line flag, time, integers, and ultimately floats
                if (recordedBytesDataLine > 0)
                {
                    uint32_t numberLines = (flow->size() - flow->pos()) / recordedBytesDataLine;
                    timestamps.reserve(timestamps.size() + numberLines);
                    intData.reserve(intData.size() + numberLines);
                    floatData.reserve(floatData.size() + numberLines);
                }

                while (flow->bytesAvailable() > 0)
                {
                    flow->seek(flow->pos() + START_LINE_TOKEN.size()); // Skip new line flag
                    flow->readData(&timestamp, sizeof(int32_t));
                    flow->readData(intDataLine.data(), integerSectionSize);
                    flow->readData(floatDataLine.data(), floatSectionSize);

                    if (!timestamps.empty() && timestamp == 0)
                    {
                        // The buffer is not full, must stop reading !
                        break;
                    }

                    timestamps.emplace_back(static_cast<float64_t>(timestamp / timeUnit));
                    intData.emplace_back(intDataLine);
                    floatData.emplace_back(floatDataLine);
                }

                // Restore the cursor position
                flow->seek(pos_old);
            }
        }
    }

    void TelemetryRecorder::getData(std::vector<std::string>             & header,
                                    std::vector<float64_t>               & timestamps,
                                    std::vector<std::vector<int32_t> >   & intData,
                                    std::vector<std::vector<float32_t> > & floatData)
    {
        std::vector<AbstractIODevice *> abstractFlows_;
        for (MemoryDevice & device: flows_)
        {
            abstractFlows_.push_back(&device);
        }

        getData(header,
                timestamps,
                intData,
                floatData,
                abstractFlows_,
                integerSectionSize_,
                floatSectionSize_,
                headerSize_,
                recordedBytesDataLine_);
    }
}