///////////////////////////////////////////////////////////////////////////////
///
/// \brief TelemetryRecorder Implementation.
///
//////////////////////////////////////////////////////////////////////////////

#include <iomanip>
#include <fstream>

#include "jiminy/core/telemetry/TelemetryRecorder.h"


namespace jiminy
{
    TelemetryRecorder::TelemetryRecorder(std::shared_ptr<TelemetryData const> const & telemetryDataInstance) :
    telemetryData_(telemetryDataInstance),
    flows_(),
    isInitialized_(false),
    recordedBytesLimits_(0),
    recordedBytesDataLine_(0),
    recordedBytes_(0),
    headerSize_(0),
    integersAddress_(),
    integerSectionSize_(0),
    floatsAddress_(),
    floatSectionSize_(0)
    {
        // Empty.
    }

    TelemetryRecorder::~TelemetryRecorder()
    {
        if (!flows_.empty())
        {
            flows_.back().close();
        }
    }

    hresult_t TelemetryRecorder::initialize(void)
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        if (isInitialized_)
        {
            std::cout << "Error - TelemetryRecorder::initialize - TelemetryRecorder already initialized." << std::endl;
            returnCode = hresult_t::ERROR_INIT_FAILED;
        }

        std::vector<char_t> header;
        if (returnCode == hresult_t::SUCCESS)
        {
            // Clear the MemoryDevice buffer
            flows_.clear();

            // Get telemetry data infos.
            telemetryData_->getData(integersAddress_,
                                    integerSectionSize_,
                                    floatsAddress_,
                                    floatSectionSize_);
            recordedBytesDataLine_ = integerSectionSize_ + floatSectionSize_
                                   + static_cast<int64_t>(START_LINE_TOKEN.size() + sizeof(uint32_t));

            // Get the header
            telemetryData_->formatHeader(header);
            headerSize_ = header.size();

            // Create a new MemoryDevice and open it.
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

    bool_t const & TelemetryRecorder::getIsInitialized(void)
    {
        return isInitialized_;
    }

    void TelemetryRecorder::reset(void)
    {
        // Close the current MemoryDevice, if any and if it was opened.
        if (!flows_.empty())
        {
            flows_.back().close();
        }

        isInitialized_ = false;
    }

    hresult_t TelemetryRecorder::createNewChunk()
    {
        hresult_t returnCode = hresult_t::SUCCESS;

        // Close the current MemoryDevice, if any and if it was opened.
        if (!flows_.empty())
        {
            flows_.back().close();
        }

        // Create a new one.
        uint32_t isHeaderThere = flows_.empty();
        uint32_t maxRecordedDataLines = ((MAX_BUFFER_SIZE - isHeaderThere * headerSize_) / recordedBytesDataLine_);
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
            flows_.back().write(static_cast<int32_t>(timestamp * 1e6));

            // Write data, integers first.
            flows_.back().write(reinterpret_cast<uint8_t const*>(integersAddress_), integerSectionSize_);

            // Write data, floats last.
            flows_.back().write(reinterpret_cast<uint8_t const*>(floatsAddress_), floatSectionSize_);

            // Update internal counter
            recordedBytes_ += recordedBytesDataLine_;
        }

        return returnCode;
    }

    hresult_t TelemetryRecorder::writeDataBinary(std::string const & filename)
    {
        std::ofstream myFile = std::ofstream(filename,
                                             std::ios::out |
                                             std::ios::binary |
                                             std::ofstream::trunc);
        if (myFile.is_open())
        {
            for (uint32_t i=0; i<flows_.size(); i++)
            {
                int64_t pos_old = flows_[i].pos();
                flows_[i].seek(0);

                std::vector<uint8_t> bufferChunk;
                bufferChunk.resize(pos_old);
                flows_[i].readData(bufferChunk.data(), pos_old);
                myFile.write(reinterpret_cast<char_t *>(bufferChunk.data()), pos_old);

                if (i == flows_.size() - 1)
                {
                    flows_[i].seek(pos_old);
                }
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

            for (uint32_t i=0; i<flows.size(); i++)
            {
                int64_t pos_old = flows[i]->pos();
                flows[i]->seek(0);

                /* Dealing with version flag, constants, header, and descriptor.
                It makes the reasonable assumption that it does not overlap on several chunks. */
                if (i == 0)
                {
                    int64_t header_version_length = sizeof(int32_t);
                    flows[i]->seek(header_version_length); // Skip the version flag
                    std::vector<char_t> headerCharBuffer;
                    headerCharBuffer.resize(headerSize - header_version_length);
                    flows[i]->readData(headerCharBuffer.data(), headerSize - header_version_length);
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
                }

                /* Dealing with data lines, starting with new line flag, time, integers, and ultimately floats. */
                if (recordedBytesDataLine > 0)
                {
                    uint32_t numberLines = (flows[i]->size() - flows[i]->pos()) / recordedBytesDataLine;
                    timestamps.reserve(timestamps.size() + numberLines);
                    intData.reserve(intData.size() + numberLines);
                    floatData.reserve(floatData.size() + numberLines);
                }

                while (flows[i]->bytesAvailable() > 0)
                {
                    flows[i]->seek(flows[i]->pos() + START_LINE_TOKEN.size()); // Skip new line flag
                    flows[i]->readData(reinterpret_cast<uint8_t *>(&timestamp), sizeof(int32_t));
                    flows[i]->readData(reinterpret_cast<uint8_t *>(intDataLine.data()), integerSectionSize);
                    flows[i]->readData(reinterpret_cast<uint8_t *>(floatDataLine.data()), floatSectionSize);

                    if (!timestamps.empty() && timestamp == 0)
                    {
                        // The buffer is not full, must stop reading !
                        break;
                    }

                    timestamps.emplace_back(static_cast<float64_t>(timestamp * 1e-6));
                    intData.emplace_back(intDataLine);
                    floatData.emplace_back(floatDataLine);
                }

                if (i == flows.size() - 1)
                {
                    flows[i]->seek(pos_old);
                }
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