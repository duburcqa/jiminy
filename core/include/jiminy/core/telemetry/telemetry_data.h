#ifndef JIMINY_TELEMETRY_DATA_H
#define JIMINY_TELEMETRY_DATA_H

#include <deque>

#include "jiminy/core/macros.h"
#include "jiminy/core/types.h"


namespace jiminy
{
    /// \brief Version of the telemetry format.
    const int32_t TELEMETRY_VERSION = 1;
    /// \brief Number of integers in the data section.
    const std::string NUM_INTS("NumIntEntries=");
    /// \brief Number of floats in the data section.
    const std::string NUM_FLOATS("NumFloatEntries=");
    /// \brief Special column
    const std::string GLOBAL_TIME("Global.Time");
    /// \brief Special constant
    const std::string TIME_UNIT("Global.TIME_UNIT");
    /// \brief Marker of the beginning the constants section.
    const std::string START_CONSTANTS("StartConstants");
    /// \brief Marker of the beginning the columns section.
    const std::string START_COLUMNS("StartColumns");
    /// \brief Marker of the beginning of a line of data.
    const std::string START_LINE_TOKEN("StartLine");
    /// \brief Marker of the beginning of the data section.
    const std::string START_DATA("StartData");

    /// \brief This class manages the data structures of the telemetry.
    class JIMINY_DLLAPI TelemetryData
    {
    public:
        DISABLE_COPY(TelemetryData)

    public:
        TelemetryData();
        ~TelemetryData() = default;

        /// \brief Reset the telemetry before starting to use the telemetry.
        void reset();

        /// \brief Register a new variable in for telemetry.
        ///
        /// \warning The only supported types are int64_t and float64_t.
        ///
        /// \param[in] variableNameIn Name of the variable to register.
        /// \param[out] positionInBufferOut Pointer on the allocated buffer holding the variable.
        ///
        /// \return S_OK if successful, the corresponding telemetry error otherwise.
        template<typename T>
        hresult_t registerVariable(const std::string & variableNameIn, T *& positionInBufferOut);

        /// \brief Register a constant for the telemetry.
        ///
        /// \param[in] invariantNameIn Name of the invariant.
        /// \param[in] valueIn Value of the invariant.
        ///
        /// \return S_OK if successful, the corresponding telemetry error otherwise.
        hresult_t registerConstant(const std::string & invariantNameIn,
                                   const std::string & valueIn);

        /// \brief Format the telemetry header with the current recorded informations.
        ///
        /// \warning Calling this method will disable further registrations.
        ///
        /// \param[out] header Header to populate.
        void formatHeader(std::vector<char_t> & header);

        template<typename T>
        std::deque<std::pair<std::string, T>> * getRegistry();

    private:
        // Must use dequeue to preserve pointer addresses after resize

        /// \brief Memory to handle constants.
        std::deque<std::pair<std::string, std::string>> constantsRegistry_;
        /// \brief Memory to handle integers.
        std::deque<std::pair<std::string, int64_t>> integersRegistry_;
        /// \brief Memory to handle floats.
        std::deque<std::pair<std::string, float64_t>> floatsRegistry_;
        /// \brief Whether registering is available.
        bool_t isRegisteringAvailable_;
    };
}

#include "jiminy/core/telemetry/telemetry_data.hxx"

#endif  // JIMINY_TELEMETRY_DATA_H
