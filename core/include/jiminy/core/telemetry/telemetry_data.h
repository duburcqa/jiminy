#ifndef JIMINY_TELEMETRY_DATA_H
#define JIMINY_TELEMETRY_DATA_H

#include <deque>

#include "jiminy/core/fwd.h"


namespace jiminy
{
    /// \brief Version of the telemetry format.
    inline constexpr int32_t TELEMETRY_VERSION{1};
    /// \brief Number of integers in the data section.
    inline constexpr std::string_view NUM_INTS{"NumIntEntries"};
    /// \brief Number of floats in the data section.
    inline constexpr std::string_view NUM_FLOATS{"NumFloatEntries"};
    /// \brief Marker of the beginning the constants section.
    inline constexpr std::string_view START_CONSTANTS{"StartConstants"};
    /// \brief Marker of the beginning the columns section.
    inline constexpr std::string_view START_COLUMNS{"StartColumns"};
    /// \brief Marker of the beginning of a line of data.
    inline constexpr std::string_view START_LINE_TOKEN{"StartLine"};
    /// \brief Marker of the beginning of the data section.
    inline constexpr std::string_view START_DATA{"StartData"};

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
        /// \warning The only supported types are int64_t and double.
        ///
        /// \param[in] name Name of the variable to register.
        /// \param[out] positionInBuffer Pointer on the allocated buffer holding the variable.
        ///
        /// \return S_OK if successful, the corresponding telemetry error otherwise.
        template<typename T>
        hresult_t registerVariable(const std::string & name, T *& positionInBuffer);

        /// \brief Register a constant for the telemetry.
        ///
        /// \param[in] name Name of the invariant.
        /// \param[in] value Value of the invariant.
        ///
        /// \return S_OK if successful, the corresponding telemetry error otherwise.
        hresult_t registerConstant(const std::string & name, const std::string & value);

        /// \brief Format the telemetry header with the current recorded informations.
        ///
        /// \warning Calling this method will disable further registrations.
        ///
        /// \param[out] header Header to populate.
        void formatHeader(std::vector<char> & header);

        template<typename T>
        std::deque<std::pair<std::string, T>> * getRegistry();

    private:
        // Must use dequeue to preserve pointer addresses after resize

        /// \brief Memory to handle constants.
        std::deque<std::pair<std::string, std::string>> constantsRegistry_;
        /// \brief Memory to handle integers.
        std::deque<std::pair<std::string, int64_t>> integersRegistry_;
        /// \brief Memory to handle floats.
        std::deque<std::pair<std::string, double>> floatsRegistry_;
        /// \brief Whether registering is available.
        bool isRegisteringAvailable_;
    };
}

#include "jiminy/core/telemetry/telemetry_data.hxx"

#endif  // JIMINY_TELEMETRY_DATA_H
