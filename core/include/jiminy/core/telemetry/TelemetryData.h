///////////////////////////////////////////////////////////////////////////////
///
/// \brief   Manage the data structures of the telemetry.
///
///////////////////////////////////////////////////////////////////////////////

#ifndef JIMINY_TELEMETRY_DATA_H
#define JIMINY_TELEMETRY_DATA_H

#include <iostream>
#include <deque>

#include "jiminy/core/telemetry/TelemetrySender.h"
#include "jiminy/core/Macros.h"
#include "jiminy/core/Types.h"


namespace jiminy
{
    int32_t     const TELEMETRY_VERSION = 1;              ///< Version of the telemetry format.
    std::string const NUM_INTS("NumIntEntries=");         ///< Number of integers in the data section.
    std::string const NUM_FLOATS("NumFloatEntries=");     ///< Number of floats in the data section.
    std::string const GLOBAL_TIME("Global.Time");         ///< Special column
    std::string const TIME_UNIT("Global.TIME_UNIT");      ///< Special constant
    std::string const START_CONSTANTS("StartConstants");  ///< Marker of the beginning the constants section.
    std::string const START_COLUMNS("StartColumns");      ///< Marker of the beginning the columns section.
    std::string const START_LINE_TOKEN("StartLine");      ///< Marker of the beginning of a line of data.
    std::string const START_DATA("StartData");            ///< Marker of the beginning of the data section.

    ////////////////////////////////////////////////////////////////////////
    /// \class TelemetryData
    /// \brief Manage the telemetry buffers.
    ////////////////////////////////////////////////////////////////////////
    class TelemetryData
    {
    public:
        // Disable the copy of the class
        TelemetryData(TelemetryData const &) = delete;
        TelemetryData & operator=(TelemetryData const &) = delete;

    public:
        TelemetryData(void);
        ~TelemetryData(void) = default;

        ////////////////////////////////////////////////////////////////////////
        /// \brief Reset the telemetry before starting to use the telemetry.
        ////////////////////////////////////////////////////////////////////////
        void reset(void);

        ////////////////////////////////////////////////////////////////////////
        /// \brief Register a new variable in for telemetry.
        /// \warning The only supported types are int64_t and float64_t.
        ///
        /// \param[in]  variableNameIn       Name of the variable to register.
        /// \param[out] positionInBufferOut  Pointer on the allocated buffer that will hold the variable.
        ///
        /// \return S_OK if successful, the corresponding telemetry error otherwise.
        ////////////////////////////////////////////////////////////////////////
        template<typename T>
        hresult_t registerVariable(std::string const   & variableNameIn,
                                   T                 * & positionInBufferOut);

        ////////////////////////////////////////////////////////////////////////
        /// \brief Register a constant for the telemetry.
        ///
        /// \param[in] invariantNameIn  Name of the invariant.
        /// \param[in] valueIn          Value of the invariant.
        ///
        /// \return S_OK if successful, the corresponding telemetry error otherwise.
        ////////////////////////////////////////////////////////////////////////
        hresult_t registerConstant(std::string const & invariantNameIn,
                                   std::string const & valueIn);

        ////////////////////////////////////////////////////////////////////////
        /// \brief Format the telemetry header with the current recorded informations.
        /// \warning Calling this method will disable further registrations.
        ///
        /// \param[out] header  header to populate.
        ////////////////////////////////////////////////////////////////////////
        void formatHeader(std::vector<char_t> & header);

        template<typename T>
        std::deque<std::pair<std::string, T> > * getRegistry(void);

    private:
        // Must use dequeue to preserve pointer addresses after resize
        std::deque<std::pair<std::string, std::string> > constantsRegistry_;  ///< Memory to handle constants
        std::deque<std::pair<std::string, int64_t> > integersRegistry_;       ///< Memory to handle integers
        std::deque<std::pair<std::string, float64_t> > floatsRegistry_;       ///< Memory to handle floats
        bool_t isRegisteringAvailable_;                                       ///< Whether registering is available
    };
} // namespace jiminy

#include "TelemetryData.tpp"

#endif // JIMINY_TELEMETRY_DATA_H