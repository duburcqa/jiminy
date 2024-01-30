#ifndef JIMINY_TELEMETRY_DATA_H
#define JIMINY_TELEMETRY_DATA_H

#include "jiminy/core/fwd.h"


namespace jiminy
{
    /// \brief This class manages the data structures of the telemetry.
    class JIMINY_DLLAPI TelemetryData
    {
    public:
        DISABLE_COPY(TelemetryData)

    public:
        explicit TelemetryData() = default;

        /// \brief Reset the telemetry before starting to use the telemetry.
        void reset() noexcept;

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
        static_map_t<std::string, T, false> * getRegistry();

    private:
        // Must use dequeue to preserve pointer addresses after resize

        /// \brief Memory to handle constants.
        static_map_t<std::string, std::string, false> constantsRegistry_{};
        /// \brief Memory to handle integers.
        static_map_t<std::string, int64_t, false> integersRegistry_{};
        /// \brief Memory to handle floats.
        static_map_t<std::string, double, false> floatsRegistry_{};
        /// \brief Whether registering is available.
        bool isRegisteringAvailable_{true};
    };
}

#include "jiminy/core/telemetry/telemetry_data.hxx"

#endif  // JIMINY_TELEMETRY_DATA_H
