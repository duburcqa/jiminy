#ifndef JIMINY_TELEMETRY_DATA_H
#define JIMINY_TELEMETRY_DATA_H

#include "jiminy/core/fwd.h"


namespace jiminy
{
    /// \brief This class manages the data structures of the telemetry.
    class JIMINY_DLLAPI TelemetryData
    {
    public:
        JIMINY_DISABLE_COPY(TelemetryData)

    public:
        explicit TelemetryData() = default;

        /// \brief Reset the telemetry before starting to use the telemetry.
        void reset() noexcept;

        /// \brief Register a new variable in for telemetry.
        ///
        /// \warning The only supported types are int64_t and double.
        ///
        /// \param[in] name Name of the variable to register.
        ///
        /// \return Pointer on the allocated buffer holding the variable.
        template<typename T>
        T * registerVariable(const std::string & name);

        /// \brief Register a constant (so-called invariant) to the telemetry.
        ///
        /// \details The user is responsible to convert it as a byte array (eg `std::string`).
        ///
        /// \param[in] name Name of the constant.
        /// \param[in] value Value of the constant.
        void registerConstant(const std::string & name, const std::string & value);

        /// \brief Format the telemetry header with the current recorded informations.
        ///
        /// \warning Calling this method will disable further registrations.
        ///
        /// \param[out] header Header to populate.
        std::vector<char> formatHeader();

        template<typename T>
        static_map_t<std::string, T, false> * getRegistry();

    private:
        // Must use dequeue to preserve pointer addresses after resize

        /// \brief Memory to handle constants.
        static_map_t<std::string, std::string, false> constantRegistry_{};
        /// \brief Memory to handle integers.
        static_map_t<std::string, int64_t, false> integerRegistry_{};
        /// \brief Memory to handle floats.
        static_map_t<std::string, double, false> floatRegistry_{};
        /// \brief Whether registering is available.
        bool isRegisteringAvailable_{true};
    };
}

#include "jiminy/core/telemetry/telemetry_data.hxx"

#endif  // JIMINY_TELEMETRY_DATA_H
