#ifndef JIMINY_TELEMETRY_CLIENT_CLASS_H
#define JIMINY_TELEMETRY_CLIENT_CLASS_H

#include <string>
#include <memory>
#include <variant>
#include <unordered_map>

#include "jiminy/core/fwd.h"


namespace jiminy
{
    class TelemetryData;

    /// \brief Class to inherit for sending telemetry data.
    class JIMINY_DLLAPI TelemetrySender
    {
    public:
        JIMINY_DISABLE_COPY(TelemetrySender)

    public:
        template<typename T>
        using telemetry_data_pair_t = std::pair<const T * const, T * const>;

        template<typename... T>
        using telemetry_data_registry_t = std::vector<std::variant<telemetry_data_pair_t<T>...>>;

        explicit TelemetrySender() = default;
        explicit TelemetrySender(TelemetrySender &&) = default;

        /// \brief Configure the object.
        ///
        /// \remarks Should only be used when default constructor is called for delayed
        ///          configuration. Should be set before registering any entry.
        ///
        /// \param[in] telemetryData Shared pointer to the telemetry instance.
        /// \param[in] name Name of the object.
        void configure(std::shared_ptr<TelemetryData> telemetryData,
                       const std::string_view & name);

        /// \brief Register a new variable to the telemetry.
        ///
        /// \details A variable must be registered to be taken into account by the telemetry. The
        ///          user is responsible for managing its lifetime and updating it in-place. The
        ///          telemetry sender will fetch its value when calling 'updateValues'.
        ///
        /// \param[in] name Name of the field to record in the telemetry.
        /// \param[in] value Pointer to the corresponding scalar value.
        template<typename Scalar>
        std::enable_if_t<std::is_arithmetic_v<Scalar>, void>
        registerVariable(const std::string & name, const Scalar * valuePtr);

        template<typename KeyType, typename Derived>
        void registerVariable(const std::vector<KeyType> & fieldnames,
                              const Eigen::MatrixBase<Derived> & values);

        /// \brief Register a constant (so-called invariant) to the telemetry.
        ///
        /// \details The user is responsible to convert it as a byte array (eg `std::string`).
        ///
        /// \param[in] name Name of the constant.
        /// \param[in] value Value of the constant.
        void registerConstant(const std::string & name, const std::string & value);

        /// \brief Update all registered variables in the telemetry buffer.
        void updateValues();

        /// \brief The number of registered entries.
        uint32_t getNumEntries() const;

        /// \brief The object name.
        const std::string & getName() const;

    protected:
        /// \brief Name of the logged object.
        std::string objectName_{"Uninitialized Object"};

    private:
        std::shared_ptr<TelemetryData> telemetryData_{nullptr};
        /// \brief Associate each variable pointer provided by the user to their reserved position
        ///        in the contiguous storage of telemetry data.
        telemetry_data_registry_t<double, int64_t> bufferPosition_{};
    };
}

#include "jiminy/core/telemetry/telemetry_sender.hxx"

#endif  // JIMINY_TELEMETRY_CLIENT_CLASS_H
